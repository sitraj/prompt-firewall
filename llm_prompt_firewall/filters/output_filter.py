"""
Output Filter
==============

The OutputFilter inspects LLM responses for data leakage before they are
returned to the caller. It is the last line of defence — even when a prompt
passes the input firewall cleanly, the LLM may still leak sensitive context.

What it detects:

  1. SECRETS IN RESPONSES
     API keys, JWT tokens, private keys, passwords, and other credentials
     matching known format signatures. A curated library of 15+ secret
     pattern types covers the most common credential formats.

  2. SYSTEM PROMPT ECHOING
     When the response contains verbatim or near-verbatim content from the
     system prompt. Detected via SHA-256 fragment matching (fast, exact) and
     optionally via embedding similarity (catches paraphrased leaks).
     The system prompt is never passed to the filter directly — only its
     SHA-256 hash is used (consistent with the no-raw-prompt-in-logs policy).

  3. EXFILTRATION VECTORS
     Email addresses and URLs in LLM responses that arrive in a session
     flagged as suspicious. A URL in a response to a benign factual query
     is normal; a URL in a response to a prompt the input firewall flagged
     with risk > 0.40 may be a successful exfiltration attempt.

Actions:
  REDACT  — Replace detected secrets with type-specific placeholders.
             The response is returned to the caller with redactions.
  BLOCK   — When system prompt echoing is detected or exfiltration vectors
             appear in high-risk context. The response is suppressed entirely.

Design decisions:

  - PATTERN-FIRST, NOT MODEL-FIRST
    Secret detection uses compiled regex patterns, not another LLM call.
    Adding a second LLM call to the output path would double latency for
    every request. Regex patterns are faster, deterministic, and auditable.

  - REDACTED SAMPLES IN AUDIT LOGS, NOT FULL SECRETS
    When a secret is found, the audit log stores a masked preview
    (e.g. 'AKIA***...***ABCD') not the full value. This ensures that
    secrets in LLM responses are never written to logs in plaintext.

  - CONTEXT-AWARE EXFILTRATION DETECTION
    A URL in a response is not inherently suspicious. The filter only
    flags exfiltration vectors when the input risk score exceeded the
    log threshold (0.40). This prevents every helpful response containing
    a link from being flagged.
"""

from __future__ import annotations

import hashlib
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Final

from llm_prompt_firewall.models.schemas import (
    FirewallAction,
    OutputInspectionResult,
    SecretMatch,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Secret pattern library
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SecretPattern:
    """A compiled pattern for detecting a specific credential type."""

    secret_type: str
    pattern: re.Pattern[str]
    severity: float  # [0.0–1.0] — how bad a leak of this type is
    redaction_label: str  # replacement string in redacted output
    mask_preview_fn: Callable[..., str] | None = None  # optional: how to mask for logs


def _mask_middle(value: str, keep: int = 4) -> str:
    """Show first and last `keep` chars, mask the rest."""
    if len(value) <= keep * 2:
        return "*" * len(value)
    return value[:keep] + "***...***" + value[-keep:]


# All patterns use re.IGNORECASE where appropriate.
# Patterns ordered roughly by specificity (most specific first to reduce
# false positives from greedy overlap).
SECRET_PATTERNS: list[SecretPattern] = [
    SecretPattern(
        secret_type="openai_api_key",
        pattern=re.compile(r"sk-(?:proj-)?[A-Za-z0-9_\-]{40,60}\b"),
        severity=1.0,
        redaction_label="[OPENAI_API_KEY_REDACTED]",
    ),
    SecretPattern(
        secret_type="anthropic_api_key",
        pattern=re.compile(r"sk-ant-[A-Za-z0-9_\-]{80,120}\b"),
        severity=1.0,
        redaction_label="[ANTHROPIC_API_KEY_REDACTED]",
    ),
    SecretPattern(
        secret_type="aws_access_key_id",
        pattern=re.compile(r"\b(AKIA|ASIA|AROA|AIDA|ANPA|ANVA|APKA)[A-Z0-9]{16}\b"),
        severity=1.0,
        redaction_label="[AWS_ACCESS_KEY_REDACTED]",
    ),
    SecretPattern(
        secret_type="aws_secret_access_key",
        # 40 base64 chars following common context keywords
        pattern=re.compile(
            r"(?i)(?:aws.?secret.?(?:access.?)?key|SecretAccessKey)\s*[=:\"'\s]+([A-Za-z0-9/+]{40})\b"
        ),
        severity=1.0,
        redaction_label="[AWS_SECRET_KEY_REDACTED]",
    ),
    SecretPattern(
        secret_type="github_token",
        pattern=re.compile(r"(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{36}\b"),
        severity=1.0,
        redaction_label="[GITHUB_TOKEN_REDACTED]",
    ),
    SecretPattern(
        secret_type="github_pat",
        pattern=re.compile(r"github_pat_[A-Za-z0-9_]{59}\b"),
        severity=1.0,
        redaction_label="[GITHUB_PAT_REDACTED]",
    ),
    SecretPattern(
        secret_type="jwt_token",
        # Header.Payload.Signature — all base64url encoded
        pattern=re.compile(r"\beyJ[A-Za-z0-9\-_]+\.[A-Za-z0-9\-_]+\.[A-Za-z0-9\-_\.+/=]*\b"),
        severity=0.90,
        redaction_label="[JWT_TOKEN_REDACTED]",
    ),
    SecretPattern(
        secret_type="rsa_private_key",
        pattern=re.compile(
            r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
            re.IGNORECASE,
        ),
        severity=1.0,
        redaction_label="[PRIVATE_KEY_REDACTED]",
    ),
    SecretPattern(
        secret_type="google_api_key",
        pattern=re.compile(r"\bAIza[0-9A-Za-z\-_]{35}\b"),
        severity=0.95,
        redaction_label="[GOOGLE_API_KEY_REDACTED]",
    ),
    SecretPattern(
        secret_type="stripe_secret_key",
        pattern=re.compile(r"\bsk_(?:live|test)_[0-9A-Za-z]{24,99}\b"),
        severity=1.0,
        redaction_label="[STRIPE_SECRET_KEY_REDACTED]",
    ),
    SecretPattern(
        secret_type="stripe_publishable_key",
        pattern=re.compile(r"\bpk_(?:live|test)_[0-9A-Za-z]{24,99}\b"),
        severity=0.60,
        redaction_label="[STRIPE_PUBLIC_KEY_REDACTED]",
    ),
    SecretPattern(
        secret_type="slack_token",
        pattern=re.compile(r"\bxox[baprs]-[0-9A-Za-z\-]{10,50}\b"),
        severity=0.90,
        redaction_label="[SLACK_TOKEN_REDACTED]",
    ),
    SecretPattern(
        secret_type="generic_password_field",
        # Matches: password = "...", password: "..." etc.
        pattern=re.compile(
            r'(?i)(?:password|passwd|secret|token|api.?key)\s*[=:]\s*["\']([^"\']{8,})["\']'
        ),
        severity=0.80,
        redaction_label="[CREDENTIAL_REDACTED]",
    ),
    SecretPattern(
        secret_type="email_address",
        # Broad email pattern — used for exfiltration vector detection, lower severity
        pattern=re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"),
        severity=0.30,  # Low severity alone; elevated when combined with high input risk
        redaction_label="[EMAIL_REDACTED]",
    ),
    SecretPattern(
        secret_type="url_with_query",
        # URLs with query params (potential exfiltration endpoint)
        pattern=re.compile(r"https?://[A-Za-z0-9.\-_~:/?#\[\]@!$&'()*+,;=%]{20,}"),
        severity=0.20,  # Low severity alone; elevated in high-risk context
        redaction_label="[URL_REDACTED]",
    ),
]

# Index by type for O(1) lookup
_SECRET_PATTERN_INDEX: dict[str, SecretPattern] = {p.secret_type: p for p in SECRET_PATTERNS}

# Severity threshold above which a secret match triggers BLOCK (not just REDACT)
_BLOCK_SEVERITY_THRESHOLD: Final[float] = 0.85

# Input risk score above which exfiltration vectors (email, URLs) become suspicious
_EXFILTRATION_RISK_THRESHOLD: Final[float] = 0.40


# ---------------------------------------------------------------------------
# OutputFilter
# ---------------------------------------------------------------------------


class OutputFilter:
    """
    Inspects LLM responses for credential leakage, system prompt echoing,
    and exfiltration vectors.

    Thread-safe: all state is read-only after construction (compiled patterns
    are shared). The filter is designed to be a long-lived singleton, created
    once at firewall startup.

    Usage:
        filt = OutputFilter()
        result = filt.inspect(
            response_text=llm_response,
            input_risk_score=0.72,
            system_prompt_hash=my_system_prompt_sha256,
        )
        if not result.clean:
            safe_text = filt.redact(llm_response, result)
    """

    def __init__(
        self,
        patterns: list[SecretPattern] | None = None,
        block_severity_threshold: float = _BLOCK_SEVERITY_THRESHOLD,
        exfiltration_risk_threshold: float = _EXFILTRATION_RISK_THRESHOLD,
    ) -> None:
        self._patterns = patterns or SECRET_PATTERNS
        self._block_severity = block_severity_threshold
        self._exfil_threshold = exfiltration_risk_threshold

    def inspect(
        self,
        response_text: str,
        input_risk_score: float = 0.0,
        system_prompt_hash: str | None = None,
        response_contains_system_prompt_fragments: bool = False,
    ) -> OutputInspectionResult:
        """
        Inspect an LLM response for policy violations.

        Args:
            response_text: The raw LLM response to inspect.
            input_risk_score: The risk score assigned to the input prompt by
                              the firewall. Used to contextualise low-severity
                              exfiltration signals (email, URLs).
            system_prompt_hash: SHA-256 of the system prompt. If a fragment
                                 of the response hashes to this value, system
                                 prompt echoing is detected.
            response_contains_system_prompt_fragments: Pre-computed flag from
                                 an embedding-based system prompt similarity
                                 check (optional, set by PromptAnalyzer).

        Returns:
            OutputInspectionResult. If `clean=False`, call redact() to obtain
            a response safe to return to the caller.
        """
        import time

        start = time.perf_counter()

        secret_matches: list[SecretMatch] = []
        high_severity_found = False

        # --- Secret detection ---
        for sp in self._patterns:
            for m in sp.pattern.finditer(response_text):
                # Always start with the full match for the length/offset record.
                full_match = m.group(0)
                matched_value = full_match

                # For group-capturing patterns (like generic_password_field),
                # group(1) is the credential value itself (e.g. the password after
                # the "password =" prefix). Use it for masking only if it is
                # long enough to be the actual secret — short groups (e.g. the
                # 4-char "AKIA" prefix in aws_access_key_id) are not the secret.
                try:
                    captured = m.group(1)
                    if captured and len(captured) >= 6:
                        matched_value = captured
                except IndexError:
                    pass

                # Skip very short matches from broad patterns (noise reduction)
                if len(matched_value) < 6:
                    continue

                masked = _mask_middle(matched_value)
                secret_matches.append(
                    SecretMatch(
                        secret_type=sp.secret_type,
                        pattern_id=f"secret:{sp.secret_type}",
                        offset=m.start(),
                        redacted_sample=masked,
                        severity=sp.severity,
                    )
                )

                if sp.severity >= self._block_severity:
                    high_severity_found = True

                logger.warning(
                    "OutputFilter: detected %s in response (offset=%d, preview=%s)",
                    sp.secret_type,
                    m.start(),
                    masked,
                )

        # --- System prompt echo detection ---
        system_prompt_echo = response_contains_system_prompt_fragments
        if system_prompt_hash and not system_prompt_echo:
            system_prompt_echo = _detect_system_prompt_echo(response_text, system_prompt_hash)

        if system_prompt_echo:
            logger.warning("OutputFilter: possible system prompt echo detected in response.")

        # --- Exfiltration vector detection ---
        # Only flag emails/URLs as exfiltration when the input was already suspicious.
        # Email/URL patterns run in the main secret-detection loop above, so they
        # are already present in secret_matches. Elevate them to exfiltration status
        # when the input risk score crosses the threshold.
        _EXFIL_VECTOR_TYPES: frozenset[str] = frozenset({"email_address", "url_with_query"})
        exfil_detected = False
        if input_risk_score >= self._exfil_threshold:
            # Check whether email/URL were caught by the main secret-detection loop.
            exfil_detected = any(sm.secret_type in _EXFIL_VECTOR_TYPES for sm in secret_matches)
            if not exfil_detected:
                # Fallback for custom pattern sets that may not include the default
                # email/URL patterns: run _detect_exfiltration_vectors directly.
                exfil_detected = _detect_exfiltration_vectors(response_text, secret_matches)
            if exfil_detected:
                logger.warning(
                    "OutputFilter: exfiltration vector detected in high-risk context "
                    "(input_risk_score=%.2f).",
                    input_risk_score,
                )

        # --- Determine recommended action ---
        clean = not secret_matches and not system_prompt_echo and not exfil_detected

        if system_prompt_echo or high_severity_found:
            recommended = FirewallAction.BLOCK
        elif secret_matches or exfil_detected:
            recommended = FirewallAction.SANITIZE  # SANITIZE = redact in output context
        else:
            recommended = FirewallAction.ALLOW

        processing_ms = (time.perf_counter() - start) * 1000

        return OutputInspectionResult(
            clean=clean,
            secret_matches=secret_matches,
            system_prompt_echo_detected=system_prompt_echo,
            exfiltration_vector_detected=exfil_detected,
            recommended_action=recommended,
            processing_time_ms=round(processing_ms, 3),
        )

    def redact(
        self,
        response_text: str,
        result: OutputInspectionResult,
    ) -> tuple[str, list[str]]:
        """
        Redact all detected secrets from the response text.

        Processes matches in reverse offset order to preserve earlier offsets.
        Returns (redacted_text, list_of_redaction_descriptions).

        Note: This method re-runs pattern matching rather than using stored
        offsets because the OutputInspectionResult is immutable and offsets
        may be stale if the response was modified between inspect() and redact().
        """
        redacted = response_text
        redactions: list[str] = []

        for sp in self._patterns:
            if not any(sm.secret_type == sp.secret_type for sm in result.secret_matches):
                continue

            def _replace(
                m: re.Match[str],
                label: str = sp.redaction_label,
                stype: str = sp.secret_type,
            ) -> str:
                redactions.append(
                    f"Redacted {stype} at offset {m.start()} ({_mask_middle(m.group(0))})."
                )
                return label

            redacted = sp.pattern.sub(_replace, redacted)

        if result.system_prompt_echo_detected:
            redactions.append("Response suppressed: system prompt echo detected.")

        return redacted, redactions


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _detect_system_prompt_echo(response: str, system_prompt_hash: str) -> bool:
    """
    Check whether any 512-character sliding window of the response has a
    SHA-256 hash that matches the system prompt hash.

    This is a fast, exact check for verbatim system prompt repetition.
    It does NOT catch paraphrased leaks — that requires an embedding comparison
    which is handled optionally by the PromptAnalyzer.

    Window size 512 chars balances sensitivity (smaller = more false positives
    from coincidental hash matches) and specificity (larger = misses short
    system prompts).
    """
    window = 512
    step = 128
    for i in range(0, max(1, len(response) - window + 1), step):
        fragment = response[i : i + window]
        fragment_hash = hashlib.sha256(fragment.encode("utf-8")).hexdigest()
        if fragment_hash == system_prompt_hash:
            return True
    return False


def _detect_exfiltration_vectors(
    response: str,
    secret_matches: list[SecretMatch],
) -> bool:
    """
    Return True if the response contains email addresses or suspicious URLs
    that are NOT already captured as higher-severity secret matches.

    The caller is responsible for only calling this when input_risk_score
    exceeds the exfiltration threshold.
    """
    already_matched_types = {sm.secret_type for sm in secret_matches}

    # If email or URL was already caught as a secret match, don't double-flag
    email_pattern = _SECRET_PATTERN_INDEX.get("email_address")
    url_pattern = _SECRET_PATTERN_INDEX.get("url_with_query")

    if (
        email_pattern
        and "email_address" not in already_matched_types
        and email_pattern.pattern.search(response)
    ):
        return True

    return bool(
        url_pattern
        and "url_with_query" not in already_matched_types
        and url_pattern.pattern.search(response)
    )
