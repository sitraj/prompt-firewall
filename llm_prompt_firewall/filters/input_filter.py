"""
Input Filter
=============

The InputFilter is the sanitization stage applied to a prompt AFTER the
detection pipeline has run and the PolicyEngine has decided the action is
SANITIZE (rather than BLOCK or ALLOW-as-is).

Responsibility:
  - Strip or redact the specific injection phrases that were matched.
  - Remove invisible/zero-width characters used in evasion.
  - Apply unicode normalisation before the prompt is forwarded to the LLM.
  - Record every transformation made, so the audit trail is complete.

What it does NOT do:
  - Make security decisions (that is the PolicyEngine's job).
  - Inspect output (that is the OutputFilter's job).
  - Detect attacks (that is the detector pipeline's job).

The filter takes the raw prompt and the PatternSignal from the detection
pipeline. Matched injection phrases are replaced with the REDACTION_MARKER.
This preserves the overall structure of the prompt (sentence lengths, context
tokens) while neutralising the adversarial payload.

Design decision — why replace rather than delete?
  Replacing a matched phrase with a marker rather than deleting it prevents
  the surrounding benign text from accidentally merging into a new attack
  phrase. It also signals clearly in the audit log what was stripped, rather
  than silently shortening the prompt.
"""

from __future__ import annotations

import logging
import unicodedata
from dataclasses import dataclass, field

from llm_prompt_firewall.models.schemas import (
    PatternSignal,
    PromptContext,
    SanitizedPrompt,
)
from llm_prompt_firewall.policy.policy_engine import SanitizationPolicy

logger = logging.getLogger(__name__)

# Marker inserted where an injection phrase was stripped.
REDACTION_MARKER: str = "[REDACTED]"

# Characters stripped in invisible-char removal (mirrors the pattern detector's
# list so both layers are consistent).
_INVISIBLE_CHARS: frozenset[str] = frozenset([
    "\u200b", "\u200c", "\u200d", "\u00ad", "\u2060",
    "\ufeff", "\u202a", "\u202b", "\u202c", "\u202d",
    "\u202e", "\u2066", "\u2067", "\u2069",
])


@dataclass(frozen=True)
class InputFilterResult:
    """
    Output of InputFilter.sanitize().

    Contains the sanitized text ready for forwarding to the LLM, plus a
    full record of every transformation applied, for audit logging.
    """
    sanitized_text: str
    original_sha256: str
    modifications: list[str]
    chars_removed: int

    def to_sanitized_prompt(self) -> SanitizedPrompt:
        return SanitizedPrompt(
            sanitized_text=self.sanitized_text,
            original_sha256=self.original_sha256,
            modifications=list(self.modifications),
            chars_removed=self.chars_removed,
        )


class InputFilter:
    """
    Sanitizes prompts flagged for SANITIZE action by the PolicyEngine.

    The filter is stateless and thread-safe. All configuration is read from
    the SanitizationPolicy passed at construction time.

    Usage:
        policy = SanitizationPolicy()
        filt = InputFilter(policy)
        result = filt.sanitize(prompt_context, pattern_signal)
        safe_text = result.sanitized_text
    """

    def __init__(self, policy: SanitizationPolicy | None = None) -> None:
        self._policy = policy or SanitizationPolicy()

    def sanitize(
        self,
        prompt_context: PromptContext,
        pattern_signal: PatternSignal | None = None,
    ) -> InputFilterResult:
        """
        Apply all configured sanitization operations to the prompt.

        Operations are applied in this order:
          1. Strip invisible characters (if policy.strip_invisible_chars)
          2. Unicode NFKC normalisation (if policy.normalize_unicode)
          3. Redact matched injection phrases (if policy.strip_injection_phrases
             and pattern_signal has matches)

        Args:
            prompt_context: The PromptContext being sanitized.
            pattern_signal: Optional PatternSignal from the detection pipeline.
                            If provided, the matched phrases are redacted.

        Returns:
            InputFilterResult with the sanitized text and audit record.
        """
        original = prompt_context.raw_prompt
        text = original
        original_sha = prompt_context.prompt_sha256()
        modifications: list[str] = []
        original_len = len(text)

        # Step 1: Strip invisible characters
        if self._policy.strip_invisible_chars:
            text, removed_count = _strip_invisible(text)
            if removed_count > 0:
                modifications.append(
                    f"Stripped {removed_count} invisible/zero-width character(s)."
                )

        # Step 2: Unicode normalisation
        if self._policy.normalize_unicode:
            normalised = unicodedata.normalize("NFKC", text)
            if normalised != text:
                modifications.append("Applied NFKC unicode normalisation.")
                text = normalised

        # Step 3: Redact matched injection phrases
        if self._policy.strip_injection_phrases and pattern_signal and pattern_signal.matched:
            text, phrase_mods = _redact_matched_phrases(text, pattern_signal)
            modifications.extend(phrase_mods)

        chars_removed = original_len - len(text)

        if modifications:
            logger.info(
                "InputFilter sanitized prompt %s: %d modification(s), %d chars removed.",
                original_sha[:12],
                len(modifications),
                chars_removed,
            )
        else:
            logger.debug(
                "InputFilter: no modifications needed for prompt %s.",
                original_sha[:12],
            )

        return InputFilterResult(
            sanitized_text=text,
            original_sha256=original_sha,
            modifications=modifications,
            chars_removed=max(0, chars_removed),
        )

    def apply_pre_detection_normalization(self, text: str) -> str:
        """
        Apply lightweight normalization for detection purposes only.

        This is the normalization applied before the DetectorPipeline runs —
        not the full sanitization (which is only applied on SANITIZE action).
        Strips invisible chars and applies NFKC so detectors see clean text,
        but the original is preserved in PromptContext.raw_prompt.
        """
        text, _ = _strip_invisible(text)
        return unicodedata.normalize("NFKC", text)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _strip_invisible(text: str) -> tuple[str, int]:
    """Remove invisible Unicode characters. Returns (cleaned_text, count_removed)."""
    cleaned = "".join(ch for ch in text if ch not in _INVISIBLE_CHARS)
    return cleaned, len(text) - len(cleaned)


def _redact_matched_phrases(
    text: str,
    pattern_signal: PatternSignal,
) -> tuple[str, list[str]]:
    """
    Replace each matched injection phrase with REDACTION_MARKER.

    Processes matches in reverse offset order so earlier redactions do not
    shift the character indices of later matches.

    We redact by matched_text (the actual substring) rather than re-running
    the regex, because:
      1. The matched text is already captured in the PatternSignal.
      2. Re-running regexes on the potentially modified text could produce
         different matches.
      3. Using the captured offset + length is the most precise replacement.
    """
    mods: list[str] = []

    # De-duplicate matches by (offset, matched_text) to avoid double-redaction
    # when multiple pattern rules matched the same substring.
    seen: set[tuple[int, str]] = set()
    unique_matches = []
    for m in pattern_signal.matches:
        key = (m.offset, m.matched_text)
        if key not in seen:
            seen.add(key)
            unique_matches.append(m)

    # Sort descending by offset so replacements don't shift subsequent indices
    sorted_matches = sorted(unique_matches, key=lambda m: m.offset, reverse=True)

    for match in sorted_matches:
        start = match.offset
        end = start + len(match.matched_text)

        # Guard against stale offsets (text may have shifted from prior steps)
        if start >= len(text) or text[start:end] != match.matched_text:
            # Fall back to string replacement (case-insensitive)
            lower_text = text.lower()
            lower_phrase = match.matched_text.lower()
            idx = lower_text.find(lower_phrase)
            if idx == -1:
                logger.debug(
                    "InputFilter: could not locate matched phrase '%s' for redaction — skipping.",
                    match.matched_text[:40],
                )
                continue
            start, end = idx, idx + len(match.matched_text)

        text = text[:start] + REDACTION_MARKER + text[end:]
        mods.append(
            f"Redacted injection phrase '{match.matched_text[:40]}' "
            f"(pattern: {match.pattern_id}, category: {match.category.value})."
        )

    return text, mods
