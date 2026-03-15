"""
PromptFirewall
==============

The top-level SDK entry point. PromptFirewall integrates the full
input-inspection pipeline (PromptAnalyzer) with the output-inspection layer
(OutputFilter) behind a single, stable API that application code calls.

Typical usage:

    from llm_prompt_firewall import PromptFirewall
    from llm_prompt_firewall.models.schemas import PromptContext

    firewall = PromptFirewall.from_default_config()

    # 1. Inspect input before sending to the LLM
    decision = firewall.inspect_input(PromptContext(raw_prompt=user_text))

    if decision.action == FirewallAction.BLOCK:
        return decision.block_reason   # or a generic safe message

    # 2. Call your LLM with decision.effective_prompt
    llm_response = your_llm(decision.effective_prompt)

    # 3. Inspect the LLM response before returning it to the user
    output = firewall.inspect_output(llm_response, decision)

    if isinstance(output, SafeResponse):
        return output.content
    elif isinstance(output, RedactedResponse):
        return output.redacted_content
    else:  # BlockedResponse
        return output.safe_message

Design principles:

  THIN FACADE
    PromptFirewall owns no detection logic. It delegates to PromptAnalyzer
    (input side) and OutputFilter (output side). Adding or replacing a
    detector requires changing only those components; the firewall API
    remains stable.

  AUDIT HOOK
    An optional `audit_logger` callable receives an AuditEvent after every
    inspect_input() call. This decouples observability from the firewall core
    — plug in a Kafka producer, Splunk forwarder, or structured logger without
    touching the firewall code.

  ASYNC-FIRST INPUT, SYNC OUTPUT
    inspect_input_async() runs the full async pipeline. inspect_input() wraps
    it. Output inspection (inspect_output) is intentionally synchronous —
    the OutputFilter is regex-based and sub-millisecond; an async wrapper
    would add overhead with no benefit.

  THREAD-SAFE SINGLETON
    PromptFirewall holds only references to stateless, thread-safe components.
    A single instance can be shared across all request handlers in a
    multi-threaded or async application.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Callable

from llm_prompt_firewall.core.prompt_analyzer import AnalyzerConfig, PromptAnalyzer
from llm_prompt_firewall.filters.output_filter import OutputFilter
from llm_prompt_firewall.models.schemas import (
    AuditEvent,
    BlockedResponse,
    FirewallAction,
    FirewallDecision,
    OutputInspectionResult,
    PromptContext,
    RedactedResponse,
    SafeResponse,
    ThreatCategory,
)

logger = logging.getLogger(__name__)

# Type alias for the optional audit logger hook
AuditLoggerFn = Callable[[AuditEvent], None]

# Union type for inspect_output return values
OutputResult = SafeResponse | BlockedResponse | RedactedResponse


class PromptFirewall:
    """
    Single entry point for LLM prompt injection protection.

    Combines input inspection (PromptAnalyzer) with output inspection
    (OutputFilter) into a two-phase API that wraps any LLM call site.

    Thread-safe: safe to share as a singleton across concurrent requests.

    Usage:
        firewall = PromptFirewall.from_default_config()

        # Input gate
        decision = firewall.inspect_input(PromptContext(raw_prompt=user_text))
        if decision.action == FirewallAction.BLOCK:
            return "Request blocked."

        # LLM call
        response = llm.complete(decision.effective_prompt)

        # Output gate
        result = firewall.inspect_output(response, decision)
        if isinstance(result, SafeResponse):
            return result.content
    """

    def __init__(
        self,
        analyzer: PromptAnalyzer | None = None,
        output_filter: OutputFilter | None = None,
        audit_logger: AuditLoggerFn | None = None,
    ) -> None:
        self._analyzer = analyzer or PromptAnalyzer()
        self._output_filter = output_filter or OutputFilter()
        self._audit_logger = audit_logger

        logger.info(
            "PromptFirewall ready: analyzer=%s, output_filter=%s, audit_logger=%s",
            type(self._analyzer).__name__,
            type(self._output_filter).__name__,
            "configured" if audit_logger else "none",
        )

    # ------------------------------------------------------------------
    # Factory constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_default_config(
        cls,
        audit_logger: AuditLoggerFn | None = None,
    ) -> "PromptFirewall":
        """
        Build a PromptFirewall using the bundled attack dataset and default policy.

        This is the recommended constructor for production deployments.
        The pattern detector is always built; the embedding detector is built
        only if sentence-transformers is installed; the LLM classifier is off
        by default (requires an API key — enable via `from_config()`).

        Args:
            audit_logger: Optional callable invoked after every inspect_input().
                          Receives a fully populated AuditEvent.

        Returns:
            A ready-to-use PromptFirewall instance.
        """
        analyzer = PromptAnalyzer.from_default_config()
        return cls(analyzer=analyzer, audit_logger=audit_logger)

    @classmethod
    def from_config(
        cls,
        config: AnalyzerConfig,
        audit_logger: AuditLoggerFn | None = None,
    ) -> "PromptFirewall":
        """
        Build a PromptFirewall from a custom AnalyzerConfig.

        Use this when you need a non-default dataset, custom policy file,
        or want to enable the LLM classifier.

        Args:
            config:       AnalyzerConfig controlling which detectors to enable
                          and where to find the dataset and policy files.
            audit_logger: Optional audit event callback.

        Returns:
            A configured PromptFirewall instance.
        """
        analyzer = PromptAnalyzer.from_config(config)
        return cls(analyzer=analyzer, audit_logger=audit_logger)

    @classmethod
    def from_config_file(
        cls,
        policy_path: Path,
        audit_logger: AuditLoggerFn | None = None,
    ) -> "PromptFirewall":
        """
        Build a PromptFirewall with a custom policy file.

        Uses the bundled attack dataset and all default detector settings,
        but loads the policy thresholds and rules from the provided YAML file.

        Args:
            policy_path:  Path to a YAML policy file (overrides default policy).
            audit_logger: Optional audit event callback.

        Returns:
            A configured PromptFirewall instance.
        """
        config = AnalyzerConfig(policy_path=policy_path)
        return cls.from_config(config, audit_logger=audit_logger)

    # ------------------------------------------------------------------
    # Input inspection
    # ------------------------------------------------------------------

    async def inspect_input_async(
        self,
        prompt_context: PromptContext,
        *,
        application_id: str | None = None,
        user_id: str | None = None,
        ip_address: str | None = None,
    ) -> FirewallDecision:
        """
        Inspect a user prompt asynchronously.

        Runs the full detection and policy pipeline via PromptAnalyzer.
        Emits an AuditEvent via the configured audit_logger (if any).

        Args:
            prompt_context: The prompt to inspect.
            application_id: Optional application identifier for audit logs.
            user_id:        Optional user identifier — anonymised in audit log.
            ip_address:     Optional source IP — anonymised in audit log.

        Returns:
            FirewallDecision. Check `decision.action` to determine next step:
              ALLOW / LOG → call LLM with `decision.effective_prompt`
              SANITIZE    → call LLM with `decision.effective_prompt` (sanitized)
              BLOCK       → do not call LLM; use `decision.block_reason`
        """
        decision = await self._analyzer.inspect_async(prompt_context)

        if self._audit_logger is not None:
            try:
                event = self._analyzer.build_audit_event(
                    decision,
                    application_id=application_id,
                    user_id=user_id,
                    ip_address=ip_address,
                )
                self._audit_logger(event)
            except Exception as exc:
                # Audit failures must never block the request path
                logger.error(
                    "PromptFirewall: audit_logger raised unexpectedly: %s — "
                    "continuing without audit record for decision %s.",
                    exc,
                    decision.decision_id,
                )

        return decision

    def inspect_input(
        self,
        prompt_context: PromptContext,
        *,
        application_id: str | None = None,
        user_id: str | None = None,
        ip_address: str | None = None,
    ) -> FirewallDecision:
        """
        Synchronous wrapper around inspect_input_async().

        Use inspect_input_async() directly when your application already
        runs in an async context (FastAPI, asyncio).

        Args:
            prompt_context: The prompt to inspect.
            application_id: Optional application identifier for audit logs.
            user_id:        Optional user identifier — anonymised in audit log.
            ip_address:     Optional source IP — anonymised in audit log.

        Returns:
            FirewallDecision.
        """
        return self._analyzer.inspect(prompt_context)
        # Note: audit logging happens in inspect_input_async(). The sync path
        # skips audit logging to avoid async/sync bridging complexity.
        # Callers who need audit logs in sync contexts should call
        # build_audit_event() themselves after inspect_input() returns.

    # ------------------------------------------------------------------
    # Output inspection
    # ------------------------------------------------------------------

    def inspect_output(
        self,
        response_text: str,
        decision: FirewallDecision,
    ) -> OutputResult:
        """
        Inspect an LLM response for data leakage before returning it to the caller.

        Must be called with the FirewallDecision from the corresponding
        inspect_input() call. The input risk score and system prompt hash
        from that decision are used to contextualise the output inspection.

        Args:
            response_text: The raw LLM response text.
            decision:      The FirewallDecision from inspect_input().

        Returns:
            One of:
              SafeResponse     — no violations; `content` is safe to return.
              RedactedResponse — secrets redacted; `redacted_content` is safe.
              BlockedResponse  — response suppressed; use `safe_message`.
        """
        input_risk_score = decision.risk_score.score
        system_prompt_hash = decision.prompt_context.system_prompt_hash

        inspection: OutputInspectionResult = self._output_filter.inspect(
            response_text=response_text,
            input_risk_score=input_risk_score,
            system_prompt_hash=system_prompt_hash,
        )

        logger.info(
            "PromptFirewall.inspect_output: clean=%s action=%s decision_id=%s",
            inspection.clean,
            inspection.recommended_action.value,
            decision.decision_id,
        )

        # Route to the appropriate response type
        if inspection.clean:
            return SafeResponse(
                content=response_text,
                decision_id=decision.decision_id,
                output_inspection_result=inspection,
            )

        if inspection.recommended_action == FirewallAction.BLOCK:
            reason = _build_block_reason(inspection)
            return BlockedResponse(
                decision_id=decision.decision_id,
                reason=reason,
                risk_score=input_risk_score,
                threat_category=decision.risk_score.primary_threat,
            )

        # SANITIZE → redact and return
        redacted_text, redaction_list = self._output_filter.redact(
            response_text, inspection
        )
        original_hash = hashlib.sha256(response_text.encode("utf-8")).hexdigest()
        return RedactedResponse(
            decision_id=decision.decision_id,
            original_sha256=original_hash,
            redacted_content=redacted_text,
            redactions=redaction_list,
        )

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def build_audit_event(
        self,
        decision: FirewallDecision,
        *,
        application_id: str | None = None,
        user_id: str | None = None,
        ip_address: str | None = None,
        output_result: OutputResult | None = None,
    ) -> AuditEvent:
        """
        Build an AuditEvent from a FirewallDecision, optionally including
        output inspection outcome metadata.

        Call this after inspect_input() (sync path) when you need an audit
        record. In the async path, AuditEvents are emitted automatically.

        Args:
            decision:       The FirewallDecision to audit.
            application_id: Optional application identifier.
            user_id:        Optional user identifier — anonymised.
            ip_address:     Optional source IP — anonymised.
            output_result:  Optional result from inspect_output(). When provided,
                            output_blocked/output_redacted/secrets_detected_count
                            are populated from it.

        Returns:
            AuditEvent ready for emission.
        """
        output_blocked = False
        output_redacted = False
        secrets_count = 0

        if isinstance(output_result, BlockedResponse):
            output_blocked = True
        elif isinstance(output_result, RedactedResponse):
            output_redacted = True
        elif isinstance(output_result, SafeResponse):
            result = output_result.output_inspection_result
            secrets_count = len(result.secret_matches)

        return self._analyzer.build_audit_event(
            decision,
            application_id=application_id,
            user_id=user_id,
            ip_address=ip_address,
            output_blocked=output_blocked,
            output_redacted=output_redacted,
            secrets_detected_count=secrets_count,
        )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _build_block_reason(inspection: OutputInspectionResult) -> str:
    """Synthesise a concise block reason from an OutputInspectionResult."""
    reasons: list[str] = []
    if inspection.system_prompt_echo_detected:
        reasons.append("system prompt echo detected in response")
    if inspection.exfiltration_vector_detected:
        reasons.append("exfiltration vector detected in response")
    high_sev = [sm for sm in inspection.secret_matches if sm.severity >= 0.85]
    if high_sev:
        types = ", ".join(sorted({sm.secret_type for sm in high_sev}))
        reasons.append(f"high-severity credential leak ({types})")
    return "; ".join(reasons) if reasons else "output policy violation"
