"""
Tests for PromptFirewall (firewall.py).

All heavy dependencies (PromptAnalyzer, OutputFilter) are mocked so that tests
are fast, deterministic, and require no model downloads or API calls.

Test structure:
  - TestFirewallInit           — __init__ wires components correctly
  - TestFactoryMethods         — from_default_config / from_config / from_config_file
  - TestInspectInputAsync      — async path: decision forwarding, audit hook, error isolation
  - TestInspectInputSync       — sync path: delegates to analyzer.inspect(), skips audit
  - TestInspectOutput          — routing: SafeResponse / BlockedResponse / RedactedResponse
  - TestBuildAuditEvent        — output_result type drives output_blocked / output_redacted
  - TestBuildBlockReason       — private helper: synthesises concise block reason strings
"""

from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from llm_prompt_firewall.firewall import PromptFirewall, _build_block_reason
from llm_prompt_firewall.models.schemas import (
    AuditEvent,
    BlockedResponse,
    DetectorEnsemble,
    DetectorType,
    FirewallAction,
    FirewallDecision,
    OutputInspectionResult,
    PatternSignal,
    PromptContext,
    RedactedResponse,
    RiskLevel,
    RiskScore,
    SafeResponse,
    SanitizedPrompt,
    SecretMatch,
    ThreatCategory,
)

# ---------------------------------------------------------------------------
# Shared factories
# ---------------------------------------------------------------------------


def _ctx(text: str = "What is the weather today?") -> PromptContext:
    return PromptContext(raw_prompt=text)


def _risk_score(
    score: float = 0.05,
    level: RiskLevel = RiskLevel.SAFE,
    threat: ThreatCategory = ThreatCategory.UNKNOWN,
) -> RiskScore:
    return RiskScore(
        score=score,
        level=level,
        primary_threat=threat,
        contributing_detectors=[DetectorType.PATTERN],
        weights_applied={"pattern": 1.0},
        explanation="Test.",
    )


def _ensemble(short_circuited: bool = False) -> DetectorEnsemble:
    return DetectorEnsemble(
        prompt_sha256="a" * 64,
        pattern_signal=PatternSignal(
            matched=False,
            matches=[],
            confidence=0.0,
            processing_time_ms=0.5,
        ),
        total_pipeline_time_ms=1.0,
        pipeline_short_circuited=short_circuited,
    )


def _allow_decision(ctx: PromptContext | None = None) -> FirewallDecision:
    ctx = ctx or _ctx()
    return FirewallDecision(
        prompt_context=ctx,
        ensemble=_ensemble(),
        risk_score=_risk_score(),
        action=FirewallAction.ALLOW,
        effective_prompt=ctx.raw_prompt,
    )


def _log_decision(ctx: PromptContext | None = None) -> FirewallDecision:
    ctx = ctx or _ctx()
    return FirewallDecision(
        prompt_context=ctx,
        ensemble=_ensemble(),
        risk_score=_risk_score(0.45, RiskLevel.SUSPICIOUS),
        action=FirewallAction.LOG,
        effective_prompt=ctx.raw_prompt,
    )


def _block_decision(ctx: PromptContext | None = None) -> FirewallDecision:
    ctx = ctx or _ctx()
    return FirewallDecision(
        prompt_context=ctx,
        ensemble=_ensemble(),
        risk_score=_risk_score(1.0, RiskLevel.CRITICAL, ThreatCategory.INSTRUCTION_OVERRIDE),
        action=FirewallAction.BLOCK,
        effective_prompt=None,
        block_reason="Hard block: injection detected.",
    )


def _sanitize_decision(ctx: PromptContext | None = None) -> FirewallDecision:
    ctx = ctx or _ctx()
    original_sha = hashlib.sha256(ctx.raw_prompt.encode()).hexdigest()
    return FirewallDecision(
        prompt_context=ctx,
        ensemble=_ensemble(),
        risk_score=_risk_score(0.75, RiskLevel.HIGH, ThreatCategory.INSTRUCTION_OVERRIDE),
        action=FirewallAction.SANITIZE,
        sanitized_prompt=SanitizedPrompt(
            sanitized_text="[REDACTED]",
            original_sha256=original_sha,
            modifications=["Removed injection phrase"],
            chars_removed=5,
        ),
        effective_prompt="[REDACTED]",
    )


def _clean_inspection() -> OutputInspectionResult:
    return OutputInspectionResult(
        clean=True,
        secret_matches=[],
        system_prompt_echo_detected=False,
        exfiltration_vector_detected=False,
        recommended_action=FirewallAction.ALLOW,
        processing_time_ms=0.1,
    )


def _dirty_inspection(
    action: FirewallAction = FirewallAction.SANITIZE,
    system_prompt_echo: bool = False,
    exfil: bool = False,
    secrets: list[SecretMatch] | None = None,
) -> OutputInspectionResult:
    return OutputInspectionResult(
        clean=False,
        secret_matches=secrets or [],
        system_prompt_echo_detected=system_prompt_echo,
        exfiltration_vector_detected=exfil,
        recommended_action=action,
        processing_time_ms=0.5,
    )


def _secret_match(severity: float = 0.90, secret_type: str = "aws_access_key") -> SecretMatch:
    return SecretMatch(
        secret_type=secret_type,
        pattern_id=f"test:{secret_type}",
        offset=0,
        redacted_sample="AKIA***ABCD",
        severity=severity,
    )


def _audit_event(decision: FirewallDecision) -> AuditEvent:
    return AuditEvent(
        decision_id=decision.decision_id,
        session_id=decision.prompt_context.session.session_id,
        prompt_sha256=decision.prompt_context.prompt_sha256(),
        prompt_preview_redacted=decision.prompt_context.redacted_preview(),
        risk_score=decision.risk_score.score,
        risk_level=decision.risk_score.level,
        primary_threat=decision.risk_score.primary_threat,
        action_taken=decision.action,
        total_latency_ms=1.0,
    )


def _make_firewall(
    decision: FirewallDecision | None = None,
    inspection: OutputInspectionResult | None = None,
    audit_logger=None,
) -> tuple[PromptFirewall, MagicMock, MagicMock]:
    """Build a PromptFirewall with mocked analyzer and output_filter."""
    dec = decision or _allow_decision()
    insp = inspection or _clean_inspection()

    analyzer = MagicMock()
    analyzer.inspect_async = AsyncMock(return_value=dec)
    analyzer.inspect.return_value = dec
    analyzer.build_audit_event.return_value = _audit_event(dec)

    output_filter = MagicMock()
    output_filter.inspect.return_value = insp
    output_filter.redact.return_value = ("[REDACTED output]", ["redacted: aws_access_key"])

    fw = PromptFirewall(
        analyzer=analyzer,
        output_filter=output_filter,
        audit_logger=audit_logger,
    )
    return fw, analyzer, output_filter


# ---------------------------------------------------------------------------
# TestFirewallInit
# ---------------------------------------------------------------------------


class TestFirewallInit:
    def test_custom_components_stored(self):
        analyzer = MagicMock()
        output_filter = MagicMock()
        audit_logger = MagicMock()

        fw = PromptFirewall(
            analyzer=analyzer,
            output_filter=output_filter,
            audit_logger=audit_logger,
        )

        assert fw._analyzer is analyzer
        assert fw._output_filter is output_filter
        assert fw._audit_logger is audit_logger

    def test_no_audit_logger_stored_as_none(self):
        fw = PromptFirewall(analyzer=MagicMock(), output_filter=MagicMock())
        assert fw._audit_logger is None

    def test_defaults_created_when_no_args(self):
        """Constructing with no args must not raise (uses real components)."""
        with (
            patch("llm_prompt_firewall.firewall.PromptAnalyzer") as MockAnalyzer,
            patch("llm_prompt_firewall.firewall.OutputFilter") as MockFilter,
        ):
            PromptFirewall()
            MockAnalyzer.assert_called_once()
            MockFilter.assert_called_once()


# ---------------------------------------------------------------------------
# TestFactoryMethods
# ---------------------------------------------------------------------------


class TestFactoryMethods:
    def test_from_default_config_returns_firewall(self):
        with patch("llm_prompt_firewall.firewall.PromptAnalyzer") as MockAnalyzer:
            MockAnalyzer.from_default_config.return_value = MagicMock()
            fw = PromptFirewall.from_default_config()
            assert isinstance(fw, PromptFirewall)
            MockAnalyzer.from_default_config.assert_called_once()

    def test_from_default_config_passes_audit_logger(self):
        logger_fn = MagicMock()
        with patch("llm_prompt_firewall.firewall.PromptAnalyzer") as MockAnalyzer:
            MockAnalyzer.from_default_config.return_value = MagicMock()
            fw = PromptFirewall.from_default_config(audit_logger=logger_fn)
            assert fw._audit_logger is logger_fn

    def test_from_config_uses_provided_config(self):
        from llm_prompt_firewall.core.prompt_analyzer import AnalyzerConfig

        config = AnalyzerConfig()
        with patch("llm_prompt_firewall.firewall.PromptAnalyzer") as MockAnalyzer:
            MockAnalyzer.from_config.return_value = MagicMock()
            fw = PromptFirewall.from_config(config)
            assert isinstance(fw, PromptFirewall)
            MockAnalyzer.from_config.assert_called_once_with(config)

    def test_from_config_passes_audit_logger(self):
        from llm_prompt_firewall.core.prompt_analyzer import AnalyzerConfig

        logger_fn = MagicMock()
        config = AnalyzerConfig()
        with patch("llm_prompt_firewall.firewall.PromptAnalyzer") as MockAnalyzer:
            MockAnalyzer.from_config.return_value = MagicMock()
            fw = PromptFirewall.from_config(config, audit_logger=logger_fn)
            assert fw._audit_logger is logger_fn

    def test_from_config_file_creates_config_from_path(self):
        policy_path = Path("/fake/policy.yaml")
        with patch("llm_prompt_firewall.firewall.PromptAnalyzer") as MockAnalyzer:
            MockAnalyzer.from_config.return_value = MagicMock()
            fw = PromptFirewall.from_config_file(policy_path)
            assert isinstance(fw, PromptFirewall)
            # from_config is called under the hood
            MockAnalyzer.from_config.assert_called_once()

    def test_from_config_file_passes_audit_logger(self):
        logger_fn = MagicMock()
        policy_path = Path("/fake/policy.yaml")
        with patch("llm_prompt_firewall.firewall.PromptAnalyzer") as MockAnalyzer:
            MockAnalyzer.from_config.return_value = MagicMock()
            fw = PromptFirewall.from_config_file(policy_path, audit_logger=logger_fn)
            assert fw._audit_logger is logger_fn


# ---------------------------------------------------------------------------
# TestInspectInputAsync
# ---------------------------------------------------------------------------


class TestInspectInputAsync:
    def test_returns_firewall_decision(self):
        fw, analyzer, _ = _make_firewall()
        decision = asyncio.run(fw.inspect_input_async(_ctx()))
        assert isinstance(decision, FirewallDecision)

    def test_delegates_to_analyzer_inspect_async(self):
        fw, analyzer, _ = _make_firewall()
        ctx = _ctx("Hello world")
        asyncio.run(fw.inspect_input_async(ctx))
        analyzer.inspect_async.assert_called_once_with(ctx)

    def test_audit_logger_called_when_configured(self):
        audit_fn = MagicMock()
        fw, analyzer, _ = _make_firewall(audit_logger=audit_fn)
        asyncio.run(fw.inspect_input_async(_ctx()))
        audit_fn.assert_called_once()

    def test_audit_logger_receives_audit_event(self):
        audit_fn = MagicMock()
        fw, analyzer, _ = _make_firewall(audit_logger=audit_fn)
        asyncio.run(fw.inspect_input_async(_ctx()))
        event = audit_fn.call_args[0][0]
        assert isinstance(event, AuditEvent)

    def test_audit_logger_build_called_with_metadata(self):
        audit_fn = MagicMock()
        fw, analyzer, _ = _make_firewall(audit_logger=audit_fn)
        asyncio.run(
            fw.inspect_input_async(
                _ctx(),
                application_id="app-1",
                user_id="user-42",
                ip_address="10.0.0.5",
            )
        )
        analyzer.build_audit_event.assert_called_once()
        kwargs = analyzer.build_audit_event.call_args[1]
        assert kwargs["application_id"] == "app-1"
        assert kwargs["user_id"] == "user-42"
        assert kwargs["ip_address"] == "10.0.0.5"

    def test_no_audit_logger_does_not_call_build_audit_event(self):
        fw, analyzer, _ = _make_firewall(audit_logger=None)
        asyncio.run(fw.inspect_input_async(_ctx()))
        analyzer.build_audit_event.assert_not_called()

    def test_audit_logger_error_does_not_propagate(self):
        """An exception from the audit logger must never block the request path."""

        def bad_logger(event):
            raise RuntimeError("Kafka is down!")

        fw, _, _ = _make_firewall(audit_logger=bad_logger)
        # Should not raise
        decision = asyncio.run(fw.inspect_input_async(_ctx()))
        assert isinstance(decision, FirewallDecision)

    def test_audit_error_decision_still_returned(self):
        """Even when audit fails, the correct decision is returned."""
        expected = _block_decision()

        def bad_logger(event):
            raise RuntimeError("network error")

        fw, analyzer, _ = _make_firewall(decision=expected, audit_logger=bad_logger)
        result = asyncio.run(fw.inspect_input_async(_ctx()))
        assert result.action == FirewallAction.BLOCK

    def test_allow_action_returned_async(self):
        fw, _, _ = _make_firewall(decision=_allow_decision())
        decision = asyncio.run(fw.inspect_input_async(_ctx()))
        assert decision.action == FirewallAction.ALLOW

    def test_block_action_returned_async(self):
        fw, _, _ = _make_firewall(decision=_block_decision())
        decision = asyncio.run(fw.inspect_input_async(_ctx()))
        assert decision.action == FirewallAction.BLOCK


# ---------------------------------------------------------------------------
# TestInspectInputSync
# ---------------------------------------------------------------------------


class TestInspectInputSync:
    def test_returns_firewall_decision(self):
        fw, _, _ = _make_firewall()
        decision = fw.inspect_input(_ctx())
        assert isinstance(decision, FirewallDecision)

    def test_delegates_to_analyzer_inspect(self):
        fw, analyzer, _ = _make_firewall()
        ctx = _ctx("sync test prompt")
        fw.inspect_input(ctx)
        analyzer.inspect.assert_called_once_with(ctx)

    def test_sync_does_not_call_audit_logger(self):
        """Sync path intentionally skips audit logging (documented behaviour)."""
        audit_fn = MagicMock()
        fw, _, _ = _make_firewall(audit_logger=audit_fn)
        fw.inspect_input(_ctx())
        audit_fn.assert_not_called()

    def test_allow_action_returned_sync(self):
        fw, _, _ = _make_firewall(decision=_allow_decision())
        decision = fw.inspect_input(_ctx())
        assert decision.action == FirewallAction.ALLOW

    def test_block_action_returned_sync(self):
        fw, _, _ = _make_firewall(decision=_block_decision())
        decision = fw.inspect_input(_ctx())
        assert decision.action == FirewallAction.BLOCK

    def test_sanitize_action_returned_sync(self):
        fw, _, _ = _make_firewall(decision=_sanitize_decision())
        decision = fw.inspect_input(_ctx())
        assert decision.action == FirewallAction.SANITIZE


# ---------------------------------------------------------------------------
# TestInspectOutput
# ---------------------------------------------------------------------------


class TestInspectOutput:
    # --- SafeResponse path ---

    def test_clean_response_returns_safe_response(self):
        fw, _, output_filter = _make_firewall(inspection=_clean_inspection())
        result = fw.inspect_output("Hello, the weather is sunny.", _allow_decision())
        assert isinstance(result, SafeResponse)

    def test_safe_response_contains_original_content(self):
        fw, _, _ = _make_firewall(inspection=_clean_inspection())
        text = "The weather is sunny today."
        result = fw.inspect_output(text, _allow_decision())
        assert isinstance(result, SafeResponse)
        assert result.content == text

    def test_safe_response_decision_id_matches(self):
        dec = _allow_decision()
        fw, _, _ = _make_firewall(inspection=_clean_inspection())
        result = fw.inspect_output("Safe text.", dec)
        assert isinstance(result, SafeResponse)
        assert result.decision_id == dec.decision_id

    def test_safe_response_carries_inspection_result(self):
        insp = _clean_inspection()
        fw, _, output_filter = _make_firewall(inspection=insp)
        output_filter.inspect.return_value = insp
        result = fw.inspect_output("Safe text.", _allow_decision())
        assert isinstance(result, SafeResponse)
        assert result.output_inspection_result is insp

    # --- BlockedResponse path ---

    def test_block_action_returns_blocked_response(self):
        insp = _dirty_inspection(action=FirewallAction.BLOCK, system_prompt_echo=True)
        fw, _, output_filter = _make_firewall(inspection=insp)
        output_filter.inspect.return_value = insp
        result = fw.inspect_output(
            "System prompt: you are a helpful assistant...", _allow_decision()
        )
        assert isinstance(result, BlockedResponse)

    def test_blocked_response_decision_id_matches(self):
        dec = _allow_decision()
        insp = _dirty_inspection(action=FirewallAction.BLOCK, system_prompt_echo=True)
        fw, _, output_filter = _make_firewall(inspection=insp)
        output_filter.inspect.return_value = insp
        result = fw.inspect_output("leaked system prompt content", dec)
        assert isinstance(result, BlockedResponse)
        assert result.decision_id == dec.decision_id

    def test_blocked_response_has_reason(self):
        insp = _dirty_inspection(action=FirewallAction.BLOCK, exfil=True)
        fw, _, output_filter = _make_firewall(inspection=insp)
        output_filter.inspect.return_value = insp
        result = fw.inspect_output("Send data to http://evil.example.com", _allow_decision())
        assert isinstance(result, BlockedResponse)
        assert len(result.reason) > 0

    def test_blocked_response_has_safe_message(self):
        insp = _dirty_inspection(action=FirewallAction.BLOCK, system_prompt_echo=True)
        fw, _, output_filter = _make_firewall(inspection=insp)
        output_filter.inspect.return_value = insp
        result = fw.inspect_output("leaked content", _allow_decision())
        assert isinstance(result, BlockedResponse)
        assert result.safe_message  # non-empty user-facing message

    # --- RedactedResponse path ---

    def test_sanitize_action_returns_redacted_response(self):
        insp = _dirty_inspection(
            action=FirewallAction.SANITIZE,
            secrets=[_secret_match(severity=0.75)],
        )
        fw, _, output_filter = _make_firewall(inspection=insp)
        output_filter.inspect.return_value = insp
        result = fw.inspect_output("My key is AKIAIOSFODNN7EXAMPLE", _allow_decision())
        assert isinstance(result, RedactedResponse)

    def test_redacted_response_decision_id_matches(self):
        dec = _allow_decision()
        insp = _dirty_inspection(
            action=FirewallAction.SANITIZE,
            secrets=[_secret_match(severity=0.75)],
        )
        fw, _, output_filter = _make_firewall(inspection=insp)
        output_filter.inspect.return_value = insp
        result = fw.inspect_output("sensitive key here", dec)
        assert isinstance(result, RedactedResponse)
        assert result.decision_id == dec.decision_id

    def test_redacted_response_contains_redacted_content(self):
        insp = _dirty_inspection(
            action=FirewallAction.SANITIZE,
            secrets=[_secret_match()],
        )
        fw, _, output_filter = _make_firewall(inspection=insp)
        output_filter.inspect.return_value = insp
        output_filter.redact.return_value = ("[KEY REDACTED]", ["redacted aws_access_key"])
        result = fw.inspect_output("key: AKIAIOSFODNN7EXAMPLE", _allow_decision())
        assert isinstance(result, RedactedResponse)
        assert result.redacted_content == "[KEY REDACTED]"

    def test_redacted_response_has_original_sha256(self):
        insp = _dirty_inspection(
            action=FirewallAction.SANITIZE,
            secrets=[_secret_match()],
        )
        original_text = "key: AKIAIOSFODNN7EXAMPLE"
        expected_hash = hashlib.sha256(original_text.encode()).hexdigest()
        fw, _, output_filter = _make_firewall(inspection=insp)
        output_filter.inspect.return_value = insp
        output_filter.redact.return_value = ("[REDACTED]", ["redacted aws_access_key"])
        result = fw.inspect_output(original_text, _allow_decision())
        assert isinstance(result, RedactedResponse)
        assert result.original_sha256 == expected_hash

    def test_output_filter_inspect_called_with_correct_args(self):
        """inspect() is called with input risk score and system prompt hash."""
        dec = _allow_decision()
        fw, _, output_filter = _make_firewall(inspection=_clean_inspection())
        output_filter.inspect.return_value = _clean_inspection()

        fw.inspect_output("response text", dec)

        output_filter.inspect.assert_called_once_with(
            response_text="response text",
            input_risk_score=dec.risk_score.score,
            system_prompt_hash=dec.prompt_context.system_prompt_hash,
        )

    def test_redact_called_only_for_sanitize_action(self):
        """OutputFilter.redact() must not be called for clean responses."""
        fw, _, output_filter = _make_firewall(inspection=_clean_inspection())
        output_filter.inspect.return_value = _clean_inspection()
        fw.inspect_output("safe text", _allow_decision())
        output_filter.redact.assert_not_called()

    def test_redact_called_for_sanitize_action(self):
        insp = _dirty_inspection(
            action=FirewallAction.SANITIZE,
            secrets=[_secret_match()],
        )
        fw, _, output_filter = _make_firewall(inspection=insp)
        output_filter.inspect.return_value = insp
        fw.inspect_output("sensitive content", _allow_decision())
        output_filter.redact.assert_called_once()


# ---------------------------------------------------------------------------
# TestBuildAuditEvent
# ---------------------------------------------------------------------------


class TestBuildAuditEvent:
    def test_no_output_result_defaults(self):
        fw, analyzer, _ = _make_firewall()
        dec = _allow_decision()
        fw.build_audit_event(dec)
        analyzer.build_audit_event.assert_called_once()
        kwargs = analyzer.build_audit_event.call_args[1]
        assert kwargs["output_blocked"] is False
        assert kwargs["output_redacted"] is False
        assert kwargs["secrets_detected_count"] == 0

    def test_blocked_response_sets_output_blocked(self):
        fw, analyzer, _ = _make_firewall()
        dec = _allow_decision()
        blocked = BlockedResponse(
            decision_id=dec.decision_id,
            reason="system prompt echo",
            risk_score=0.9,
            threat_category=ThreatCategory.PROMPT_EXTRACTION,
        )
        fw.build_audit_event(dec, output_result=blocked)
        kwargs = analyzer.build_audit_event.call_args[1]
        assert kwargs["output_blocked"] is True
        assert kwargs["output_redacted"] is False

    def test_redacted_response_sets_output_redacted(self):
        fw, analyzer, _ = _make_firewall()
        dec = _allow_decision()
        redacted = RedactedResponse(
            decision_id=dec.decision_id,
            original_sha256="a" * 64,
            redacted_content="[REDACTED]",
            redactions=["aws_access_key"],
        )
        fw.build_audit_event(dec, output_result=redacted)
        kwargs = analyzer.build_audit_event.call_args[1]
        assert kwargs["output_redacted"] is True
        assert kwargs["output_blocked"] is False

    def test_safe_response_populates_secrets_count(self):
        fw, analyzer, _ = _make_firewall()
        dec = _allow_decision()
        insp = OutputInspectionResult(
            clean=True,
            secret_matches=[_secret_match(), _secret_match()],
            recommended_action=FirewallAction.ALLOW,
            processing_time_ms=0.1,
        )
        safe = SafeResponse(
            content="text",
            decision_id=dec.decision_id,
            output_inspection_result=insp,
        )
        fw.build_audit_event(dec, output_result=safe)
        kwargs = analyzer.build_audit_event.call_args[1]
        assert kwargs["secrets_detected_count"] == 2

    def test_optional_metadata_forwarded(self):
        fw, analyzer, _ = _make_firewall()
        dec = _allow_decision()
        fw.build_audit_event(
            dec,
            application_id="myapp",
            user_id="user-1",
            ip_address="192.168.1.5",
        )
        kwargs = analyzer.build_audit_event.call_args[1]
        assert kwargs["application_id"] == "myapp"
        assert kwargs["user_id"] == "user-1"
        assert kwargs["ip_address"] == "192.168.1.5"

    def test_returns_audit_event(self):
        fw, analyzer, _ = _make_firewall()
        dec = _allow_decision()
        event = fw.build_audit_event(dec)
        assert isinstance(event, AuditEvent)


# ---------------------------------------------------------------------------
# TestBuildBlockReason  (private helper)
# ---------------------------------------------------------------------------


class TestBuildBlockReason:
    def test_system_prompt_echo_reason(self):
        insp = _dirty_inspection(action=FirewallAction.BLOCK, system_prompt_echo=True)
        reason = _build_block_reason(insp)
        assert "system prompt echo" in reason

    def test_exfil_vector_reason(self):
        insp = _dirty_inspection(action=FirewallAction.BLOCK, exfil=True)
        reason = _build_block_reason(insp)
        assert "exfiltration vector" in reason

    def test_high_severity_secret_included(self):
        secret = _secret_match(severity=0.90, secret_type="jwt_token")
        insp = _dirty_inspection(action=FirewallAction.BLOCK, secrets=[secret])
        reason = _build_block_reason(insp)
        assert "jwt_token" in reason
        assert "high-severity" in reason

    def test_low_severity_secret_excluded(self):
        """Secrets below 0.85 severity are not included in the block reason."""
        secret = _secret_match(severity=0.70, secret_type="generic_password")
        insp = _dirty_inspection(action=FirewallAction.BLOCK, secrets=[secret])
        reason = _build_block_reason(insp)
        # Low-severity secret should not appear in block reason
        assert "generic_password" not in reason

    def test_combined_reasons_joined_with_semicolon(self):
        secret = _secret_match(severity=0.90, secret_type="private_key")
        insp = _dirty_inspection(
            action=FirewallAction.BLOCK,
            system_prompt_echo=True,
            exfil=True,
            secrets=[secret],
        )
        reason = _build_block_reason(insp)
        assert ";" in reason

    def test_fallback_reason_when_no_specific_violations(self):
        """When inspection is dirty but no specific flag is set, return fallback."""
        insp = OutputInspectionResult(
            clean=False,
            secret_matches=[],
            system_prompt_echo_detected=False,
            exfiltration_vector_detected=False,
            recommended_action=FirewallAction.BLOCK,
            processing_time_ms=0.1,
        )
        reason = _build_block_reason(insp)
        assert reason == "output policy violation"

    def test_multiple_high_severity_types_listed(self):
        secrets = [
            _secret_match(severity=0.90, secret_type="aws_access_key"),
            _secret_match(severity=0.95, secret_type="private_key"),
        ]
        insp = _dirty_inspection(action=FirewallAction.BLOCK, secrets=secrets)
        reason = _build_block_reason(insp)
        assert "aws_access_key" in reason
        assert "private_key" in reason
