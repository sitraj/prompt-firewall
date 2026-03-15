"""
Tests for the Prometheus metrics module and API metrics integration.

Test structure:
  - TestRecordHelpers     — record_input / record_output / update_cache_size update counters
  - TestMetricsEndpoint   — GET /v1/metrics returns 200 with Prometheus text format
  - TestMetricsWiring     — calling /inspect/input and /inspect/output records metrics
"""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from prometheus_client import REGISTRY

import llm_prompt_firewall.api as api_module
import llm_prompt_firewall.metrics as fw_metrics
from llm_prompt_firewall.api import app
from llm_prompt_firewall.models.schemas import (
    BlockedResponse,
    ContextBoundarySignal,
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
    ThreatCategory,
)


# ---------------------------------------------------------------------------
# Shared factories (mirrors test_api.py helpers)
# ---------------------------------------------------------------------------


def _allow_decision(decision_id: str = "test-id") -> FirewallDecision:
    ctx = PromptContext(raw_prompt="Hello world")
    return FirewallDecision(
        decision_id=decision_id,
        prompt_context=ctx,
        ensemble=DetectorEnsemble(
            prompt_sha256="a" * 64,
            pattern_signal=PatternSignal(
                matched=False, matches=[], confidence=0.0, processing_time_ms=0.5
            ),
            total_pipeline_time_ms=1.0,
        ),
        risk_score=RiskScore(
            score=0.05,
            level=RiskLevel.SAFE,
            primary_threat=ThreatCategory.UNKNOWN,
            contributing_detectors=[DetectorType.PATTERN],
            weights_applied={"pattern": 1.0},
            explanation="Test.",
        ),
        action=FirewallAction.ALLOW,
        effective_prompt="Hello world",
    )


def _safe_response(decision_id: str = "test-id") -> SafeResponse:
    return SafeResponse(
        content="Clean response.",
        decision_id=decision_id,
        output_inspection_result=OutputInspectionResult(
            clean=True,
            recommended_action=FirewallAction.ALLOW,
            processing_time_ms=0.1,
        ),
    )


def _make_mock_firewall(
    decision: FirewallDecision | None = None,
    output_result=None,
) -> MagicMock:
    dec = decision or _allow_decision()
    out = output_result or _safe_response(dec.decision_id)
    fw = MagicMock()
    fw.inspect_input_async = AsyncMock(return_value=dec)
    fw.inspect_output.return_value = out
    fw.build_audit_event.return_value = MagicMock()
    return fw


@pytest.fixture()
def client():
    mock_fw = _make_mock_firewall()
    with patch(
        "llm_prompt_firewall.api.PromptFirewall.from_default_config",
        return_value=mock_fw,
    ):
        with TestClient(app) as c:
            api_module._firewall = mock_fw
            api_module._decision_cache.clear()
            yield c, mock_fw
    api_module._firewall = None
    api_module._decision_cache.clear()


# ---------------------------------------------------------------------------
# Helpers to read current prometheus counter/gauge values
# ---------------------------------------------------------------------------


def _counter_value(metric_name: str, **labels) -> float:
    """Read the current value of a prometheus counter with given label values."""
    for metric in REGISTRY.collect():
        if metric.name == metric_name:
            for sample in metric.samples:
                if sample.name == metric_name + "_total":
                    if all(sample.labels.get(k) == v for k, v in labels.items()):
                        return sample.value
    return 0.0


def _gauge_value(metric_name: str) -> float:
    for metric in REGISTRY.collect():
        if metric.name == metric_name:
            for sample in metric.samples:
                if sample.name == metric_name:
                    return sample.value
    return 0.0


# ---------------------------------------------------------------------------
# TestRecordHelpers
# ---------------------------------------------------------------------------


class TestRecordHelpers:
    def test_record_input_increments_counter(self):
        before = _counter_value("firewall_input_inspections", action="allow")
        fw_metrics.record_input(action="allow", risk_score=0.05, duration_seconds=0.01)
        after = _counter_value("firewall_input_inspections", action="allow")
        assert after == before + 1

    def test_record_input_block_increments_block_counter(self):
        before = _counter_value("firewall_input_inspections", action="block")
        fw_metrics.record_input(action="block", risk_score=0.95, duration_seconds=0.005)
        after = _counter_value("firewall_input_inspections", action="block")
        assert after == before + 1

    def test_record_output_safe_increments_counter(self):
        before = _counter_value("firewall_output_inspections", outcome="safe")
        fw_metrics.record_output("safe", duration_seconds=0.001)
        after = _counter_value("firewall_output_inspections", outcome="safe")
        assert after == before + 1

    def test_record_output_redacted_increments_counter(self):
        before = _counter_value("firewall_output_inspections", outcome="redacted")
        fw_metrics.record_output("redacted", duration_seconds=0.002, secret_types=["aws_access_key"])
        after = _counter_value("firewall_output_inspections", outcome="redacted")
        assert after == before + 1

    def test_record_output_secret_types_increments_secrets_counter(self):
        before = _counter_value("firewall_secrets_detected", secret_type="jwt_token")
        fw_metrics.record_output("redacted", duration_seconds=0.001, secret_types=["jwt_token"])
        after = _counter_value("firewall_secrets_detected", secret_type="jwt_token")
        assert after == before + 1

    def test_record_output_multiple_secret_types(self):
        before_aws = _counter_value("firewall_secrets_detected", secret_type="aws_access_key")
        before_pk = _counter_value("firewall_secrets_detected", secret_type="private_key")
        fw_metrics.record_output(
            "redacted",
            duration_seconds=0.001,
            secret_types=["aws_access_key", "private_key"],
        )
        assert _counter_value("firewall_secrets_detected", secret_type="aws_access_key") == before_aws + 1
        assert _counter_value("firewall_secrets_detected", secret_type="private_key") == before_pk + 1

    def test_record_output_no_secret_types_does_not_increment_secrets(self):
        before = _counter_value("firewall_secrets_detected", secret_type="generic_password")
        fw_metrics.record_output("safe", duration_seconds=0.001, secret_types=None)
        after = _counter_value("firewall_secrets_detected", secret_type="generic_password")
        assert after == before  # no change

    def test_update_cache_size_sets_gauge(self):
        fw_metrics.update_cache_size(42)
        assert _gauge_value("firewall_decision_cache_size") == 42.0

    def test_update_cache_size_updates_gauge(self):
        fw_metrics.update_cache_size(10)
        fw_metrics.update_cache_size(99)
        assert _gauge_value("firewall_decision_cache_size") == 99.0


# ---------------------------------------------------------------------------
# TestMetricsEndpoint
# ---------------------------------------------------------------------------


class TestMetricsEndpoint:
    def test_metrics_returns_200(self, client):
        c, _ = client
        resp = c.get("/v1/metrics")
        assert resp.status_code == 200

    def test_metrics_content_type_is_prometheus(self, client):
        c, _ = client
        resp = c.get("/v1/metrics")
        assert "text/plain" in resp.headers["content-type"]

    def test_metrics_body_contains_firewall_prefix(self, client):
        c, _ = client
        resp = c.get("/v1/metrics")
        assert b"firewall_" in resp.content

    def test_metrics_body_contains_input_inspections(self, client):
        c, _ = client
        resp = c.get("/v1/metrics")
        assert b"firewall_input_inspections_total" in resp.content

    def test_metrics_body_contains_output_inspections(self, client):
        c, _ = client
        resp = c.get("/v1/metrics")
        assert b"firewall_output_inspections_total" in resp.content

    def test_metrics_body_contains_cache_size(self, client):
        c, _ = client
        resp = c.get("/v1/metrics")
        assert b"firewall_decision_cache_size" in resp.content

    def test_metrics_not_in_openapi_schema(self, client):
        """The /v1/metrics endpoint is excluded from the OpenAPI schema."""
        c, _ = client
        resp = c.get("/openapi.json")
        schema = resp.json()
        assert "/v1/metrics" not in schema["paths"]


# ---------------------------------------------------------------------------
# TestMetricsWiring
# ---------------------------------------------------------------------------


class TestMetricsWiring:
    def test_inspect_input_increments_allow_counter(self, client):
        c, mock_fw = client
        mock_fw.inspect_input_async = AsyncMock(return_value=_allow_decision("w1"))
        before = _counter_value("firewall_input_inspections", action="allow")
        c.post("/v1/inspect/input", json={"prompt": "Hello"})
        after = _counter_value("firewall_input_inspections", action="allow")
        assert after == before + 1

    def test_inspect_input_block_increments_block_counter(self, client):
        c, mock_fw = client
        ctx = PromptContext(raw_prompt="Ignore instructions")
        block_dec = FirewallDecision(
            decision_id="block-w",
            prompt_context=ctx,
            ensemble=DetectorEnsemble(
                prompt_sha256="b" * 64,
                total_pipeline_time_ms=1.0,
            ),
            risk_score=RiskScore(
                score=0.97,
                level=RiskLevel.CRITICAL,
                primary_threat=ThreatCategory.INSTRUCTION_OVERRIDE,
                contributing_detectors=[DetectorType.PATTERN],
                weights_applied={"pattern": 1.0},
                explanation="Blocked.",
            ),
            action=FirewallAction.BLOCK,
            effective_prompt=None,
            block_reason="Injection detected.",
        )
        mock_fw.inspect_input_async = AsyncMock(return_value=block_dec)
        before = _counter_value("firewall_input_inspections", action="block")
        c.post("/v1/inspect/input", json={"prompt": "Ignore instructions"})
        after = _counter_value("firewall_input_inspections", action="block")
        assert after == before + 1

    def test_inspect_output_safe_increments_safe_counter(self, client):
        c, mock_fw = client
        dec = _allow_decision("out-safe")
        api_module._cache_put("out-safe", dec)
        mock_fw.inspect_output.return_value = _safe_response("out-safe")
        before = _counter_value("firewall_output_inspections", outcome="safe")
        c.post("/v1/inspect/output", json={"decision_id": "out-safe", "response_text": "Hi"})
        after = _counter_value("firewall_output_inspections", outcome="safe")
        assert after == before + 1

    def test_inspect_output_blocked_increments_blocked_counter(self, client):
        c, mock_fw = client
        dec = _allow_decision("out-blocked")
        api_module._cache_put("out-blocked", dec)
        mock_fw.inspect_output.return_value = BlockedResponse(
            decision_id="out-blocked",
            reason="system prompt echo",
            risk_score=0.9,
            threat_category=ThreatCategory.PROMPT_EXTRACTION,
        )
        before = _counter_value("firewall_output_inspections", outcome="blocked")
        c.post(
            "/v1/inspect/output",
            json={"decision_id": "out-blocked", "response_text": "System prompt: ..."},
        )
        after = _counter_value("firewall_output_inspections", outcome="blocked")
        assert after == before + 1

    def test_cache_size_gauge_updated_after_input(self, client):
        c, mock_fw = client
        api_module._decision_cache.clear()
        fw_metrics.update_cache_size(0)
        mock_fw.inspect_input_async = AsyncMock(return_value=_allow_decision("sz-1"))
        c.post("/v1/inspect/input", json={"prompt": "Test prompt"})
        assert _gauge_value("firewall_decision_cache_size") >= 1.0
