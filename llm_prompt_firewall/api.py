"""
FastAPI REST Server for the LLM Prompt Firewall
================================================

Exposes the PromptFirewall as an HTTP service that any application can call,
regardless of language or runtime. The server is stateless — the PromptFirewall
instance is created once at startup and shared across all requests.

Endpoints:

  POST /v1/inspect/input
    Inspect a user prompt before sending it to an LLM.
    Returns a FirewallDecision (action, risk score, effective prompt).

  POST /v1/inspect/output
    Inspect an LLM response before returning it to the user.
    Requires the decision_id from the prior /inspect/input call.
    Returns SafeResponse | BlockedResponse | RedactedResponse.

  GET  /v1/health
    Liveness probe. Returns {"status": "ok"}.

  GET  /v1/ready
    Readiness probe. Returns 200 when the firewall is fully initialised.

Design decisions:

  SINGLETON FIREWALL
    PromptFirewall is initialised once during app startup (lifespan hook)
    and shared across all request handlers. All components are stateless
    and thread-safe; there is no per-request state.

  NO RAW PROMPT IN RESPONSES
    The /inspect/input response includes an effective_prompt field only when
    action != BLOCK. In all cases, the raw prompt is never echoed back —
    callers already have it.

  AUDIT LOGGER WIRED TO STRUCTURED LOG
    If FIREWALL_AUDIT_LOG_FILE is set, audit events are appended as newline-
    delimited JSON to that file. Otherwise, they go to the root logger at INFO.

  CORRELATION VIA decision_id
    The client must pass the decision_id from /inspect/input to /inspect/output.
    This ties the two calls together in audit logs and enables end-to-end tracing.

Environment variables:

  FIREWALL_POLICY_PATH   Path to a YAML policy file (uses bundled default if unset).
  FIREWALL_AUDIT_LOG_FILE  Path for NDJSON audit log output (stdout if unset).
  FIREWALL_LOG_LEVEL     Logging level (default: INFO).
"""

from __future__ import annotations

import logging
import os
import time
from collections import OrderedDict
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock
from typing import Any

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field

from llm_prompt_firewall import metrics as fw_metrics
from llm_prompt_firewall.firewall import PromptFirewall
from llm_prompt_firewall.models.schemas import (
    AuditEvent,
    FirewallAction,
    FirewallDecision,
    PromptContext,
    PromptRole,
    RedactedResponse,
    SafeResponse,
    SessionMetadata,
    ThreatCategory,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global firewall singleton (populated in lifespan)
# ---------------------------------------------------------------------------

_firewall: PromptFirewall | None = None

# ---------------------------------------------------------------------------
# Decision cache with TTL + LRU eviction
#
# Decisions are cached so /inspect/output can look up the FirewallDecision
# produced by /inspect/input using the decision_id.
#
# Bounded to _CACHE_MAX_SIZE entries (LRU eviction) with a per-entry TTL of
# _CACHE_TTL_SECONDS. Both limits prevent unbounded memory growth in a
# long-running server.
#
# Thread-safe: all mutations are protected by _cache_lock.
#
# In production multi-replica deployments, replace with Redis or another
# shared store so that /inspect/input and /inspect/output can be handled
# by different instances.
# ---------------------------------------------------------------------------

_CACHE_MAX_SIZE: int = int(os.environ.get("FIREWALL_CACHE_MAX_SIZE", "10000"))
_CACHE_TTL_SECONDS: float = float(os.environ.get("FIREWALL_CACHE_TTL_SECONDS", "300"))  # 5 min

# Each entry: decision_id → (FirewallDecision, inserted_at_epoch_seconds)
_decision_cache: OrderedDict[str, tuple[FirewallDecision, float]] = OrderedDict()
_cache_lock = Lock()


def _cache_put(decision_id: str, decision: FirewallDecision) -> None:
    """Insert a decision into the cache, evicting LRU or expired entries."""
    now = time.monotonic()
    with _cache_lock:
        # Evict expired entries first (scan from oldest)
        expired = [k for k, (_, ts) in _decision_cache.items() if now - ts > _CACHE_TTL_SECONDS]
        for k in expired:
            del _decision_cache[k]

        # Evict LRU entry if still at capacity
        if len(_decision_cache) >= _CACHE_MAX_SIZE:
            _decision_cache.popitem(last=False)

        _decision_cache[decision_id] = (decision, now)
        _decision_cache.move_to_end(decision_id)


def _cache_get(decision_id: str) -> FirewallDecision | None:
    """Look up a decision by ID. Returns None if missing or expired."""
    now = time.monotonic()
    with _cache_lock:
        entry = _decision_cache.get(decision_id)
        if entry is None:
            return None
        decision, inserted_at = entry
        if now - inserted_at > _CACHE_TTL_SECONDS:
            del _decision_cache[decision_id]
            return None
        # Move to end (most recently used)
        _decision_cache.move_to_end(decision_id)
        return decision


# ---------------------------------------------------------------------------
# Audit logger
# ---------------------------------------------------------------------------


def _make_audit_logger() -> Any:
    """Return a callable that writes AuditEvents to the configured sink."""
    audit_path = os.environ.get("FIREWALL_AUDIT_LOG_FILE")
    if audit_path:
        audit_file = open(audit_path, "a", buffering=1, encoding="utf-8")  # noqa: SIM115

        def _file_logger(event: AuditEvent) -> None:
            line = event.model_dump_json()
            audit_file.write(line + "\n")

        return _file_logger
    else:

        def _log_logger(event: AuditEvent) -> None:
            logger.info("AUDIT %s", event.model_dump_json())

        return _log_logger


# ---------------------------------------------------------------------------
# Lifespan: initialise firewall on startup
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _firewall
    log_level = os.environ.get("FIREWALL_LOG_LEVEL", "INFO").upper()
    logging.basicConfig(level=getattr(logging, log_level, logging.INFO))

    policy_path_env = os.environ.get("FIREWALL_POLICY_PATH")
    if policy_path_env:
        _firewall = PromptFirewall.from_config_file(
            Path(policy_path_env),
            audit_logger=_make_audit_logger(),
        )
    else:
        _firewall = PromptFirewall.from_default_config(
            audit_logger=_make_audit_logger(),
        )

    logger.info("PromptFirewall API ready (policy=%s)", policy_path_env or "default")
    yield
    logger.info("PromptFirewall API shutting down")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="LLM Prompt Firewall",
    description=(
        "Production-grade prompt injection detection and output sanitization API. "
        "Wrap any LLM call site with /inspect/input and /inspect/output."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------


class InspectInputRequest(BaseModel):
    """Request body for POST /v1/inspect/input."""

    prompt: str = Field(
        description="The raw user prompt to inspect (untrusted input).",
        min_length=1,
        max_length=128_000,
    )
    role: PromptRole = Field(
        default=PromptRole.USER,
        description="Conversational role of the message.",
    )
    session_id: str | None = Field(
        default=None,
        description="Stable session identifier for multi-turn attack detection.",
    )
    user_id: str | None = Field(
        default=None,
        description="Opaque user identifier (anonymised in audit logs).",
    )
    application_id: str | None = Field(
        default=None,
        description="Identifier of the application making this request.",
    )
    ip_address: str | None = Field(
        default=None,
        description="Source IP address (last octet zeroed in audit logs).",
    )
    prior_turns: list[str] = Field(
        default_factory=list,
        description="Recent prior user turns for multi-turn attack detection.",
        max_length=10,
    )
    system_prompt_hash: str | None = Field(
        default=None,
        description="SHA-256 hash of the current system prompt (never the raw text).",
    )
    turn_index: int = Field(
        default=0,
        ge=0,
        description="Zero-based index of this message within the session.",
    )


class InspectInputResponse(BaseModel):
    """Response body for POST /v1/inspect/input."""

    decision_id: str = Field(description="Unique ID — pass to /inspect/output.")
    action: FirewallAction
    risk_score: float
    risk_level: str
    primary_threat: ThreatCategory
    effective_prompt: str | None = Field(
        default=None,
        description="Use this prompt when calling your LLM. None when action=BLOCK.",
    )
    block_reason: str | None = None
    pipeline_short_circuited: bool
    pattern_confidence: float | None = None
    embedding_similarity: float | None = None
    context_boundary_confidence: float | None = None


class InspectOutputRequest(BaseModel):
    """Request body for POST /v1/inspect/output."""

    decision_id: str = Field(
        description="The decision_id from the corresponding /inspect/input call.",
    )
    response_text: str = Field(
        description="The raw LLM response to inspect.",
        min_length=1,
    )


class InspectOutputResponse(BaseModel):
    """Response body for POST /v1/inspect/output."""

    decision_id: str
    outcome: str = Field(
        description="One of: 'safe', 'redacted', 'blocked'.",
    )
    content: str | None = Field(
        default=None,
        description="The response text to return to the user (None when blocked).",
    )
    safe_message: str | None = Field(
        default=None,
        description="User-facing message when the response is blocked.",
    )
    redactions: list[str] = Field(
        default_factory=list,
        description="What was redacted and why (populated for 'redacted' outcome).",
    )
    block_reason: str | None = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/v1/health", tags=["ops"])
async def health() -> dict[str, str]:
    """Liveness probe. Always returns 200 if the server is running."""
    return {"status": "ok"}


@app.get("/v1/metrics", tags=["ops"], include_in_schema=False)
async def metrics() -> Response:
    """
    Prometheus metrics endpoint.

    Returns metrics in the standard Prometheus text exposition format.
    Scrape this endpoint with your Prometheus server at the configured
    scrape_interval (recommended: 15s).
    """
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/v1/ready", tags=["ops"])
async def ready() -> dict[str, str]:
    """Readiness probe. Returns 503 if the firewall is not yet initialised."""
    if _firewall is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Firewall not yet initialised.",
        )
    return {"status": "ready"}


@app.post("/v1/inspect/input", response_model=InspectInputResponse, tags=["firewall"])
async def inspect_input(request: InspectInputRequest) -> InspectInputResponse:
    """
    Inspect a user prompt before sending it to an LLM.

    Returns a decision with action, risk score, and the effective_prompt to
    forward to your LLM. Store the decision_id — you need it for /inspect/output.

    Actions:
      - ALLOW / LOG: Forward effective_prompt to your LLM.
      - SANITIZE:    Forward effective_prompt (injection phrases removed).
      - BLOCK:       Do not call your LLM. Return block_reason to the user.
    """
    if _firewall is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Firewall not yet initialised.",
        )

    session = SessionMetadata(
        session_id=request.session_id or SessionMetadata().session_id,
        user_id=request.user_id,
        turn_index=request.turn_index,
        ip_address=request.ip_address,
        application_id=request.application_id,
    )

    ctx = PromptContext(
        raw_prompt=request.prompt,
        role=request.role,
        session=session,
        prior_turns=request.prior_turns,
        system_prompt_hash=request.system_prompt_hash,
    )

    t0 = time.monotonic()
    decision: FirewallDecision = await _firewall.inspect_input_async(
        ctx,
        application_id=request.application_id,
        user_id=request.user_id,
        ip_address=request.ip_address,
    )
    elapsed = time.monotonic() - t0

    # Cache decision so /inspect/output can retrieve it by decision_id.
    _cache_put(decision.decision_id, decision)

    # Record Prometheus metrics.
    fw_metrics.record_input(
        action=decision.action.value,
        risk_score=decision.risk_score.score,
        duration_seconds=elapsed,
    )
    fw_metrics.update_cache_size(len(_decision_cache))

    ens = decision.ensemble
    return InspectInputResponse(
        decision_id=decision.decision_id,
        action=decision.action,
        risk_score=decision.risk_score.score,
        risk_level=decision.risk_score.level.value,
        primary_threat=decision.risk_score.primary_threat,
        effective_prompt=decision.effective_prompt,
        block_reason=decision.block_reason,
        pipeline_short_circuited=ens.pipeline_short_circuited,
        pattern_confidence=(ens.pattern_signal.confidence if ens.pattern_signal else None),
        embedding_similarity=(
            ens.embedding_signal.similarity_score if ens.embedding_signal else None
        ),
        context_boundary_confidence=(
            ens.context_boundary_signal.confidence if ens.context_boundary_signal else None
        ),
    )


@app.post("/v1/inspect/output", response_model=InspectOutputResponse, tags=["firewall"])
async def inspect_output(request: InspectOutputRequest) -> InspectOutputResponse:
    """
    Inspect an LLM response before returning it to the user.

    Requires the decision_id from the corresponding /inspect/input call.
    Returns the outcome ('safe', 'redacted', or 'blocked') and the content
    to return to the user.
    """
    if _firewall is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Firewall not yet initialised.",
        )

    decision = _cache_get(request.decision_id)
    if decision is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"No decision found for decision_id={request.decision_id!r}. "
                "Call /inspect/input first."
            ),
        )

    t0 = time.monotonic()
    result = _firewall.inspect_output(request.response_text, decision)
    elapsed = time.monotonic() - t0

    if isinstance(result, SafeResponse):
        fw_metrics.record_output("safe", elapsed)
        return InspectOutputResponse(
            decision_id=request.decision_id,
            outcome="safe",
            content=result.content,
        )
    elif isinstance(result, RedactedResponse):
        secret_types = [r.split(":")[0] for r in result.redactions]
        fw_metrics.record_output("redacted", elapsed, secret_types)
        return InspectOutputResponse(
            decision_id=request.decision_id,
            outcome="redacted",
            content=result.redacted_content,
            redactions=result.redactions,
        )
    else:  # BlockedResponse
        fw_metrics.record_output("blocked", elapsed)
        return InspectOutputResponse(
            decision_id=request.decision_id,
            outcome="blocked",
            content=None,
            safe_message=result.safe_message,
            block_reason=result.reason,
        )
