"""
Prompt Analyzer
===============

The PromptAnalyzer is the central orchestration layer of the firewall. It
assembles all detection and policy components into a single, ordered pipeline
and returns a typed FirewallDecision for every prompt it inspects.

Pipeline execution order:

  1. PRE-DETECTION NORMALISATION
     Strip invisible characters and apply NFKC unicode normalisation on the
     raw prompt text. The normalised text is used exclusively for detection
     — the original raw prompt is preserved in PromptContext and returned
     to the caller in ALLOW/LOG decisions.

  2. PATTERN DETECTION  (always runs if configured)
     The PatternDetector applies compiled regex signatures against the
     normalised text. If confidence == 1.0 (a hard CRITICAL match), the
     pipeline short-circuits: embedding and LLM detectors are skipped.
     This makes the common BLOCK path sub-millisecond.

  3. EMBEDDING DETECTION  (skipped on short-circuit; optional)
     The EmbeddingDetector measures semantic distance from the attack corpus.
     If the sentence-transformers library is not installed, this detector is
     absent and its weight is redistributed to active detectors by the scorer.

  4. LLM CLASSIFICATION  (skipped on short-circuit; optional, async)
     The LLMClassifier sends a structured query to a small LLM (gpt-4o-mini or
     claude-haiku) to reason about the prompt's intent. The primary path is
     async (inspect_async); inspect() wraps it with asyncio.run().

  5. CONTEXT BOUNDARY DETECTION  (always runs; never short-circuited)
     The ContextBoundaryDetector applies rule-based heuristics against the
     full PromptContext (prompt + prior_turns). It catches structural attacks
     missed by patterns and embeddings: indirect system-prompt probing,
     RAG injection markers, and multi-turn escalation.

  6. ENSEMBLE ASSEMBLY + RISK SCORING
     All available signals are assembled into a DetectorEnsemble. The
     RiskScorer aggregates them into a single [0.0–1.0] RiskScore with
     dynamic renormalisation over active detectors.

  7. POLICY EVALUATION
     The PolicyEngine maps the RiskScore to a FirewallAction, applying
     explicit block/allow pattern rules first, then score thresholds.

  8. SANITIZATION  (only when action == SANITIZE)
     The InputFilter redacts matched injection phrases, strips invisible
     characters, and applies NFKC normalisation to the effective prompt
     forwarded to the LLM.

  9. FIREWALL DECISION ASSEMBLY
     All components are assembled into a FirewallDecision (the canonical
     return type) and optionally an AuditEvent (for SIEM/observability).

Design decisions:

  STATELESS PER-REQUEST
    The PromptAnalyzer holds references to its component instances but no
    per-request state. All request state lives in local variables and in the
    returned FirewallDecision. The analyzer is safe to share across threads.

  DEPENDENCY INJECTION OVER AUTO-CONSTRUCTION
    Components are passed to __init__ rather than constructed internally.
    This allows tests to inject mocks, lets callers choose which detectors
    to enable, and avoids hidden startup costs (model loading, API warmup)
    in the core orchestration layer.

  ASYNC-FIRST, SYNC WRAPPER
    inspect_async() is the primary entry point (the LLM classifier is
    inherently async). inspect() is a convenience wrapper that runs the
    coroutine in a new event loop. In production, callers with existing
    event loops should use inspect_async() directly.

  GRACEFUL DEGRADATION
    Missing detectors contribute zero weight and are excluded from the
    ensemble. The risk scorer renormalises over active detectors. The
    pipeline never raises because a detector is absent.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from llm_prompt_firewall.core.injection_detector import ContextBoundaryDetector
from llm_prompt_firewall.core.risk_scoring import RiskScorer
from llm_prompt_firewall.filters.input_filter import InputFilter
from llm_prompt_firewall.models.schemas import (
    AuditEvent,
    ContextBoundarySignal,
    DetectorEnsemble,
    EmbeddingSignal,
    FirewallAction,
    FirewallDecision,
    LLMClassifierSignal,
    PatternSignal,
    PromptContext,
    RiskScore,
    SanitizedPrompt,
)
from llm_prompt_firewall.policy.policy_engine import PolicyEngine

if TYPE_CHECKING:
    from llm_prompt_firewall.detectors.embedding_detector import EmbeddingDetector
    from llm_prompt_firewall.detectors.llm_classifier import LLMClassifier
    from llm_prompt_firewall.detectors.pattern_detector import PatternDetector

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Analyser configuration dataclass
# ---------------------------------------------------------------------------


@dataclass
class AnalyzerConfig:
    """
    Configuration for PromptAnalyzer factory methods.

    All paths default to the bundled package assets. Override to use
    custom datasets or policy files without subclassing.
    """

    dataset_path: Path = field(
        default_factory=lambda: (
            Path(__file__).parent.parent / "datasets" / "prompt_injection_attacks.json"
        )
    )
    policy_path: Path | None = field(
        default_factory=lambda: (
            Path(__file__).parent.parent.parent / "config" / "default_policy.yaml"
        )
    )
    # Which detectors to enable (all default True)
    enable_pattern_detector: bool = True
    enable_embedding_detector: bool = True
    enable_llm_classifier: bool = False  # Off by default — requires API key config
    enable_context_boundary_detector: bool = True

    # Short-circuit: skip embedding+LLM when pattern confidence == 1.0
    enable_short_circuit: bool = True


# ---------------------------------------------------------------------------
# PromptAnalyzer
# ---------------------------------------------------------------------------


class PromptAnalyzer:
    """
    Orchestrates the full firewall detection and policy pipeline.

    Construct via one of the factory classmethods for production use,
    or pass pre-built components directly for testing.

    Usage (production):
        analyzer = PromptAnalyzer.from_default_config()
        decision = analyzer.inspect(prompt_context)

    Usage (async production):
        analyzer = PromptAnalyzer.from_default_config()
        decision = await analyzer.inspect_async(prompt_context)

    Usage (test / custom):
        analyzer = PromptAnalyzer(
            pattern_detector=my_detector,
            policy_engine=my_policy_engine,
        )
    """

    def __init__(
        self,
        pattern_detector: PatternDetector | None = None,
        embedding_detector: EmbeddingDetector | None = None,
        llm_classifier: LLMClassifier | None = None,
        context_detector: ContextBoundaryDetector | None = None,
        risk_scorer: RiskScorer | None = None,
        policy_engine: PolicyEngine | None = None,
        input_filter: InputFilter | None = None,
        enable_short_circuit: bool = True,
    ) -> None:
        self._pattern_detector = pattern_detector
        self._embedding_detector = embedding_detector
        self._llm_classifier = llm_classifier
        self._context_detector = context_detector or ContextBoundaryDetector()
        self._risk_scorer = risk_scorer or RiskScorer()
        self._policy_engine = policy_engine or PolicyEngine.with_defaults()
        self._input_filter = input_filter or InputFilter()
        self._short_circuit = enable_short_circuit

        logger.info(
            "PromptAnalyzer ready: pattern=%s, embedding=%s, llm=%s, "
            "context_boundary=%s, short_circuit=%s",
            "on" if pattern_detector else "off",
            "on" if embedding_detector else "off",
            "on" if llm_classifier else "off",
            "on",
            enable_short_circuit,
        )

    # ------------------------------------------------------------------
    # Factory constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config: AnalyzerConfig) -> PromptAnalyzer:
        """
        Build a PromptAnalyzer from an AnalyzerConfig.

        Loads the attack dataset and policy file. Attempts to build the
        embedding detector; if sentence-transformers is not installed,
        the detector is set to None and its weight is redistributed.
        """
        from llm_prompt_firewall.detectors.pattern_detector import PatternDetector

        # Pattern detector — always built if dataset exists
        pattern_detector = None
        if config.enable_pattern_detector and config.dataset_path.exists():
            try:
                pattern_detector = PatternDetector.from_dataset_file(config.dataset_path)
            except Exception as exc:
                logger.error("PromptAnalyzer: failed to build PatternDetector: %s", exc)

        # Embedding detector — optional; degrades gracefully if unavailable
        embedding_detector = None
        if config.enable_embedding_detector and config.dataset_path.exists():
            try:
                import json

                from llm_prompt_firewall.detectors.embedding_detector import (
                    EmbeddingDetector,
                )

                with config.dataset_path.open(encoding="utf-8") as fh:
                    raw = json.load(fh)
                from datetime import datetime

                from llm_prompt_firewall.models.schemas import AttackDataset

                for dt_field in ("created_at", "updated_at"):
                    if isinstance(raw.get(dt_field), str):
                        raw[dt_field] = datetime.fromisoformat(raw[dt_field].replace("Z", "+00:00"))
                dataset = AttackDataset(**raw)
                embedding_detector = EmbeddingDetector.from_dataset(dataset)
            except Exception as exc:
                logger.warning(
                    "PromptAnalyzer: embedding detector unavailable (%s) — continuing without it.",
                    exc,
                )

        # Policy engine
        policy_engine = None
        if config.policy_path and config.policy_path.exists():
            try:
                policy_engine = PolicyEngine.from_file(config.policy_path)
            except Exception as exc:
                logger.warning(
                    "PromptAnalyzer: failed to load policy from %s (%s) — using built-in defaults.",
                    config.policy_path,
                    exc,
                )
        if policy_engine is None:
            policy_engine = PolicyEngine.with_defaults()

        return cls(
            pattern_detector=pattern_detector,
            embedding_detector=embedding_detector,
            llm_classifier=None,  # LLM classifier requires explicit configuration
            policy_engine=policy_engine,
            enable_short_circuit=config.enable_short_circuit,
        )

    @classmethod
    def from_default_config(cls) -> PromptAnalyzer:
        """Build a PromptAnalyzer using bundled dataset and default policy."""
        return cls.from_config(AnalyzerConfig())

    # ------------------------------------------------------------------
    # Core inspection — async (primary entry point)
    # ------------------------------------------------------------------

    async def inspect_async(self, prompt_context: PromptContext) -> FirewallDecision:
        """
        Run the full detection and policy pipeline asynchronously.

        The LLM classifier (if configured) runs as a true async call. All other
        detectors are synchronous and run in the calling thread/coroutine.

        Args:
            prompt_context: The PromptContext to inspect.

        Returns:
            FirewallDecision with the resolved action, risk score, and ensemble.
        """
        pipeline_start = time.perf_counter()

        # Step 1: Pre-detection normalisation (for detection only)
        normalized = self._input_filter.apply_pre_detection_normalization(prompt_context.raw_prompt)

        # Step 2: Pattern detection
        pattern_signal: PatternSignal | None = None
        short_circuited = False
        if self._pattern_detector is not None:
            pattern_signal = self._pattern_detector.inspect(normalized)
            if self._short_circuit and pattern_signal.matched and pattern_signal.confidence >= 1.0:
                short_circuited = True
                logger.info(
                    "PromptAnalyzer: pipeline short-circuited on pattern hit "
                    "(confidence=1.0, prompt=%.12s)",
                    prompt_context.prompt_sha256(),
                )

        # Step 3: Embedding detection (skipped on short-circuit)
        embedding_signal: EmbeddingSignal | None = None
        if not short_circuited and self._embedding_detector is not None:
            try:
                embedding_signal = self._embedding_detector.inspect(normalized)
            except Exception as exc:
                logger.warning(
                    "PromptAnalyzer: embedding detector raised unexpectedly: %s — "
                    "treating as absent.",
                    exc,
                )

        # Step 4: LLM classification (skipped on short-circuit; async)
        llm_signal: LLMClassifierSignal | None = None
        if not short_circuited and self._llm_classifier is not None:
            try:
                llm_signal = await self._llm_classifier.inspect_async(normalized)
            except Exception as exc:
                logger.warning(
                    "PromptAnalyzer: LLM classifier raised unexpectedly: %s — "
                    "treating as degraded.",
                    exc,
                )

        # Step 5: Context boundary detection (always runs)
        context_signal: ContextBoundarySignal = self._context_detector.inspect(prompt_context)

        # Step 6: Ensemble assembly + risk scoring
        pipeline_time_ms = (time.perf_counter() - pipeline_start) * 1000
        ensemble = DetectorEnsemble(
            prompt_sha256=prompt_context.prompt_sha256(),
            pattern_signal=pattern_signal,
            embedding_signal=embedding_signal,
            llm_classifier_signal=llm_signal,
            context_boundary_signal=context_signal,
            pipeline_short_circuited=short_circuited,
            total_pipeline_time_ms=round(pipeline_time_ms, 3),
        )
        risk_score: RiskScore = self._risk_scorer.score(ensemble)

        # Step 7: Policy evaluation
        policy_decision = self._policy_engine.evaluate(risk_score, normalized)

        # Step 8: Sanitization (only when action == SANITIZE)
        action = policy_decision.action
        sanitized_prompt: SanitizedPrompt | None = None
        effective_prompt: str | None = None
        block_reason: str | None = None

        if action == FirewallAction.BLOCK:
            block_reason = policy_decision.explanation
            effective_prompt = None

        elif action == FirewallAction.SANITIZE:
            filter_result = self._input_filter.sanitize(prompt_context, pattern_signal)
            sanitized_prompt = filter_result.to_sanitized_prompt()
            effective_prompt = sanitized_prompt.sanitized_text

        else:  # ALLOW or LOG
            # Pass the original raw prompt — normalization is for detection only
            effective_prompt = prompt_context.raw_prompt

        # Step 9: FirewallDecision assembly
        total_ms = (time.perf_counter() - pipeline_start) * 1000
        decision = FirewallDecision(
            prompt_context=prompt_context,
            ensemble=ensemble,
            risk_score=risk_score,
            action=action,
            sanitized_prompt=sanitized_prompt,
            effective_prompt=effective_prompt,
            block_reason=block_reason,
        )

        logger.info(
            "PromptAnalyzer: decision=%s risk=%.3f threat=%s latency=%.1fms sha256=%.12s",
            action.value,
            risk_score.score,
            risk_score.primary_threat.value,
            total_ms,
            prompt_context.prompt_sha256(),
        )

        return decision

    # ------------------------------------------------------------------
    # Synchronous wrapper
    # ------------------------------------------------------------------

    def inspect(self, prompt_context: PromptContext) -> FirewallDecision:
        """
        Synchronous entry point. Runs inspect_async() in a new event loop.

        Use inspect_async() directly when your caller already has a running
        event loop (FastAPI, asyncio application). This method is provided for
        compatibility with synchronous callers.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None and loop.is_running():
            # Caller is inside an async context — cannot use asyncio.run().
            # Run the coroutine in a separate thread to avoid deadlock.
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(asyncio.run, self.inspect_async(prompt_context))
                return future.result()

        return asyncio.run(self.inspect_async(prompt_context))

    # ------------------------------------------------------------------
    # Audit event builder
    # ------------------------------------------------------------------

    def build_audit_event(
        self,
        decision: FirewallDecision,
        *,
        application_id: str | None = None,
        user_id: str | None = None,
        ip_address: str | None = None,
        output_blocked: bool = False,
        output_redacted: bool = False,
        secrets_detected_count: int = 0,
    ) -> AuditEvent:
        """
        Construct an AuditEvent from a FirewallDecision.

        Anonymises user_id (SHA-256) and IP address (zeroes last octet).
        Raw PII never appears in the audit log.

        Args:
            decision:               The FirewallDecision to audit.
            application_id:         Optional application/service identifier.
            user_id:                Optional user identifier — will be hashed.
            ip_address:             Optional source IP — last octet zeroed.
            output_blocked:         True if the LLM response was subsequently blocked.
            output_redacted:        True if the LLM response was subsequently redacted.
            secrets_detected_count: Number of secrets found in the LLM response.

        Returns:
            AuditEvent ready for emission to SIEM / logging pipeline.
        """
        ctx = decision.prompt_context
        ensemble = decision.ensemble
        risk = decision.risk_score

        # Anonymise user_id
        user_id_hash: str | None = None
        if user_id:
            user_id_hash = hashlib.sha256(user_id.encode("utf-8")).hexdigest()

        # Anonymise IP: zero the last octet
        ip_prefix: str | None = None
        if ip_address:
            parts = ip_address.split(".")
            if len(parts) == 4:
                ip_prefix = ".".join(parts[:3]) + ".0"

        return AuditEvent(
            decision_id=decision.decision_id,
            session_id=ctx.session.session_id,
            application_id=application_id,
            prompt_sha256=ctx.prompt_sha256(),
            prompt_preview_redacted=ctx.redacted_preview(),
            risk_score=risk.score,
            risk_level=risk.level,
            primary_threat=risk.primary_threat,
            action_taken=decision.action,
            pattern_confidence=(
                ensemble.pattern_signal.confidence if ensemble.pattern_signal is not None else None
            ),
            embedding_similarity=(
                ensemble.embedding_signal.similarity_score
                if ensemble.embedding_signal is not None
                else None
            ),
            llm_classifier_score=(
                ensemble.llm_classifier_signal.risk_score
                if ensemble.llm_classifier_signal is not None
                and not ensemble.llm_classifier_signal.degraded
                else None
            ),
            context_boundary_confidence=(
                ensemble.context_boundary_signal.confidence
                if ensemble.context_boundary_signal is not None
                else None
            ),
            pipeline_short_circuited=ensemble.pipeline_short_circuited,
            total_latency_ms=ensemble.total_pipeline_time_ms,
            output_blocked=output_blocked,
            output_redacted=output_redacted,
            secrets_detected_count=secrets_detected_count,
            user_id_hash=user_id_hash,
            ip_prefix=ip_prefix,
        )
