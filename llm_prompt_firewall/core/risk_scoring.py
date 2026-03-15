"""
Risk Scoring Engine
====================

The RiskScorer aggregates signals from all active detectors into a single
normalised risk score, maps it to a RiskLevel tier, and determines the
primary threat category and recommended FirewallAction.

Design decisions:

  1. WEIGHTED ENSEMBLE WITH DYNAMIC RENORMALISATION
     Each detector has a configured weight. When a detector is absent
     (not configured), skipped (short-circuit), or degraded (LLM timeout),
     its weight is redistributed proportionally across the remaining active
     detectors. This ensures the score always lives in [0.0, 1.0] and that
     a missing detector does not silently deflate the score.

     Example: if the LLM classifier (weight 0.35) is degraded, pattern
     (0.30) and embedding (0.25) are renormalised to weights of 0.545 and
     0.455 respectively — preserving their relative importance.

  2. SHORT-CIRCUIT PROPAGATION
     When the pattern detector short-circuited (PatternSignal.confidence==1.0
     AND ensemble.pipeline_short_circuited==True), the ensemble score is set
     to 1.0 immediately. No renormalisation, no ensemble math — a CRITICAL
     hard match is unconditionally the highest possible score.

  3. SCORE IS NOT THE ACTION
     The RiskScorer outputs a numeric score and a RiskLevel. The FirewallAction
     is determined by the PolicyEngine (a separate concern). The scorer does
     recommend an action based purely on score thresholds, but the policy
     engine may override it (e.g. a user-configured whitelist can ALLOW a
     prompt the scorer rated HIGH).

  4. THREAT CATEGORY RESOLUTION
     Primary threat comes from the highest-fidelity available detector in
     priority order: LLM classifier (most reasoned) → pattern (most specific)
     → embedding (most general). The category is only assigned from a detector
     that actually fired above its signal threshold.

  5. CONFIGURABLE WEIGHTS AND THRESHOLDS
     WeightConfig and ThresholdConfig are dataclasses, not hardcoded constants.
     The PolicyEngine passes them through from the loaded YAML policy. This
     allows per-deployment tuning without code changes.

  6. EXPLANATION SYNTHESIS
     Every RiskScore carries a human-readable explanation synthesised from
     the contributing signals. This is for operator review and audit — not
     shown to end users.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from llm_prompt_firewall.models.schemas import (
    ContextBoundarySignal,
    DetectorEnsemble,
    DetectorType,
    EmbeddingSignal,
    FirewallAction,
    LLMClassifierSignal,
    PatternSignal,
    RiskLevel,
    RiskScore,
    ThreatCategory,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------


@dataclass
class WeightConfig:
    """
    Per-detector weights used in the ensemble score calculation.

    Weights do not need to sum to exactly 1.0 — the scorer renormalises
    over active detectors. However, their relative magnitudes determine
    each detector's influence on the final score.

    Default rationale:
      LLM classifier (0.35): highest-fidelity signal, slowest, most expensive.
          Highest weight because it reasons about intent, not just surface form.
      Pattern (0.30): deterministic, zero false negatives on known signatures.
          High weight because a pattern hit is unambiguous evidence.
      Embedding (0.25): semantic generalisation. Lower weight because cosine
          similarity is noisier than a hard pattern match.
      Context boundary (0.10): structural heuristics. Supplementary signal,
          lower weight — it fires broadly and needs other detectors to confirm.
    """

    pattern: float = 0.30
    embedding: float = 0.25
    llm_classifier: float = 0.35
    context_boundary: float = 0.10

    def for_detector(self, detector: DetectorType) -> float:
        return {
            DetectorType.PATTERN: self.pattern,
            DetectorType.EMBEDDING: self.embedding,
            DetectorType.LLM_CLASSIFIER: self.llm_classifier,
            DetectorType.CONTEXT_BOUNDARY: self.context_boundary,
        }.get(detector, 0.0)


@dataclass
class ThresholdConfig:
    """
    Score thresholds that map numeric risk scores to RiskLevel tiers
    and recommended FirewallActions.

    [0.00 – safe_max]        → SAFE       → ALLOW
    (safe_max – suspicious_max] → SUSPICIOUS → LOG
    (suspicious_max – high_max] → HIGH       → SANITIZE
    (high_max – 1.00]        → CRITICAL   → BLOCK
    """

    safe_max: float = 0.39
    suspicious_max: float = 0.69
    high_max: float = 0.84

    def risk_level(self, score: float) -> RiskLevel:
        if score <= self.safe_max:
            return RiskLevel.SAFE
        if score <= self.suspicious_max:
            return RiskLevel.SUSPICIOUS
        if score <= self.high_max:
            return RiskLevel.HIGH
        return RiskLevel.CRITICAL

    def recommended_action(self, level: RiskLevel) -> FirewallAction:
        return {
            RiskLevel.SAFE: FirewallAction.ALLOW,
            RiskLevel.SUSPICIOUS: FirewallAction.LOG,
            RiskLevel.HIGH: FirewallAction.SANITIZE,
            RiskLevel.CRITICAL: FirewallAction.BLOCK,
        }[level]


# Default instances used when no custom config is supplied
DEFAULT_WEIGHTS = WeightConfig()
DEFAULT_THRESHOLDS = ThresholdConfig()


# ---------------------------------------------------------------------------
# Internal signal extraction helpers
# ---------------------------------------------------------------------------


def _pattern_score(signal: PatternSignal | None) -> float | None:
    """Extract the usable score from a PatternSignal. None = no signal."""
    if signal is None:
        return None
    return signal.confidence


def _embedding_score(signal: EmbeddingSignal | None) -> float | None:
    """
    Extract the usable score from an EmbeddingSignal.

    We use similarity_score (raw cosine similarity) rather than a binary
    exceeded_threshold flag. This gives the ensemble finer granularity —
    a similarity of 0.78 (below 0.82 threshold) still contributes useful
    signal, just lower than a 0.95 hit.
    """
    if signal is None:
        return None
    return signal.similarity_score


def _llm_score(signal: LLMClassifierSignal | None) -> float | None:
    """
    Extract the usable score from an LLMClassifierSignal.
    Returns None for degraded signals — they must not contribute to the ensemble.
    """
    if signal is None:
        return None
    if signal.degraded:
        return None
    return signal.risk_score


def _context_score(signal: ContextBoundarySignal | None) -> float | None:
    """Extract the usable score from a ContextBoundarySignal."""
    if signal is None:
        return None
    return signal.confidence


# ---------------------------------------------------------------------------
# Threat category resolution
# ---------------------------------------------------------------------------


def _resolve_primary_threat(
    pattern: PatternSignal | None,
    embedding: EmbeddingSignal | None,
    llm: LLMClassifierSignal | None,
    context: ContextBoundarySignal | None,
) -> ThreatCategory:
    """
    Determine the primary threat category from available detector signals.

    Priority order (highest fidelity first):
      1. LLM classifier (reasoned, specific category)
      2. Pattern detector (unambiguous signature match with known category)
      3. Embedding detector (nearest attack category from index)
      4. Context boundary (detected boundary type)
      5. UNKNOWN (fallback when no detector fired with a category)
    """
    # LLM classifier: highest fidelity, most specific reasoning
    if llm is not None and not llm.degraded and llm.risk_score > 0.30:
        return llm.threat_category

    # Pattern: deterministic, maps directly to attack category
    if pattern is not None and pattern.matched and pattern.matches:
        # Take the category of the highest-severity match
        best = max(pattern.matches, key=lambda m: m.severity)
        return best.category

    # Embedding: nearest attack in index
    if (
        embedding is not None
        and embedding.nearest_attack_category is not None
        and embedding.similarity_score > 0.50
    ):
        return embedding.nearest_attack_category

    # Context boundary: coarse structural detection
    if context is not None and context.boundary_violation_detected:
        return ThreatCategory.PROMPT_EXTRACTION  # most common context crossing

    return ThreatCategory.UNKNOWN


# ---------------------------------------------------------------------------
# Explanation synthesis
# ---------------------------------------------------------------------------


def _build_explanation(
    score: float,
    level: RiskLevel,
    pattern: PatternSignal | None,
    embedding: EmbeddingSignal | None,
    llm: LLMClassifierSignal | None,
    context: ContextBoundarySignal | None,
    short_circuited: bool,
) -> str:
    """
    Build a concise human-readable explanation of the risk score.

    Aimed at security operators reviewing flagged prompts. Summarises which
    detectors fired and with what signal strength.
    """
    parts: list[str] = [f"Risk score: {score:.2f} ({level.value})."]

    if short_circuited:
        parts.append("Pipeline short-circuited on CRITICAL pattern match.")

    if pattern and pattern.matched:
        top_patterns = [m.pattern_id for m in pattern.matches[:3]]
        parts.append(
            f"Pattern detector: {len(pattern.matches)} match(es) "
            f"(confidence {pattern.confidence:.2f}). "
            f"Top signatures: {', '.join(top_patterns)}."
        )

    if embedding and embedding.similarity_score > 0.0:
        parts.append(
            f"Embedding detector: similarity {embedding.similarity_score:.3f} "
            f"(threshold {embedding.threshold_used:.2f}) "
            f"to attack '{embedding.nearest_attack_id}'."
        )

    if llm and not llm.degraded:
        parts.append(
            f"LLM classifier ({llm.model_used}): score {llm.risk_score:.2f}, "
            f"category '{llm.threat_category.value}'. "
            f"Reasoning: {llm.reasoning[:100]}."
        )
    elif llm and llm.degraded:
        parts.append("LLM classifier: DEGRADED (excluded from ensemble).")

    if context and context.boundary_violation_detected:
        parts.append(f"Context boundary: violations {context.violated_boundaries}.")

    return " ".join(parts)


# ---------------------------------------------------------------------------
# RiskScorer
# ---------------------------------------------------------------------------


class RiskScorer:
    """
    Aggregates detector signals from a DetectorEnsemble into a RiskScore.

    The scorer is stateless — all configuration is passed at construction
    time and never mutated. It is safe to share across threads and coroutines.

    Usage:
        scorer = RiskScorer()
        risk = scorer.score(ensemble)
    """

    def __init__(
        self,
        weights: WeightConfig = DEFAULT_WEIGHTS,
        thresholds: ThresholdConfig = DEFAULT_THRESHOLDS,
    ) -> None:
        self._weights = weights
        self._thresholds = thresholds

    def score(self, ensemble: DetectorEnsemble) -> RiskScore:
        """
        Compute the aggregate risk score from a DetectorEnsemble.

        Args:
            ensemble: The complete set of detector signals for a single prompt.

        Returns:
            RiskScore with numeric score, risk level, primary threat category,
            explanation, and metadata about the ensemble.
        """
        pattern = ensemble.pattern_signal
        embedding = ensemble.embedding_signal
        llm = ensemble.llm_classifier_signal
        context = ensemble.context_boundary_signal

        # --- Short-circuit propagation ---
        # When the pattern detector flagged a CRITICAL match and the pipeline
        # exited early, we skip all ensemble math. The score is 1.0.
        if ensemble.pipeline_short_circuited and pattern is not None and pattern.confidence >= 1.0:
            level = RiskLevel.CRITICAL
            primary_threat = _resolve_primary_threat(pattern, embedding, llm, context)
            explanation = _build_explanation(
                1.0, level, pattern, embedding, llm, context, short_circuited=True
            )
            logger.debug(
                "Short-circuit score: 1.0 (CRITICAL) for prompt %s",
                ensemble.prompt_sha256[:12],
            )
            return RiskScore(
                score=1.0,
                level=level,
                primary_threat=primary_threat,
                contributing_detectors=[DetectorType.PATTERN],
                weights_applied={DetectorType.PATTERN.value: 1.0},
                explanation=explanation,
            )

        # --- Collect active signals ---
        # Build a list of (detector_type, raw_score) for detectors that
        # produced a usable (non-None, non-degraded) signal.
        raw_signals: list[tuple[DetectorType, float]] = []

        p_score = _pattern_score(pattern)
        if p_score is not None:
            raw_signals.append((DetectorType.PATTERN, p_score))

        e_score = _embedding_score(embedding)
        if e_score is not None:
            raw_signals.append((DetectorType.EMBEDDING, e_score))

        l_score = _llm_score(llm)
        if l_score is not None:
            raw_signals.append((DetectorType.LLM_CLASSIFIER, l_score))

        c_score = _context_score(context)
        if c_score is not None:
            raw_signals.append((DetectorType.CONTEXT_BOUNDARY, c_score))

        # --- No signals at all ---
        if not raw_signals:
            logger.debug(
                "RiskScorer: no active signals for prompt %s — score 0.0",
                ensemble.prompt_sha256[:12],
            )
            return RiskScore(
                score=0.0,
                level=RiskLevel.SAFE,
                primary_threat=ThreatCategory.UNKNOWN,
                contributing_detectors=[],
                weights_applied={},
                explanation="No detector signals available. Score defaulted to 0.0.",
            )

        # --- Weighted ensemble with renormalisation ---
        # Sum the configured weights for active detectors only.
        total_weight = sum(self._weights.for_detector(det) for det, _ in raw_signals)

        if total_weight == 0.0:
            # All active detectors have zero weight — shouldn't happen with
            # sane config, but guard against division by zero.
            logger.warning(
                "RiskScorer: total weight is zero for active detectors %s",
                [det.value for det, _ in raw_signals],
            )
            total_weight = 1.0

        weighted_sum = sum(
            self._weights.for_detector(det) * raw_score for det, raw_score in raw_signals
        )

        # Renormalised score: divide by active weight sum, not 1.0
        ensemble_score = weighted_sum / total_weight

        # Clamp to [0, 1] — fp arithmetic can produce tiny overflows
        ensemble_score = max(0.0, min(1.0, ensemble_score))
        ensemble_score = round(ensemble_score, 4)

        # Track applied weights for auditability
        weights_applied = {
            det.value: round(self._weights.for_detector(det) / total_weight, 4)
            for det, _ in raw_signals
        }

        # --- Map to level and action ---
        level = self._thresholds.risk_level(ensemble_score)
        primary_threat = _resolve_primary_threat(pattern, embedding, llm, context)
        contributing = [det for det, _ in raw_signals]

        explanation = _build_explanation(
            ensemble_score,
            level,
            pattern,
            embedding,
            llm,
            context,
            short_circuited=False,
        )

        logger.debug(
            "RiskScore: %.4f (%s) | detectors=%s | threat=%s | prompt=%s",
            ensemble_score,
            level.value,
            [det.value for det in contributing],
            primary_threat.value,
            ensemble.prompt_sha256[:12],
        )

        return RiskScore(
            score=ensemble_score,
            level=level,
            primary_threat=primary_threat,
            contributing_detectors=contributing,
            weights_applied=weights_applied,
            explanation=explanation,
        )

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def score_from_signals(
        self,
        *,
        prompt_sha256: str,
        pattern: PatternSignal | None = None,
        embedding: EmbeddingSignal | None = None,
        llm: LLMClassifierSignal | None = None,
        context: ContextBoundarySignal | None = None,
        pipeline_short_circuited: bool = False,
        total_pipeline_time_ms: float = 0.0,
    ) -> RiskScore:
        """
        Convenience method: build a DetectorEnsemble inline and score it.

        Useful in tests and the PromptAnalyzer when building the ensemble
        incrementally rather than up-front.
        """
        ensemble = DetectorEnsemble(
            prompt_sha256=prompt_sha256,
            pattern_signal=pattern,
            embedding_signal=embedding,
            llm_classifier_signal=llm,
            context_boundary_signal=context,
            pipeline_short_circuited=pipeline_short_circuited,
            total_pipeline_time_ms=total_pipeline_time_ms,
        )
        return self.score(ensemble)
