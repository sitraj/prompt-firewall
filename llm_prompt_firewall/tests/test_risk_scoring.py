"""
Tests for RiskScorer.

All tests are pure unit tests — no detector calls, no I/O.
Signals are constructed directly from schema models using known values.

Test coverage:
  - WeightConfig: per-detector weight lookup, custom config.
  - ThresholdConfig: boundary conditions for all four RiskLevel tiers,
    and the recommended FirewallAction for each tier.
  - RiskScorer.score():
      short-circuit propagation (confidence=1.0 + pipeline_short_circuited)
      single-detector ensemble (each detector type individually)
      multi-detector ensemble with expected weighted average
      degraded LLM signal is excluded and weights renormalised
      missing signals produce clean zero score
      score always clamped to [0.0, 1.0]
  - Primary threat resolution priority order.
  - Explanation synthesis: key fields present in explanation text.
  - score_from_signals() convenience method.
  - RiskScore model integrity: frozen, valid ranges.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from llm_prompt_firewall.core.risk_scoring import (
    DEFAULT_THRESHOLDS,
    DEFAULT_WEIGHTS,
    RiskScorer,
    ThresholdConfig,
    WeightConfig,
    _resolve_primary_threat,
)
from llm_prompt_firewall.models.schemas import (
    ContextBoundarySignal,
    DetectorEnsemble,
    DetectorType,
    EmbeddingSignal,
    FirewallAction,
    LLMClassifierSignal,
    PatternMatch,
    PatternSignal,
    RiskLevel,
    RiskScore,
    ThreatCategory,
)


# ---------------------------------------------------------------------------
# Signal factories — build minimal valid signal objects for testing
# ---------------------------------------------------------------------------


def _pattern(
    confidence: float = 0.0,
    matched: bool = False,
    category: ThreatCategory = ThreatCategory.INSTRUCTION_OVERRIDE,
) -> PatternSignal:
    matches = []
    if matched:
        matches = [
            PatternMatch(
                pattern_id="PI-001:sig0",
                pattern_text="ignore previous instructions",
                matched_text="ignore previous instructions",
                category=category,
                severity=confidence,
                offset=0,
            )
        ]
    return PatternSignal(
        matched=matched,
        matches=matches,
        confidence=confidence,
        processing_time_ms=0.1,
    )


def _embedding(
    similarity: float = 0.0,
    threshold: float = 0.82,
    attack_id: str = "PI-001",
    category: ThreatCategory = ThreatCategory.INSTRUCTION_OVERRIDE,
) -> EmbeddingSignal:
    exceeded = similarity >= threshold
    return EmbeddingSignal(
        similarity_score=similarity,
        nearest_attack_id=attack_id if similarity > 0 else None,
        nearest_attack_category=category if similarity > 0 else None,
        chunk_index=0 if similarity > 0 else None,
        threshold_used=threshold,
        exceeded_threshold=exceeded,
        processing_time_ms=10.0,
    )


def _llm(
    risk_score: float = 0.0,
    category: ThreatCategory = ThreatCategory.INSTRUCTION_OVERRIDE,
    degraded: bool = False,
    reasoning: str = "Test reasoning.",
) -> LLMClassifierSignal:
    return LLMClassifierSignal(
        risk_score=risk_score,
        threat_category=category,
        reasoning=reasoning,
        degraded=degraded,
        model_used="gpt-4o-mini",
        processing_time_ms=250.0,
    )


def _context(
    confidence: float = 0.0,
    violated: bool = False,
) -> ContextBoundarySignal:
    return ContextBoundarySignal(
        boundary_violation_detected=violated,
        violated_boundaries=["system_prompt"] if violated else [],
        indirect_injection_suspected=False,
        multi_turn_escalation=False,
        confidence=confidence,
        processing_time_ms=1.0,
    )


def _ensemble(
    pattern: PatternSignal | None = None,
    embedding: EmbeddingSignal | None = None,
    llm: LLMClassifierSignal | None = None,
    context: ContextBoundarySignal | None = None,
    short_circuited: bool = False,
    sha: str = "abc123def456",
) -> DetectorEnsemble:
    return DetectorEnsemble(
        prompt_sha256=sha,
        pattern_signal=pattern,
        embedding_signal=embedding,
        llm_classifier_signal=llm,
        context_boundary_signal=context,
        pipeline_short_circuited=short_circuited,
        total_pipeline_time_ms=50.0,
    )


# ---------------------------------------------------------------------------
# WeightConfig tests
# ---------------------------------------------------------------------------


class TestWeightConfig:
    def test_defaults_are_positive(self):
        w = WeightConfig()
        assert w.pattern > 0
        assert w.embedding > 0
        assert w.llm_classifier > 0
        assert w.context_boundary > 0

    def test_for_detector_lookup(self):
        w = WeightConfig(pattern=0.40, embedding=0.30, llm_classifier=0.20, context_boundary=0.10)
        assert w.for_detector(DetectorType.PATTERN) == pytest.approx(0.40)
        assert w.for_detector(DetectorType.EMBEDDING) == pytest.approx(0.30)
        assert w.for_detector(DetectorType.LLM_CLASSIFIER) == pytest.approx(0.20)
        assert w.for_detector(DetectorType.CONTEXT_BOUNDARY) == pytest.approx(0.10)

    def test_unknown_detector_returns_zero(self):
        w = WeightConfig()
        # Pass a value not in the map — should return 0.0 safely
        assert w.for_detector("nonexistent") == 0.0  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# ThresholdConfig tests
# ---------------------------------------------------------------------------


class TestThresholdConfig:
    def test_safe_boundary(self):
        t = ThresholdConfig()
        assert t.risk_level(0.00) == RiskLevel.SAFE
        assert t.risk_level(0.39) == RiskLevel.SAFE
        assert t.risk_level(0.40) == RiskLevel.SUSPICIOUS

    def test_suspicious_boundary(self):
        t = ThresholdConfig()
        assert t.risk_level(0.40) == RiskLevel.SUSPICIOUS
        assert t.risk_level(0.69) == RiskLevel.SUSPICIOUS
        assert t.risk_level(0.70) == RiskLevel.HIGH

    def test_high_boundary(self):
        t = ThresholdConfig()
        assert t.risk_level(0.70) == RiskLevel.HIGH
        assert t.risk_level(0.84) == RiskLevel.HIGH
        assert t.risk_level(0.85) == RiskLevel.CRITICAL

    def test_critical_boundary(self):
        t = ThresholdConfig()
        assert t.risk_level(0.85) == RiskLevel.CRITICAL
        assert t.risk_level(1.00) == RiskLevel.CRITICAL

    def test_recommended_action_mapping(self):
        t = ThresholdConfig()
        assert t.recommended_action(RiskLevel.SAFE) == FirewallAction.ALLOW
        assert t.recommended_action(RiskLevel.SUSPICIOUS) == FirewallAction.LOG
        assert t.recommended_action(RiskLevel.HIGH) == FirewallAction.SANITIZE
        assert t.recommended_action(RiskLevel.CRITICAL) == FirewallAction.BLOCK

    def test_custom_thresholds(self):
        t = ThresholdConfig(safe_max=0.20, suspicious_max=0.50, high_max=0.75)
        assert t.risk_level(0.20) == RiskLevel.SAFE
        assert t.risk_level(0.21) == RiskLevel.SUSPICIOUS
        assert t.risk_level(0.50) == RiskLevel.SUSPICIOUS
        assert t.risk_level(0.51) == RiskLevel.HIGH
        assert t.risk_level(0.76) == RiskLevel.CRITICAL


# ---------------------------------------------------------------------------
# RiskScorer.score() — short-circuit propagation
# ---------------------------------------------------------------------------


class TestShortCircuit:
    def test_short_circuit_returns_score_1(self):
        scorer = RiskScorer()
        ens = _ensemble(
            pattern=_pattern(confidence=1.0, matched=True),
            short_circuited=True,
        )
        risk = scorer.score(ens)
        assert risk.score == pytest.approx(1.0)
        assert risk.level == RiskLevel.CRITICAL

    def test_short_circuit_contributing_detectors_is_pattern_only(self):
        scorer = RiskScorer()
        ens = _ensemble(
            pattern=_pattern(confidence=1.0, matched=True),
            embedding=_embedding(similarity=0.9),
            llm=_llm(risk_score=0.8),
            short_circuited=True,
        )
        risk = scorer.score(ens)
        # When short-circuited, only pattern is listed as contributor
        assert risk.contributing_detectors == [DetectorType.PATTERN]

    def test_short_circuit_requires_both_flag_and_confidence_1(self):
        scorer = RiskScorer()
        # pipeline_short_circuited=True but confidence < 1.0 → normal scoring
        ens = _ensemble(
            pattern=_pattern(confidence=0.80, matched=True),
            short_circuited=True,
        )
        risk = scorer.score(ens)
        # Should NOT get 1.0 — the short-circuit condition isn't fully met
        assert risk.score < 1.0


# ---------------------------------------------------------------------------
# RiskScorer.score() — single detector ensembles
# ---------------------------------------------------------------------------


class TestSingleDetectorEnsembles:
    def test_pattern_only_score_equals_confidence(self):
        scorer = RiskScorer()
        ens = _ensemble(pattern=_pattern(confidence=0.75, matched=True))
        risk = scorer.score(ens)
        # With only one detector, renormalised weight is 1.0
        # Score = pattern_confidence * 1.0 = 0.75
        assert risk.score == pytest.approx(0.75, abs=1e-3)
        assert DetectorType.PATTERN in risk.contributing_detectors

    def test_embedding_only_score_equals_similarity(self):
        scorer = RiskScorer()
        ens = _ensemble(embedding=_embedding(similarity=0.88))
        risk = scorer.score(ens)
        assert risk.score == pytest.approx(0.88, abs=1e-3)
        assert DetectorType.EMBEDDING in risk.contributing_detectors

    def test_llm_only_score_equals_risk_score(self):
        scorer = RiskScorer()
        ens = _ensemble(llm=_llm(risk_score=0.65))
        risk = scorer.score(ens)
        assert risk.score == pytest.approx(0.65, abs=1e-3)

    def test_context_only_score_equals_confidence(self):
        scorer = RiskScorer()
        ens = _ensemble(context=_context(confidence=0.55, violated=True))
        risk = scorer.score(ens)
        assert risk.score == pytest.approx(0.55, abs=1e-3)


# ---------------------------------------------------------------------------
# RiskScorer.score() — multi-detector ensemble math
# ---------------------------------------------------------------------------


class TestMultiDetectorEnsemble:
    def test_two_detector_weighted_average(self):
        """
        With pattern (w=0.30, score=1.0) and embedding (w=0.25, score=0.0):
        total_weight = 0.55
        weighted_sum = 0.30 * 1.0 + 0.25 * 0.0 = 0.30
        expected_score = 0.30 / 0.55 ≈ 0.5455
        """
        scorer = RiskScorer()
        ens = _ensemble(
            pattern=_pattern(confidence=1.0, matched=True),
            embedding=_embedding(similarity=0.0),
        )
        risk = scorer.score(ens)
        expected = (0.30 * 1.0 + 0.25 * 0.0) / (0.30 + 0.25)
        assert risk.score == pytest.approx(expected, abs=1e-3)

    def test_three_detector_weighted_average(self):
        """
        pattern=0.90 (w=0.30), embedding=0.85 (w=0.25), llm=0.80 (w=0.35)
        total = 0.90, weighted_sum = 0.30*0.90 + 0.25*0.85 + 0.35*0.80
                                   = 0.270 + 0.2125 + 0.280 = 0.7625
        score = 0.7625 / 0.90 ≈ 0.8472
        """
        scorer = RiskScorer()
        ens = _ensemble(
            pattern=_pattern(confidence=0.90, matched=True),
            embedding=_embedding(similarity=0.85),
            llm=_llm(risk_score=0.80),
        )
        risk = scorer.score(ens)
        tw = 0.30 + 0.25 + 0.35
        expected = (0.30 * 0.90 + 0.25 * 0.85 + 0.35 * 0.80) / tw
        assert risk.score == pytest.approx(expected, abs=1e-3)

    def test_degraded_llm_excluded_from_ensemble(self):
        """
        Degraded LLM (risk_score irrelevant) must not contribute.
        With pattern=0.70 (w=0.30) and degraded llm:
        total_weight = 0.30 (llm excluded)
        score = 0.70
        """
        scorer = RiskScorer()
        ens = _ensemble(
            pattern=_pattern(confidence=0.70, matched=True),
            llm=_llm(risk_score=0.99, degraded=True),  # degraded — must not inflate score
        )
        risk = scorer.score(ens)
        assert risk.score == pytest.approx(0.70, abs=1e-3)
        assert DetectorType.LLM_CLASSIFIER not in risk.contributing_detectors

    def test_degraded_llm_weights_renormalised(self):
        """
        After excluding degraded LLM, remaining weights renormalise.
        pattern=1.0 (w=0.30), embedding=0.0 (w=0.25), llm=degraded
        total = 0.55, weighted_sum = 0.30
        score ≈ 0.5455
        """
        scorer = RiskScorer()
        ens = _ensemble(
            pattern=_pattern(confidence=1.0, matched=True),
            embedding=_embedding(similarity=0.0),
            llm=_llm(risk_score=0.0, degraded=True),
        )
        risk = scorer.score(ens)
        expected = 0.30 / (0.30 + 0.25)
        assert risk.score == pytest.approx(expected, abs=1e-3)

    def test_all_detectors_zero_score_is_safe(self):
        scorer = RiskScorer()
        ens = _ensemble(
            pattern=_pattern(confidence=0.0, matched=False),
            embedding=_embedding(similarity=0.0),
            llm=_llm(risk_score=0.0),
        )
        risk = scorer.score(ens)
        assert risk.score == pytest.approx(0.0, abs=1e-3)
        assert risk.level == RiskLevel.SAFE

    def test_custom_weights_change_score(self):
        # Give LLM all the weight
        weights = WeightConfig(
            pattern=0.0, embedding=0.0, llm_classifier=1.0, context_boundary=0.0
        )
        scorer = RiskScorer(weights=weights)
        ens = _ensemble(
            pattern=_pattern(confidence=1.0, matched=True),
            llm=_llm(risk_score=0.50),
        )
        risk = scorer.score(ens)
        # Only LLM contributes (pattern weight=0 so it's effectively excluded from math,
        # but it's still in the active signals list — weight 0 makes contribution 0)
        # total_weight = 0 + 1.0 = 1.0, weighted_sum = 0*1.0 + 1.0*0.50 = 0.50
        assert risk.score == pytest.approx(0.50, abs=1e-3)


# ---------------------------------------------------------------------------
# RiskScorer — empty ensemble
# ---------------------------------------------------------------------------


class TestEmptyEnsemble:
    def test_no_signals_returns_zero_score(self):
        scorer = RiskScorer()
        ens = _ensemble()  # all signals None
        risk = scorer.score(ens)
        assert risk.score == 0.0
        assert risk.level == RiskLevel.SAFE
        assert risk.contributing_detectors == []

    def test_zero_score_is_allow_action(self):
        t = ThresholdConfig()
        assert t.recommended_action(RiskLevel.SAFE) == FirewallAction.ALLOW


# ---------------------------------------------------------------------------
# Score clamping
# ---------------------------------------------------------------------------


class TestScoreClamping:
    def test_score_never_exceeds_1(self):
        scorer = RiskScorer()
        # All detectors at maximum
        ens = _ensemble(
            pattern=_pattern(confidence=1.0, matched=True),
            embedding=_embedding(similarity=1.0),
            llm=_llm(risk_score=1.0),
            context=_context(confidence=1.0, violated=True),
        )
        risk = scorer.score(ens)
        assert risk.score <= 1.0

    def test_score_never_below_0(self):
        scorer = RiskScorer()
        ens = _ensemble(
            pattern=_pattern(confidence=0.0),
            embedding=_embedding(similarity=0.0),
            llm=_llm(risk_score=0.0),
        )
        risk = scorer.score(ens)
        assert risk.score >= 0.0


# ---------------------------------------------------------------------------
# Primary threat resolution
# ---------------------------------------------------------------------------


class TestPrimaryThreatResolution:
    def test_llm_classifier_takes_priority(self):
        """LLM classifier category wins over pattern when both fire."""
        pattern_sig = _pattern(confidence=0.9, matched=True, category=ThreatCategory.JAILBREAK)
        llm_sig = _llm(risk_score=0.85, category=ThreatCategory.DATA_EXFILTRATION)
        threat = _resolve_primary_threat(pattern_sig, None, llm_sig, None)
        assert threat == ThreatCategory.DATA_EXFILTRATION

    def test_pattern_used_when_llm_absent(self):
        pattern_sig = _pattern(confidence=0.9, matched=True, category=ThreatCategory.PROMPT_EXTRACTION)
        threat = _resolve_primary_threat(pattern_sig, None, None, None)
        assert threat == ThreatCategory.PROMPT_EXTRACTION

    def test_embedding_used_when_pattern_and_llm_absent(self):
        emb_sig = _embedding(similarity=0.90, category=ThreatCategory.TOOL_ABUSE)
        threat = _resolve_primary_threat(None, emb_sig, None, None)
        assert threat == ThreatCategory.TOOL_ABUSE

    def test_degraded_llm_falls_through_to_pattern(self):
        """A degraded LLM should not be used for category resolution."""
        pattern_sig = _pattern(confidence=0.8, matched=True, category=ThreatCategory.JAILBREAK)
        degraded_llm = _llm(risk_score=0.0, category=ThreatCategory.DATA_EXFILTRATION, degraded=True)
        threat = _resolve_primary_threat(pattern_sig, None, degraded_llm, None)
        assert threat == ThreatCategory.JAILBREAK

    def test_no_signals_returns_unknown(self):
        threat = _resolve_primary_threat(None, None, None, None)
        assert threat == ThreatCategory.UNKNOWN

    def test_low_llm_score_falls_through(self):
        """LLM with score < 0.30 should not determine the category."""
        pattern_sig = _pattern(confidence=0.8, matched=True, category=ThreatCategory.JAILBREAK)
        low_llm = _llm(risk_score=0.15, category=ThreatCategory.DATA_EXFILTRATION, degraded=False)
        threat = _resolve_primary_threat(pattern_sig, None, low_llm, None)
        assert threat == ThreatCategory.JAILBREAK


# ---------------------------------------------------------------------------
# Explanation synthesis
# ---------------------------------------------------------------------------


class TestExplanation:
    def test_explanation_contains_score(self):
        scorer = RiskScorer()
        ens = _ensemble(pattern=_pattern(confidence=0.85, matched=True))
        risk = scorer.score(ens)
        assert "0.85" in risk.explanation or "0." in risk.explanation

    def test_explanation_contains_level(self):
        scorer = RiskScorer()
        ens = _ensemble(pattern=_pattern(confidence=0.85, matched=True))
        risk = scorer.score(ens)
        assert risk.level.value in risk.explanation

    def test_explanation_mentions_pattern_matches(self):
        scorer = RiskScorer()
        ens = _ensemble(pattern=_pattern(confidence=0.85, matched=True))
        risk = scorer.score(ens)
        assert "Pattern" in risk.explanation or "pattern" in risk.explanation

    def test_explanation_mentions_degraded_llm(self):
        # A degraded-only ensemble has no active signals — the "no signals" early
        # return fires before _build_explanation(). Add a pattern signal so the
        # explanation path is reached and the degraded note appears.
        scorer = RiskScorer()
        ens = _ensemble(
            pattern=_pattern(confidence=0.50, matched=True),
            llm=_llm(risk_score=0.0, degraded=True),
        )
        risk = scorer.score(ens)
        assert "DEGRADED" in risk.explanation

    def test_short_circuit_explanation_notes_it(self):
        scorer = RiskScorer()
        ens = _ensemble(
            pattern=_pattern(confidence=1.0, matched=True),
            short_circuited=True,
        )
        risk = scorer.score(ens)
        assert "short-circuit" in risk.explanation.lower()


# ---------------------------------------------------------------------------
# score_from_signals() convenience method
# ---------------------------------------------------------------------------


class TestScoreFromSignals:
    def test_convenience_method_matches_direct_score(self):
        scorer = RiskScorer()

        pattern_sig = _pattern(confidence=0.80, matched=True)
        emb_sig = _embedding(similarity=0.75)

        direct = scorer.score(_ensemble(pattern=pattern_sig, embedding=emb_sig, sha="test123"))
        convenience = scorer.score_from_signals(
            prompt_sha256="test123",
            pattern=pattern_sig,
            embedding=emb_sig,
        )
        assert direct.score == pytest.approx(convenience.score, abs=1e-4)
        assert direct.level == convenience.level

    def test_convenience_method_works_with_no_signals(self):
        scorer = RiskScorer()
        risk = scorer.score_from_signals(prompt_sha256="empty123")
        assert risk.score == 0.0


# ---------------------------------------------------------------------------
# RiskScore model integrity
# ---------------------------------------------------------------------------


class TestRiskScoreIntegrity:
    def test_risk_score_is_frozen(self):
        from pydantic import ValidationError
        scorer = RiskScorer()
        risk = scorer.score(_ensemble(pattern=_pattern(confidence=0.5, matched=True)))
        with pytest.raises((TypeError, AttributeError, ValidationError)):
            risk.score = 0.0  # type: ignore[misc]

    def test_score_in_valid_range(self):
        scorer = RiskScorer()
        for conf in [0.0, 0.25, 0.50, 0.75, 1.0]:
            risk = scorer.score(_ensemble(pattern=_pattern(confidence=conf, matched=conf > 0)))
            assert 0.0 <= risk.score <= 1.0

    def test_contributing_detectors_are_valid_enum_values(self):
        scorer = RiskScorer()
        ens = _ensemble(
            pattern=_pattern(confidence=0.5, matched=True),
            llm=_llm(risk_score=0.6),
        )
        risk = scorer.score(ens)
        for det in risk.contributing_detectors:
            assert isinstance(det, DetectorType)

    def test_primary_threat_is_valid_enum(self):
        scorer = RiskScorer()
        risk = scorer.score(_ensemble(pattern=_pattern(confidence=0.8, matched=True)))
        assert isinstance(risk.primary_threat, ThreatCategory)
