"""
Pattern-Based Prompt Injection Detector
========================================

The PatternDetector is the first and fastest stage of the firewall's detection
pipeline. It inspects prompts against a compiled library of known attack
signatures using regex matching.

Design principles:
  - Zero per-request compilation cost. All patterns are compiled into
    re.Pattern objects once at startup and cached on the PatternLibrary.
  - Unicode normalisation is applied before matching. This defeats homoglyph
    substitution (e.g. Cyrillic 'а' standing in for Latin 'a') and leet-speak
    tricks (normalised via a custom mapping before pattern matching).
  - Invisible character stripping. Zero-width spaces, soft hyphens, and
    bidirectional overrides are removed before comparison.
  - Short-circuit semantics. A CRITICAL severity hit immediately returns with
    confidence=1.0 without evaluating remaining patterns — the pipeline can
    skip expensive embedding/LLM calls for clear-cut attacks.
  - Patterns are partitioned by ThreatCategory so the detector can report
    which attack family triggered, not just that something matched.
  - False-positive awareness. Each PatternEntry carries an is_anchored flag.
    Unanchored broad patterns (e.g. bare 'ignore') have their confidence
    dampened so a single broad hit does not auto-block a legitimate request.
"""

from __future__ import annotations

import logging
import re
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

from llm_prompt_firewall.models.schemas import (
    AttackDataset,
    AttackSample,
    FirewallAction,
    PatternMatch,
    PatternSignal,
    RiskLevel,
    ThreatCategory,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Minimum pattern length to accept from the dataset.
# Patterns shorter than this are too broad and would generate excessive
# false positives (e.g. a 3-char pattern like "dan" would match surnames).
MIN_PATTERN_LENGTH: Final[int] = 6

# Severity weight assigned to each RiskLevel tier.
# These are used when computing the aggregate confidence score from multiple
# matched patterns. Weights are intentionally asymmetric — a CRITICAL match
# dominates the score regardless of how many LOW matches also fired.
SEVERITY_WEIGHTS: Final[dict[str, float]] = {
    RiskLevel.CRITICAL: 1.00,
    RiskLevel.HIGH: 0.75,
    RiskLevel.SUSPICIOUS: 0.45,
    RiskLevel.SAFE: 0.10,
}

# Confidence dampening factor applied to broad/unanchored patterns.
# An unanchored pattern that matches is less certain evidence of an attack
# than a tightly anchored one.
BROAD_PATTERN_DAMPEN: Final[float] = 0.70

# Regex flags applied to all compiled patterns.
# IGNORECASE: attacks are case-insensitive by nature.
# UNICODE: ensure \w, \b etc work correctly across scripts.
PATTERN_FLAGS: Final[int] = re.IGNORECASE | re.UNICODE

# Invisible / zero-width Unicode characters commonly used to split attack
# keywords and evade detection. These are stripped before matching.
INVISIBLE_CHARS: Final[frozenset[str]] = frozenset(
    [
        "\u200b",  # ZERO WIDTH SPACE
        "\u200c",  # ZERO WIDTH NON-JOINER
        "\u200d",  # ZERO WIDTH JOINER
        "\u00ad",  # SOFT HYPHEN
        "\u2060",  # WORD JOINER
        "\ufeff",  # ZERO WIDTH NO-BREAK SPACE (BOM)
        "\u202a",  # LEFT-TO-RIGHT EMBEDDING
        "\u202b",  # RIGHT-TO-LEFT EMBEDDING
        "\u202c",  # POP DIRECTIONAL FORMATTING
        "\u202d",  # LEFT-TO-RIGHT OVERRIDE
        "\u202e",  # RIGHT-TO-LEFT OVERRIDE
        "\u2066",  # LEFT-TO-RIGHT ISOLATE
        "\u2067",  # RIGHT-TO-LEFT ISOLATE
        "\u2069",  # POP DIRECTIONAL ISOLATE
    ]
)

# Leet-speak normalisation table. Maps common leet substitutions back to the
# canonical ASCII letter so that "1gnore" is treated the same as "ignore".
# Applied after unicode normalisation and invisible char stripping.
LEET_MAP: Final[dict[str, str]] = {
    "0": "o",
    "1": "i",
    "3": "e",
    "4": "a",
    "5": "s",
    "7": "t",
    "@": "a",
    "$": "s",
    "|": "i",
    "!": "i",
}


# ---------------------------------------------------------------------------
# Internal data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PatternEntry:
    """
    A single compiled attack pattern with its associated metadata.

    Frozen dataclass — entries are built once at startup and never mutated.
    """

    pattern_id: str
    """Unique ID in the format 'CATEGORY-NNN:signature_index'."""

    compiled: re.Pattern[str]
    """The compiled regex. Never re-compiled at request time."""

    raw_pattern: str
    """The original regex string, stored for audit reporting."""

    category: ThreatCategory

    severity: RiskLevel

    expected_action: FirewallAction

    is_broad: bool = False
    """
    True when the pattern lacks word boundaries or anchors and could produce
    false positives on benign text. Broad patterns have their confidence score
    dampened (multiplied by BROAD_PATTERN_DAMPEN) to avoid auto-blocking
    legitimate requests that happen to contain a trigger substring.
    """


@dataclass
class PatternLibrary:
    """
    The compiled pattern library, built once at startup from an AttackDataset.

    Holds all PatternEntry objects partitioned by ThreatCategory for efficient
    per-category analysis and reporting.

    Design note: PatternLibrary is a dataclass (not a Pydantic model) because
    it holds compiled re.Pattern objects which are not JSON-serialisable. It
    is an internal implementation detail of PatternDetector, never serialised.
    """

    entries: list[PatternEntry] = field(default_factory=list)
    entries_by_category: dict[ThreatCategory, list[PatternEntry]] = field(default_factory=dict)
    total_patterns: int = 0
    dataset_version: str = "unknown"

    @classmethod
    def from_dataset(cls, dataset: AttackDataset) -> PatternLibrary:
        """
        Build a PatternLibrary from a validated AttackDataset.

        Patterns shorter than MIN_PATTERN_LENGTH are skipped and logged.
        Invalid regex patterns are caught and skipped with a WARNING so a
        single bad entry in the dataset does not bring down the firewall.
        """
        lib = cls(dataset_version=dataset.version)

        for sample in dataset.samples:
            lib._load_sample_patterns(sample)

        lib.total_patterns = len(lib.entries)
        logger.info(
            "PatternLibrary built: %d patterns across %d categories from dataset v%s",
            lib.total_patterns,
            len(lib.entries_by_category),
            lib.dataset_version,
        )
        return lib

    def _load_sample_patterns(self, sample: AttackSample) -> None:
        for idx, raw in enumerate(sample.pattern_signatures):
            if len(raw) < MIN_PATTERN_LENGTH:
                logger.debug(
                    "Skipping short pattern '%s' from sample %s (len=%d < %d)",
                    raw,
                    sample.id,
                    len(raw),
                    MIN_PATTERN_LENGTH,
                )
                continue

            try:
                compiled = re.compile(raw, PATTERN_FLAGS)
            except re.error as exc:
                logger.warning(
                    "Invalid regex in sample %s, signature index %d: %s — skipping. Error: %s",
                    sample.id,
                    idx,
                    raw,
                    exc,
                )
                continue

            is_broad = _is_broad_pattern(raw)

            entry = PatternEntry(
                pattern_id=f"{sample.id}:sig{idx}",
                compiled=compiled,
                raw_pattern=raw,
                category=sample.category,
                severity=sample.severity,
                expected_action=sample.expected_action,
                is_broad=is_broad,
            )

            self.entries.append(entry)
            self.entries_by_category.setdefault(sample.category, []).append(entry)


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------


def _is_broad_pattern(pattern: str) -> bool:
    """
    Heuristic: a pattern is 'broad' if it lacks anchors or word boundaries.

    Broad patterns match anywhere in the string without requiring word context.
    We consider a pattern anchored if it uses \\b, ^, $, or has a minimum
    length of 20+ characters (long literals are naturally selective).
    """
    has_boundary = r"\b" in pattern or pattern.startswith("^") or pattern.endswith("$")
    is_long_literal = len(pattern) >= 20 and not any(c in pattern for c in r"?+*{[")
    return not (has_boundary or is_long_literal)


def _strip_invisible(text: str) -> str:
    """Remove invisible and zero-width Unicode characters from text."""
    return "".join(ch for ch in text if ch not in INVISIBLE_CHARS)


def _unicode_normalise(text: str) -> str:
    """
    Apply NFKC normalisation.

    NFKC (Compatibility Decomposition + Canonical Composition) maps compatibility
    characters to their canonical equivalents. This handles a wide range of
    homoglyph tricks:
      - Mathematical bold/italic letters → ASCII letters
      - Full-width ASCII → ASCII
      - Ligatures (ﬁ → fi)
      - Superscript/subscript digits → plain digits
    It does NOT handle cross-script homoglyphs (Cyrillic 'а' stays Cyrillic).
    Those require the confusables mapping applied in the embedding detector.
    """
    return unicodedata.normalize("NFKC", text)


def _apply_leet_normalisation(text: str) -> str:
    """
    Map leet-speak digit/symbol substitutions back to ASCII letters.

    Applied character-by-character. Only substitutes characters that are
    in the LEET_MAP. This is intentionally conservative — we only map the
    most unambiguous substitutions to avoid distorting legitimate numeric text.
    """
    return "".join(LEET_MAP.get(ch, ch) for ch in text)


def normalise_for_matching(text: str) -> str:
    """
    Full normalisation pipeline applied to a prompt before pattern matching.

    Order matters:
      1. Strip invisible chars — removes split-keyword tricks before anything else.
      2. Unicode NFKC — maps compatibility variants to canonical forms.
      3. Leet normalisation — maps digit/symbol substitutions to letters.

    The original text is preserved in PromptContext.raw_prompt. This function
    returns a normalised copy used only for matching — it is never forwarded
    to the LLM.
    """
    text = _strip_invisible(text)
    text = _unicode_normalise(text)
    text = _apply_leet_normalisation(text)
    return text


# ---------------------------------------------------------------------------
# PatternDetector
# ---------------------------------------------------------------------------


class PatternDetector:
    """
    Detects known prompt injection attack signatures via compiled regex matching.

    Lifecycle:
        detector = PatternDetector.from_dataset(dataset)
        signal = detector.inspect(prompt_context)

    The detector is stateless with respect to individual requests — the only
    state is the compiled PatternLibrary, which is shared and read-only across
    concurrent requests.

    Thread safety: PatternDetector is safe to share across threads/coroutines.
    re.Pattern.search() holds the GIL but releases it between calls. The
    library itself is never mutated after construction.
    """

    def __init__(self, library: PatternLibrary) -> None:
        self._library = library
        logger.info(
            "PatternDetector initialised with %d compiled patterns (dataset v%s)",
            self._library.total_patterns,
            self._library.dataset_version,
        )

    @classmethod
    def from_dataset(cls, dataset: AttackDataset) -> PatternDetector:
        """Convenience constructor that builds the PatternLibrary from an AttackDataset."""
        library = PatternLibrary.from_dataset(dataset)
        return cls(library)

    @classmethod
    def from_dataset_file(cls, path: Path) -> PatternDetector:
        """
        Load an AttackDataset from a JSON file on disk and build the detector.

        Validates the JSON against the AttackDataset schema. Raises ValueError
        if the file is missing, malformed, or fails schema validation.
        """
        import json

        if not path.exists():
            raise ValueError(f"Attack dataset file not found: {path}")

        try:
            with path.open(encoding="utf-8") as fh:
                raw = json.load(fh)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Attack dataset JSON is malformed: {exc}") from exc

        # Datetime fields need coercion from ISO strings
        from datetime import datetime

        for dt_field in ("created_at", "updated_at"):
            if isinstance(raw.get(dt_field), str):
                raw[dt_field] = datetime.fromisoformat(raw[dt_field].replace("Z", "+00:00"))

        dataset = AttackDataset(**raw)
        return cls.from_dataset(dataset)

    # ------------------------------------------------------------------
    # Core inspection
    # ------------------------------------------------------------------

    def inspect(self, raw_prompt: str) -> PatternSignal:
        """
        Inspect a prompt against the full pattern library.

        Args:
            raw_prompt: The untrusted prompt text to inspect. Must be the raw,
                        unmodified text — normalisation is applied internally.

        Returns:
            PatternSignal with matched patterns, aggregate confidence, and
            processing time.

        Design:
            - Patterns are evaluated in severity order (CRITICAL first) so that
              a hard hit can short-circuit immediately.
            - Each match records the specific substring that triggered it and
              its character offset, enabling precise audit reporting.
            - Confidence is derived from the maximum severity weight of all
              matched patterns, boosted by a multi-match bonus when several
              independent patterns fire simultaneously.
        """
        start = time.perf_counter()

        normalised = normalise_for_matching(raw_prompt)
        matches: list[PatternMatch] = []

        # Evaluate CRITICAL patterns first to enable early exit
        sorted_entries = sorted(
            self._library.entries,
            key=lambda e: SEVERITY_WEIGHTS.get(e.severity, 0.0),
            reverse=True,
        )

        for entry in sorted_entries:
            match = entry.compiled.search(normalised)
            if match is None:
                continue

            matched_text = match.group(0)
            severity_weight = SEVERITY_WEIGHTS.get(entry.severity, 0.5)
            # Dampen broad patterns to avoid over-confident false positives
            effective_weight = (
                severity_weight * BROAD_PATTERN_DAMPEN if entry.is_broad else severity_weight
            )

            pattern_match = PatternMatch(
                pattern_id=entry.pattern_id,
                pattern_text=entry.raw_pattern,
                matched_text=matched_text,
                category=entry.category,
                severity=effective_weight,
                offset=match.start(),
            )
            matches.append(pattern_match)

            # Short-circuit: undampened CRITICAL hit → maximum confidence block
            if entry.severity == RiskLevel.CRITICAL and not entry.is_broad:
                logger.debug(
                    "Short-circuit triggered by CRITICAL pattern '%s' at offset %d",
                    entry.pattern_id,
                    match.start(),
                )
                processing_ms = (time.perf_counter() - start) * 1000
                return PatternSignal(
                    matched=True,
                    matches=matches,
                    confidence=1.0,
                    processing_time_ms=round(processing_ms, 3),
                )

        processing_ms = (time.perf_counter() - start) * 1000

        if not matches:
            return PatternSignal(
                matched=False,
                matches=[],
                confidence=0.0,
                processing_time_ms=round(processing_ms, 3),
            )

        confidence = _compute_confidence(matches)
        return PatternSignal(
            matched=True,
            matches=matches,
            confidence=confidence,
            processing_time_ms=round(processing_ms, 3),
        )

    def inspect_category(self, raw_prompt: str, category: ThreatCategory) -> PatternSignal:
        """
        Inspect against patterns from a single ThreatCategory only.

        Used by the RAG injection scanner, which only needs to check the
        rag_injection category against retrieved document content rather than
        running the full library (which would generate noise on retrieved text).
        """
        start = time.perf_counter()
        normalised = normalise_for_matching(raw_prompt)
        entries = self._library.entries_by_category.get(category, [])
        matches: list[PatternMatch] = []

        for entry in entries:
            match = entry.compiled.search(normalised)
            if match is None:
                continue

            severity_weight = SEVERITY_WEIGHTS.get(entry.severity, 0.5)
            effective_weight = (
                severity_weight * BROAD_PATTERN_DAMPEN if entry.is_broad else severity_weight
            )
            matches.append(
                PatternMatch(
                    pattern_id=entry.pattern_id,
                    pattern_text=entry.raw_pattern,
                    matched_text=match.group(0),
                    category=entry.category,
                    severity=effective_weight,
                    offset=match.start(),
                )
            )

        processing_ms = (time.perf_counter() - start) * 1000

        if not matches:
            return PatternSignal(
                matched=False,
                matches=[],
                confidence=0.0,
                processing_time_ms=round(processing_ms, 3),
            )

        return PatternSignal(
            matched=True,
            matches=matches,
            confidence=_compute_confidence(matches),
            processing_time_ms=round(processing_ms, 3),
        )

    @property
    def pattern_count(self) -> int:
        return self._library.total_patterns

    @property
    def dataset_version(self) -> str:
        return self._library.dataset_version


# ---------------------------------------------------------------------------
# Confidence scoring
# ---------------------------------------------------------------------------


def _compute_confidence(matches: list[PatternMatch]) -> float:
    """
    Derive an aggregate confidence score from a list of pattern matches.

    Algorithm:
        base      = max severity weight across all matches
                    (a single CRITICAL match dominates)
        bonus     = diminishing returns bonus for N additional independent matches
                    bonus = (N - 1) * 0.05, capped at 0.15
                    Rationale: multiple independent patterns firing simultaneously
                    is stronger evidence than a single match, but we do not want
                    N matches to make the score unreasonably high on its own.
        confidence = min(base + bonus, 1.0)

    The bonus is only applied when matches come from more than one distinct
    ThreatCategory — multiple patterns from the same category may be redundant
    (e.g. two synonymous instruction-override patterns). Cross-category hits
    are genuinely independent signals.
    """
    if not matches:
        return 0.0

    base = max(m.severity for m in matches)

    # Count unique categories that fired
    unique_categories = {m.category for m in matches}
    cross_category_count = len(unique_categories)
    bonus = min((cross_category_count - 1) * 0.05, 0.15)

    return round(min(base + bonus, 1.0), 4)
