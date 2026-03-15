"""
Detector implementations for the LLM Prompt Injection Firewall.

Each detector is an independent, composable unit that takes a prompt (or
normalised text) and returns a typed signal model. Detectors are assembled
into a pipeline by the PromptAnalyzer in the core layer.

Available detectors (in pipeline execution order):
  1. PatternDetector   — regex/literal signature matching (~microseconds)
  2. EmbeddingDetector — semantic similarity via sentence embeddings (~10–50ms)
  3. LLMClassifier     — secondary LLM risk classifier (~200–800ms)
  4. ContextBoundaryDetector — structural context-crossing heuristics (~1ms)
"""

from llm_prompt_firewall.detectors.pattern_detector import (
    PatternDetector,
    PatternLibrary,
    PatternEntry,
    normalise_for_matching,
)
from llm_prompt_firewall.detectors.embedding_detector import (
    EmbeddingDetector,
    EmbeddingDetectorUnavailable,
    EmbeddingIndex,
    IndexEntry,
    chunk_text,
    build_index,
)

__all__ = [
    "PatternDetector",
    "PatternLibrary",
    "PatternEntry",
    "normalise_for_matching",
    "EmbeddingDetector",
    "EmbeddingDetectorUnavailable",
    "EmbeddingIndex",
    "IndexEntry",
    "chunk_text",
    "build_index",
]
