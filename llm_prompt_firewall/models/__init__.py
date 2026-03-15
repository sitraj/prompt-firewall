"""
Data models for the LLM Prompt Injection Firewall.

This module exports all Pydantic schemas used across the firewall system.
Centralising schema definitions here enforces a single source of truth and
prevents circular imports between detector/filter/core layers.
"""

from llm_prompt_firewall.models.schemas import (
    # Enumerations
    ThreatCategory,
    DetectorType,
    RiskLevel,
    FirewallAction,
    PromptRole,
    # Input / context models
    PromptContext,
    SessionMetadata,
    # Per-detector signal models
    PatternMatch,
    PatternSignal,
    EmbeddingSignal,
    LLMClassifierSignal,
    ContextBoundarySignal,
    DetectorSignal,
    # Aggregation models
    DetectorEnsemble,
    RiskScore,
    # Decision models
    FirewallDecision,
    SanitizedPrompt,
    # Response models
    SafeResponse,
    BlockedResponse,
    RedactedResponse,
    # Output inspection models
    SecretMatch,
    OutputInspectionResult,
    # Audit / observability
    AuditEvent,
    # Dataset models
    AttackVariant,
    AttackSample,
    AttackDataset,
)

__all__ = [
    "ThreatCategory",
    "DetectorType",
    "RiskLevel",
    "FirewallAction",
    "PromptRole",
    "PromptContext",
    "SessionMetadata",
    "PatternMatch",
    "PatternSignal",
    "EmbeddingSignal",
    "LLMClassifierSignal",
    "ContextBoundarySignal",
    "DetectorSignal",
    "DetectorEnsemble",
    "RiskScore",
    "FirewallDecision",
    "SanitizedPrompt",
    "SafeResponse",
    "BlockedResponse",
    "RedactedResponse",
    "SecretMatch",
    "OutputInspectionResult",
    "AuditEvent",
    "AttackVariant",
    "AttackSample",
    "AttackDataset",
]
