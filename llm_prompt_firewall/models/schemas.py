"""
Core Pydantic schemas for the LLM Prompt Injection Firewall.

Design philosophy:
  - Every inter-component boundary is typed. No dicts passed between layers.
  - Enums are used for all categorical values — no stringly-typed flags.
  - Sensitive fields (raw prompt text) are isolated and never serialised to
    audit logs by default. Use .audit_safe() on models that carry raw text.
  - Immutability is enforced via model_config frozen=True on signal models so
    detector outputs cannot be mutated after the fact (important for audit
    integrity — a detector's signal must reflect what it computed, not what
    downstream code wished it had computed).
  - All float scores are bounded via Annotated validators. Unbounded floats
    are a common source of bugs when aggregating multi-detector ensembles.
"""

from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Annotated, Any

from pydantic import BaseModel, Field, field_validator, model_validator

# ---------------------------------------------------------------------------
# Bounded numeric type aliases
# ---------------------------------------------------------------------------

# All risk/confidence scores are normalised to [0.0, 1.0].
Score = Annotated[float, Field(ge=0.0, le=1.0)]

# Similarity scores from embedding models (cosine similarity in [-1, 1]).
# We clamp to [0.0, 1.0] because negative cosine similarity is not meaningful
# in the context of attack similarity — they are simply "not similar".
SimilarityScore = Annotated[float, Field(ge=0.0, le=1.0)]


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class ThreatCategory(str, Enum):
    """
    Taxonomy of prompt injection and related attack categories.

    These map directly to attack families in the dataset. Keeping them as an
    enum (rather than free-form strings) means new categories must be
    explicitly registered here — preventing silent taxonomy drift between the
    dataset, detectors, and reporting layers.
    """

    INSTRUCTION_OVERRIDE = "instruction_override"
    """Direct attempts to override or ignore system/user instructions."""

    JAILBREAK = "jailbreak"
    """Attempts to make the model abandon its safety guardrails via role-play,
    fictional framing, or claimed special modes."""

    PROMPT_EXTRACTION = "prompt_extraction"
    """Attempts to cause the model to repeat or reveal its system prompt,
    few-shot examples, or hidden instructions."""

    DATA_EXFILTRATION = "data_exfiltration"
    """Attempts to extract private data, documents, or context the model has
    access to (e.g., RAG corpus, tool outputs, prior turns)."""

    TOOL_ABUSE = "tool_abuse"
    """Attempts to misuse model-connected tools (email, code execution, file
    access, web browsing) for malicious purposes."""

    RAG_INJECTION = "rag_injection"
    """Indirect injection: malicious instructions embedded in retrieved
    documents, web pages, or other external content ingested by the model."""

    ENCODING_OBFUSCATION = "encoding_obfuscation"
    """Use of Base64, URL encoding, unicode homoglyphs, l33tspeak, or other
    encoding tricks to evade pattern-based detection."""

    POLICY_BYPASS = "policy_bypass"
    """Attempts to convince the model that its safety policies do not apply in
    the current context (hypothetical framing, authority claims, etc.)."""

    CONTEXT_MANIPULATION = "context_manipulation"
    """Multi-turn attacks that gradually shift model behaviour across a
    conversation rather than attacking in a single turn."""

    UNKNOWN = "unknown"
    """Catch-all for anomalous prompts that do not fit established categories.
    High unknown scores should trigger human review."""


class DetectorType(str, Enum):
    """Identifies which detector produced a signal."""

    PATTERN = "pattern"
    EMBEDDING = "embedding"
    LLM_CLASSIFIER = "llm_classifier"
    CONTEXT_BOUNDARY = "context_boundary"


class RiskLevel(str, Enum):
    """
    Human-readable risk tier derived from the numeric risk score.

    Thresholds are defined in policy configuration, not hardcoded here.
    This enum is for reporting and UI only.
    """

    SAFE = "safe"             # [0.00 – 0.39]
    SUSPICIOUS = "suspicious" # [0.40 – 0.69]
    HIGH = "high"             # [0.70 – 0.84]
    CRITICAL = "critical"     # [0.85 – 1.00]


class FirewallAction(str, Enum):
    """
    Actions the policy engine can instruct the firewall to take.

    Actions are ordered by escalation severity. The policy engine selects
    the highest-severity action warranted by the risk score and active rules.
    """

    ALLOW = "allow"
    """Prompt is clean; pass through to the LLM unchanged."""

    LOG = "log"
    """Prompt is suspicious but allowed. Log at WARNING level for review."""

    SANITIZE = "sanitize"
    """Strip or redact detected injection phrases before forwarding."""

    BLOCK = "block"
    """Reject the prompt entirely. Return a safe error message to the caller."""

    CHALLENGE = "challenge"
    """Request additional verification from the caller (future: CAPTCHA,
    user confirmation flow). Reserved for interactive deployments."""


class PromptRole(str, Enum):
    """
    The conversational role of the message being inspected.

    Mirrors the OpenAI / Anthropic message role taxonomy. Role is significant
    because injection attacks arriving via the 'user' role are more dangerous
    than the same text in an 'assistant' role (which the app controls).
    The context boundary detector uses role to calibrate sensitivity.
    """

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


# ---------------------------------------------------------------------------
# Input / context models
# ---------------------------------------------------------------------------


class SessionMetadata(BaseModel):
    """
    Immutable metadata about the session in which a prompt was generated.

    Session context is critical for detecting multi-turn attacks. A single
    low-risk turn may be benign; a sequence of escalating turns is a pattern.
    The ContextBoundaryDetector uses this to maintain per-session state.
    """

    model_config = {"frozen": True}

    session_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Stable identifier for the conversation session.",
    )
    user_id: str | None = Field(
        default=None,
        description=(
            "Opaque identifier for the authenticated user. Never store PII "
            "here — use an internal user ID that maps to PII in a separate "
            "store."
        ),
    )
    turn_index: int = Field(
        default=0,
        ge=0,
        description="Zero-based index of this message within the session.",
    )
    ip_address: str | None = Field(
        default=None,
        description=(
            "Source IP of the request. Used for rate-limit correlation. "
            "Must be anonymised (last octet zeroed) before storage."
        ),
    )
    application_id: str | None = Field(
        default=None,
        description="Identifier of the application using the firewall.",
    )
    extra: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary application-supplied metadata (e.g., tenant ID).",
    )


class PromptContext(BaseModel):
    """
    The primary input to the firewall's inspection pipeline.

    Wraps the raw prompt text together with all contextual signals the
    detectors need: conversational role, session state, prior turns.

    Design note on raw_prompt:
      The field is intentionally named raw_prompt (not 'prompt' or 'text')
      to make it obvious at call sites that this contains untrusted, potentially
      malicious input that has not yet been inspected.
    """

    model_config = {"frozen": True}

    raw_prompt: str = Field(
        description="The untrusted prompt text to be inspected.",
        min_length=1,
        max_length=128_000,  # ~32k tokens; adjust for model context window
    )
    role: PromptRole = Field(
        default=PromptRole.USER,
        description="Conversational role of this message.",
    )
    session: SessionMetadata = Field(
        default_factory=SessionMetadata,
        description="Session context for multi-turn attack detection.",
    )
    prior_turns: list[str] = Field(
        default_factory=list,
        description=(
            "Recent prior turns (user messages only) for context window "
            "analysis. Cap at 10 turns to bound memory usage."
        ),
        max_length=10,
    )
    system_prompt_hash: str | None = Field(
        default=None,
        description=(
            "SHA-256 hash of the current system prompt. Never pass the "
            "system prompt text itself — only the hash. The output filter "
            "uses this to detect if the model is echoing its system prompt."
        ),
    )

    @field_validator("raw_prompt")
    @classmethod
    def prompt_must_not_be_whitespace_only(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("raw_prompt must contain non-whitespace content.")
        return v

    def prompt_sha256(self) -> str:
        """
        SHA-256 hash of the raw prompt.

        Used in audit logs instead of the raw text, ensuring the audit trail
        is tamper-evident without storing potentially sensitive prompt content.
        """
        return hashlib.sha256(self.raw_prompt.encode("utf-8")).hexdigest()

    def redacted_preview(self, max_chars: int = 64) -> str:
        """
        Return a short, safe preview of the prompt for log messages.

        Truncates to max_chars and appends '...' if truncated. Never use
        this for security decisions — it is for human-readable log context only.
        """
        preview = self.raw_prompt[:max_chars]
        if len(self.raw_prompt) > max_chars:
            preview += "..."
        return preview


# ---------------------------------------------------------------------------
# Per-detector signal models
# All signal models are frozen — detector outputs are immutable after creation.
# ---------------------------------------------------------------------------


class PatternMatch(BaseModel):
    """A single pattern that matched in the prompt."""

    model_config = {"frozen": True}

    pattern_id: str = Field(description="Unique ID from the pattern library.")
    pattern_text: str = Field(description="The regex or literal that matched.")
    matched_text: str = Field(description="The substring in the prompt that matched.")
    category: ThreatCategory
    severity: Score = Field(description="Per-pattern severity weight [0.0–1.0].")
    offset: int = Field(ge=0, description="Character offset of match start in prompt.")


class PatternSignal(BaseModel):
    """
    Output of the PatternDetector for a single prompt.

    The PatternDetector is the cheapest detector in the pipeline (~microseconds)
    and runs first. A high-confidence pattern hit can short-circuit the pipeline,
    avoiding expensive embedding/LLM calls.
    """

    model_config = {"frozen": True}

    detector: DetectorType = Field(default=DetectorType.PATTERN)
    matched: bool
    matches: list[PatternMatch] = Field(default_factory=list)
    confidence: Score = Field(
        description=(
            "Aggregate confidence derived from matched pattern severities. "
            "max(severity) if a single critical pattern matched; "
            "weighted combination for multiple lower-severity matches."
        )
    )
    processing_time_ms: float = Field(ge=0.0)


class EmbeddingSignal(BaseModel):
    """
    Output of the EmbeddingDetector.

    Reports the nearest attack vector found in the embedding index and
    the cosine similarity score. When multiple prompt chunks are evaluated,
    reports the chunk with the highest similarity (worst-case view).
    """

    model_config = {"frozen": True}

    detector: DetectorType = Field(default=DetectorType.EMBEDDING)
    similarity_score: SimilarityScore = Field(
        description="Cosine similarity to the nearest known attack vector [0.0–1.0]."
    )
    nearest_attack_id: str | None = Field(
        default=None,
        description="ID of the closest attack sample from the dataset.",
    )
    nearest_attack_category: ThreatCategory | None = None
    chunk_index: int | None = Field(
        default=None,
        description="Which chunk of the prompt produced the highest similarity.",
    )
    threshold_used: SimilarityScore = Field(
        description="The similarity threshold configured at detection time."
    )
    exceeded_threshold: bool
    processing_time_ms: float = Field(ge=0.0)


class LLMClassifierSignal(BaseModel):
    """
    Output of the LLM-based risk classifier.

    The classifier prompt is a fixed, hardcoded template. The user input is
    inserted as a data literal inside a clearly delimited block — it is never
    concatenated as instructions. This design ensures the classifier itself
    cannot be prompt-injected.

    The classifier returns a structured JSON response which is parsed into
    this model. If the LLM returns malformed JSON or an out-of-range score,
    the signal is marked as degraded and excluded from ensemble scoring.
    """

    model_config = {"frozen": True}

    detector: DetectorType = Field(default=DetectorType.LLM_CLASSIFIER)
    risk_score: Score
    threat_category: ThreatCategory
    reasoning: str = Field(
        description=(
            "The classifier's chain-of-thought reasoning. Stored for audit "
            "and human review. Never shown to end users."
        )
    )
    degraded: bool = Field(
        default=False,
        description=(
            "True if the classifier returned malformed output, timed out, "
            "or produced an invalid score. Degraded signals are excluded from "
            "ensemble scoring and logged for reliability monitoring."
        ),
    )
    model_used: str = Field(description="The LLM model ID used for classification.")
    processing_time_ms: float = Field(ge=0.0)


class ContextBoundarySignal(BaseModel):
    """
    Output of the ContextBoundaryDetector.

    Detects structural attempts to probe or cross context boundaries:
    references to the system prompt, prior retrieved documents, tool outputs,
    or implicit knowledge the model is expected to have.
    """

    model_config = {"frozen": True}

    detector: DetectorType = Field(default=DetectorType.CONTEXT_BOUNDARY)
    boundary_violation_detected: bool
    violated_boundaries: list[str] = Field(
        default_factory=list,
        description=(
            "Human-readable list of context boundaries the prompt attempted "
            "to cross (e.g., 'system_prompt', 'rag_corpus', 'tool_output')."
        ),
    )
    indirect_injection_suspected: bool = Field(
        default=False,
        description=(
            "True when the prompt appears to originate from retrieved external "
            "content rather than direct user input (RAG injection pattern)."
        ),
    )
    multi_turn_escalation: bool = Field(
        default=False,
        description=(
            "True when prior_turns analysis reveals a progressive escalation "
            "pattern consistent with a multi-turn manipulation attack."
        ),
    )
    confidence: Score
    processing_time_ms: float = Field(ge=0.0)


class DetectorSignal(BaseModel):
    """
    Union wrapper for any detector signal type.

    Using an explicit union model (rather than Any) keeps the ensemble
    aggregation layer type-safe. The discriminator field is 'detector'.
    """

    model_config = {"frozen": True}

    detector: DetectorType
    signal: PatternSignal | EmbeddingSignal | LLMClassifierSignal | ContextBoundarySignal


# ---------------------------------------------------------------------------
# Aggregation models
# ---------------------------------------------------------------------------


class DetectorEnsemble(BaseModel):
    """
    All detector signals for a single prompt, collected and frozen before
    the risk scorer runs. Immutability here is critical — the ensemble
    represents the ground truth of what the detectors observed, and must
    not be modifiable by the scoring or policy layers.
    """

    model_config = {"frozen": True}

    prompt_sha256: str = Field(description="SHA-256 of the inspected prompt.")
    pattern_signal: PatternSignal | None = None
    embedding_signal: EmbeddingSignal | None = None
    llm_classifier_signal: LLMClassifierSignal | None = None
    context_boundary_signal: ContextBoundarySignal | None = None
    pipeline_short_circuited: bool = Field(
        default=False,
        description=(
            "True when the pipeline exited early (e.g., pattern confidence "
            "exceeded hard-block threshold) and not all detectors ran."
        ),
    )
    total_pipeline_time_ms: float = Field(ge=0.0)


class RiskScore(BaseModel):
    """
    The aggregated risk assessment for a single prompt.

    Produced by the RiskScorer from a DetectorEnsemble. Contains the numeric
    score, human-readable tier, primary threat category, and the per-detector
    weights applied during aggregation (for auditability).
    """

    model_config = {"frozen": True}

    score: Score = Field(description="Weighted ensemble risk score [0.0–1.0].")
    level: RiskLevel
    primary_threat: ThreatCategory
    contributing_detectors: list[DetectorType] = Field(
        description="Detectors that contributed to the score (excludes skipped/degraded)."
    )
    weights_applied: dict[str, float] = Field(
        description="The weight vector used in ensemble scoring, keyed by DetectorType value."
    )
    explanation: str = Field(
        description=(
            "Human-readable explanation of why this score was assigned. "
            "Synthesised from detector signals for operator review."
        )
    )


# ---------------------------------------------------------------------------
# Decision models
# ---------------------------------------------------------------------------


class SanitizedPrompt(BaseModel):
    """
    A prompt that has passed through the input sanitization filter.

    Carries both the sanitized text and a record of what was modified,
    so that callers can understand what the firewall changed and why.
    """

    model_config = {"frozen": True}

    sanitized_text: str
    original_sha256: str = Field(description="SHA-256 of the pre-sanitization prompt.")
    modifications: list[str] = Field(
        description="Human-readable list of transformations applied.",
        default_factory=list,
    )
    chars_removed: int = Field(ge=0)


class FirewallDecision(BaseModel):
    """
    The complete, final output of the firewall inspection pipeline for a
    single input prompt.

    This is the primary return type of PromptFirewall.inspect_input().
    Everything downstream (LLM connector, application layer, audit system)
    reads from this model.

    Design note: FirewallDecision is NOT frozen. The caller may attach
    application-specific metadata to 'extra' after the decision is made.
    All security-relevant fields within it (risk_score, action, ensemble)
    reference frozen child models and are therefore tamper-evident.
    """

    decision_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique ID for this inspection event.",
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )
    prompt_context: PromptContext
    ensemble: DetectorEnsemble
    risk_score: RiskScore
    action: FirewallAction
    sanitized_prompt: SanitizedPrompt | None = Field(
        default=None,
        description="Populated only when action == SANITIZE.",
    )
    effective_prompt: str | None = Field(
        default=None,
        description=(
            "The prompt text that should be forwarded to the LLM. "
            "None if action == BLOCK. "
            "Equal to sanitized_prompt.sanitized_text if action == SANITIZE. "
            "Equal to raw_prompt if action == ALLOW or LOG."
        ),
    )
    block_reason: str | None = Field(
        default=None,
        description="Human-readable reason for blocking. Populated only when action == BLOCK.",
    )
    extra: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_decision_consistency(self) -> "FirewallDecision":
        """
        Enforce internal consistency constraints on the decision.

        These checks catch logic bugs in the policy engine — e.g., a decision
        that says BLOCK but still has an effective_prompt, or a SANITIZE
        decision with no sanitized output.
        """
        if self.action == FirewallAction.BLOCK:
            if self.effective_prompt is not None:
                raise ValueError(
                    "FirewallDecision with action=BLOCK must not have an effective_prompt."
                )
            if self.block_reason is None:
                raise ValueError(
                    "FirewallDecision with action=BLOCK must include a block_reason."
                )
        elif self.action == FirewallAction.SANITIZE:
            if self.sanitized_prompt is None:
                raise ValueError(
                    "FirewallDecision with action=SANITIZE must include a sanitized_prompt."
                )
            if self.effective_prompt is None:
                raise ValueError(
                    "FirewallDecision with action=SANITIZE must have an effective_prompt."
                )
        elif self.action in (FirewallAction.ALLOW, FirewallAction.LOG):
            if self.effective_prompt is None:
                raise ValueError(
                    f"FirewallDecision with action={self.action} must have an effective_prompt."
                )
        return self


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class SafeResponse(BaseModel):
    """
    A clean LLM response that passed output inspection.

    Returned by PromptFirewall.inspect_output() when no sensitive data
    leakage or policy violations were detected.
    """

    model_config = {"frozen": True}

    content: str
    decision_id: str = Field(description="Ties back to the input FirewallDecision.")
    output_inspection_result: "OutputInspectionResult"


class BlockedResponse(BaseModel):
    """
    Represents a blocked interaction — either input was blocked before
    reaching the LLM, or the LLM response was blocked by the output filter.
    """

    model_config = {"frozen": True}

    decision_id: str
    action: FirewallAction = FirewallAction.BLOCK
    reason: str
    risk_score: Score
    threat_category: ThreatCategory
    safe_message: str = Field(
        description="The safe, user-facing error message to return to the caller.",
        default=(
            "I'm unable to process this request. "
            "If you believe this is an error, please contact support."
        ),
    )


class RedactedResponse(BaseModel):
    """
    An LLM response where sensitive content was detected and redacted
    rather than fully blocked.
    """

    model_config = {"frozen": True}

    decision_id: str
    original_sha256: str = Field(description="Hash of the pre-redaction response.")
    redacted_content: str = Field(description="Response with sensitive sections replaced.")
    redactions: list[str] = Field(
        description="Human-readable list of what was redacted and why."
    )


# ---------------------------------------------------------------------------
# Output inspection models
# ---------------------------------------------------------------------------


class SecretMatch(BaseModel):
    """A sensitive pattern detected in an LLM response."""

    model_config = {"frozen": True}

    secret_type: str = Field(
        description="Type of secret detected (e.g., 'aws_access_key', 'jwt_token', 'private_key')."
    )
    pattern_id: str
    offset: int = Field(ge=0)
    redacted_sample: str = Field(
        description=(
            "A safe, partially redacted preview for audit logs. "
            "Never store the actual secret value."
        )
    )
    severity: Score


class OutputInspectionResult(BaseModel):
    """
    Result of the OutputFilter's analysis of an LLM response.

    The output filter is the last line of defence against data leakage. Even
    when the input firewall allows a prompt through, the model may leak
    sensitive data. This model captures what was found.
    """

    model_config = {"frozen": True}

    clean: bool = Field(description="True only if no violations were detected.")
    secret_matches: list[SecretMatch] = Field(default_factory=list)
    system_prompt_echo_detected: bool = Field(
        default=False,
        description=(
            "True when the response contains content with high similarity "
            "to the system prompt (checked via hash + embedding comparison)."
        ),
    )
    exfiltration_vector_detected: bool = Field(
        default=False,
        description=(
            "True when the response contains URLs, email addresses, or other "
            "potential exfiltration vectors in a context that was flagged as "
            "suspicious on input."
        ),
    )
    recommended_action: FirewallAction
    processing_time_ms: float = Field(ge=0.0)


# ---------------------------------------------------------------------------
# Audit / observability
# ---------------------------------------------------------------------------


class AuditEvent(BaseModel):
    """
    Structured audit log entry for a complete firewall interaction.

    Design principles:
      1. No raw prompt text. Only SHA-256 hash. Compliance-safe by default.
      2. Every field used for alerting/dashboarding is a first-class typed
         field — not buried in a generic 'extra' blob.
      3. Immutable after creation. Audit events must not be modifiable.

    Downstream consumers: SIEM, Elasticsearch, Splunk, DataDog.
    """

    model_config = {"frozen": True}

    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    decision_id: str
    session_id: str
    application_id: str | None = None

    # Risk assessment (no raw prompt)
    prompt_sha256: str
    prompt_preview_redacted: str = Field(
        description="First 64 chars of prompt with injection phrases redacted."
    )
    risk_score: Score
    risk_level: RiskLevel
    primary_threat: ThreatCategory
    action_taken: FirewallAction

    # Detector signals (scores only, no text)
    pattern_confidence: Score | None = None
    embedding_similarity: SimilarityScore | None = None
    llm_classifier_score: Score | None = None
    context_boundary_confidence: Score | None = None

    # Pipeline metadata
    pipeline_short_circuited: bool = False
    total_latency_ms: float

    # Output inspection
    output_blocked: bool = False
    output_redacted: bool = False
    secrets_detected_count: int = Field(default=0, ge=0)

    # Attribution (anonymised)
    user_id_hash: str | None = Field(
        default=None,
        description="SHA-256 of user_id if provided. Never the raw user_id.",
    )
    ip_prefix: str | None = Field(
        default=None,
        description=(
            "First three octets of source IP (last octet zeroed for anonymisation). "
            "E.g., '192.168.1.0'."
        ),
    )


# ---------------------------------------------------------------------------
# Attack dataset models
# ---------------------------------------------------------------------------


class AttackVariant(BaseModel):
    """
    A paraphrase or variant of a canonical attack prompt.

    Variants are used to train/seed the embedding detector's attack index.
    Each variant should be semantically equivalent to its parent attack
    but lexically distinct, so the embedding detector learns generalisation.
    """

    model_config = {"frozen": True}

    text: str = Field(min_length=1)
    obfuscation_technique: str | None = Field(
        default=None,
        description=(
            "If this variant uses an obfuscation technique, name it here. "
            "E.g., 'base64', 'unicode_homoglyph', 'leet_speak', 'synonym_substitution'."
        ),
    )
    language: str = Field(
        default="en",
        description="ISO 639-1 language code. Multi-lingual attacks are tracked separately.",
    )


class AttackSample(BaseModel):
    """
    A canonical attack sample with metadata.

    The attack dataset is the ground truth used to:
      1. Compile the PatternDetector's signature library
      2. Seed the EmbeddingDetector's attack vector index
      3. Run adversarial tests against the full pipeline

    Each sample has a unique ID that propagates through detection signals and
    audit logs, enabling traceability from a blocked prompt back to the
    specific attack template it matched.
    """

    model_config = {"frozen": True}

    id: str = Field(
        description=(
            "Stable unique ID in the format 'CATEGORY-NNN'. "
            "E.g., 'PI-001', 'JB-042'. IDs must not be reused or reordered."
        )
    )
    category: ThreatCategory
    severity: RiskLevel = Field(
        description="Severity of this attack class if successfully executed."
    )
    canonical_prompt: str = Field(
        description="The canonical, unobfuscated form of the attack.",
        min_length=1,
    )
    variants: list[AttackVariant] = Field(
        default_factory=list,
        description="Semantic variants for embedding detector training.",
    )
    pattern_signatures: list[str] = Field(
        default_factory=list,
        description=(
            "Regex or literal strings extracted from this attack for the "
            "pattern detector. Should match the canonical form and its "
            "most common surface variants."
        ),
    )
    tags: list[str] = Field(
        default_factory=list,
        description=(
            "Free-form tags for filtering and grouping. "
            "E.g., ['multi-turn', 'role-play', 'base64', 'rag']."
        ),
    )
    expected_action: FirewallAction = Field(
        description="The action the firewall should take when this attack is detected."
    )
    description: str = Field(
        description="Human-readable description of the attack's mechanism and goal."
    )
    references: list[str] = Field(
        default_factory=list,
        description="Links to research papers, CVEs, or blog posts describing this attack.",
    )

    @field_validator("id")
    @classmethod
    def id_format(cls, v: str) -> str:
        import re
        if not re.match(r"^[A-Z]+-\d{3}$", v):
            raise ValueError(
                f"Attack ID '{v}' must match pattern 'CATEGORY-NNN' (e.g., 'PI-001')."
            )
        return v


class AttackDataset(BaseModel):
    """
    The complete attack dataset loaded at firewall startup.

    Versioned so that changes to the dataset can be tracked and rolled back.
    The checksum is computed over the serialised samples to detect tampering
    or corruption of the dataset file.
    """

    version: str = Field(description="Semantic version of the dataset, e.g., '1.0.0'.")
    description: str
    created_at: datetime
    updated_at: datetime
    checksum: str | None = Field(
        default=None,
        description=(
            "SHA-256 of the canonical JSON serialisation of 'samples'. "
            "Computed and verified at load time."
        ),
    )
    samples: list[AttackSample]

    @property
    def sample_count(self) -> int:
        return len(self.samples)

    def by_category(self, category: ThreatCategory) -> list[AttackSample]:
        return [s for s in self.samples if s.category == category]

    def by_severity(self, severity: RiskLevel) -> list[AttackSample]:
        return [s for s in self.samples if s.severity == severity]

    def get_by_id(self, attack_id: str) -> AttackSample | None:
        return next((s for s in self.samples if s.id == attack_id), None)
