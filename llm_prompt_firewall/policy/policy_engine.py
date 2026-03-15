"""
Policy Engine
==============

The PolicyEngine is the decision layer that sits between the RiskScorer and
the final FirewallAction. It translates a numeric risk score into an
enforceable action by evaluating the score against a configurable set of
rules loaded from a YAML policy file.

Separation of concerns:
  - RiskScorer: WHAT is the threat and HOW severe?
  - PolicyEngine: WHAT do we DO about it given the organisation's rules?

This distinction matters because the appropriate response to a given risk
score varies by deployment context. A customer-facing chatbot may BLOCK at
0.70; an internal developer tool may LOG at 0.85 and only BLOCK above 0.95.
The scoring logic stays constant; the policy adapts per environment.

Rule resolution order (first match wins):
  1. explicit_block_patterns  — regex/literal patterns that always BLOCK,
                                 regardless of risk score.
  2. explicit_allow_patterns  — whitelist patterns that always ALLOW,
                                 regardless of risk score (override blocks).
  3. block_threat_categories  — BLOCK if primary threat is in the listed set.
  4. score >= block_threshold  — BLOCK.
  5. score >= sanitize_threshold — SANITIZE.
  6. score >= log_threshold    — LOG.
  7. default_action            — fallback (default: ALLOW).

Design decisions:

  1. POLICY AS DATA, NOT CODE
     Security teams should be able to update blocking rules without a
     deployment. YAML policy files can be reloaded at runtime via the
     reload() method (called by a file watcher or admin API).

  2. VALIDATION AT LOAD TIME
     Policy files are validated against a Pydantic schema the moment they
     are loaded. A malformed policy fails loudly at startup, not silently
     at request time. An invalid policy update does not replace the current
     live policy — the old policy continues to serve traffic while the error
     is logged.

  3. WHITELIST TAKES PRIORITY OVER RISK SCORE
     An explicit allow_pattern match overrides the numeric risk score. This
     is intentional — a security operator who has vetted a specific pattern
     as safe should not be second-guessed by the scorer. Whitelist entries
     should be as specific as possible to avoid inadvertently allowing attacks.

  4. IMMUTABLE POLICY SNAPSHOT PER REQUEST
     The engine takes a snapshot of the current policy at the beginning of
     each evaluate() call. Hot-reload cannot cause a race condition where
     half a request is evaluated under one policy and half under another.

  5. AUDIT-COMPLETE DECISIONS
     Every PolicyDecision records which rule triggered it and why. The audit
     layer can reconstruct exactly why a specific prompt was blocked or allowed.
"""

from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

import yaml
from pydantic import BaseModel, Field, field_validator

from llm_prompt_firewall.core.risk_scoring import ThresholdConfig, WeightConfig
from llm_prompt_firewall.models.schemas import (
    FirewallAction,
    RiskLevel,
    RiskScore,
    ThreatCategory,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Policy schema models (validated at load time)
# ---------------------------------------------------------------------------


class ThresholdPolicy(BaseModel):
    """Score thresholds that gate each action tier."""

    block: float = Field(default=0.85, ge=0.0, le=1.0)
    sanitize: float = Field(default=0.70, ge=0.0, le=1.0)
    log: float = Field(default=0.40, ge=0.0, le=1.0)

    @field_validator("sanitize")
    @classmethod
    def sanitize_below_block(cls, v: float, info: object) -> float:
        # Access other field values via info.data
        data = getattr(info, "data", {})
        block = data.get("block", 0.85)
        if v >= block:
            raise ValueError(
                f"sanitize threshold ({v}) must be strictly less than block threshold ({block})."
            )
        return v

    @field_validator("log")
    @classmethod
    def log_below_sanitize(cls, v: float, info: object) -> float:
        data = getattr(info, "data", {})
        sanitize = data.get("sanitize", 0.70)
        if v >= sanitize:
            raise ValueError(
                f"log threshold ({v}) must be strictly less than sanitize threshold ({sanitize})."
            )
        return v

    def to_threshold_config(self) -> ThresholdConfig:
        return ThresholdConfig(
            safe_max=self.log - 0.001,
            suspicious_max=self.sanitize - 0.001,
            high_max=self.block - 0.001,
        )


class WeightPolicy(BaseModel):
    """Optional per-deployment detector weight overrides."""

    pattern: float = Field(default=0.30, ge=0.0, le=1.0)
    embedding: float = Field(default=0.25, ge=0.0, le=1.0)
    llm_classifier: float = Field(default=0.35, ge=0.0, le=1.0)
    context_boundary: float = Field(default=0.10, ge=0.0, le=1.0)

    def to_weight_config(self) -> WeightConfig:
        return WeightConfig(
            pattern=self.pattern,
            embedding=self.embedding,
            llm_classifier=self.llm_classifier,
            context_boundary=self.context_boundary,
        )


class SanitizationPolicy(BaseModel):
    """Controls what the InputFilter strips when action == SANITIZE."""

    redact_secrets: bool = True
    strip_injection_phrases: bool = True
    strip_invisible_chars: bool = True
    normalize_unicode: bool = True


class PolicyConfig(BaseModel):
    """
    Complete policy configuration loaded from a YAML file.

    All fields have safe defaults so a minimal policy file (e.g. only
    overriding thresholds) is valid without specifying every field.
    """

    version: str = Field(default="1.0")
    description: str = Field(default="")

    # Core thresholds
    thresholds: ThresholdPolicy = Field(default_factory=ThresholdPolicy)

    # Detector weights (optional override)
    weights: WeightPolicy = Field(default_factory=WeightPolicy)

    # Explicit block rules — applied before risk score thresholds
    block_patterns: list[str] = Field(
        default_factory=list,
        description=(
            "Regex or literal patterns that unconditionally BLOCK a prompt. "
            "Evaluated before the risk score thresholds."
        ),
    )
    block_threat_categories: list[str] = Field(
        default_factory=list,
        description=(
            "ThreatCategory values that unconditionally BLOCK a prompt. "
            "E.g. ['tool_abuse', 'data_exfiltration']."
        ),
    )

    # Whitelist — applied after block_patterns, override risk score
    allow_patterns: list[str] = Field(
        default_factory=list,
        description=(
            "Regex or literal patterns that unconditionally ALLOW a prompt, "
            "even if other rules would block it. Use sparingly."
        ),
    )

    # Sanitization settings
    sanitization: SanitizationPolicy = Field(default_factory=SanitizationPolicy)

    # Fallback action when no threshold is exceeded
    default_action: str = Field(
        default="allow",
        description="Fallback action when risk score is below all thresholds.",
    )

    @field_validator("default_action")
    @classmethod
    def validate_default_action(cls, v: str) -> str:
        valid = {a.value for a in FirewallAction}
        if v not in valid:
            raise ValueError(
                f"default_action '{v}' is not a valid FirewallAction. Valid values: {sorted(valid)}"
            )
        return v

    @field_validator("block_threat_categories")
    @classmethod
    def validate_threat_categories(cls, v: list[str]) -> list[str]:
        valid = {c.value for c in ThreatCategory}
        for cat in v:
            if cat not in valid:
                raise ValueError(
                    f"Unknown threat category '{cat}' in block_threat_categories. "
                    f"Valid values: {sorted(valid)}"
                )
        return v


# ---------------------------------------------------------------------------
# Compiled policy — post-load, runtime-ready form
# ---------------------------------------------------------------------------


@dataclass
class CompiledPolicy:
    """
    A validated, compiled PolicyConfig ready for per-request evaluation.

    Patterns are compiled into re.Pattern objects at load time (zero
    compilation cost at request time). The config object is retained for
    threshold and weight access.

    Frozen via lock: the CompiledPolicy object itself is mutable only by the
    PolicyEngine._reload_lock. Callers receive a reference snapshot; concurrent
    reloads cannot modify a snapshot that is mid-evaluation.
    """

    config: PolicyConfig
    compiled_block_patterns: list[re.Pattern[str]] = field(default_factory=list)
    compiled_allow_patterns: list[re.Pattern[str]] = field(default_factory=list)
    block_categories: frozenset[ThreatCategory] = field(default_factory=frozenset)
    default_action: FirewallAction = FirewallAction.ALLOW

    @classmethod
    def from_config(cls, config: PolicyConfig) -> CompiledPolicy:
        block_compiled: list[re.Pattern[str]] = []
        for pat in config.block_patterns:
            try:
                block_compiled.append(re.compile(pat, re.IGNORECASE | re.UNICODE))
            except re.error as exc:
                logger.warning("Invalid block_pattern regex '%s': %s — skipping.", pat, exc)

        allow_compiled: list[re.Pattern[str]] = []
        for pat in config.allow_patterns:
            try:
                allow_compiled.append(re.compile(pat, re.IGNORECASE | re.UNICODE))
            except re.error as exc:
                logger.warning("Invalid allow_pattern regex '%s': %s — skipping.", pat, exc)

        block_cats: frozenset[ThreatCategory] = frozenset(
            ThreatCategory(c) for c in config.block_threat_categories
        )

        return cls(
            config=config,
            compiled_block_patterns=block_compiled,
            compiled_allow_patterns=allow_compiled,
            block_categories=block_cats,
            default_action=FirewallAction(config.default_action),
        )


# ---------------------------------------------------------------------------
# PolicyDecision — the output of a single evaluation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PolicyDecision:
    """
    The result of evaluating a RiskScore against a PolicyConfig.

    Carries enough information to reconstruct exactly why an action was taken,
    for the audit layer and for debugging. Never shown to end users.
    """

    action: FirewallAction
    rule_triggered: str
    """Human-readable name of the rule that determined the action."""

    explanation: str
    """Full explanation for audit logs."""

    risk_score: float
    risk_level: RiskLevel
    primary_threat: ThreatCategory
    matched_block_pattern: str | None = None
    matched_allow_pattern: str | None = None


# ---------------------------------------------------------------------------
# PolicyEngine
# ---------------------------------------------------------------------------

# Default policy file path (relative to the package root)
DEFAULT_POLICY_PATH: Final[Path] = (
    Path(__file__).resolve().parents[2] / "config" / "default_policy.yaml"
)


class PolicyEngine:
    """
    Evaluates a RiskScore against a loaded policy and returns a PolicyDecision.

    The engine is thread-safe. The current policy is stored under a read-write
    lock so hot-reloads (via reload()) do not interfere with in-flight evaluate()
    calls.

    Usage:
        engine = PolicyEngine.from_file(Path("config/policy.yaml"))
        decision = engine.evaluate(risk_score, prompt_text)

        # Hot-reload after a policy update
        engine.reload()
    """

    def __init__(self, policy: CompiledPolicy) -> None:
        self._policy = policy
        self._policy_path: Path | None = None
        self._reload_lock = threading.RLock()
        logger.info(
            "PolicyEngine loaded: version=%s, block_patterns=%d, allow_patterns=%d, "
            "block_categories=%s, default_action=%s",
            policy.config.version,
            len(policy.compiled_block_patterns),
            len(policy.compiled_allow_patterns),
            [c.value for c in policy.block_categories],
            policy.default_action.value,
        )

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_file(cls, path: Path) -> PolicyEngine:
        """
        Load a policy from a YAML file and return a PolicyEngine.

        Raises ValueError if the file is missing, malformed YAML, or fails
        Pydantic schema validation. The error message identifies the specific
        validation failure so operators can fix it quickly.
        """
        config = _load_policy_file(path)
        compiled = CompiledPolicy.from_config(config)
        engine = cls(compiled)
        engine._policy_path = path
        return engine

    @classmethod
    def with_defaults(cls) -> PolicyEngine:
        """
        Build a PolicyEngine using only default values (no YAML file).

        Suitable for tests and minimal deployments that do not need
        custom policy rules.
        """
        config = PolicyConfig()
        compiled = CompiledPolicy.from_config(config)
        return cls(compiled)

    @classmethod
    def from_default_file(cls) -> PolicyEngine:
        """Load from the package's built-in default_policy.yaml."""
        if DEFAULT_POLICY_PATH.exists():
            return cls.from_file(DEFAULT_POLICY_PATH)
        logger.warning(
            "Default policy file not found at %s — using built-in defaults.",
            DEFAULT_POLICY_PATH,
        )
        return cls.with_defaults()

    # ------------------------------------------------------------------
    # Hot-reload
    # ------------------------------------------------------------------

    def reload(self) -> bool:
        """
        Reload the policy from the file it was originally loaded from.

        Returns True on success, False if no file path is set or if the
        new policy fails validation (the old policy continues in that case).

        Thread-safe: acquires the reload lock. In-flight evaluate() calls
        complete under the old policy; new calls after reload() use the new one.
        """
        if self._policy_path is None:
            logger.warning("PolicyEngine.reload(): no policy file path set — nothing to reload.")
            return False

        try:
            new_config = _load_policy_file(self._policy_path)
            new_compiled = CompiledPolicy.from_config(new_config)
        except Exception as exc:
            logger.error(
                "PolicyEngine.reload(): new policy at %s is invalid: %s — "
                "keeping current policy active.",
                self._policy_path,
                exc,
            )
            return False

        with self._reload_lock:
            self._policy = new_compiled

        logger.info(
            "PolicyEngine reloaded from %s (version=%s)",
            self._policy_path,
            new_compiled.config.version,
        )
        return True

    # ------------------------------------------------------------------
    # Core evaluation
    # ------------------------------------------------------------------

    def evaluate(self, risk_score: RiskScore, prompt_text: str = "") -> PolicyDecision:
        """
        Evaluate a RiskScore against the current policy.

        Args:
            risk_score: The aggregated risk score from the RiskScorer.
            prompt_text: The (possibly normalised) prompt text. Used for
                         pattern matching against block/allow rules.
                         Defaults to empty string if not provided (disables
                         pattern-based rules; threshold rules still apply).

        Returns:
            PolicyDecision with the resolved action and full audit trail.

        Resolution order (first match wins):
          1. explicit block_patterns
          2. explicit allow_patterns (whitelist override)
          3. block_threat_categories
          4. score >= block threshold
          5. score >= sanitize threshold
          6. score >= log threshold
          7. default_action
        """
        # Snapshot the policy at the start of evaluation — hot-reload cannot
        # modify this reference mid-evaluation.
        with self._reload_lock:
            policy = self._policy

        score = risk_score.score
        level = risk_score.level
        threat = risk_score.primary_threat
        thresholds = policy.config.thresholds

        # --- 1. Explicit block patterns ---
        for pattern in policy.compiled_block_patterns:
            match = pattern.search(prompt_text)
            if match:
                matched = match.group(0)
                return PolicyDecision(
                    action=FirewallAction.BLOCK,
                    rule_triggered="explicit_block_pattern",
                    explanation=(
                        f"Explicit block pattern matched: '{pattern.pattern}' "
                        f"→ matched text: '{matched[:60]}'. "
                        f"Risk score: {score:.2f}."
                    ),
                    risk_score=score,
                    risk_level=level,
                    primary_threat=threat,
                    matched_block_pattern=pattern.pattern,
                )

        # --- 2. Explicit allow patterns (whitelist) ---
        for pattern in policy.compiled_allow_patterns:
            match = pattern.search(prompt_text)
            if match:
                matched = match.group(0)
                return PolicyDecision(
                    action=FirewallAction.ALLOW,
                    rule_triggered="explicit_allow_pattern",
                    explanation=(
                        f"Whitelisted by allow pattern: '{pattern.pattern}' "
                        f"→ matched text: '{matched[:60]}'. "
                        f"Risk score {score:.2f} overridden."
                    ),
                    risk_score=score,
                    risk_level=level,
                    primary_threat=threat,
                    matched_allow_pattern=pattern.pattern,
                )

        # --- 3. Block by threat category ---
        if threat in policy.block_categories:
            return PolicyDecision(
                action=FirewallAction.BLOCK,
                rule_triggered="block_threat_category",
                explanation=(
                    f"Threat category '{threat.value}' is in block_threat_categories. "
                    f"Risk score: {score:.2f}."
                ),
                risk_score=score,
                risk_level=level,
                primary_threat=threat,
            )

        # --- 4. Score-based threshold rules ---
        if score >= thresholds.block:
            return PolicyDecision(
                action=FirewallAction.BLOCK,
                rule_triggered="score_threshold_block",
                explanation=(
                    f"Risk score {score:.4f} >= block threshold {thresholds.block}. "
                    f"Level: {level.value}. Threat: {threat.value}."
                ),
                risk_score=score,
                risk_level=level,
                primary_threat=threat,
            )

        if score >= thresholds.sanitize:
            return PolicyDecision(
                action=FirewallAction.SANITIZE,
                rule_triggered="score_threshold_sanitize",
                explanation=(
                    f"Risk score {score:.4f} >= sanitize threshold {thresholds.sanitize}. "
                    f"Level: {level.value}. Threat: {threat.value}."
                ),
                risk_score=score,
                risk_level=level,
                primary_threat=threat,
            )

        if score >= thresholds.log:
            return PolicyDecision(
                action=FirewallAction.LOG,
                rule_triggered="score_threshold_log",
                explanation=(
                    f"Risk score {score:.4f} >= log threshold {thresholds.log}. "
                    f"Suspicious but allowed. Threat: {threat.value}."
                ),
                risk_score=score,
                risk_level=level,
                primary_threat=threat,
            )

        # --- 5. Default action ---
        return PolicyDecision(
            action=policy.default_action,
            rule_triggered="default_action",
            explanation=(
                f"Risk score {score:.4f} is below all thresholds. "
                f"Default action: {policy.default_action.value}."
            ),
            risk_score=score,
            risk_level=level,
            primary_threat=threat,
        )

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def current_version(self) -> str:
        with self._reload_lock:
            return self._policy.config.version

    @property
    def weight_config(self) -> WeightConfig:
        with self._reload_lock:
            return self._policy.config.weights.to_weight_config()

    @property
    def threshold_config(self) -> ThresholdConfig:
        with self._reload_lock:
            return self._policy.config.thresholds.to_threshold_config()

    @property
    def sanitization_config(self) -> SanitizationPolicy:
        with self._reload_lock:
            return self._policy.config.sanitization


# ---------------------------------------------------------------------------
# YAML loader
# ---------------------------------------------------------------------------


def _load_policy_file(path: Path) -> PolicyConfig:
    """
    Load and validate a PolicyConfig from a YAML file.

    Raises:
        ValueError: if the file is missing, not valid YAML, or fails
                    Pydantic schema validation.
    """
    if not path.exists():
        raise ValueError(f"Policy file not found: {path}")

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"Policy file '{path}' contains invalid YAML: {exc}") from exc

    if raw is None:
        raw = {}

    try:
        config = PolicyConfig(**raw)
    except Exception as exc:
        raise ValueError(f"Policy file '{path}' failed schema validation: {exc}") from exc

    logger.debug("Loaded policy from %s (version=%s)", path, config.version)
    return config
