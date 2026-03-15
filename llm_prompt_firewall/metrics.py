"""
Prometheus metrics for the LLM Prompt Firewall API.

All metrics use the `firewall_` namespace. Import and call the record_*
helpers from api.py after each inspection to keep metrics in sync.

Metrics exposed:

  firewall_input_inspections_total{action}
      Counter — total input inspections, labelled by action taken
      (allow, log, sanitize, block).

  firewall_input_pipeline_duration_seconds
      Histogram — end-to-end input inspection wall-clock time.
      Buckets tuned for the expected range of 1ms (pattern-only short-circuit)
      to ~5s (LLM classifier enabled).

  firewall_input_risk_score
      Histogram — distribution of risk scores across all inspections.
      Useful for detecting drift (e.g. sudden spike in high-risk traffic).

  firewall_output_inspections_total{outcome}
      Counter — total output inspections, labelled by outcome
      (safe, redacted, blocked).

  firewall_output_pipeline_duration_seconds
      Histogram — output filter wall-clock time.
      OutputFilter is regex-based so buckets are in the sub-millisecond range.

  firewall_secrets_detected_total{secret_type}
      Counter — secrets found in LLM responses, labelled by type
      (e.g. aws_access_key, jwt_token, private_key).

  firewall_decision_cache_size
      Gauge — current number of entries in the in-process decision cache.
      Helps detect cache exhaustion or abnormally long-lived sessions.
"""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

# ---------------------------------------------------------------------------
# Input-side metrics
# ---------------------------------------------------------------------------

INPUT_INSPECTIONS = Counter(
    "firewall_input_inspections_total",
    "Total input inspections by action taken.",
    ["action"],
)

INPUT_LATENCY = Histogram(
    "firewall_input_pipeline_duration_seconds",
    "End-to-end input inspection pipeline duration in seconds.",
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)

INPUT_RISK_SCORE = Histogram(
    "firewall_input_risk_score",
    "Distribution of input risk scores [0.0–1.0].",
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

# ---------------------------------------------------------------------------
# Output-side metrics
# ---------------------------------------------------------------------------

OUTPUT_INSPECTIONS = Counter(
    "firewall_output_inspections_total",
    "Total output inspections by outcome.",
    ["outcome"],
)

OUTPUT_LATENCY = Histogram(
    "firewall_output_pipeline_duration_seconds",
    "Output inspection pipeline duration in seconds.",
    buckets=[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.025, 0.05, 0.1],
)

SECRETS_DETECTED = Counter(
    "firewall_secrets_detected_total",
    "Secrets detected in LLM responses by type.",
    ["secret_type"],
)

# ---------------------------------------------------------------------------
# Infrastructure metrics
# ---------------------------------------------------------------------------

DECISION_CACHE_SIZE = Gauge(
    "firewall_decision_cache_size",
    "Current number of entries in the in-process decision cache.",
)


# ---------------------------------------------------------------------------
# Record helpers
# ---------------------------------------------------------------------------


def record_input(
    action: str,
    risk_score: float,
    duration_seconds: float,
) -> None:
    """Record metrics for a completed input inspection."""
    INPUT_INSPECTIONS.labels(action=action).inc()
    INPUT_LATENCY.observe(duration_seconds)
    INPUT_RISK_SCORE.observe(risk_score)


def record_output(
    outcome: str,
    duration_seconds: float,
    secret_types: list[str] | None = None,
) -> None:
    """Record metrics for a completed output inspection."""
    OUTPUT_INSPECTIONS.labels(outcome=outcome).inc()
    OUTPUT_LATENCY.observe(duration_seconds)
    for stype in secret_types or []:
        SECRETS_DETECTED.labels(secret_type=stype).inc()


def update_cache_size(size: int) -> None:
    """Update the decision cache size gauge."""
    DECISION_CACHE_SIZE.set(size)
