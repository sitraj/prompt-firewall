# ── Stage 1: build ──────────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /build

# Install build deps and the package with API + Prometheus extras.
# We copy pyproject.toml first so that this layer is cached unless deps change.
COPY pyproject.toml ./
COPY llm_prompt_firewall/ ./llm_prompt_firewall/
COPY config/ ./config/

RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir --prefix=/install \
    ".[api]" \
    prometheus-client


# ── Stage 2: runtime ────────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

LABEL org.opencontainers.image.title="LLM Prompt Firewall" \
      org.opencontainers.image.description="Production-grade prompt injection detection for LLMs" \
      org.opencontainers.image.source="https://github.com/shounakitraj/prompt-firewall"

# Non-root user for security
RUN groupadd --gid 1001 firewall \
 && useradd --uid 1001 --gid firewall --no-create-home firewall

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application source
COPY --from=builder /build/llm_prompt_firewall ./llm_prompt_firewall
COPY --from=builder /build/config ./config

USER firewall

EXPOSE 8000

# Liveness probe used by Docker and Kubernetes
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/v1/health')"

# Environment variable defaults (override at runtime)
ENV FIREWALL_LOG_LEVEL=INFO \
    FIREWALL_CACHE_MAX_SIZE=10000 \
    FIREWALL_CACHE_TTL_SECONDS=300

CMD ["uvicorn", "llm_prompt_firewall.api:app", \
     "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "1", "--log-level", "info"]
