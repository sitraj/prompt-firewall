# ADR-002: Data Flow Diagram

**Date:** 2026-03-15
**Status:** Accepted

---

## Level 0 — Context Diagram

Shows the system boundary and what flows in and out.

```mermaid
flowchart LR
    User(["👤 User / Client App"])
    LLM(["🤖 LLM Provider\n(OpenAI / Anthropic / etc.)"])
    Audit(["📋 Audit Store\n(file / SIEM / stdout)"])
    FW["🛡️ llm-prompt-firewall"]

    User -->|"raw prompt"| FW
    FW -->|"FirewallDecision\n(action + effective_prompt)"| User
    User -->|"effective_prompt"| LLM
    LLM -->|"llm_response"| User
    User -->|"llm_response + decision_id"| FW
    FW -->|"SafeResponse / RedactedResponse\n/ BlockedResponse"| User
    FW -.->|"AuditEvent (NDJSON)"| Audit
```

---

## Level 1 — Input Inspection Pipeline

What happens inside `firewall.inspect_input()`.

```mermaid
flowchart TD
    A(["Application"])

    subgraph INPUT["inspect_input()"]
        direction TB
        N["1. Pre-Normaliser\n─────────────────\n• Strip invisible chars\n• NFKC unicode\n• Deduplicate whitespace"]

        P["2. Pattern Detector\n─────────────────\n• Regex signatures\n• ~microseconds\n• Always runs"]

        SC{{"Short-circuit?\nconfidence = 1.0"}}

        E["3. Embedding Detector\n─────────────────\n• Local ML model\n• all-MiniLM-L6-v2\n• ~10–50ms\n• Optional extra"]

        L["4. LLM Classifier\n─────────────────\n• OpenAI / Anthropic API\n• ~200–800ms\n• OFF by default"]

        CB["5. Context Boundary\n─────────────────\n• Structural heuristics\n• Delimiter abuse\n• Role-switch markers\n• ~1ms, always runs"]

        RS["6. Risk Scorer\n─────────────────\n• Weighted ensemble\n• Renormalises missing\n  detector weights\n• Maps score → RiskLevel"]

        PE["7. Policy Engine\n─────────────────\n• Block patterns\n• Allow patterns\n• Threat categories\n• Score thresholds\n• YAML-driven"]

        ACT{{"Action?"}}

        IF["8a. Input Filter\n─────────────────\n• Strip injection phrases\n• Remove invisible chars\n• NFKC normalise"]

        DA["9. Decision Assembly\n─────────────────\n• FirewallDecision\n• decision_id (UUID)\n• effective_prompt\n• risk_score + level\n• primary_threat"]

        CACHE[("Decision Cache\nLRU + TTL\nOrderedDict")]
        AUD[("Audit Logger\nNDJSON")]
    end

    DS1[("Pattern\nSignatures\nYAML")]
    DS2[("Embedding\nIndex\nfloat32 matrix")]
    DS3[("Policy\nConfig\nYAML")]

    A -->|"PromptContext\n(raw_prompt + metadata)"| N
    N --> P
    DS1 -.->|"compiled regex"| P
    P --> SC
    SC -->|"yes → skip detectors"| CB
    SC -->|"no"| E
    DS2 -.->|"attack vectors"| E
    E --> L
    L -->|"LLMClassifierSignal\nor DEGRADED"| CB
    CB --> RS
    RS --> PE
    DS3 -.->|"thresholds\n+ rules"| PE
    PE --> ACT
    ACT -->|"SANITIZE"| IF
    ACT -->|"ALLOW / LOG / BLOCK"| DA
    IF --> DA
    DA --> CACHE
    DA -.-> AUD
    DA -->|"FirewallDecision"| A
```

---

## Level 1 — Output Inspection Pipeline

What happens inside `firewall.inspect_output()`.

```mermaid
flowchart TD
    A(["Application"])

    subgraph OUTPUT["inspect_output()"]
        direction TB
        CL["1. Cache Lookup\n─────────────────\n• Retrieve FirewallDecision\n  by decision_id\n• 503 if not found or expired"]

        OF["2. Output Filter\n─────────────────\n• Regex-only, no ML\n• Sub-millisecond"]

        subgraph CHECKS["Detection checks (run in parallel)"]
            S["Secret Detection\n──────────────\n• AWS keys\n• OpenAI / Anthropic tokens\n• GitHub tokens\n• JWT tokens\n• RSA private keys\n• Generic API key patterns\n• 15+ types total"]
            SP["System Prompt Echo\n──────────────\n• SHA-256 fragment match\n  against known prompt hash"]
            EX["Exfiltration Vectors\n──────────────\n• URLs + emails in response\n• Only flagged when input\n  risk_score ≥ 0.40"]
        end

        ACT{{"Violations\nfound?"}}

        RD["Redact\n─────────────────\n• Replace secret with\n  [TYPE_REDACTED]\n• Preserve rest of response"]

        BL["Block\n─────────────────\n• Suppress entire response\n• Return safe_message only"]

        OUT["3. Result Assembly"]
    end

    CACHE[("Decision Cache")]

    A -->|"response_text\n+ decision_id"| CL
    CACHE -.->|"FirewallDecision"| CL
    CL --> OF
    OF --> S
    OF --> SP
    OF --> EX
    S --> ACT
    SP --> ACT
    EX --> ACT
    ACT -->|"none"| OUT
    ACT -->|"secret / echo"| RD
    ACT -->|"blocked output"| BL
    RD --> OUT
    BL --> OUT
    OUT -->|"SafeResponse"| A
    OUT -->|"RedactedResponse\n(redacted_content)"| A
    OUT -->|"BlockedResponse\n(safe_message)"| A
```

---

## Level 1 — REST API Data Flow

How data moves when using the HTTP server instead of the library directly.

```mermaid
flowchart LR
    APP(["Client App\n(any language)"])

    subgraph API["FastAPI Server  :8080"]
        direction TB
        H1["POST /v1/inspect/input"]
        H2["POST /v1/inspect/output"]
        H3["GET  /v1/health"]
        H4["GET  /v1/ready"]
        H5["GET  /v1/metrics"]

        FW["PromptFirewall\nsingleton"]
        MC["Metrics Collector\nPrometheus"]
        CACHE2[("Decision Cache\nLRU + TTL")]
    end

    LLM(["LLM Provider"])
    PROM(["Prometheus\n/ Grafana"])

    APP -->|"{ prompt, role, session_id... }"| H1
    H1 --> FW
    FW --> CACHE2
    FW -->|"{ decision_id, action,\neffective_prompt, risk_score }"| H1
    H1 --> MC
    H1 -->|"InspectInputResponse"| APP

    APP -->|"effective_prompt"| LLM
    LLM -->|"llm_response"| APP

    APP -->|"{ decision_id, response_text }"| H2
    H2 --> CACHE2
    H2 --> FW
    FW -->|"SafeResponse / Redacted\n/ Blocked"| H2
    H2 --> MC
    H2 -->|"InspectOutputResponse"| APP

    APP -->|"liveness check"| H3
    APP -->|"readiness check"| H4
    PROM -->|"scrape"| H5
    MC -.->|"counters / histograms\n/ gauges"| H5
```

---

## Data Stores Reference

| Store | Type | Contents | Lifecycle |
|---|---|---|---|
| Pattern Signatures | YAML (bundled) | Regex patterns + threat categories | Loaded once at startup |
| Embedding Index | In-memory float32 matrix | Normalised attack vectors | Built once at startup from dataset |
| Policy Config | YAML (file or default) | Thresholds, block/allow rules | Loaded at startup, hot-reload supported |
| Decision Cache | In-memory OrderedDict | `decision_id → (FirewallDecision, timestamp)` | LRU eviction + TTL expiry (default 5 min, max 10k entries) |
| Audit Log | NDJSON file or stdout | AuditEvent per inspection | Append-only, configurable via `FIREWALL_AUDIT_LOG_FILE` |

---

## Key Data Flows Summary

| Flow | From | To | Notes |
|---|---|---|---|
| `PromptContext` | Application | `inspect_input()` | Carries raw prompt + session metadata |
| `FirewallDecision` | `inspect_input()` | Application + Cache | Immutable; `decision_id` links it to output |
| `effective_prompt` | Application | LLM Provider | From `decision.effective_prompt` — may be sanitized |
| `response_text + decision_id` | Application | `inspect_output()` | App passes both together |
| `SafeResponse / RedactedResponse / BlockedResponse` | `inspect_output()` | Application | Final result to return to user |
| `AuditEvent` | Firewall | Audit Store | No raw prompts — SHA-256 hashes only |
| Prometheus scrape | Prometheus | `/v1/metrics` | Pull-based, every 15s by default |
