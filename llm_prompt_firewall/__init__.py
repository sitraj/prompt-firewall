"""
LLM Prompt Firewall
====================

Production-grade prompt injection detection and output sanitization
for LLM-powered applications.

Quick start::

    from llm_prompt_firewall import PromptFirewall
    from llm_prompt_firewall.models.schemas import PromptContext, FirewallAction

    firewall = PromptFirewall.from_default_config()

    decision = firewall.inspect_input(PromptContext(raw_prompt=user_text))
    if decision.action == FirewallAction.BLOCK:
        return decision.block_reason

    response = your_llm(decision.effective_prompt)

    result = firewall.inspect_output(response, decision)
"""

from llm_prompt_firewall.firewall import PromptFirewall
from llm_prompt_firewall.models.schemas import (
    FirewallAction,
    FirewallDecision,
    PromptContext,
    PromptRole,
    RiskLevel,
    ThreatCategory,
)

__all__ = [
    "PromptFirewall",
    "FirewallAction",
    "FirewallDecision",
    "PromptContext",
    "PromptRole",
    "RiskLevel",
    "ThreatCategory",
]

__version__ = "0.1.0"
