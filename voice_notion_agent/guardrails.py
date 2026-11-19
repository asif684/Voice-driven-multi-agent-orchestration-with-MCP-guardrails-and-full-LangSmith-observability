"""Guardrails and Human-in-the-Loop for the Voice Agent.

This module provides:
1. Input guardrails - PII detection, prompt injection blocking
2. Output guardrails - Policy violations
3. Human-in-the-loop - Optional approval for sensitive actions

All guardrails are applied as middleware around the existing agent,
making them easy to enable/disable via config.
"""
import re
from typing import Callable, Any
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig

from . import config
from .logging_utils import log_stage


# =============================================================================
# PII DETECTION PATTERNS
# =============================================================================

PII_PATTERNS = {
    "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
    "phone": re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "api_key": re.compile(r"\b(?:sk|pk)[_-]?[A-Za-z0-9]{32,}\b", re.IGNORECASE),
    "credit_card": re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b"),
}

# Prompt injection patterns to block
INJECTION_PATTERNS = [
    re.compile(r"ignore\s+previous\s+instructions", re.IGNORECASE),
    re.compile(r"system\s+prompt", re.IGNORECASE),
    re.compile(r"you\s+are\s+now", re.IGNORECASE),
    re.compile(r"forget\s+everything", re.IGNORECASE),
    re.compile(r"new\s+instructions", re.IGNORECASE),
    re.compile(r"disregard.*instructions", re.IGNORECASE),
]

# Sensitive tool calls requiring human approval (when HIL enabled)
SENSITIVE_TOOLS = {"send_email", "delete_email", "create_notion_page"}


def detect_pii(text: str) -> list[str]:
    """Detect PII in user input. Returns list of detected PII types."""
    found = []
    for pii_type, pattern in PII_PATTERNS.items():
        if pattern.search(text):
            found.append(pii_type)
    return found


def detect_injection(text: str) -> bool:
    """Detect prompt injection attempts."""
    return any(p.search(text) for p in INJECTION_PATTERNS)


def contains_sensitive_tool_call(msg: AIMessage) -> list[str]:
    """Extract sensitive tool calls from AI message."""
    if not hasattr(msg, "tool_calls") or not msg.tool_calls:
        return []
    found = []
    for tc in msg.tool_calls:
        name = tc.get("name", "")
        if name in SENSITIVE_TOOLS:
            found.append(name)
    return found


# =============================================================================
# GUARDRAIL WRAPPER
# =============================================================================

class GuardedAgentWrapper:
    """
    A middleware-style wrapper that applies guardrails around the existing agent.

    Usage:
        from voice_notion_agent.guardrails import GuardedAgentWrapper
        from voice_notion_agent.agent import build_agent

        base_agent = await build_agent()
        guarded_agent = GuardedAgentWrapper(base_agent)
        result = await guarded_agent.ainvoke({"messages": messages})
    """

    def __init__(self, original_agent: Any):
        self.agent = original_agent

    async def ainvoke(self, inputs: dict, config: RunnableConfig = None) -> dict:
        """Invoke the agent with guardrails applied."""
        messages = inputs.get("messages", [])

        if not messages:
            return {"messages": messages}

        # Get the last message (user input)
        last_msg = messages[-1]

        # ============================================
        # INPUT GUARDRAILS (always active)
        # ============================================
        if isinstance(last_msg, HumanMessage):
            text = last_msg.content
            pii = detect_pii(text)
            injection = detect_injection(text)

            if pii or injection:
                log_stage("Guardrail: Blocked PII/Injection", output={"pii": pii, "injection": injection})

                response = AIMessage(
                    content=(
                        "I can't process that request. "
                        + ("I detected sensitive information. " if pii else "")
                        + ("This appears to be a prompt injection attempt. " if injection else "")
                        + "Please rephrase."
                    )
                )
                return {"messages": messages + [response]}

        # ============================================
        # CALL THE AGENT
        # ============================================
        result = await self.agent.ainvoke(inputs, config)

        # ============================================
        # OUTPUT GUARDRAILS
        # ============================================
        output_messages = result.get("messages", [])
        if output_messages:
            last_output = output_messages[-1]
            if isinstance(last_output, AIMessage):
                sensitive = contains_sensitive_tool_call(last_output)

                if sensitive and config.ENABLE_HUMAN_IN_LOOP if config else False:
                    # Human-in-the-loop would interrupt here
                    # For now, block the action unless explicitly approved
                    log_stage("Guardrail: Blocked sensitive action", output={"actions": sensitive})
                    response = AIMessage(
                        content=(
                            f"I've detected actions that require human approval: {', '.join(sensitive)}. "
                            f"Please configure the agent with proper human-in-the-loop support to enable these."
                        )
                    )
                    output_messages = output_messages + [response]

        return {"messages": output_messages}


async def apply_guardrails(agent_or_none: Any = None) -> Any:
    """
    Convenience function to apply guardrails to an agent.

    If guardrails are disabled in config, returns the original agent unchanged.
    If enabled and an agent is provided, wraps it with guardrails.
    If enabled but no agent provided, builds one with guardrails.

    Args:
        agent_or_none: The agent to wrap, or None to build a new one

    Returns:
        The original agent (if guardrails disabled) or a guarded wrapper
    """
    if not config.ENABLE_GUARDRAILS:
        if agent_or_none:
            return agent_or_none
        from .agent import build_agent
        return await build_agent()

    if agent_or_none is None:
        from .agent import build_agent
        agent = await build_agent()
        return GuardedAgentWrapper(agent)

    return GuardedAgentWrapper(agent_or_none)


async def build_guarded_agent():
    """
    Build an agent with guardrails applied.
    Use this if you want to build the agent inline.
    """
    if not config.ENABLE_GUARDRAILS:
        from .agent import build_agent
        return await build_agent()

    from .agent import build_agent
    agent = await build_agent()
    return GuardedAgentWrapper(agent)