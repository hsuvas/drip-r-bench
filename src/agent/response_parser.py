"""
Response parser & validator — converts raw LLM output into typed objects.

Supports two parsing paths:
  A) Native tool calls returned via the provider API.
  B) JSON text in the message content (the prompt instructs JSON output).

Also validates tool names against the known registry.
"""

import json
import re
from typing import Any, Dict, List, Optional

try:
    import json_repair as _json_repair
    _HAS_JSON_REPAIR = True
except ImportError:
    _HAS_JSON_REPAIR = False

from .conversation_state import Resolution, ToolCallRecord

# ---------------------------------------------------------------------------
# JSON extraction regexes (mirrors src/dataset/output_collect patterns)
# ---------------------------------------------------------------------------

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*\}|\[.*\])\s*```", re.DOTALL)
_JSON_OBJ_RE = re.compile(r"(\{.*\})", re.DOTALL)


def _repair(s: str) -> Dict[str, Any]:
    if _HAS_JSON_REPAIR:
        result = _json_repair.repair_json(s, return_objects=True)
        if isinstance(result, dict):
            return result
    raise ValueError("json-repair unavailable or produced non-dict")


def extract_json(text: str) -> Dict[str, Any]:
    """Best-effort extraction of a JSON object from LLM text output."""
    t = text.strip()

    # Already valid JSON
    try:
        return json.loads(t)
    except Exception:
        pass

    # Fenced code block
    m = _JSON_FENCE_RE.search(t)
    if m:
        candidate = m.group(1)
        try:
            return json.loads(candidate)
        except Exception:
            return _repair(candidate)

    # Bare object
    m = _JSON_OBJ_RE.search(t)
    if m:
        candidate = m.group(1)
        try:
            return json.loads(candidate)
        except Exception:
            return _repair(candidate)

    # Last resort: repair the whole text
    return _repair(t)


# ---------------------------------------------------------------------------
# Parsed response types
# ---------------------------------------------------------------------------

class AgentResponse:
    """Parsed agent turn — may contain a message, tool calls, or both."""

    def __init__(
        self,
        message: Optional[str] = None,
        tool_calls: Optional[List[ToolCallRecord]] = None,
        facts: Optional[List[str]] = None,
        policy_refs: Optional[List[str]] = None,
        resolution: Optional[Resolution] = None,
        conclusion_reached: bool = False,
        reasoning_summary: Optional[str] = None,
        agent_persona_type: Optional[str] = None,
        raw_json: Optional[Dict[str, Any]] = None,
    ):
        self.message = message
        self.tool_calls = tool_calls
        self.facts = facts or []
        self.policy_refs = policy_refs or []
        self.resolution = resolution
        self.conclusion_reached = conclusion_reached
        self.reasoning_summary = reasoning_summary
        self.agent_persona_type = agent_persona_type
        self.raw_json = raw_json


class CustomerResponse:
    """Parsed customer turn."""

    def __init__(
        self,
        reply: str,
        information_provided: Optional[List[str]] = None,
        emotional_tone: str = "neutral",
        tool_calls: Optional[List[ToolCallRecord]] = None,
        withdraw: bool = False,
    ):
        self.reply = reply
        self.information_provided = information_provided or []
        self.emotional_tone = emotional_tone
        self.tool_calls = tool_calls
        self.withdraw = withdraw


# ---------------------------------------------------------------------------
# Agent response parsing
# ---------------------------------------------------------------------------

def parse_agent_response(llm_response: Any) -> AgentResponse:
    """Parse an ``LLMResponse`` into an ``AgentResponse``.

    Handles both native tool-call responses (Path A) and JSON-in-content
    responses (Path B).
    """
    # Path A: native tool calls from the API
    if llm_response.has_tool_calls:
        tool_calls = [
            ToolCallRecord(
                tool_name=tc["tool_name"],
                tool_call_id=tc["tool_call_id"],
                arguments=tc["arguments"],
            )
            for tc in llm_response.tool_calls
        ]
        # The content may also contain a JSON body with the full agent object
        message = None
        facts: List[str] = []
        policy_refs: List[str] = []
        resolution = None
        conclusion = False
        reasoning = None
        persona = None
        raw = None

        if llm_response.content:
            try:
                raw = extract_json(llm_response.content)
                parsed = _parse_agent_json_body(raw)
                message = parsed.message
                facts = parsed.facts
                policy_refs = parsed.policy_refs
                resolution = parsed.resolution
                conclusion = parsed.conclusion_reached
                reasoning = parsed.reasoning_summary
                persona = parsed.agent_persona_type
            except Exception:
                # Content wasn't valid JSON — use as plain message
                message = llm_response.content.strip() or None

        return AgentResponse(
            message=message,
            tool_calls=tool_calls,
            facts=facts,
            policy_refs=policy_refs,
            resolution=resolution,
            conclusion_reached=conclusion,
            reasoning_summary=reasoning,
            agent_persona_type=persona,
            raw_json=raw,
        )

    # Path B: JSON text content
    if not llm_response.content:
        raise ValueError("Empty response from agent LLM")

    raw = extract_json(llm_response.content)
    return _parse_agent_json_body(raw)


def _parse_agent_json_body(data: Dict[str, Any]) -> AgentResponse:
    """Parse the JSON body the existing agent prompt instructs the LLM to produce."""
    # Messages from conversation_flow
    messages: List[str] = []
    for item in data.get("conversation_flow", []):
        msg = (item or {}).get("message", "")
        if msg:
            messages.append(msg)
    combined_message = "\n".join(messages) if messages else None

    # Tool calls from tool_calls_made
    tool_calls: Optional[List[ToolCallRecord]] = None
    raw_tools = data.get("tool_calls_made") or []
    if raw_tools:
        tool_calls = []
        for i, tc in enumerate(raw_tools):
            tool_calls.append(
                ToolCallRecord(
                    tool_name=tc.get("tool_name", "unknown"),
                    tool_call_id=tc.get("tool_call_id", f"call_{i}"),
                    arguments=tc.get("arguments", {}),
                )
            )

    # Resolution
    resolution = None
    conclusion_str = data.get("conclusion_reached", "No")
    conclusion = conclusion_str == "Yes"
    raw_res = data.get("final_resolution")
    if raw_res and conclusion:
        try:
            resolution = Resolution(**raw_res)
        except Exception:
            resolution = None

    return AgentResponse(
        message=combined_message,
        tool_calls=tool_calls,
        facts=data.get("facts_collected_or_assumed", []),
        policy_refs=data.get("policy_references_used", []),
        resolution=resolution,
        conclusion_reached=conclusion,
        reasoning_summary=data.get("reasoning_summary"),
        agent_persona_type=data.get("agent_persona_type"),
        raw_json=data,
    )


# ---------------------------------------------------------------------------
# Customer response parsing
# ---------------------------------------------------------------------------

def parse_customer_response(llm_response: Any) -> CustomerResponse:
    """Parse an ``LLMResponse`` into a ``CustomerResponse``."""
    # Native tool calls path: customer model used API-level tool calling
    native_tool_calls: Optional[List[ToolCallRecord]] = None
    if llm_response.has_tool_calls:
        native_tool_calls = [
            ToolCallRecord(
                tool_name=tc["tool_name"],
                tool_call_id=tc["tool_call_id"],
                arguments=tc["arguments"],
            )
            for tc in llm_response.tool_calls
        ]
        # Check for withdraw_from_conversation native call
        withdraw = any(tc.tool_name == "withdraw_from_conversation" for tc in native_tool_calls)
        if not llm_response.content:
            return CustomerResponse(reply="", tool_calls=native_tool_calls, withdraw=withdraw)

    if not llm_response.content:
        raise ValueError("Empty response from customer LLM")

    data = extract_json(llm_response.content)

    # Optional tool calls
    tool_calls: Optional[List[ToolCallRecord]] = None
    raw_tools = data.get("tool_calls_made") or []
    if raw_tools:
        tool_calls = [
            ToolCallRecord(
                tool_name=tc.get("tool_name", ""),
                tool_call_id=tc.get("tool_call_id", f"cust_call_{i}"),
                arguments=tc.get("arguments", {}),
            )
            for i, tc in enumerate(raw_tools)
        ]

    # Merge native tool calls with any JSON-declared ones; native take precedence
    merged_tool_calls = native_tool_calls or tool_calls
    withdraw = (native_tool_calls is not None and any(
        tc.tool_name == "withdraw_from_conversation" for tc in native_tool_calls
    )) or data.get("withdraw_conversation", False)

    return CustomerResponse(
        reply=data.get("customer_reply", ""),
        information_provided=data.get("information_provided", []),
        emotional_tone=data.get("emotional_tone", "neutral"),
        tool_calls=merged_tool_calls,
        withdraw=withdraw,
    )


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def validate_tool_call(
    tool_call: ToolCallRecord,
    valid_tool_names: List[str],
) -> bool:
    """Return True if the tool call references a known tool name."""
    return tool_call.tool_name in valid_tool_names
