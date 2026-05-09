"""
Prompt builder — constructs messages for the agent and customer LLMs.
"""

import json
from typing import Any, Dict, List, Optional

from .conversation_state import ConversationState, ConversationTurn
from .prompts.output_gen_prompt_agent import (
    output_creation_prompt as _AGENT_PROMPT_RAW,
)
from .prompts.output_gen_prompt_customer import (
    customer_response_prompt as _CUSTOMER_PROMPT_RAW,
)
from .prompt_single import (
    single_agent_prompt as _SINGLE_AGENT_PROMPT_RAW,
    single_customer_prompt as _SINGLE_CUSTOMER_PROMPT_RAW,
)
from .tool_registry import format_tools_for_prompt_detailed, format_customer_tools_for_prompt_detailed


# ---------------------------------------------------------------------------
# Safe format-string construction
# ---------------------------------------------------------------------------

def _make_safe_format_string(raw_prompt: str, allowed_keys: List[str]) -> str:
    """Escape all ``{`` / ``}`` then restore only *allowed_keys*.

    This is required because the prompt templates contain literal JSON
    curly braces in their output-format examples.
    """
    s = raw_prompt.replace("{", "{{").replace("}", "}}")
    for k in allowed_keys:
        s = s.replace("{{" + k + "}}", "{" + k + "}")
    return s


_AGENT_ALLOWED_KEYS = [
    "agent_persona",
    "conversation_type",
    "conversation_variant_id",
    "prior_variants_brief",
    "primary_policy_text",
    "related_policies_text",
    "detail_agent",
    "conversation_history",
]

_CUSTOMER_ALLOWED_KEYS = [
    "conversation_type",
    "conversation_variant_id",
    "prior_variants_brief",
    "persona_details",
    "basic_info",
    "private_info",
    "behavior",
    "primary_policy_text",
    "conversation_history",
    "revealed_facts",
    "latest_agent_message",
    "customer_tools_text",
]

_AGENT_PROMPT = _make_safe_format_string(_AGENT_PROMPT_RAW, _AGENT_ALLOWED_KEYS)
_CUSTOMER_PROMPT = _make_safe_format_string(_CUSTOMER_PROMPT_RAW, _CUSTOMER_ALLOWED_KEYS)

# Single-conversation mode keys (no variant_id / prior_variants_brief)
_SINGLE_AGENT_ALLOWED_KEYS = [
    "agent_persona",
    "conversation_type",
    "primary_policy_text",
    "related_policies_text",
    "detail_agent",
    "conversation_history",
]

_SINGLE_CUSTOMER_ALLOWED_KEYS = [
    "conversation_type",
    "persona_details",
    "basic_info",
    "private_info",
    "behavior",
    "primary_policy_text",
    "conversation_history",
    "revealed_facts",
    "latest_agent_message",
    "customer_tools_text",
]

_SINGLE_AGENT_PROMPT = _make_safe_format_string(
    _SINGLE_AGENT_PROMPT_RAW, _SINGLE_AGENT_ALLOWED_KEYS
)
_SINGLE_CUSTOMER_PROMPT = _make_safe_format_string(
    _SINGLE_CUSTOMER_PROMPT_RAW, _SINGLE_CUSTOMER_ALLOWED_KEYS
)


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def _safe_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)


def _related_policies_to_text(related_policies: List[Dict[str, Any]]) -> str:
    if not related_policies:
        return "(none)"
    blocks = []
    for i, rp in enumerate(related_policies, start=1):
        url = rp.get("url", "(missing url)")
        txt = rp.get("text", "")
        blocks.append(f"[Related Policy {i}] {url}\n{txt}".strip())
    return "\n\n".join(blocks).strip()


def _latest_agent_message(history: List[ConversationTurn]) -> Optional[str]:
    for turn in reversed(history):
        if turn.turn == "agent" and turn.message and turn.message.strip():
            return turn.message.strip()
    return None


def _agent_visible_history(history: List[ConversationTurn]) -> List[ConversationTurn]:
    """Return history with customer tool calls and their results removed.

    The agent must gather order/product facts via its own tool calls, not by
    reading the customer's lookup results from the shared transcript. Customer
    messages are kept — the customer can still report facts in text.
    """
    customer_tool_ids: set = set()
    for turn in history:
        if turn.turn == "tool_call" and turn.tool_calls:
            for tc in turn.tool_calls:
                if tc.tool_name.startswith("customer_"):
                    customer_tool_ids.add(tc.tool_call_id)

    filtered: List[ConversationTurn] = []
    for turn in history:
        if turn.turn == "tool_call" and turn.tool_calls:
            agent_calls = [tc for tc in turn.tool_calls if not tc.tool_name.startswith("customer_")]
            if agent_calls:
                filtered.append(ConversationTurn(turn="tool_call", tool_calls=agent_calls))
            # If all calls were customer tools, omit the turn entirely
        elif turn.turn == "tool_result" and turn.tool_result:
            tid = turn.tool_result.tool_call_id
            tname = turn.tool_result.tool_name
            if tid in customer_tool_ids or tname.startswith("customer_"):
                continue
            filtered.append(turn)
        else:
            filtered.append(turn)
    return filtered


# ---------------------------------------------------------------------------
# Agent prompt construction
# ---------------------------------------------------------------------------

def build_agent_messages(
    scenario: Dict[str, Any],
    state: ConversationState,
    prior_variants_brief: str = "(none)",
    use_native_tools: bool = True,
    single_mode: bool = False,
) -> List[Dict[str, str]]:
    """Return ``[system, user]`` messages for the agent LLM call.

    When *use_native_tools* is True the system prompt is compact (tool
    definitions are passed via the API ``tools`` parameter).  When False,
    the full tool descriptions are injected into the system prompt.

    When *single_mode* is True, uses the single-conversation prompt that
    lets the agent pick the most plausible resolution instead of
    targeting a hard-coded outcome per variant.
    """
    system = _build_agent_system_prompt(use_native_tools)
    if single_mode:
        user = _build_single_agent_user_prompt(scenario, state)
    else:
        user = _build_agent_user_prompt(scenario, state, prior_variants_brief)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def _build_agent_system_prompt(use_native_tools: bool) -> str:
    base = (
        "RUNTIME CONSTRAINT (VERY IMPORTANT):\n"
        "- You are in an iterative loop.\n"
        '- If you still need customer info, set conclusion_reached to "No" '
        "and final_resolution to null.\n"
        '- Only when decision-ready, set conclusion_reached to "Yes" '
        "and output final_resolution.\n"
        "- Output MUST be valid JSON only, matching the schema in the prompt.\n"
    )
    if not use_native_tools:
        base += (
            "\n\n"
            + format_tools_for_prompt_detailed()
            + "\n\n"
            "- MANDATORY: You MUST include at least one tool call in tool_calls_made.\n"
            "- Specify which tools you would call and with what arguments.\n"
            "- Tool results will be provided back to you in the conversation history.\n"
        )
    return base


def _build_agent_user_prompt(
    scenario: Dict[str, Any],
    state: ConversationState,
    prior_variants_brief: str,
) -> str:
    primary_policy_text = scenario["Policy"]["Primary Policy"]["text"]
    related_policies = scenario["Policy"].get("Related policies", [])
    related_blob = _related_policies_to_text(related_policies)
    detail_agent = scenario.get("detail_agent", "")

    agent_history = _agent_visible_history(state.history)
    agent_history_str = json.dumps(
        [t.model_dump(exclude_none=True) for t in agent_history], ensure_ascii=False, indent=2
    ) if agent_history else "(empty)"

    return _AGENT_PROMPT.format(
        agent_persona=state.agent_persona,
        conversation_type="multitype",
        conversation_variant_id=str(state.variant_id),
        prior_variants_brief=prior_variants_brief,
        primary_policy_text=primary_policy_text,
        related_policies_text=related_blob,
        detail_agent=detail_agent,
        conversation_history=agent_history_str,
    )


def _build_single_agent_user_prompt(
    scenario: Dict[str, Any],
    state: ConversationState,
) -> str:
    primary_policy_text = scenario["Policy"]["Primary Policy"]["text"]
    related_policies = scenario["Policy"].get("Related policies", [])
    related_blob = _related_policies_to_text(related_policies)
    detail_agent = scenario.get("detail_agent", "")

    agent_history = _agent_visible_history(state.history)
    agent_history_str = json.dumps(
        [t.model_dump(exclude_none=True) for t in agent_history], ensure_ascii=False, indent=2
    ) if agent_history else "(empty)"

    return _SINGLE_AGENT_PROMPT.format(
        agent_persona=state.agent_persona,
        conversation_type="single",
        primary_policy_text=primary_policy_text,
        related_policies_text=related_blob,
        detail_agent=detail_agent,
        conversation_history=agent_history_str,
    )


# ---------------------------------------------------------------------------
# Customer prompt construction
# ---------------------------------------------------------------------------

def build_customer_messages(
    scenario: Dict[str, Any],
    state: ConversationState,
    prior_variants_brief: str = "(none)",
    single_mode: bool = False,
) -> List[Dict[str, str]]:
    """Return ``[system, user]`` messages for the customer LLM call."""
    system = "Output ONLY valid JSON matching the schema in the prompt."
    if single_mode:
        user = _build_single_customer_user_prompt(scenario, state)
    else:
        user = _build_customer_user_prompt(scenario, state, prior_variants_brief)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def _build_customer_user_prompt(
    scenario: Dict[str, Any],
    state: ConversationState,
    prior_variants_brief: str,
) -> str:
    persona_details = scenario.get("persona", {})
    primary_policy_text = scenario["Policy"]["Primary Policy"]["text"]
    ctd = scenario.get("customer_task_detail", {})
    latest = _latest_agent_message(state.history) or "(none yet)"
    revealed_text = _safe_json(state.revealed_facts) if state.revealed_facts else "[]"

    return _CUSTOMER_PROMPT.format(
        conversation_type="multitype",
        conversation_variant_id=str(state.variant_id),
        prior_variants_brief=prior_variants_brief,
        persona_details=_safe_json(persona_details),
        basic_info=ctd.get("basic_info", "(not provided)"),
        private_info=ctd.get("private_info", "(not provided)"),
        behavior=ctd.get("behavior", "(not provided)"),
        primary_policy_text=primary_policy_text,
        conversation_history=state.get_formatted_history_str(),
        revealed_facts=revealed_text,
        latest_agent_message=latest,
        customer_tools_text=format_customer_tools_for_prompt_detailed(),
    )


def build_agent_messages_chat_init(
    scenario: Dict[str, Any],
    state: ConversationState,
    prior_variants_brief: str = "(none)",
    use_native_tools: bool = True,
    single_mode: bool = False,
) -> List[Dict[str, Any]]:
    """Initial [system, user] for stateful chat mode (first call only).

    Embeds the full static context (policy, task, tool descriptions if not native)
    plus the customer's opening message. Marked with cache_control so Anthropic
    caches it after the first call; silently ignored for other providers via
    litellm.drop_params = True.
    """
    system_content = _build_agent_system_prompt(use_native_tools)
    opening = state.history[0].message if state.history else ""

    if single_mode:
        static_ctx = _SINGLE_AGENT_PROMPT.format(
            agent_persona=state.agent_persona,
            conversation_type="single",
            primary_policy_text=scenario["Policy"]["Primary Policy"]["text"],
            related_policies_text=_related_policies_to_text(
                scenario["Policy"].get("Related policies", [])
            ),
            detail_agent=scenario.get("detail_agent", ""),
            conversation_history="(provided via chat history)",
        )
    else:
        static_ctx = _AGENT_PROMPT.format(
            agent_persona=state.agent_persona,
            conversation_type="multitype",
            conversation_variant_id=str(state.variant_id),
            prior_variants_brief=prior_variants_brief,
            primary_policy_text=scenario["Policy"]["Primary Policy"]["text"],
            related_policies_text=_related_policies_to_text(
                scenario["Policy"].get("Related policies", [])
            ),
            detail_agent=scenario.get("detail_agent", ""),
            conversation_history="(provided via chat history)",
        )

    first_user_text = static_ctx + f"\n\nThe customer has contacted you with:\n{opening}"
    return [
        {"role": "system", "content": system_content},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": first_user_text, "cache_control": {"type": "ephemeral"}}
            ],
        },
    ]


def agent_history_to_api_messages(
    new_turns: List[ConversationTurn],
) -> List[Dict[str, Any]]:
    """Convert new ConversationTurn entries to API message dicts (agent view).

    Skips 'agent' and 'tool_call' turns — the agent appends those itself
    after each LLM response.
    """
    msgs = []
    for t in new_turns:
        if t.turn == "customer" and t.message:
            msgs.append({"role": "user", "content": f"Customer: {t.message}"})
        elif t.turn == "tool_result" and t.tool_result:
            msgs.append({
                "role": "tool",
                "tool_call_id": t.tool_result.tool_call_id,
                "content": json.dumps(t.tool_result.result, ensure_ascii=False),
            })
    return msgs


def build_customer_messages_chat_init(
    scenario: Dict[str, Any],
    state: ConversationState,
    prior_variants_brief: str = "(none)",
    single_mode: bool = False,
) -> List[Dict[str, Any]]:
    """Initial [system, user] for stateful customer chat mode (first call only)."""
    if single_mode:
        static_ctx = _build_single_customer_user_prompt(scenario, state)
    else:
        static_ctx = _build_customer_user_prompt(scenario, state, prior_variants_brief)

    return [
        {"role": "system", "content": "Output ONLY valid JSON matching the schema in the prompt."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": static_ctx, "cache_control": {"type": "ephemeral"}}
            ],
        },
    ]


def customer_history_to_api_messages(
    new_turns: List[ConversationTurn],
) -> List[Dict[str, Any]]:
    """Convert new ConversationTurn entries to API message dicts (customer view).

    Only agent messages are visible to the customer LLM.
    """
    msgs = []
    for t in new_turns:
        if t.turn == "agent" and t.message:
            msgs.append({"role": "user", "content": f"Agent: {t.message}"})
    return msgs


def _build_single_customer_user_prompt(
    scenario: Dict[str, Any],
    state: ConversationState,
) -> str:
    persona_details = scenario.get("persona", {})
    primary_policy_text = scenario["Policy"]["Primary Policy"]["text"]
    ctd = scenario.get("customer_task_detail", {})
    latest = _latest_agent_message(state.history) or "(none yet)"
    revealed_text = _safe_json(state.revealed_facts) if state.revealed_facts else "[]"

    return _SINGLE_CUSTOMER_PROMPT.format(
        conversation_type="single",
        persona_details=_safe_json(persona_details),
        basic_info=ctd.get("basic_info", "(not provided)"),
        private_info=ctd.get("private_info", "(not provided)"),
        behavior=ctd.get("behavior", "(not provided)"),
        primary_policy_text=primary_policy_text,
        conversation_history=state.get_formatted_history_str(),
        revealed_facts=revealed_text,
        latest_agent_message=latest,
        customer_tools_text=format_customer_tools_for_prompt_detailed(),
    )
