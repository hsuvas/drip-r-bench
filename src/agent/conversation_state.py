"""
State / Memory module — stores the rolling conversation transcript.

Provides Pydantic models for conversation turns, tool calls, tool results,
and resolutions. The ConversationState container holds per-variant mutable
state and exposes helpers that the orchestrator and prompt builder use.
"""

import json
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

RESOLUTION_TYPE_CHOICES = Literal[
    "RETURN_REFUND_FULL_BANK",
    "RETURN_REFUND_PARTIAL_BANK",
    "RETURN_REFUND_GIFT_CARD",
    "DENY_REFUND",
    "ESCALATE_HUMAN_AGENT",
    "REPLACEMENT_EXCHANGE",
    "USER_ABORT",
]

AGENT_PERSONA_CHOICES = Literal[
    "DIRECT", "FAIR", "AGREEABLE", "HELPFUL", "VERY_HELPFUL",
]


# ---------------------------------------------------------------------------
# Pydantic models (wire-compatible with src/dataset/ models)
# ---------------------------------------------------------------------------

class ToolCallRecord(BaseModel):
    """A single tool invocation."""
    tool_name: str
    tool_call_id: str
    arguments: Dict[str, Any] = Field(default_factory=dict)


class ToolResultRecord(BaseModel):
    """Result returned from the environment for a tool call."""
    tool_call_id: str
    tool_name: str
    result: Dict[str, Any]


class ConversationTurn(BaseModel):
    """One entry in the rolling transcript."""
    turn: Literal["agent", "customer", "tool_call", "tool_result"]
    message: Optional[str] = None
    tool_calls: Optional[List[ToolCallRecord]] = None
    tool_result: Optional[ToolResultRecord] = None


class Resolution(BaseModel):
    """Final resolution of a conversation variant."""
    resolution_id: str
    resolution_type: RESOLUTION_TYPE_CHOICES
    resolution_description: str
    conditions: List[str] = Field(default_factory=list)
    customer_next_steps: str


# ---------------------------------------------------------------------------
# Conversation state container
# ---------------------------------------------------------------------------

# For these tools only the primary key determines the result; extra args (like
# customer_id, detail_type, include_tracking) don't change the response and
# should not create a distinct signature that evades duplicate detection.
_DEDUP_KEY_ARGS: Dict[str, str] = {
    "get_order_details": "order_id",
    "customer_view_order_details": "order_id",
}


def _tool_call_signature(tool_name: str, arguments: Dict[str, Any]) -> str:
    """Deterministic signature for deduplicating identical tool calls."""
    key_arg = _DEDUP_KEY_ARGS.get(tool_name)
    if key_arg and key_arg in arguments:
        return f"{tool_name}::{key_arg}={arguments[key_arg]}"
    args_normalized = json.dumps(arguments, sort_keys=True, ensure_ascii=False)
    return f"{tool_name}::{args_normalized}"


class ConversationState:
    """Mutable container for one conversation variant's state."""

    def __init__(
        self,
        scenario: Dict[str, Any],
        variant_id: int,
        agent_persona: str,
    ):
        self.scenario = scenario
        self.variant_id = variant_id
        self.agent_persona = agent_persona

        # Rolling transcript
        self.history: List[ConversationTurn] = []

        # Tool interaction logs (evaluator expects separate agent / customer lists)
        self.tool_interactions: List[Dict[str, Any]] = []
        self.customer_tool_interactions: List[Dict[str, Any]] = []

        # Dedup sets
        self._seen_agent_sigs: set = set()
        self._seen_customer_sigs: set = set()

        # Termination flags
        self.finished: bool = False
        self.customer_withdrew: bool = False
        self.error_count: int = 0

        # Metadata populated by the agent
        self.resolution: Optional[Resolution] = None
        self.agent_facts: List[str] = []
        self.agent_summary: Optional[str] = None
        self.agent_persona_type: Optional[str] = None

        # Facts the customer has disclosed turn-by-turn
        self.revealed_facts: List[str] = []

        # Set to True after the first process_return verification attempt
        # so subsequent calls skip verification and execute directly.
        self.verification_attempted: bool = False

        # Stateful chat-mode messages arrays (populated only when chat_mode=True)
        self.agent_messages: List[Dict[str, Any]] = []
        self.agent_messages_ptr: int = 0
        self.customer_messages: List[Dict[str, Any]] = []
        self.customer_messages_ptr: int = 0

    # ----- append helpers -----

    def _last_speaker(self) -> Optional[str]:
        """Return the role of the last agent/customer message, ignoring tool entries."""
        for turn in reversed(self.history):
            if turn.turn in ("agent", "customer"):
                return turn.turn
        return None

    def append_customer_message(self, message: str) -> None:
        if self._last_speaker() == "customer":
            print(
                f"  [ConversationState] Skipping consecutive customer message "
                f"(no agent turn between): {message[:80]!r}"
            )
            return
        self.history.append(ConversationTurn(turn="customer", message=message))

    def append_agent_message(self, message: str) -> None:
        if self._last_speaker() == "agent":
            print(
                f"  [ConversationState] Skipping consecutive agent message "
                f"(no customer turn between): {message[:80]!r}"
            )
            return
        self.history.append(ConversationTurn(turn="agent", message=message))

    def append_tool_call(
        self,
        tool_call: ToolCallRecord,
        caller: str = "agent",
    ) -> bool:
        """Append a tool call. Returns False (and skips) if it is a duplicate.

        When a duplicate is detected, a synthetic ALREADY_CALLED tool_result is
        appended to history so the LLM knows not to repeat the call.
        """
        sig = _tool_call_signature(tool_call.tool_name, tool_call.arguments)
        seen = self._seen_agent_sigs if caller == "agent" else self._seen_customer_sigs
        if sig in seen:
            return False  # Silently skip; orchestrator handles duplicate detection

        seen.add(sig)
        self.history.append(
            ConversationTurn(turn="tool_call", tool_calls=[tool_call])
        )

        record = {
            "tool_call_id": tool_call.tool_call_id,
            "tool_name": tool_call.tool_name,
            "arguments": tool_call.arguments,
            "caller": caller,
        }
        if caller == "agent":
            self.tool_interactions.append(record)
        else:
            self.customer_tool_interactions.append(record)
        return True

    def append_tool_result(self, result: ToolResultRecord) -> None:
        self.history.append(
            ConversationTurn(turn="tool_result", tool_result=result)
        )

    # ----- serialisation helpers -----

    def get_history_dicts(self) -> List[Dict[str, Any]]:
        """Return the transcript as plain dicts (for JSONL output)."""
        return [t.model_dump() for t in self.history]

    def get_formatted_history_str(self) -> str:
        """JSON-serialised history string suitable for prompt injection."""
        dicts = self.get_history_dicts()
        if not dicts:
            return "(empty)"
        return json.dumps(dicts, ensure_ascii=False, indent=2)
