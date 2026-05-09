"""
Orchestrator — owns the turn loop and routes messages between
Agent, Customer, and Environment.

Provides ``run_conversation`` which generates a single conversation
per scenario and flattens the output into the JSONL format expected
by the evaluator.
"""

import re
import time
from typing import Any, Dict, List, Optional

from .agent import AgentInterface, LLMAgent, LLMCustomer
from .conversation_state import ConversationState, Resolution, ToolCallRecord
from .environment import Environment
from .llm_provider import LLMProvider
from .response_parser import validate_tool_call
from .tool_registry import get_agent_tools, get_customer_tools, get_tool_names


AGENT_PERSONAS = ["DIRECT", "FAIR", "AGREEABLE", "HELPFUL", "VERY_HELPFUL"]


class Orchestrator:
    """Runs the Agent ⇄ Customer ⇄ Environment turn loop for one conversation."""

    def __init__(
        self,
        agent: AgentInterface,
        customer: LLMCustomer,
        environment: Environment,
        scenario: Dict[str, Any],
        variant_id: int,
        agent_persona: str,
        max_turns: int = 20,
        max_errors: int = 3,
        sleep_between_turns: float = 0.0,
    ):
        self.agent = agent
        self.customer = customer
        self.environment = environment
        self.scenario = scenario
        self.variant_id = variant_id
        self.agent_persona = agent_persona
        self.max_turns = max_turns
        self.max_errors = max_errors
        self.sleep_between_turns = sleep_between_turns

        self.state = ConversationState(
            scenario=scenario,
            variant_id=variant_id,
            agent_persona=agent_persona,
        )

        self._valid_agent_tools = get_tool_names(get_agent_tools())
        self._valid_customer_tools = get_tool_names(get_customer_tools())

    def run(self) -> Dict[str, Any]:
        """Execute the full conversation and return a per-variant record."""

        opening = self._seed_customer_opening()
        if opening:
            self.state.append_customer_message(opening)

        _MAX_TOOL_ROUNDS = 5
        _MAX_DUPLICATE_OUTER = 3

        consecutive_duplicate_turns = 0

        for turn_i in range(self.max_turns + 1):
            agent_error = False
            agent_produced_message = False
            all_duplicates_this_turn = False
            for _round in range(_MAX_TOOL_ROUNDS):
                try:
                    agent_resp = self.agent.generate_response(
                        state=self.state,
                        prior_variants_brief="(none)",
                    )
                except Exception as e:
                    print(
                        f"  [Orchestrator] Agent error scenario={self.scenario.get('scenario_id','?')} "
                        f"variant={self.variant_id} turn={turn_i} "
                        f"model={getattr(self.agent.llm_provider, 'model', '?')} "
                        f"err_type={type(e).__name__} msg={str(e)[:200]} "
                        f"history_len={len(self.state.history)} "
                        f"error_count={self.state.error_count + 1}/{self.max_errors}"
                    )
                    self.state.error_count += 1
                    agent_error = True
                    break

                if agent_resp.tool_calls:
                    any_new = False
                    for tc in agent_resp.tool_calls:
                        if not validate_tool_call(tc, self._valid_agent_tools):
                            self.state.error_count += 1
                            continue
                        was_new = self.state.append_tool_call(tc, caller="agent")
                        if was_new:
                            any_new = True
                            tool_result = self.environment.execute_tool(tc)
                            self.state.append_tool_result(tool_result)
                    if not any_new:
                        all_duplicates_this_turn = True
                        if agent_resp.message:
                            self.state.append_agent_message(agent_resp.message)
                            agent_produced_message = True
                        break

                if agent_resp.message:
                    self.state.append_agent_message(agent_resp.message)
                    agent_produced_message = True

                if agent_resp.facts:
                    self.state.agent_facts = agent_resp.facts
                if agent_resp.reasoning_summary:
                    self.state.agent_summary = agent_resp.reasoning_summary
                if agent_resp.agent_persona_type:
                    self.state.agent_persona_type = agent_resp.agent_persona_type

                if agent_resp.conclusion_reached and agent_resp.resolution:
                    pr_result = self._execute_process_return(agent_resp.resolution)
                    if isinstance(pr_result, dict) and pr_result.get("status") == "verification_required":
                        continue
                    self.state.resolution = agent_resp.resolution
                    if pr_result:
                        confirmation = pr_result.get(
                            "message",
                            "Your return has been successfully initiated.",
                        )
                        label_url = pr_result.get("return_label_url", "")
                        if label_url:
                            confirmation += f" Return label: {label_url}"
                        self.state.append_agent_message(confirmation)
                    self.state.finished = True
                    break

                if agent_resp.message or not agent_resp.tool_calls:
                    break

            if self.state.finished:
                break

            if agent_error:
                if self.state.error_count >= self.max_errors:
                    break
                continue

            if all_duplicates_this_turn:
                consecutive_duplicate_turns += 1
                if consecutive_duplicate_turns < _MAX_DUPLICATE_OUTER:
                    if not agent_produced_message:
                        continue
                else:
                    consecutive_duplicate_turns = 0
            else:
                consecutive_duplicate_turns = 0

            if not agent_produced_message:
                continue

            if turn_i >= self.max_turns:
                break

            if self.sleep_between_turns > 0:
                time.sleep(self.sleep_between_turns)

            try:
                cust_resp = self.customer.generate_response(
                    state=self.state,
                    prior_variants_brief="(none)",
                )
            except Exception as e:
                print(
                    f"  [Orchestrator] Customer error scenario={self.scenario.get('scenario_id','?')} "
                    f"variant={self.variant_id} turn={turn_i} "
                    f"model={getattr(self.customer.llm_provider, 'model', '?')} "
                    f"err_type={type(e).__name__} msg={str(e)[:200]} "
                    f"history_len={len(self.state.history)} "
                    f"error_count={self.state.error_count + 1}/{self.max_errors}"
                )
                self.state.error_count += 1
                if self.state.error_count >= self.max_errors:
                    break
                continue

            if cust_resp.tool_calls:
                for tc in cust_resp.tool_calls:
                    if validate_tool_call(tc, self._valid_customer_tools):
                        was_new = self.state.append_tool_call(tc, caller="customer")
                        if was_new:
                            tool_result = self.environment.execute_tool(tc)
                            self.state.append_tool_result(tool_result)

            if cust_resp.reply:
                self.state.append_customer_message(cust_resp.reply)
            if cust_resp.information_provided:
                self.state.revealed_facts.extend(cust_resp.information_provided)

            if cust_resp.withdraw:
                self.state.customer_withdrew = True
                self.state.finished = True
                break

            if self.sleep_between_turns > 0:
                time.sleep(self.sleep_between_turns)

        return self._build_variant_record()

    _ORDER_TOOLS = frozenset({"get_order_details", "customer_view_order_details"})

    def _extract_order_id_from_history(self) -> Optional[str]:
        for turn in self.state.history:
            if turn.turn == "tool_result" and turn.tool_result:
                if turn.tool_result.tool_name in self._ORDER_TOOLS:
                    result = turn.tool_result.result
                    if "order_id" in result:
                        oid = str(result["order_id"])
                        if oid != self.scenario.get("scenario_id", ""):
                            return oid
                    orders = result.get("orders", [])
                    if orders and "order_id" in orders[0]:
                        return str(orders[0]["order_id"])
        return None

    def _extract_customer_id_from_history(self) -> Optional[str]:
        for turn in self.state.history:
            if turn.turn == "tool_result" and turn.tool_result:
                if turn.tool_result.tool_name in self._ORDER_TOOLS:
                    cid = str(turn.tool_result.result.get("customer_id", ""))
                    if cid and re.search(r"[0-9]", cid):
                        return cid
        return None

    def _extract_items_from_history(self) -> list:
        for turn in reversed(self.state.history):
            if turn.turn == "tool_result" and turn.tool_result:
                if turn.tool_result.tool_name in self._ORDER_TOOLS:
                    result = turn.tool_result.result
                    raw_items = result.get("items") or result.get("order_items") or []
                    items = []
                    for item in raw_items:
                        if not isinstance(item, dict):
                            continue
                        item_id = (
                            item.get("item_id")
                            or item.get("product_id")
                            or item.get("sku")
                            or item.get("item_name")
                            or ""
                        )
                        items.append({
                            "item_id": str(item_id),
                            "quantity": int(item.get("quantity", 1)),
                            "condition": "unknown",
                        })
                    if items:
                        return items
        return []

    def _execute_process_return(self, resolution: Resolution) -> Optional[Dict[str, Any]]:
        order_id = self._extract_order_id_from_history()
        if not order_id:
            task = self.scenario.get("task", {})
            order_id = (
                task.get("order_id")
                or task.get("order_number")
                or self.scenario.get("scenario_id", "UNKNOWN")
            )

        customer_id = (
            self._extract_customer_id_from_history()
            or self.scenario.get("persona", {}).get("customer_id")
            or self.scenario.get("persona", {}).get("Name", "UNKNOWN")
        )

        tc = ToolCallRecord(
            tool_name="process_return",
            tool_call_id=f"process_return_{self.variant_id}",
            arguments={
                "order_id": str(order_id),
                "customer_id": str(customer_id),
                "resolution_type": resolution.resolution_type,
                "items_to_return": self._extract_items_from_history(),
                "return_reason": "other",
                "return_reason_details": resolution.resolution_description,
            },
        )

        if not self.state.verification_attempted:
            self.state.verification_attempted = True
            verification = self.environment.verify_return(
                history=self.state.get_history_dicts(),
                arguments=tc.arguments,
            )
            if not verification.get("verified", True):
                from .conversation_state import ConversationTurn, ToolResultRecord
                self.state.history.append(
                    ConversationTurn(turn="tool_call", tool_calls=[tc])
                )
                self.state.tool_interactions.append({
                    "tool_call_id": tc.tool_call_id,
                    "tool_name": tc.tool_name,
                    "arguments": tc.arguments,
                    "caller": "agent",
                })
                hints = verification.get("verification_hints", [])
                synthetic_result = ToolResultRecord(
                    tool_call_id=tc.tool_call_id,
                    tool_name="process_return",
                    result={
                        "status": "verification_required",
                        "verified": False,
                        "discrepancies": verification.get("discrepancies", []),
                        "verification_hints": hints,
                        "message": (
                            "Return could not be processed: the information provided "
                            "does not match our records. Please clarify the following "
                            "with the customer before resubmitting: "
                            + ("; ".join(hints) if hints else "see discrepancies above")
                        ),
                    },
                )
                self.state.append_tool_result(synthetic_result)
                return synthetic_result.result

        was_new = self.state.append_tool_call(tc, caller="agent")
        if was_new:
            tool_result = self.environment.execute_tool(tc)
            self.state.append_tool_result(tool_result)
            return tool_result.result
        return None

    def _seed_customer_opening(self) -> str:
        opening = self.scenario.get("first_customer_message", "")
        if opening and isinstance(opening, str):
            return opening
        return self.scenario.get("task", {}).get("task", "")

    def _build_variant_record(self) -> Dict[str, Any]:
        if not self.state.finished:
            print(
                f"  [Orchestrator] Variant aborted scenario={self.scenario.get('scenario_id','?')} "
                f"variant={self.variant_id} history_turns={len(self.state.history)} "
                f"error_count={self.state.error_count}/{self.max_errors} "
                f"customer_withdrew={self.state.customer_withdrew} "
                f"resolution={'set' if self.state.resolution else 'none'}"
            )

        resolution_dict = None
        if self.state.resolution:
            base = self.state.resolution.model_dump()
            desc_parts = [base.get("resolution_description", "")]
            if base.get("conditions"):
                desc_parts.append("CONDITIONS: " + "; ".join(base["conditions"]))
            if base.get("customer_next_steps"):
                desc_parts.append("CUSTOMER NEXT STEPS: " + base["customer_next_steps"])
            if self.state.agent_facts:
                desc_parts.append("FACTS USED: " + "; ".join(self.state.agent_facts))
            if self.state.agent_summary:
                desc_parts.append("AGENT REASONING: " + self.state.agent_summary)
            resolution_dict = {
                "resolution_type": base["resolution_type"],
                "resolution_description": "\n\n".join(p for p in desc_parts if p.strip()),
            }

        return {
            "scenario_id": self.scenario.get("scenario_id", "unknown"),
            "conversation_type": "single",
            "conversation_variant_id": self.variant_id,
            "agent_persona": self.state.agent_persona_type or self.agent_persona,
            "finished": self.state.finished,
            "customer_withdrew": self.state.customer_withdrew,
            "conversation_history": self.state.get_history_dicts(),
            "tool_interactions": self.state.tool_interactions,
            "customer_tool_interactions": self.state.customer_tool_interactions,
            "agent_final_object": {
                "final_resolution": resolution_dict,
            },
            "source_scenario": self.scenario,
        }


def run_conversation(
    scenario: Dict[str, Any],
    llm_provider: LLMProvider,
    agent_persona: str = "FAIR",
    max_turns: int = 20,
    max_errors: int = 3,
    use_native_tools: bool = True,
    sleep_between_turns: float = 0.0,
    include_resolution: bool = True,
    customer_llm_provider: Optional[LLMProvider] = None,
) -> Dict[str, Any]:
    """Run a single conversation for one scenario.

    The agent picks the most plausible resolution; output is flattened
    into the column layout expected by the evaluator.
    """
    scenario_id = scenario.get("scenario_id", "unknown")
    cust_prov = customer_llm_provider or llm_provider
    env = Environment(scenario, cust_prov)

    agent = LLMAgent(
        llm_provider=llm_provider,
        scenario=scenario,
        use_native_tools=use_native_tools,
        single_mode=True,
    )
    customer = LLMCustomer(
        llm_provider=cust_prov,
        scenario=scenario,
        single_mode=True,
    )

    orch = Orchestrator(
        agent=agent,
        customer=customer,
        environment=env,
        scenario=scenario,
        variant_id=1,
        agent_persona=agent_persona,
        max_turns=max_turns,
        max_errors=max_errors,
        sleep_between_turns=sleep_between_turns,
    )

    rec = orch.run()
    return _flatten_to_evaluator_format(
        scenario_id=scenario_id,
        scenario=scenario,
        variant_record=rec,
        agent_persona=agent_persona,
        include_resolution=include_resolution,
    )


def _flatten_to_evaluator_format(
    scenario_id: str,
    scenario: Dict[str, Any],
    variant_record: Dict[str, Any],
    agent_persona: str,
    include_resolution: bool = True,
) -> Dict[str, Any]:
    """Flatten the conversation into evaluator-compatible columns.

    The evaluator iterates with ``while f"conversation_{conv_idx}" in transcript``
    so output uses ``conversation_1``, ``tool_interactions_1``, ``resolution_1`` etc.
    """
    out: Dict[str, Any] = {
        "scenario_id": scenario_id,
        "conversation_type": "single",
        "finished": bool(variant_record.get("finished", False)),
        "source_scenario": scenario,
    }

    vid = variant_record["conversation_variant_id"]
    out[f"conversation_{vid}"] = variant_record.get("conversation_history", [])
    out[f"tool_interactions_{vid}"] = variant_record.get("tool_interactions", [])
    out[f"customer_tool_interactions_{vid}"] = variant_record.get("customer_tool_interactions", [])
    out[f"customer_withdrew_{vid}"] = variant_record.get("customer_withdrew", False)
    out[f"agent_persona_{vid}"] = variant_record.get("agent_persona", agent_persona)

    agent_obj = variant_record.get("agent_final_object") or {}
    out[f"resolution_{vid}"] = (
        agent_obj.get("final_resolution")
        if include_resolution and isinstance(agent_obj, dict)
        else None
    )

    out["num_conversations"] = 1
    return out
