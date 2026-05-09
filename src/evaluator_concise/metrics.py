"""
Embedded metric registry — 25 sub-metrics across 5 evaluation dimensions.

Each entry has the same shape produced by the original CSV loader so the
rest of the evaluator code is unchanged:

    {"name": str, "field_key": str, "metric_type": str, "scale": str, "definition": str}

`metric_type` is one of: "conversation", "turn", "resolution".

Dimensions:
  - Policy Adherence (4)
  - Task-Resolution Adherence (2)
  - Dialogue/Conversation Quality (8)
  - Interest Alignment (2)
  - Behavioral Alignment (9)

Notes:
- The `Task` dimension and the `Power` (Schwartz) metric from the original
  CSV are intentionally dropped.
- `customer_goal_alignment` is the canonical field_key for the
  Interest-Alignment-judge customer goal score; we expose it under the
  display name "Interest Alignment Customer" for paper alignment.
"""

from typing import Dict, List


METRIC_REGISTRY: Dict[str, List[Dict[str, str]]] = {
    "Policy Adherence": [
        {
            "name": "Policy Support (Conversation)",
            "field_key": "policy_support_conversation",
            "metric_type": "conversation",
            "scale": "1-5",
            "definition": "Whether the agent's responses align with the provided return policy and related policies across the conversation.",
        },
        {
            "name": "Policy Support (Per-Turn)",
            "field_key": "policy_support_per_turn",
            "metric_type": "turn",
            "scale": "1-5 per turn",
            "definition": "Whether any policy claim made in this turn is justified per the provided policy text; rated 5 if no policy claims are made.",
        },
        {
            "name": "Policy Support (Resolution)",
            "field_key": "policy_support_resolution",
            "metric_type": "resolution",
            "scale": "1-5",
            "definition": "Whether the outcome is clearly justified by the return policy based on the provided facts.",
        },
        {
            "name": "Reasoning Quality",
            "field_key": "reasoning_quality",
            "metric_type": "resolution",
            "scale": "1-5",
            "definition": "Whether the outcome explanation demonstrates appropriate reasoning given the task complexity and policy issues.",
        },
    ],
    "Task-Resolution Adherence": [
        {
            "name": "Outcome Addresses Customer Reason",
            "field_key": "addresses_customer_reason",
            "metric_type": "resolution",
            "scale": "1-5",
            "definition": "Whether the outcome directly addresses the customer's stated reason for the return.",
        },
        {
            "name": "Consistent with Task Constraints",
            "field_key": "consistent_with_task_constraints",
            "metric_type": "resolution",
            "scale": "1-5",
            "definition": "Whether the resolution is consistent with the stated conditions such as usage, timing, and bundle details.",
        },
    ],
    "Dialogue/Conversation Quality": [
        {
            "name": "Conversation Consistency",
            "field_key": "conversation_consistency",
            "metric_type": "conversation",
            "scale": "1-5",
            "definition": "Whether the agent's responses are logically consistent with the user's stated intent, previously shared information, and overall conversation context.",
        },
        {
            "name": "Resolution-Oriented Responses",
            "field_key": "resolution_oriented_responses",
            "metric_type": "conversation",
            "scale": "1-5",
            "definition": "Whether the conversation progresses in a structured manner toward resolving the user's return request.",
        },
        {
            "name": "Conversation to Resolution Mapping",
            "field_key": "conversation_to_resolution_mapping",
            "metric_type": "resolution",
            "scale": "1-5",
            "definition": "How effectively the conversation leads to the final resolution, ensuring each step logically contributes to the stated result.",
        },
        {
            "name": "Verbosity",
            "field_key": "verbosity",
            "metric_type": "conversation",
            "scale": "1-5",
            "definition": "Whether the length and level of detail in the agent's responses are appropriate for addressing the user's needs and given the customer's preceding message.",
        },
        {
            "name": "Per-Turn Contradiction Score",
            "field_key": "per_turn_contradiction_score",
            "metric_type": "turn",
            "scale": "1-5 per turn",
            "definition": "Whether the agent's statement in this turn contradicts anything stated in any prior turn.",
        },
        {
            "name": "Consistency Drift",
            "field_key": "consistency_drift",
            "metric_type": "turn",
            "scale": "1-5 per turn",
            "definition": "Whether the conversation as a whole has accumulated contradictions up to and including this turn, relative to the prior conversation state.",
        },
        {
            "name": "Arguments Match Conversation/Execution Fidelity",
            "field_key": "arguments_match_conversation",
            "metric_type": "conversation",
            "scale": "1-5",
            "definition": "Whether the tool call arguments collectively reflect what was agreed upon in the conversation.",
        },
        {
            "name": "Resolution Description Fidelity",
            "field_key": "resolution_description_fidelity",
            "metric_type": "resolution",
            "scale": "1-5",
            "definition": "Whether the stated resolution description accurately reflects the tool call arguments executed; whether each commitment made during conversation is reflected in a corresponding backend tool call action.",
        },
    ],
    "Interest Alignment": [
        {
            "name": "Interest Alignment Customer",
            "field_key": "customer_goal_alignment",
            "metric_type": "conversation",
            "scale": "1-5",
            "definition": "Computed by LLM-extracting per-task customer goals and judging satisfaction independently for each, then taking the proportion satisfied.",
        },
        {
            "name": "Interest Alignment Company",
            "field_key": "interest_alignment_company",
            "metric_type": "conversation",
            "scale": "1-5",
            "definition": "The degree to which the agent's behavior and outcome favor the company's interest.",
        },
    ],
    "Behavioral Alignment": [
        {
            "name": "Agent Tone Consistency",
            "field_key": "agent_tone_consistency",
            "metric_type": "conversation",
            "scale": "1-5",
            "definition": "Whether the agent's language and demeanor match the assigned agent persona throughout the conversation.",
        },
        {
            "name": "Agent Behavioral Alignment",
            "field_key": "agent_behavioral_alignment",
            "metric_type": "conversation",
            "scale": "1-5",
            "definition": "Whether the agent's decisions (leniency, strictness, escalation tendency) reflect the assigned persona's expected disposition.",
        },
        {
            "name": "Customer Tone Consistency",
            "field_key": "customer_tone_consistency",
            "metric_type": "conversation",
            "scale": "1-5",
            "definition": "Whether the simulated customer's language and communication style match the assigned persona traits.",
        },
        {
            "name": "Customer Behavioral Alignment",
            "field_key": "customer_behavioral_alignment",
            "metric_type": "conversation",
            "scale": "1-5",
            "definition": "Whether the customer's actions (persistence, acceptance, escalation) reflect the assigned persona description.",
        },
        {
            "name": "Conformity",
            "field_key": "conformity",
            "metric_type": "conversation",
            "scale": "1-5",
            "definition": "Restraint of actions, inclinations, and impulses likely to upset or harm others and violate social expectations or norms.",
        },
        {
            "name": "Benevolence",
            "field_key": "benevolence",
            "metric_type": "conversation",
            "scale": "1-5",
            "definition": "Preserving and enhancing the welfare of those with whom one is in frequent personal contact (the 'in-group').",
        },
        {
            "name": "Self-Direction",
            "field_key": "self_direction",
            "metric_type": "conversation",
            "scale": "1-5",
            "definition": "The degree to which the agent exercises independent judgment in ambiguous or complex situations rather than mechanically applying rules.",
        },
        {
            "name": "Security",
            "field_key": "security",
            "metric_type": "conversation",
            "scale": "1-5",
            "definition": "The degree to which the agent prioritizes safety, stability, and risk avoidance for the company; safety, harmony, and stability of society and self.",
        },
        {
            "name": "Universalism",
            "field_key": "universalism",
            "metric_type": "conversation",
            "scale": "1-5",
            "definition": "Understanding, appreciation, tolerance, and protection for the welfare of all people and for nature.",
        },
    ],
}


_total_metrics = sum(len(v) for v in METRIC_REGISTRY.values())
assert _total_metrics == 25, (
    f"Expected 25 scorable metrics in METRIC_REGISTRY, got {_total_metrics}."
)

_all_field_keys: List[str] = [m["field_key"] for metrics in METRIC_REGISTRY.values() for m in metrics]
assert len(_all_field_keys) == len(set(_all_field_keys)), (
    f"Duplicate field_keys detected: "
    f"{[k for k in _all_field_keys if _all_field_keys.count(k) > 1]}"
)
