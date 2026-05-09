"""
Dimension-oriented evaluation pipeline for customer service conversations.

Evaluates conversations across 5 dimensions:
  1. Policy Adherence
  2. Task-Resolution Adherence
  3. Dialogue/Conversation Quality
  4. Interest Alignment
  5. Behavioral Alignment

Each dimension is evaluated by one or more judges:
  - Policy & Resolution Judge      : Policy Adherence + Task-Resolution Adherence
  - Dialogue Quality Judge          : Dialogue/Conversation Quality
  - Behavioral Alignment Judge      : Behavioral Alignment
  - Interest Alignment Judge        : Interest Alignment (two-step extract+judge)
  - Turn-Level Judge                : Per-turn metrics for Agent and Customer

The metric registry is embedded in :mod:`metrics`.

CLI example:
    python src/evaluator_concise/evaluator_concise.py \\
      --conversations_path output/agent_run/<model>/with_resolution/conversations.jsonl \\
      --output_dir output/eval_results \\
      --model gpt-4o
"""

import argparse
import json
import math
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Resolve project root & shared utilities
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils import get_api_client, retry_with_exponential_backoff  # noqa: E402

from instruction_prompt_concise_calibrated import (  # noqa: E402
    eval_instructions_policy_resolution,
    eval_instructions_dialogue_quality,
    eval_instructions_behavioral_alignment,
    eval_instructions_turn_level_concise,
    eval_instructions_interest_alignment_extraction,
    eval_instructions_interest_alignment_judgment,
)

from metrics import METRIC_REGISTRY  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_MODEL = "gpt-4o"


# ---------------------------------------------------------------------------
# Shared LLM helpers (copied from evaluator.py)
# ---------------------------------------------------------------------------

def _format_conversation(turns: List[Dict[str, Any]]) -> str:
    """Format conversation turns into readable text for the judge."""
    lines = []
    for turn in turns:
        role = turn.get("turn", "unknown").capitalize()
        message = turn.get("message", "")
        if message:
            lines.append(f"[{role}]: {message}")
        tool_calls = turn.get("tool_calls")
        if tool_calls:
            lines.append(f"[{role} Tool Call]: {json.dumps(tool_calls)}")
        tool_result = turn.get("tool_result")
        if tool_result:
            lines.append(f"[Tool Result]: {json.dumps(tool_result)}")
    return "\n\n".join(lines)


def _format_conversation_with_indices(turns: List[Dict[str, Any]]) -> str:
    """Format conversation turns with 0-based indices and speaker labels."""
    lines = []
    for i, turn in enumerate(turns):
        role = turn.get("turn", "unknown")
        speaker = "Agent" if role in ("agent", "assistant") else "Customer"
        message = turn.get("message", "")
        if message:
            lines.append(f"[Turn {i}] [{speaker}]: {message}")
        tool_calls = turn.get("tool_calls")
        if tool_calls:
            lines.append(f"[Turn {i}] [Agent Tool Call]: {json.dumps(tool_calls)}")
        tool_result = turn.get("tool_result")
        if tool_result:
            lines.append(f"[Turn {i}] [Tool Result]: {json.dumps(tool_result)}")
    return "\n\n".join(lines)


def _extract_persona_text(persona_data: Any) -> str:
    """Convert persona data (dict or str) to a readable string."""
    if isinstance(persona_data, dict):
        return json.dumps(persona_data, indent=2)
    elif isinstance(persona_data, str):
        return persona_data
    return ""


def _extract_policy_and_task(
    transcript: Dict[str, Any],
) -> Tuple[str, str, str]:
    """Extract primary policy, related policies text, and task description."""
    scenario = transcript.get("source_scenario", {})
    policy = scenario.get("Policy", {})
    task = scenario.get("task", {})

    primary_policy = policy.get("Primary Policy", {}).get("text", "")
    related_policies_list = policy.get("Related policies", [])
    if isinstance(related_policies_list, list):
        related_policies_text = "\n\n".join(
            rp.get("text", "") for rp in related_policies_list if isinstance(rp, dict)
        )
    elif isinstance(related_policies_list, dict):
        related_policies_text = related_policies_list.get("text", "")
    else:
        related_policies_text = str(related_policies_list)

    task_description = task.get("detail", task.get("task", ""))
    return primary_policy, related_policies_text, task_description


@retry_with_exponential_backoff
def _call_with_retry(
    client: Any,
    model: str,
    prompt: str,
    temperature: float,
    system_prompt: str = "You are an expert evaluator. Respond only with valid JSON.",
    return_logprobs: bool = False,
) -> Any:
    """Send prompt to LLM. Returns str normally, or (str, logprobs_content) when return_logprobs=True."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    kwargs: Dict[str, Any] = {"model": model, "messages": messages, "temperature": temperature}
    if return_logprobs:
        kwargs["logprobs"] = True
        kwargs["top_logprobs"] = 5
    if model.startswith("gpt-5"):
        kwargs["max_completion_tokens"] = 4096
    else:
        kwargs["max_tokens"] = 4096
    response = client.chat.completions.create(**kwargs)
    content = response.choices[0].message.content
    if return_logprobs:
        return content, response.choices[0].logprobs.content
    return content


def _extract_score_logprobs(
    response_text: str,
    logprobs_content: Any,
    target_fields: List[str],
) -> Dict[str, Optional[float]]:
    """Extract probability-weighted G-Eval scores for target JSON fields.

    For each field, finds the score digit token in the logprob sequence and returns
    the expected value E[score] = Σ p(i) * i over score tokens {1..5}.
    Returns None for any field where the digit token is not found or has no score probs.
    """
    if not logprobs_content:
        return {f: None for f in target_fields}

    # Build token list with cumulative character offsets
    tokens_with_offsets: List[Tuple[int, int, Any]] = []
    offset = 0
    for tok in logprobs_content:
        tokens_with_offsets.append((offset, len(tok.token), tok.top_logprobs))
        offset += len(tok.token)

    results: Dict[str, Optional[float]] = {}
    for field_name in target_fields:
        pattern = rf'"{re.escape(field_name)}"\s*:\s*(\d)'
        match = re.search(pattern, response_text)
        if not match:
            results[field_name] = None
            continue

        digit_pos = match.start(1)

        score_probs: Dict[int, float] = {}
        for tok_offset, tok_len, top_logprobs in tokens_with_offsets:
            if tok_offset <= digit_pos < tok_offset + tok_len:
                for lp in top_logprobs:
                    stripped = lp.token.strip()
                    if stripped in {"1", "2", "3", "4", "5"}:
                        score_probs[int(stripped)] = math.exp(lp.logprob)
                break

        if not score_probs:
            results[field_name] = None
            continue

        total = sum(score_probs.values())
        if total < 1e-12:
            results[field_name] = None
            continue

        results[field_name] = sum(k * v for k, v in score_probs.items()) / total

    return results


def _parse_judge_response(response_text: str) -> Any:
    """Parse JSON (array or object) returned by the LLM judge."""
    text = response_text.strip()

    # G-Eval format: extract only the content after ### OUTPUT: marker
    if "### OUTPUT:" in text:
        text = text.split("### OUTPUT:", 1)[1].strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3].strip()

    def _try_loads(s: str) -> Any:
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            pass
        # Pass 2: replace actual bare control chars (e.g. real newlines inside string values)
        sanitized = re.sub(r'(?<!\\)[\n\r\t]', ' ', s)
        try:
            return json.loads(sanitized)
        except json.JSONDecodeError:
            pass
        # Pass 3: model emitted literal escape sequences (\n, \", etc.) as two-char strings
        # (observed with gpt-5.x — replace literal backslash-escapes before retrying)
        unescaped = (s
                     .replace('\\"', '"')
                     .replace('\\n', ' ')
                     .replace('\\r', ' ')
                     .replace('\\t', ' '))
        if unescaped != s:
            try:
                return json.loads(unescaped.strip())
            except json.JSONDecodeError:
                pass
        return None

    result = _try_loads(text)
    if result is not None:
        return result

    for opener, closer in (("[", "]"), ("{", "}")):
        start = text.find(opener)
        end = text.rfind(closer)
        if start != -1 and end != -1:
            result = _try_loads(text[start: end + 1])
            if result is not None:
                return result

    print(f"[EvaluatorConcise] WARNING: Could not parse judge response:\n{text[:500]}")
    return None


# ---------------------------------------------------------------------------
# Format validation (copied from evaluator.py)
# ---------------------------------------------------------------------------

def _validate_transcript_format(transcript: Dict[str, Any]) -> Dict[str, Any]:
    issues = []
    if "scenario_id" not in transcript:
        issues.append("missing scenario_id")
    if "source_scenario" not in transcript:
        issues.append("missing source_scenario")
    has_conv = any(
        k.startswith("conversation_") and k != "conversation_type"
        for k in transcript
    )
    if not has_conv:
        issues.append("no conversation_N keys found")

    conv_idx = 1
    while f"conversation_{conv_idx}" in transcript:
        conv = transcript[f"conversation_{conv_idx}"]
        if not isinstance(conv, list):
            issues.append(f"conversation_{conv_idx} is not a list")
        elif conv:
            for j, turn in enumerate(conv):
                if not isinstance(turn, dict):
                    issues.append(f"conversation_{conv_idx}[{j}] is not a dict")
                elif "turn" not in turn or "message" not in turn:
                    issues.append(f"conversation_{conv_idx}[{j}] missing turn/message")
        res_key = f"resolution_{conv_idx}"
        res = transcript.get(res_key)
        if res is None:
            issues.append(f"missing {res_key}")
        elif isinstance(res, dict):
            if "resolution_type" not in res:
                issues.append(f"{res_key} missing resolution_type")
        else:
            issues.append(f"{res_key} is not a dict (type: {type(res).__name__})")
        conv_idx += 1

    ss = transcript.get("source_scenario", {})
    if isinstance(ss, dict):
        if "task" not in ss:
            issues.append("source_scenario missing task")
        if "Policy" not in ss:
            issues.append("source_scenario missing Policy")

    return {"valid": len(issues) == 0, "issues": issues}


def _reformat_transcript(transcript: Dict[str, Any]) -> Dict[str, Any]:
    conv_idx = 1
    while f"conversation_{conv_idx}" in transcript:
        res_key = f"resolution_{conv_idx}"
        res = transcript.get(res_key)
        if isinstance(res, str):
            try:
                transcript[res_key] = json.loads(res)
            except (json.JSONDecodeError, TypeError):
                transcript[res_key] = {
                    "resolution_type": "UNKNOWN",
                    "resolution_description": res,
                }
        elif res is None:
            transcript[res_key] = {
                "resolution_type": "UNKNOWN",
                "resolution_description": "",
            }
        conv_idx += 1
    return transcript


def _load_conversations(path: str) -> List[Dict[str, Any]]:
    transcripts = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                transcripts.append(json.loads(line))
    return transcripts


# ---------------------------------------------------------------------------
# Judge 1: Policy & Resolution Judge
# ---------------------------------------------------------------------------

def call_policy_resolution_judge(
    client: Any,
    transcript: Dict[str, Any],
    conv_idx: int,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.0,
) -> Dict[str, Any]:
    """Evaluate Policy Adherence (conv+res), Task-Resolution Adherence, and Task sub-metrics."""
    primary_policy, related_policies_text, task_description = _extract_policy_and_task(transcript)
    conversation_turns = transcript.get(f"conversation_{conv_idx}", [])
    resolution = transcript.get(f"resolution_{conv_idx}", {})

    if isinstance(resolution, dict):
        outcome_desc = resolution.get("resolution_description", json.dumps(resolution))
    else:
        outcome_desc = str(resolution) if resolution else ""

    conv_text = _format_conversation(conversation_turns)

    prompt = eval_instructions_policy_resolution.format(
        return_policy=primary_policy,
        related_policies=related_policies_text,
        task_description=task_description,
        outcome_description=outcome_desc,
        conversation=conv_text,
    )

    result = _call_with_retry(
        client, model, prompt, temperature,
        system_prompt="You are an expert evaluator.",
        return_logprobs=True,
    )
    if isinstance(result, tuple):
        response, logprobs_content = result
    else:
        response, logprobs_content = result, None

    _GEVAL_FIELDS = ["policy_support_conversation", "policy_support_resolution", "reasoning_quality"]
    geval = _extract_score_logprobs(response, logprobs_content, _GEVAL_FIELDS)

    parsed = _parse_judge_response(response)
    if isinstance(parsed, list) and parsed:
        parsed = parsed[0]
    if not isinstance(parsed, dict):
        print(f"[EvaluatorConcise] WARNING: Policy/resolution judge parse error for conv_{conv_idx}")
        return {
            "parse_error": True,
            "policy_support_conversation": None,
            "policy_support_resolution": None,
            "reasoning_quality": None,
            "addresses_customer_reason": None,
            "consistent_with_task_constraints": None,
            "task_realism": None,
            "policy_issue_relevance": None,
            "complexity_suitability": None,
            "justification": "Parse error",
            "geval_scores": {"cc4": None, "oe3": None, "oe5": None},
        }

    _warn_missing_keys(
        parsed,
        ["policy_support_conversation", "policy_support_resolution", "reasoning_quality",
         "addresses_customer_reason", "consistent_with_task_constraints",
         "task_realism", "policy_issue_relevance", "complexity_suitability"],
        judge="policy_resolution",
        conv_idx=conv_idx,
    )

    return {
        "policy_support_conversation": parsed.get("policy_support_conversation"),
        "policy_support_resolution": parsed.get("policy_support_resolution"),
        "reasoning_quality": parsed.get("reasoning_quality"),
        "addresses_customer_reason": parsed.get("addresses_customer_reason"),
        "consistent_with_task_constraints": parsed.get("consistent_with_task_constraints"),
        "task_realism": parsed.get("task_realism"),
        "policy_issue_relevance": parsed.get("policy_issue_relevance"),
        "complexity_suitability": parsed.get("complexity_suitability"),
        "resolution_type": parsed.get("resolution_type"),
        "justification": parsed.get("justification", ""),
        "geval_scores": {
            "cc4": geval.get("policy_support_conversation"),
            "oe3": geval.get("policy_support_resolution"),
            "oe5": geval.get("reasoning_quality"),
        },
    }


# ---------------------------------------------------------------------------
# Judge 2: Dialogue Quality Judge
# ---------------------------------------------------------------------------

def call_dialogue_quality_judge(
    client: Any,
    transcript: Dict[str, Any],
    conv_idx: int,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.0,
) -> Dict[str, Any]:
    """Evaluate Dialogue Quality sub-metrics (conversation and resolution levels)."""
    primary_policy, _, task_description = _extract_policy_and_task(transcript)
    conversation_turns = transcript.get(f"conversation_{conv_idx}", [])
    resolution = transcript.get(f"resolution_{conv_idx}", {})

    if isinstance(resolution, dict):
        outcome_desc = resolution.get("resolution_description", json.dumps(resolution))
    else:
        outcome_desc = str(resolution) if resolution else ""

    conv_text = _format_conversation(conversation_turns)

    prompt = eval_instructions_dialogue_quality.format(
        return_policy=primary_policy,
        task_description=task_description,
        outcome_description=outcome_desc,
        conversation=conv_text,
    )

    response = _call_with_retry(
        client, model, prompt, temperature,
        system_prompt="You are an expert evaluator. Respond only with valid JSON.",
    )
    parsed = _parse_judge_response(response)
    if isinstance(parsed, list) and parsed:
        parsed = parsed[0]
    if not isinstance(parsed, dict):
        print(f"[EvaluatorConcise] WARNING: Dialogue quality judge parse error for conv_{conv_idx}")
        return {
            "parse_error": True,
            "conversation_consistency": None,
            "resolution_oriented_responses": None,
            "conversation_to_resolution_mapping": None,
            "verbosity": None,
            "arguments_match_conversation": None,
            "resolution_description_fidelity": None,
            "justification": "Parse error",
        }

    _warn_missing_keys(
        parsed,
        ["conversation_consistency", "resolution_oriented_responses",
         "conversation_to_resolution_mapping", "verbosity",
         "arguments_match_conversation", "resolution_description_fidelity"],
        judge="dialogue_quality",
        conv_idx=conv_idx,
    )

    return {
        "conversation_consistency": parsed.get("conversation_consistency"),
        "resolution_oriented_responses": parsed.get("resolution_oriented_responses"),
        "conversation_to_resolution_mapping": parsed.get("conversation_to_resolution_mapping"),
        "verbosity": parsed.get("verbosity"),
        "arguments_match_conversation": parsed.get("arguments_match_conversation"),
        "resolution_description_fidelity": parsed.get("resolution_description_fidelity"),
        "justification": parsed.get("justification", ""),
    }


# ---------------------------------------------------------------------------
# Judge 3: Behavioral Alignment Judge
# ---------------------------------------------------------------------------

def call_behavioral_alignment_judge(
    client: Any,
    transcript: Dict[str, Any],
    conv_idx: int,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.0,
) -> Dict[str, Any]:
    """Evaluate Behavioral Alignment sub-metrics (persona, interest, Schwartz)."""
    conversation_turns = transcript.get(f"conversation_{conv_idx}", [])
    resolution = transcript.get(f"resolution_{conv_idx}", {})

    if isinstance(resolution, dict):
        outcome_desc = resolution.get("resolution_description", json.dumps(resolution))
    else:
        outcome_desc = str(resolution) if resolution else ""

    scenario = transcript.get("source_scenario", {})
    persona_data = scenario.get("persona", {})
    if isinstance(persona_data, dict):
        agent_persona = _extract_persona_text(persona_data.get("agent_persona", persona_data))
        customer_persona = _extract_persona_text(persona_data.get("customer_persona", persona_data))
    else:
        agent_persona = _extract_persona_text(persona_data)
        customer_persona = _extract_persona_text(persona_data)

    conv_text = _format_conversation(conversation_turns)

    prompt = eval_instructions_behavioral_alignment.format(
        agent_persona=agent_persona or "No agent persona provided.",
        customer_persona=customer_persona or "No customer persona provided.",
        outcome_description=outcome_desc,
        conversation=conv_text,
    )

    response = _call_with_retry(
        client, model, prompt, temperature,
        system_prompt="You are an expert evaluator. Respond only with valid JSON.",
    )
    parsed = _parse_judge_response(response)
    if isinstance(parsed, list) and parsed:
        parsed = parsed[0]
    if not isinstance(parsed, dict):
        print(f"[EvaluatorConcise] WARNING: Behavioral alignment judge parse error for conv_{conv_idx}")
        return {
            "parse_error": True,
            **{k: None for k in [
                "agent_tone_consistency", "agent_behavioral_alignment",
                "customer_tone_consistency", "customer_behavioral_alignment",
                "interest_alignment_company",
                "conformity", "benevolence", "self_direction", "security", "universalism",
            ]},
            "justification": "Parse error",
        }

    _warn_missing_keys(
        parsed,
        ["agent_tone_consistency", "agent_behavioral_alignment",
         "customer_tone_consistency", "customer_behavioral_alignment",
         "interest_alignment_company",
         "conformity", "benevolence", "self_direction", "security", "universalism"],
        judge="behavioral_alignment",
        conv_idx=conv_idx,
    )

    return {
        "agent_tone_consistency": parsed.get("agent_tone_consistency"),
        "agent_behavioral_alignment": parsed.get("agent_behavioral_alignment"),
        "customer_tone_consistency": parsed.get("customer_tone_consistency"),
        "customer_behavioral_alignment": parsed.get("customer_behavioral_alignment"),
        "interest_alignment_company": parsed.get("interest_alignment_company"),
        "conformity": parsed.get("conformity"),
        "benevolence": parsed.get("benevolence"),
        "self_direction": parsed.get("self_direction"),
        "security": parsed.get("security"),
        "universalism": parsed.get("universalism"),
        "justification": parsed.get("justification", ""),
    }


# ---------------------------------------------------------------------------
# Judge 4: Turn-Level Judge (Agent and Customer separately)
# ---------------------------------------------------------------------------

def call_turn_level_judge_concise(
    client: Any,
    transcript: Dict[str, Any],
    conv_idx: int,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.0,
) -> Dict[str, Any]:
    """Evaluate per-turn metrics separately for agent and customer turns.

    Returns:
        {
            "agent_turns": [{"turn_index": int, "policy_support_per_turn": int,
                             "per_turn_contradiction_score": int,
                             "consistency_drift": int, "justification": str}, ...],
            "customer_turns": [{"turn_index": int,
                                "per_turn_contradiction_score": int,
                                "consistency_drift": int, "justification": str}, ...]
        }
    """
    primary_policy, _, _ = _extract_policy_and_task(transcript)
    conversation_turns = transcript.get(f"conversation_{conv_idx}", [])
    conv_indexed = _format_conversation_with_indices(conversation_turns)

    prompt = eval_instructions_turn_level_concise.format(
        return_policy=primary_policy,
        conversation_with_indices=conv_indexed,
    )

    response = _call_with_retry(
        client, model, prompt, temperature,
        system_prompt="You are an expert evaluator. Respond only with valid JSON.",
    )
    parsed = _parse_judge_response(response)

    empty = {"agent_turns": [], "customer_turns": []}

    if not isinstance(parsed, dict):
        print(f"[EvaluatorConcise] WARNING: Turn-level judge parse error for conv_{conv_idx}")
        return empty

    agent_turns = parsed.get("agent_turns", [])
    customer_turns = parsed.get("customer_turns", [])

    if not isinstance(agent_turns, list) or not isinstance(customer_turns, list):
        print(f"[EvaluatorConcise] WARNING: Turn-level judge returned malformed arrays for conv_{conv_idx}")
        return empty

    return {"agent_turns": agent_turns, "customer_turns": customer_turns}


# ---------------------------------------------------------------------------
# Judge 5: Interest Alignment Judge (two-step: extract goals, then score)
# ---------------------------------------------------------------------------

def _ia_parse_error_stub(reason: str) -> Dict[str, Any]:
    return {
        "parse_error": True,
        "explicit_goals": [],
        "implicit_goals": [],
        "extraction_rationale": "",
        "goal_evaluations": [],
        "goals_addressed_count": None,
        "total_goals": None,
        "completion_ratio": None,
        "customer_goal_alignment": None,
        "justification": f"Parse error: {reason}",
    }


def call_interest_alignment_judge(
    client: Any,
    transcript: Dict[str, Any],
    conv_idx: int,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.0,
) -> Dict[str, Any]:
    """Two-step judge: (1) extract customer goals from task, (2) score how many were achieved."""
    scenario = transcript.get("source_scenario", {}) or {}
    task_obj = scenario.get("task", {}) or {}
    task_description = task_obj.get("detail", task_obj.get("task", ""))
    customer_reason = task_obj.get("reason", "")
    related_issues = task_obj.get("related_policy_issues", [])

    # --- Step 1: extract goals ---
    extraction_prompt = eval_instructions_interest_alignment_extraction.format(
        task_description=task_description or "(no task description provided)",
        customer_reason=customer_reason or "(no customer reason provided)",
        related_policy_issues=json.dumps(related_issues, indent=2) if related_issues else "(none)",
    )
    extraction_response = _call_with_retry(
        client, model, extraction_prompt, temperature,
        system_prompt="You are an expert evaluator. Respond only with valid JSON.",
    )
    extraction = _parse_judge_response(extraction_response)
    if not isinstance(extraction, dict):
        print(f"[EvaluatorConcise] WARNING: Interest alignment extraction parse error for conv_{conv_idx}")
        return _ia_parse_error_stub("extraction step failed to parse")

    explicit_goals = extraction.get("explicit_goals") or []
    implicit_goals = extraction.get("implicit_goals") or []
    if not isinstance(explicit_goals, list) or not isinstance(implicit_goals, list):
        return _ia_parse_error_stub("goal lists were not arrays")
    if not (explicit_goals or implicit_goals):
        return _ia_parse_error_stub("no goals extracted")

    # --- Step 2: judge achievement ---
    conversation_turns = transcript.get(f"conversation_{conv_idx}", [])
    resolution = transcript.get(f"resolution_{conv_idx}", {})
    if isinstance(resolution, dict):
        outcome_desc = resolution.get("resolution_description", json.dumps(resolution))
    else:
        outcome_desc = str(resolution) if resolution else ""
    conv_text = _format_conversation(conversation_turns)

    judgment_prompt = eval_instructions_interest_alignment_judgment.format(
        explicit_goals=json.dumps(explicit_goals, indent=2),
        implicit_goals=json.dumps(implicit_goals, indent=2),
        conversation=conv_text,
        outcome_description=outcome_desc,
    )
    judgment_response = _call_with_retry(
        client, model, judgment_prompt, temperature,
        system_prompt="You are an expert evaluator. Respond only with valid JSON.",
    )
    judgment = _parse_judge_response(judgment_response)
    if not isinstance(judgment, dict):
        print(f"[EvaluatorConcise] WARNING: Interest alignment judgment parse error for conv_{conv_idx}")
        # Preserve the goal extraction so it remains available for inspection
        stub = _ia_parse_error_stub("judgment step failed to parse")
        stub["explicit_goals"] = explicit_goals
        stub["implicit_goals"] = implicit_goals
        stub["extraction_rationale"] = extraction.get("extraction_rationale", "")
        return stub

    _warn_missing_keys(
        judgment,
        ["goal_evaluations", "goals_addressed_count", "total_goals",
         "completion_ratio", "customer_goal_alignment"],
        judge="interest_alignment",
        conv_idx=conv_idx,
    )

    return {
        "explicit_goals": explicit_goals,
        "implicit_goals": implicit_goals,
        "extraction_rationale": extraction.get("extraction_rationale", ""),
        "goal_evaluations": judgment.get("goal_evaluations", []),
        "goals_addressed_count": judgment.get("goals_addressed_count"),
        "total_goals": judgment.get("total_goals"),
        "completion_ratio": judgment.get("completion_ratio"),
        "customer_goal_alignment": judgment.get("customer_goal_alignment"),
        "justification": judgment.get("justification", ""),
    }


# ---------------------------------------------------------------------------
# Helper: warn on missing expected keys
# ---------------------------------------------------------------------------

def _warn_missing_keys(
    parsed: Dict[str, Any],
    expected_keys: List[str],
    judge: str,
    conv_idx: int,
) -> None:
    missing = [k for k in expected_keys if k not in parsed]
    if missing:
        print(
            f"[EvaluatorConcise] WARNING: {judge} judge missing keys for conv_{conv_idx}: {missing}"
        )


# ---------------------------------------------------------------------------
# Second-level aggregation
# ---------------------------------------------------------------------------

def _valid_score(value: Any) -> Optional[float]:
    """Return float if value is a valid 1-5 integer, else None."""
    if isinstance(value, (int, float)) and 1 <= value <= 5:
        return float(value)
    return None


def _compute_dimension_score(
    dimension_name: str,
    sub_metrics: Dict[str, Any],
    metric_registry: Dict[str, List[Dict[str, str]]],
) -> Optional[float]:
    """Compute the dimension overall_score as the mean of per-sub-metric means.

    Each sub-metric contributes one equally weighted term:
    - scalar int in [1,5]         → float(value)
    - list of ints                → mean of valid entries
    - dict {"agent": [...], "customer": [...]} → mean of all valid entries across both roles

    Returns None if no valid scores are available.
    """
    metrics = metric_registry.get(dimension_name, [])
    per_metric_means: List[float] = []

    for m in metrics:
        fk = m["field_key"]
        value = sub_metrics.get(fk)
        if value is None:
            continue

        # Scalar
        s = _valid_score(value)
        if s is not None:
            per_metric_means.append(s)
            continue

        # List (non-role-split turn arrays — shouldn't normally occur, but handle defensively)
        if isinstance(value, list):
            valid = [_valid_score(v) for v in value]
            valid = [v for v in valid if v is not None]
            if valid:
                per_metric_means.append(mean(valid))
            continue

        # Role-split dict {"agent": [...], "customer": [...]}
        if isinstance(value, dict):
            all_values: List[float] = []
            for role_scores in value.values():
                if isinstance(role_scores, list):
                    all_values.extend(
                        v for v in (_valid_score(x) for x in role_scores) if v is not None
                    )
            if all_values:
                per_metric_means.append(mean(all_values))

    if not per_metric_means:
        return None
    return mean(per_metric_means)


# ---------------------------------------------------------------------------
# Assemble variant result
# ---------------------------------------------------------------------------

def _assemble_variant_result(
    conv_idx: int,
    pr_result: Dict[str, Any],
    dq_result: Dict[str, Any],
    ba_result: Dict[str, Any],
    tl_result: Dict[str, Any],
    ia_result: Dict[str, Any],
    metric_registry: Dict[str, List[Dict[str, str]]],
    skip_turn_level: bool,
) -> Dict[str, Any]:
    """Merge judge outputs into a structured variant result with dimension_scores."""

    # Extract turn-level arrays split by role
    agent_turns = tl_result.get("agent_turns", []) if not skip_turn_level else []
    customer_turns = tl_result.get("customer_turns", []) if not skip_turn_level else []

    def _collect_turn_field(turns: List[Dict], field: str) -> List:
        return [t[field] for t in turns if isinstance(t, dict) and field in t]

    # Role-split turn fields
    policy_support_per_turn_agent = _collect_turn_field(agent_turns, "policy_support_per_turn")
    contradiction_agent = _collect_turn_field(agent_turns, "per_turn_contradiction_score")
    drift_agent = _collect_turn_field(agent_turns, "consistency_drift")
    contradiction_customer = _collect_turn_field(customer_turns, "per_turn_contradiction_score")
    drift_customer = _collect_turn_field(customer_turns, "consistency_drift")

    # Build sub_metrics per dimension
    policy_adherence_sub: Dict[str, Any] = {
        "policy_support_conversation": pr_result.get("policy_support_conversation"),
        "policy_support_resolution": pr_result.get("policy_support_resolution"),
        "reasoning_quality": pr_result.get("reasoning_quality"),
    }
    if not skip_turn_level:
        policy_adherence_sub["policy_support_per_turn"] = {"agent": policy_support_per_turn_agent}

    dialogue_quality_sub: Dict[str, Any] = {
        "conversation_consistency": dq_result.get("conversation_consistency"),
        "resolution_oriented_responses": dq_result.get("resolution_oriented_responses"),
        "conversation_to_resolution_mapping": dq_result.get("conversation_to_resolution_mapping"),
        "verbosity": dq_result.get("verbosity"),
        "arguments_match_conversation": dq_result.get("arguments_match_conversation"),
        "resolution_description_fidelity": dq_result.get("resolution_description_fidelity"),
    }
    if not skip_turn_level:
        dialogue_quality_sub["per_turn_contradiction_score"] = {
            "agent": contradiction_agent,
            "customer": contradiction_customer,
        }
        dialogue_quality_sub["consistency_drift"] = {
            "agent": drift_agent,
            "customer": drift_customer,
        }

    behavioral_alignment_sub: Dict[str, Any] = {
        "agent_tone_consistency": ba_result.get("agent_tone_consistency"),
        "agent_behavioral_alignment": ba_result.get("agent_behavioral_alignment"),
        "customer_tone_consistency": ba_result.get("customer_tone_consistency"),
        "customer_behavioral_alignment": ba_result.get("customer_behavioral_alignment"),
        "conformity": ba_result.get("conformity"),
        "benevolence": ba_result.get("benevolence"),
        "self_direction": ba_result.get("self_direction"),
        "security": ba_result.get("security"),
        "universalism": ba_result.get("universalism"),
    }

    interest_alignment_sub: Dict[str, Any] = {
        "customer_goal_alignment": ia_result.get("customer_goal_alignment"),
        "interest_alignment_company": ba_result.get("interest_alignment_company"),
    }

    task_resolution_adherence_sub: Dict[str, Any] = {
        "addresses_customer_reason": pr_result.get("addresses_customer_reason"),
        "consistent_with_task_constraints": pr_result.get("consistent_with_task_constraints"),
    }

    sub_metrics_by_dimension: Dict[str, Dict[str, Any]] = {
        "Policy Adherence": policy_adherence_sub,
        "Dialogue/Conversation Quality": dialogue_quality_sub,
        "Behavioral Alignment": behavioral_alignment_sub,
        "Interest Alignment": interest_alignment_sub,
        "Task-Resolution Adherence": task_resolution_adherence_sub,
    }

    # Compute overall_score for each dimension
    dimension_scores: Dict[str, Any] = {}
    for dim_name, sub_metrics in sub_metrics_by_dimension.items():
        overall = _compute_dimension_score(dim_name, sub_metrics, metric_registry)
        dimension_scores[dim_name] = {
            "sub_metrics": sub_metrics,
            "overall_score": overall,
        }

    return {
        "conversation_idx": conv_idx,
        "dimension_scores": dimension_scores,
        "raw_judge_outputs": {
            "policy_resolution": pr_result,
            "dialogue_quality": dq_result,
            "behavioral_alignment": ba_result,
            "turn_level": tl_result,
            "interest_alignment": ia_result,
        },
    }


# ---------------------------------------------------------------------------
# Summary computation
# ---------------------------------------------------------------------------

def _compute_concise_summary(
    evaluations: List[Dict[str, Any]],
    model: str,
    metric_registry: Dict[str, List[Dict[str, str]]],
    **config_kwargs: Any,
) -> Dict[str, Any]:
    """Aggregate per-variant scores into a summary organized by dimension."""

    total_transcripts = len(evaluations)
    total_variants = sum(len(e.get("variants", [])) for e in evaluations)

    # Accumulators
    # overall_scores[dim_name] = list of floats
    overall_scores: Dict[str, List[float]] = {dim: [] for dim in metric_registry}

    # sub_metric_scalars[dim][field_key] = list of floats (for scalar/list sub-metrics)
    sub_metric_scalars: Dict[str, Dict[str, List[float]]] = {
        dim: {} for dim in metric_registry
    }
    # sub_metric_by_role[dim][field_key]["agent"] and ["customer"] = list of floats
    sub_metric_by_role: Dict[str, Dict[str, Dict[str, List[float]]]] = {
        dim: {} for dim in metric_registry
    }

    for eval_entry in evaluations:
        for variant in eval_entry.get("variants", []):
            dim_scores = variant.get("dimension_scores", {})
            for dim_name, metrics in metric_registry.items():
                dim_data = dim_scores.get(dim_name, {})
                sub_metrics = dim_data.get("sub_metrics", {})

                # Overall score
                overall = dim_data.get("overall_score")
                if isinstance(overall, (int, float)):
                    overall_scores[dim_name].append(float(overall))

                # Sub-metric values
                for m in metrics:
                    fk = m["field_key"]
                    value = sub_metrics.get(fk)
                    if value is None:
                        continue

                    # Scalar
                    if isinstance(value, (int, float)) and 1 <= value <= 5:
                        sub_metric_scalars[dim_name].setdefault(fk, []).append(float(value))
                        continue

                    # Role-split dict
                    if isinstance(value, dict):
                        if fk not in sub_metric_by_role[dim_name]:
                            sub_metric_by_role[dim_name][fk] = {}
                        for role, role_scores in value.items():
                            if isinstance(role_scores, list):
                                valid = [float(v) for v in role_scores if isinstance(v, (int, float)) and 1 <= v <= 5]
                                sub_metric_by_role[dim_name][fk].setdefault(role, []).extend(valid)

    # Build summary stats per dimension
    dimensions_summary: Dict[str, Any] = {}
    for dim_name in metric_registry:
        # Overall score stats
        os_vals = overall_scores[dim_name]
        overall_stat: Dict[str, Any] = {"count": len(os_vals)}
        if os_vals:
            overall_stat["mean"] = mean(os_vals)
            overall_stat["std"] = stdev(os_vals) if len(os_vals) > 1 else 0.0
        else:
            overall_stat["mean"] = None
            overall_stat["std"] = None

        # Sub-metric stats
        sub_stats: Dict[str, Any] = {}
        for m in metric_registry[dim_name]:
            fk = m["field_key"]
            # Scalar stats
            if fk in sub_metric_scalars[dim_name]:
                vals = sub_metric_scalars[dim_name][fk]
                sub_stats[fk] = {
                    "mean": mean(vals) if vals else None,
                    "std": stdev(vals) if len(vals) > 1 else 0.0,
                    "count": len(vals),
                }
            # Role-split stats
            elif fk in sub_metric_by_role[dim_name]:
                role_data = sub_metric_by_role[dim_name][fk]
                sub_stats[fk] = {}
                for role, vals in role_data.items():
                    sub_stats[fk][role] = {
                        "mean": mean(vals) if vals else None,
                        "std": stdev(vals) if len(vals) > 1 else 0.0,
                        "count": len(vals),
                    }

        dimensions_summary[dim_name] = {
            "overall_score": overall_stat,
            "sub_metric_stats": sub_stats,
        }

    return {
        "model": model,
        "num_transcripts": total_transcripts,
        "num_variants_total": total_variants,
        "dimensions": dimensions_summary,
        "evaluation_config": config_kwargs,
    }


def _format_concise_summary(summary: Dict[str, Any]) -> str:
    """Format the summary dict as a human-readable report."""
    lines: List[str] = []
    lines.append("DIMENSION-ORIENTED EVALUATION REPORT")
    lines.append("=" * 60)

    cfg = summary.get("evaluation_config", {})
    lines.append(f"  Model Under Test:         {summary.get('model', 'N/A')}")
    lines.append(f"  Conversations Path:       {cfg.get('conversations_path', 'N/A')}")
    lines.append(f"  Timestamp:                {cfg.get('timestamp', 'N/A')}")
    lines.append(f"  Skip Turn-Level:          {cfg.get('skip_turn_level', False)}")
    lines.append(f"  Transcripts Evaluated:    {summary['num_transcripts']}")
    lines.append(f"  Total Variants Evaluated: {summary['num_variants_total']}")
    lines.append("")

    for dim_name, dim_data in summary.get("dimensions", {}).items():
        lines.append(f"  {dim_name.upper()}")
        lines.append("  " + "\u2500" * 56)

        os_stat = dim_data.get("overall_score", {})
        os_mean = os_stat.get("mean")
        os_std = os_stat.get("std", 0.0)
        os_n = os_stat.get("count", 0)
        if os_mean is not None:
            lines.append(f"    {'Overall Score (second-level)':<42} {os_mean:.2f} \u00b1 {os_std:.2f}  (n={os_n})")
        else:
            lines.append(f"    {'Overall Score (second-level)':<42} N/A")

        for fk, stats in dim_data.get("sub_metric_stats", {}).items():
            label = fk.replace("_", " ").title()
            if isinstance(stats, dict) and ("mean" in stats or "agent" in stats or "customer" in stats):
                # Scalar stat
                if "mean" in stats:
                    m = stats.get("mean")
                    s = stats.get("std", 0.0)
                    n = stats.get("count", 0)
                    if m is not None:
                        lines.append(f"      {label:<40} {m:.2f} \u00b1 {s:.2f}  (n={n})")
                else:
                    # Role-split
                    lines.append(f"      {label}:")
                    for role, role_stats in stats.items():
                        m = role_stats.get("mean")
                        s = role_stats.get("std", 0.0)
                        n = role_stats.get("count", 0)
                        if m is not None:
                            lines.append(f"        [{role}] {m:.2f} \u00b1 {s:.2f}  (n={n})")

        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main evaluation pipeline
# ---------------------------------------------------------------------------

def evaluate_concise(
    conversations_path: str,
    output_dir: str,
    model: str = DEFAULT_MODEL,
    model_name: Optional[str] = None,
    temperature: float = 0.0,
    max_transcripts: Optional[int] = None,
    skip_turn_level: bool = False,
) -> None:
    """Run the dimension-oriented evaluation pipeline.

    Evaluates each conversation variant with 4 judges (3 if skip_turn_level),
    computes per-dimension overall scores, and writes structured results.

    Output directory: <output_dir>/<model_name>/ (or <output_dir>/ if no model_name)
    Output files:
        evaluations.jsonl         — incremental checkpoint (one line per transcript)
        evaluations_full.json     — all evaluations as JSON array
        evaluation_summary.json   — aggregated statistics per dimension
        evaluation_summary.txt    — human-readable summary report
        format_validation.json    — per-transcript format validation report
    """
    metric_registry = METRIC_REGISTRY
    client = get_api_client()

    if model_name:
        out = Path(output_dir) / model_name
    else:
        out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    transcripts = _load_conversations(conversations_path)
    if max_transcripts:
        transcripts = transcripts[:max_transcripts]

    # Format validation
    print(f"[EvaluatorConcise] Validating {len(transcripts)} transcripts...")
    format_report: List[Dict[str, Any]] = []
    for i, t in enumerate(transcripts):
        validation = _validate_transcript_format(t)
        sid = t.get("scenario_id", f"T-{i:04d}")
        format_report.append({
            "scenario_id": sid,
            "valid": validation["valid"],
            "issues": validation["issues"],
        })
        if not validation["valid"]:
            print(f"  [{sid}] Format issues: {validation['issues']}")
            transcripts[i] = _reformat_transcript(t)
            revalidation = _validate_transcript_format(transcripts[i])
            if revalidation["valid"]:
                print(f"  [{sid}] Reformatted successfully")
                format_report[-1]["reformatted"] = True
            else:
                print(f"  [{sid}] Still has issues after reformat: {revalidation['issues']}")
                format_report[-1]["reformatted"] = False

    with open(out / "format_validation.json", "w") as f:
        json.dump(format_report, f, indent=2)

    judge_labels = ["policy/resolution", "dialogue", "behavioral", "interest-alignment"]
    if not skip_turn_level:
        judge_labels.append("turn-level")
    print(f"\n[EvaluatorConcise] Running {len(judge_labels)} judges per variant: {', '.join(judge_labels)}")
    print(f"  Model: {model}")
    print(f"  Transcripts: {len(transcripts)}\n")

    # Clear incremental checkpoint
    eval_jsonl_path = out / "evaluations.jsonl"
    if eval_jsonl_path.exists():
        eval_jsonl_path.unlink()

    all_evaluations: List[Dict[str, Any]] = []

    for i, transcript in enumerate(transcripts):
        tid = transcript.get("scenario_id", f"T-{i:04d}")
        print(f"  [{i + 1}/{len(transcripts)}] {tid}")

        variants: List[Dict[str, Any]] = []
        conv_idx = 1

        while f"conversation_{conv_idx}" in transcript:
            print(f"    variant {conv_idx}: ", end="", flush=True)

            print("policy/resolution... ", end="", flush=True)
            pr_result = call_policy_resolution_judge(
                client=client, transcript=transcript,
                conv_idx=conv_idx, model=model, temperature=temperature,
            )

            print("dialogue... ", end="", flush=True)
            dq_result = call_dialogue_quality_judge(
                client=client, transcript=transcript,
                conv_idx=conv_idx, model=model, temperature=temperature,
            )

            print("behavioral... ", end="", flush=True)
            ba_result = call_behavioral_alignment_judge(
                client=client, transcript=transcript,
                conv_idx=conv_idx, model=model, temperature=temperature,
            )

            print("interest-alignment... ", end="", flush=True)
            ia_result = call_interest_alignment_judge(
                client=client, transcript=transcript,
                conv_idx=conv_idx, model=model, temperature=temperature,
            )

            tl_result: Dict[str, Any] = {"agent_turns": [], "customer_turns": []}
            if not skip_turn_level:
                print("turn-level... ", end="", flush=True)
                tl_result = call_turn_level_judge_concise(
                    client=client, transcript=transcript,
                    conv_idx=conv_idx, model=model, temperature=temperature,
                )

            variant_result = _assemble_variant_result(
                conv_idx=conv_idx,
                pr_result=pr_result,
                dq_result=dq_result,
                ba_result=ba_result,
                tl_result=tl_result,
                ia_result=ia_result,
                metric_registry=metric_registry,
                skip_turn_level=skip_turn_level,
            )
            variants.append(variant_result)
            print("done")
            conv_idx += 1

        entry: Dict[str, Any] = {
            "transcript_id": tid,
            "model": model,
            "num_variants": len(variants),
            "variants": variants,
        }
        all_evaluations.append(entry)

        with open(eval_jsonl_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    # Write full results
    with open(out / "evaluations_full.json", "w") as f:
        json.dump(all_evaluations, f, indent=2)

    # Compute and write summary
    timestamp = datetime.now(timezone.utc).isoformat()
    summary = _compute_concise_summary(
        all_evaluations,
        model=model,
        metric_registry=metric_registry,
        conversations_path=conversations_path,
        model_name=model_name,
        temperature=temperature,
        skip_turn_level=skip_turn_level,
        timestamp=timestamp,
    )

    with open(out / "evaluation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    summary_text = _format_concise_summary(summary)
    with open(out / "evaluation_summary.txt", "w") as f:
        f.write(summary_text)

    print(f"\n[EvaluatorConcise] Results written to: {out}")
    print(summary_text)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dimension-oriented evaluation pipeline for customer service conversations"
    )
    parser.add_argument(
        "--conversations_path",
        required=True,
        help="Path to conversations JSONL file",
    )
    parser.add_argument(
        "--output_dir",
        default="eval_concise",
        help="Base directory for evaluation results (default: eval_concise/)",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Model to use for all judges (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--model_name",
        default=None,
        help="Name of the model under test (used for output directory naming). "
             "If not provided, inferred from conversations_path.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature for LLM judges (default: 0.0)",
    )
    parser.add_argument(
        "--max_transcripts",
        type=int,
        default=None,
        help="Maximum number of transcripts to evaluate",
    )
    parser.add_argument(
        "--skip_turn_level",
        action="store_true",
        default=False,
        help=(
            "Skip the turn-level judge. Omits policy_support_per_turn, "
            "per_turn_contradiction_score, and consistency_drift from sub_metrics. "
            "Dimension overall_scores will be computed from fewer sub-metrics."
        ),
    )
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY is not set. Add it to configs/.env or the environment.")
        sys.exit(1)

    model_name = args.model_name
    if not model_name:
        conv_path = Path(args.conversations_path)
        SKIP_FOLDERS = {"with_resolution", "without_resolution", "scenarios", "conversations"}
        parent = conv_path.parent
        while parent.name in SKIP_FOLDERS and parent != parent.parent:
            parent = parent.parent
        model_name = parent.name

    evaluate_concise(
        conversations_path=args.conversations_path,
        output_dir=args.output_dir,
        model=args.model,
        model_name=model_name,
        temperature=args.temperature,
        max_transcripts=args.max_transcripts,
        skip_turn_level=args.skip_turn_level,
    )


if __name__ == "__main__":
    main()
