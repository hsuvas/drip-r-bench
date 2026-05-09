"""
CLI entry point for the agent simulation system.

Usage::

    python src/agent/run.py \
        --input_path data/sample_scenarios.jsonl \
        --output_dir output/agent_run \
        --num_scenarios 10 \
        --agent_model gpt-4o
        # defaults: --customer_model gpt-4o-mini, --max_turns 20
"""

import argparse
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_SRC_DIR = str(Path(__file__).resolve().parent.parent)

if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from dotenv import load_dotenv  # noqa: E402

from agent.llm_provider import LLMProvider  # noqa: E402
from agent.orchestrator import run_conversation, AGENT_PERSONAS  # noqa: E402


def load_scenarios(path: str) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    if p.suffix.lower() == ".jsonl":
        items: List[Dict[str, Any]] = []
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    items.append(json.loads(line))
        return items

    if p.suffix.lower() == ".json":
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else [data]

    raise ValueError(f"Unsupported file format: {p.suffix}")


def _load_completion_status(output_dir: str) -> Dict[str, bool]:
    per_dir = os.path.join(output_dir, "scenarios")
    status: Dict[str, bool] = {}
    if os.path.isdir(per_dir):
        for fname in os.listdir(per_dir):
            if fname.endswith(".json"):
                status[fname[:-5]] = True
    return status


def _rebuild_jsonl(per_scenario_dir: str, out_jsonl: str) -> None:
    files = sorted(f for f in os.listdir(per_scenario_dir) if f.endswith(".json"))
    with open(out_jsonl, "w", encoding="utf-8") as out:
        for fname in files:
            fpath = os.path.join(per_scenario_dir, fname)
            with open(fpath, "r", encoding="utf-8") as f:
                rec = json.load(f)
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[REBUILD] Wrote {len(files)} records to {out_jsonl}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Agent simulation — generates customer support conversations.",
    )

    p.add_argument("--input_path", required=True, help="Path to scenarios JSONL/JSON")
    p.add_argument("--output_dir", required=True, help="Output directory")

    p.add_argument("--agent_model", default="gpt-4o")
    p.add_argument("--customer_model", default="gpt-4o-mini")

    p.add_argument("--agent_temperature", type=float, default=0.7)
    p.add_argument("--customer_temperature", type=float, default=0.8)
    p.add_argument("--top_p", type=float, default=1.0)

    p.add_argument("--max_turns", type=int, default=20)
    p.add_argument("--num_scenarios", type=int, default=None,
                   help="Maximum number of NEW (uncompleted) scenarios to process")
    p.add_argument("--scenario_offset", type=int, default=0,
                   help="Skip the first N scenarios before slicing")
    p.add_argument("--max_output_tokens_agent", type=int, default=4096)
    p.add_argument("--max_output_tokens_customer", type=int, default=800)

    p.add_argument("--agent_persona", default="RANDOM",
                   choices=AGENT_PERSONAS + ["RANDOM"],
                   help="Agent persona (or RANDOM to sample per scenario)")

    p.add_argument("--use_native_tools", action="store_true", default=True,
                   help="Use native function calling API")
    p.add_argument("--no_native_tools", dest="use_native_tools",
                   action="store_false",
                   help="Inject tools into prompt instead of using API")

    p.add_argument("--include_resolution", action="store_true", default=True)
    p.add_argument("--no_resolution", dest="include_resolution", action="store_false",
                   help="Strip resolution from output (agent still reasons internally)")

    p.add_argument("--max_policy_chars", type=int, default=None,
                   help="Truncate primary policy text to this many characters.")

    p.add_argument("--concurrency", type=int, default=2)
    p.add_argument("--sleep_s", type=float, default=0.0)
    p.add_argument("--fresh", action="store_true", default=False,
                   help="Discard previous outputs and start from scratch")

    p.add_argument("--env_path", type=str, default=None,
                   help="Path to .env file (defaults to ./.env or configs/.env)")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.env_path:
        load_dotenv(args.env_path)
    else:
        for candidate in (_PROJECT_ROOT / ".env", _PROJECT_ROOT / "configs" / ".env"):
            if candidate.exists():
                load_dotenv(candidate)
                break

    if not (os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")):
        print("ERROR: Set OPENAI_API_KEY or ANTHROPIC_API_KEY in the environment / .env file.")
        sys.exit(1)

    agent_model = args.agent_model
    customer_model = args.customer_model

    model_tag = args.agent_model.replace("/", "_").replace(":", "_")
    resolution_tag = "with_resolution" if args.include_resolution else "no_resolution"
    output_dir = os.path.join(args.output_dir, model_tag, resolution_tag)

    os.makedirs(output_dir, exist_ok=True)
    per_scenario_dir = os.path.join(output_dir, "scenarios")
    os.makedirs(per_scenario_dir, exist_ok=True)

    out_jsonl = os.path.join(output_dir, "conversations.jsonl")
    if args.fresh:
        for fname in os.listdir(per_scenario_dir):
            if fname.endswith(".json"):
                os.remove(os.path.join(per_scenario_dir, fname))
        if os.path.exists(out_jsonl):
            os.remove(out_jsonl)
        completion_status: Dict[str, bool] = {}
        print("[FRESH] Cleared previous outputs")
    else:
        completion_status = _load_completion_status(output_dir)
        n_done = len(completion_status)
        if n_done:
            print(f"[RESUME] {n_done} scenarios already completed, skipping them")

    scenarios = load_scenarios(args.input_path)
    scenarios = scenarios[args.scenario_offset:]
    if args.num_scenarios:
        new_scenarios: List[Dict[str, Any]] = []
        for sc in scenarios:
            if len(new_scenarios) >= args.num_scenarios:
                break
            if not completion_status.get(sc.get("scenario_id", "")):
                new_scenarios.append(sc)
        scenarios = new_scenarios

    if args.max_policy_chars:
        for sc in scenarios:
            txt = sc["Policy"]["Primary Policy"]["text"]
            if len(txt) > args.max_policy_chars:
                sc["Policy"]["Primary Policy"]["text"] = txt[: args.max_policy_chars]
        print(f"[Truncate] Primary policy text capped at {args.max_policy_chars} chars")

    print(f"Output dir: {output_dir}")
    print(f"Loaded {len(scenarios)} scenarios")
    print(f"Agent model: {args.agent_model}")
    print(f"Customer model: {args.customer_model}")
    print(f"Max turns: {args.max_turns}")
    print(f"Include resolution: {args.include_resolution}")
    print(f"Agent persona: {args.agent_persona}")
    print(f"Native tools: {args.use_native_tools}")
    print(f"Concurrency: {args.concurrency}")

    failures: List[Dict[str, Any]] = []
    t_start = time.time()

    def _process_scenario(idx: int, sc: Dict[str, Any], persona: str) -> Dict[str, Any]:
        provider = LLMProvider(
            model=agent_model,
            temperature=args.agent_temperature,
            max_tokens=args.max_output_tokens_agent,
            top_p=args.top_p,
        )
        cust_provider = LLMProvider(
            model=customer_model,
            temperature=args.customer_temperature,
            max_tokens=args.max_output_tokens_customer,
            top_p=args.top_p,
        )
        return run_conversation(
            scenario=sc,
            llm_provider=provider,
            agent_persona=persona,
            max_turns=args.max_turns,
            use_native_tools=args.use_native_tools,
            sleep_between_turns=args.sleep_s,
            include_resolution=args.include_resolution,
            customer_llm_provider=cust_provider,
        )

    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures: Dict[Any, tuple] = {}
        for idx, sc in enumerate(scenarios, start=1):
            scenario_id = sc.get("scenario_id", f"scenario_{idx:05d}")
            row_id = scenario_id

            if completion_status.get(row_id):
                continue

            persona = (
                random.choice(AGENT_PERSONAS)
                if args.agent_persona == "RANDOM"
                else args.agent_persona
            )

            fut = executor.submit(_process_scenario, idx, sc, persona)
            futures[fut] = (idx, row_id, scenario_id)

        for fut in as_completed(futures):
            idx, row_id, scenario_id = futures[fut]
            try:
                rec = fut.result()
                rec["row_id"] = row_id
                rec["user_idx"] = idx
                rec["scenario_idx"] = idx

                row_path = os.path.join(per_scenario_dir, f"{row_id}.json")
                with open(row_path, "w") as rf:
                    json.dump(rec, rf, indent=2, ensure_ascii=False)

                completion_status[row_id] = True
                print(f"[{idx}/{len(scenarios)}] Saved: {row_id}")

            except Exception as e:
                print(f"[{idx}/{len(scenarios)}] FAILED: {row_id} -> {e}")
                failures.append({"row_id": row_id, "error": str(e)})

    _rebuild_jsonl(per_scenario_dir, out_jsonl)

    if failures:
        fail_path = os.path.join(output_dir, "failures.json")
        with open(fail_path, "w") as f:
            json.dump(failures, f, indent=2)
        print(f"\n{len(failures)} failures saved to {fail_path}")

    total_completed = len([f for f in os.listdir(per_scenario_dir) if f.endswith(".json")])
    elapsed = time.time() - t_start
    print(f"\nDone. {total_completed} total completed scenarios in {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    print(f"Output: {out_jsonl}")


if __name__ == "__main__":
    main()
