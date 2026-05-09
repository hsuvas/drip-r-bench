# DRIP-R: A Benchmark for Decision-Making and Reasoning Under Real-World Policy Ambiguity in the Retail Domain

![DRIP-R benchmark overview](assets/benchmark_overview_v4%20%281%29.png)

Figure 1. Overview of the DRIP-R benchmark components and workflow.


This repository contains the code for DRIP-R, a benchmark for evaluating LLMs as agents in complex, real-world retail scenarios. The repository consists of two main components: an agent simulation that generates conversations between an agent and a customer based on provided scenarios, and an evaluation framework that scores these conversations across multiple dimensions.

This package ships two single-command workflows:

1. **Agent simulation** (`src/agent/run.py`) — runs sequential conversations on a JSONL of scenarios.
2. **Evaluation** (`src/evaluator_concise/evaluator_concise.py`) — scores the resulting conversations across 25 sub-metrics in 5 dimensions.

A small sample dataset (`data/sample_scenarios.jsonl`, 5 scenarios) is included
so the pipeline can be run end-to-end immediately.

---

## 1. Setup

```bash
pip install -r requirements.txt
cp .env.example .env       # then edit .env to add your OPENAI_API_KEY
```

Required: `OPENAI_API_KEY` (used by the evaluator judges and by the default
agent / customer models). Optional: `ANTHROPIC_API_KEY` if you set
`--agent_model` to a Claude model.

---

## 2. Run the agent simulation

```bash
python src/agent/run.py \
  --input_path data/sample_scenarios.jsonl \
  --output_dir output/agent_run \
  --num_scenarios 5 \
  --agent_model gpt-4o
```

Defaults: `--customer_model gpt-4o-mini`, `--max_turns 20`,
`--agent_persona RANDOM`, `--use_native_tools`, `--include_resolution`.

Output:

```
output/agent_run/<agent_model_tag>/with_resolution/
├── conversations.jsonl       # one record per scenario
└── scenarios/
    ├── <scenario_id>.json    # source-of-truth per scenario (resume-friendly)
    └── ...
```

Re-running the command resumes from the per-scenario files. Pass `--fresh` to
discard previous output.

### Useful flags

| Flag | Default | Meaning |
|---|---|---|
| `--agent_model` | `gpt-4o` | LLM under test |
| `--customer_model` | `gpt-4o-mini` | LLM that simulates the customer |
| `--max_turns` | `20` | Max agent + customer turns per conversation |
| `--num_scenarios` | (all) | Process at most N **uncompleted** scenarios |
| `--agent_persona` | `RANDOM` | One of: `DIRECT`, `FAIR`, `AGREEABLE`, `HELPFUL`, `VERY_HELPFUL`, `RANDOM` |
| `--no_native_tools` | off | Inject tool schemas into the prompt instead of using function-calling |
| `--no_resolution` | off | Strip the resolution from the saved record (agent still reasons internally) |
| `--concurrency` | `2` | Threads in the scenario worker pool |
| `--fresh` | off | Discard previous outputs and start over |

---

## 3. Run the evaluation

```bash
python src/evaluator_concise/evaluator_concise.py \
  --conversations_path output/agent_run/gpt-4o/with_resolution/conversations.jsonl \
  --output_dir output/eval_results \
  --model gpt-4o
```

Output:

```
output/eval_results/<model_name>/
├── evaluations.jsonl           # incremental checkpoint (one transcript per line)
├── evaluations_full.json       # all evaluations as a JSON array
├── evaluation_summary.json     # aggregated stats per dimension
├── evaluation_summary.txt      # human-readable summary
└── format_validation.json      # per-transcript validation issues
```

Useful flags:

| Flag | Default | Meaning |
|---|---|---|
| `--model` | `gpt-4o` | Judge model |
| `--temperature` | `0.0` | Judge sampling temperature |
| `--max_transcripts` | (all) | Only evaluate the first N transcripts |
| `--skip_turn_level` | off | Skip the turn-level judge (3 judges instead of 4) |

### Metric registry (25 metrics, 5 dimensions)

Defined in `src/evaluator_concise/metrics.py`:

| Dimension | # metrics |
|---|---|
| Policy Adherence | 4 |
| Task-Resolution Adherence | 2 |
| Dialogue/Conversation Quality | 8 |
| Interest Alignment | 2 |
| Behavioral Alignment | 9 |

---

[View Figure 2 (PDF) — DRIP-R evaluation framework grid](assets/eval_framework_grid_transposed_v4.pdf)

Figure 2. Overview of the DRIP-R benchmark evaluation metrics.


## 4. Layout

```
drip-r-bench/
├── README.md
├── requirements.txt
├── .env.example
├── data/
│   └── sample_scenarios.jsonl
└── src/
    ├── agent/                       # multi-turn simulation
    │   ├── run.py                   # CLI entry point
    │   ├── orchestrator.py          # Agent ↔ Customer ↔ Environment turn loop
    │   ├── agent.py                 # LLMAgent / LLMCustomer
    │   ├── environment.py           # LLM-simulated tool responses
    │   ├── llm_provider.py          # LiteLLM wrapper
    │   ├── conversation_state.py
    │   ├── prompt_builder.py
    │   ├── response_parser.py
    │   ├── toolset.py
    │   ├── tool_registry.py
    │   ├── prompt_single.py
    │   └── prompts/                 # agent + customer system-prompt templates
    └── evaluator_concise/           # LLM-as-judge scoring
        ├── evaluator_concise.py     # CLI entry point
        ├── metrics.py               # embedded 25-metric registry
        ├── utils.py
        └── instruction_prompt_concise_calibrated.py
```

---

## 5. Scenario format

Each line of the input JSONL is one scenario. Required keys:

- `scenario_id` — unique string
- `Policy.Primary Policy.text` — primary return policy
- `Policy.Related policies` — list of `{"text": ...}` (may be empty)
- `task` — task description and constraints
- `persona` — customer persona attributes

See `data/sample_scenarios.jsonl` for a complete example.




## Citation
If you use this codebase in your research, please cite: TBD
```



```