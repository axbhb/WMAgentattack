# World-Model-Guided Skill Trajectory Attack

This workspace currently contains the first deployment milestone: running a
local Qwen2.5-7B-Instruct model through the official AgentDojo task, tool, and
evaluation pipeline. Only sandbox benchmark tasks are in scope.

## Environment

The deployment uses the Conda environment `wmagentattack`:

```powershell
conda activate wmagentattack
```

The checked-out AgentDojo source is installed in editable mode from
`external/agentdojo`. The local model is read directly from the existing
Hugging Face snapshot under `F:\hf_cache`; it is not copied or modified.

## Smoke tests

Verify CUDA:

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Load the model without running a benchmark:

```powershell
python scripts/run_agentdojo_qwen.py --load-only
```

Run one clean AgentDojo task:

```powershell
python scripts/run_agentdojo_qwen.py --suite workspace --user-task user_task_0
```

Run the small clean baseline with one model load:

```powershell
python scripts/run_agentdojo_qwen.py --baseline --quantization 4bit
python scripts/summarize_agentdojo_runs.py
```

Native Windows PyTorch does not expose a Flash Attention kernel on this host.
Very large tool results can therefore exhaust 24 GB even with 4-bit weights.
The runner defaults to compacting tool outputs above 12,000 characters in the
model context while retaining the complete unmodified output in AgentDojo's raw
JSON trace. Set `--max-tool-output-chars 0` to disable this behavior. Attack
experiments must report this setting because an injected string can reside in a
compacted field.

Raw AgentDojo task records are written beneath `runs/`. Attack experiments are
deliberately not enabled in this initial deployment step.

## Verified deployment status

Verified on an NVIDIA RTX 4090 with:

- Python 3.11
- PyTorch 2.10.0+cu128
- AgentDojo 0.1.35 at commit `089ed468c`
- Qwen2.5-7B-Instruct loaded in BF16 from the local snapshot

The following checks have completed successfully:

- CUDA and BF16 availability
- offline loading of all Qwen model shards
- AgentDojo's test suite (`32 passed`)
- end-to-end clean task execution with parsed tool calls, sandbox tool
  execution, utility evaluation, and JSON trace output

The initial `workspace/user_task_0` and `workspace/user_task_3` runs received
`utility=false`. Their traces show valid multi-step tool interaction; the model
incorrectly assumed the year 2023 despite the system instruction and, in the
first task, omitted one participant from its final answer. These are model
baseline failures rather than deployment failures.

## Small clean baseline

Eight clean `workspace` tasks were run with Qwen2.5-7B, 4-bit NF4 weights, and
the 12,000-character context compaction threshold.

| Configuration | Utility | Tool error rate | Parse failure rate |
|---|---:|---:|---:|
| Base AgentDojo local prompt | 3/8 (37.5%) | 60.0% | 6.7% |
| Robust tool-use prompt | 4/8 (50.0%) | 10.0% | 0.0% |

The robust profile requires date lookup before inferring a year, distinguishes
file-content search from metadata inspection, and enforces the function-call
syntax. Its full summary is saved at
`runs/clean_baseline_qwen25_7b_4bit_compact12000_robust.json`.

The remaining failures are useful victim-model behavior rather than integration
errors: an exact output-format mismatch, incorrect temporal reasoning, and
incomplete comparison of long file lists. This is sufficient to begin raw-trace
normalization without further prompt tuning on the baseline task set.

## End-to-end MVP pipeline

The complete research smoke-test pipeline is now runnable with:

```powershell
python scripts/run_mvp_pipeline.py
```

It performs:

1. normalization of raw AgentDojo traces into validated step-level and
   trajectory-level JSONL;
2. abstraction of concrete tools into generalized skills while retaining
   `selected_tool` for provenance;
3. deterministic synthetic clean and symbolic-perturbation data generation;
4. trajectory-level train/validation/test splitting;
5. training of TF-IDF plus logistic-regression heads for next skill, future
   attack success, and task utility;
6. held-out world-model evaluation;
7. one-step ranking of benchmark-safe symbolic perturbations;
8. synthetic-only attack-baseline and defense-detector evaluation.

Current smoke-test outputs:

- `data/processed/agentdojo_steps.jsonl`
- `data/processed/agentdojo_trajectories.jsonl`
- `data/splits/*_steps.jsonl`
- `artifacts/world_model.joblib`
- `artifacts/world_model_metrics.json`
- `artifacts/planner_ranking.json`
- `artifacts/attack_defense_metrics.json`

With 3,000 synthetic trajectories, the current held-out smoke test reaches
approximately 91.5% next-skill accuracy and 0.83 risk AUC. The synthetic
world-model planner reaches about 68.2% ASR versus 52.5% for random selection.
These numbers validate code flow only and are not research evidence. The next
scientific milestone is collecting AgentDojo attack trajectories and replacing
synthetic risk labels with benchmark security evaluations.

## Real AgentDojo attack collection

Small sandbox-only attack batches can be collected with:

```powershell
python scripts/07_collect_agentdojo_attacks.py --attack direct
python scripts/07_collect_agentdojo_attacks.py --attack ignore_previous --attack injecagent
python scripts/08_prepare_real_agentdojo_dataset.py
python scripts/09_split_real_agentdojo_dataset.py
```

The collector uses AgentDojo's own injection placeholders, attacks, stateful
environment, utility checks, and security checks. The first audit intentionally
keeps these real traces separate from synthetic training data. If the collected
attack set contains only one label, `audit.json` marks the real risk head as not
trainable rather than silently fitting a meaningless classifier.

The current real-data audit contains 8 clean and 86 completed attack
trajectories across the `workspace` and `slack` suites. The early `workspace`
matrix produced no successful attacker goals, but a focused `slack` probe with
webpage/channel-message injection surfaces produced 30 positive attack
trajectories. This makes the real-data risk head trainable:

- Attack success: 30/86 trajectories (34.9%)
- Benign utility preserved under attack: 61/86 trajectories (70.9%)
- Positive attack domains: `slack`
- Negative-heavy domains: `workspace`

The normalizer records terminal responses as a `finish` skill and places
untrusted tool output on the following decision step, preventing future-output
leakage. Incomplete traces from interrupted long-running batches are excluded
and counted in the audit.

The current real split is trajectory-level and lightly stratified by domain and
attack-success label:

- train: 65 trajectories, 288 steps, 21 positive attack trajectories
- validation: 13 trajectories, 56 steps, 4 positive attack trajectories
- test: 16 trajectories, 74 steps, 5 positive attack trajectories

A first real-data world-model smoke test is saved at:

- `artifacts/real_agentdojo_world_model.joblib`
- `artifacts/real_agentdojo_world_model_metrics.json`
- `artifacts/real_agentdojo_planner_ranking.json`
- `artifacts/real_agentdojo_attack_defense_metrics.json`

On the current tiny held-out test set, the real model reaches 63.5%
next-skill accuracy, 87.8% top-3 next-skill accuracy, 0.83 risk AUC, and 0.88
risk F1. These are pipeline-health numbers only: positive samples are still
concentrated in `slack`, so the next scientific step is to diversify positives
across `travel` and `banking`.

The runner also supports Qwen2.5's native tool-calling template:

```powershell
python scripts/run_agentdojo_qwen.py --protocol native --user-task user_task_1
python scripts/07_collect_agentdojo_attacks.py --protocol native --attack tool_knowledge
```

Native mode produces cleaner `<tool_call>{"name", "arguments"}</tool_call>`
outputs and is preferred for future collection. The first useful positive
batch was collected in native mode with the base AgentDojo prompt on the
`slack` suite.

## Implementation note

AgentDojo's built-in `local` provider expects an OpenAI-compatible model server
such as vLLM. For this native Windows deployment,
`TransformersQwenLLM` is a custom AgentDojo pipeline element that loads Qwen
directly with Transformers while retaining AgentDojo's official environments,
tools, task suites, utility checks, and logging.
