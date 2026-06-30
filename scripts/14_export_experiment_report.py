"""Export consolidated experiment tables and a Markdown report.

This script is intentionally read-only with respect to raw experiment outputs:
it reads existing AgentDojo/world-model artifacts and writes a compact report
bundle for sharing or manuscript drafting.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{100 * value:.2f}%"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _method_rows_from_summary(summary_json: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "top_k": row["top_k"],
            "method": row["method"],
            "seeds": row["seeds"],
            "count_per_seed": row["count_per_seed"],
            "targeted_asr_mean": row["targeted_asr_mean"],
            "targeted_asr_values": row["targeted_asr_values"],
            "utility_rate_mean": row["utility_rate_mean"],
            "utility_rate_values": row["utility_rate_values"],
        }
        for row in summary_json["summary"]
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    root = args.project_root.resolve()
    artifacts = root / "artifacts"
    data_root = root / "data" / "agentdojo_full_llama31_8b"
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    audit = _load_json(data_root / "audit.json")
    split_summary = _load_json(data_root / "splits" / "summary.json")
    world_metrics = _load_json(
        artifacts / "agentdojo_full_llama31_8b_world_model_metrics.json"
    )
    one_step_k24_k32 = _load_json(
        artifacts / "clean_prefix_disjoint_replay_k24_k32_main_table.json"
    )
    full_method_k24 = _load_json(
        artifacts / "clean_prefix_disjoint_replay_top24_full_method_table.json"
    )
    rollout_k32 = _load_json(
        artifacts / "clean_prefix_rollout_h3_disjoint_replay_top32_summary.json"
    )

    baseline_rows = [
        {
            "scope": "full_attack_baseline",
            "count": audit["attack_trajectories"],
            "attack_success": audit["attack_positive_labels"],
            "targeted_asr": audit["attack_success_rate"],
            "utility_rate": audit["utility_preservation_rate_under_attack"],
            "notes": "All full-suite attacked trajectories.",
        },
        {
            "scope": "clean_baseline",
            "count": audit["clean_trajectories"],
            "attack_success": "",
            "targeted_asr": "",
            "utility_rate": "",
            "notes": "Clean trajectories used for prefix scoring.",
        },
    ]

    world_model_rows = [
        {"metric": key, "value": value} for key, value in world_metrics.items()
    ]

    main_rows = _method_rows_from_summary(one_step_k24_k32)
    rollout_rows = _method_rows_from_summary(rollout_k32)
    ablation_rows = _method_rows_from_summary(full_method_k24)

    _write_csv(out_dir / "baseline_table.csv", baseline_rows)
    _write_csv(out_dir / "world_model_metrics.csv", world_model_rows)
    _write_csv(out_dir / "main_table_clean_prefix_k24_k32.csv", main_rows)
    _write_csv(out_dir / "ablation_table_clean_prefix_k24.csv", ablation_rows)
    _write_csv(out_dir / "rollout_table_clean_prefix_h3_k32.csv", rollout_rows)

    payload = {
        "scope": "WMagentattack consolidated experiment report",
        "project_root": str(root),
        "source_files": {
            "audit": str(data_root / "audit.json"),
            "split_summary": str(data_root / "splits" / "summary.json"),
            "world_model_metrics": str(
                artifacts / "agentdojo_full_llama31_8b_world_model_metrics.json"
            ),
            "one_step_k24_k32": str(
                artifacts / "clean_prefix_disjoint_replay_k24_k32_main_table.json"
            ),
            "full_method_k24": str(
                artifacts / "clean_prefix_disjoint_replay_top24_full_method_table.json"
            ),
            "rollout_k32": str(
                artifacts / "clean_prefix_rollout_h3_disjoint_replay_top32_summary.json"
            ),
        },
        "baseline": baseline_rows,
        "world_model_metrics": world_model_rows,
        "main_table": main_rows,
        "ablation_table": ablation_rows,
        "rollout_table": rollout_rows,
        "split_summary": split_summary,
    }
    (out_dir / "consolidated_results.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    report = f"""# WMAgentAttack Experiment Report — 2026-06-30

## 1. Scope

This report consolidates the current AgentDojo + world-model experiments for the
World-Model-Guided attack pipeline. The current victim model is
`Meta-Llama-3.1-8B-Instruct` running through the AgentDojo full-suite setup.

The current pipeline is:

```text
AgentDojo clean/attack traces
→ standardized trajectory/step records
→ train/val/test split
→ TF-IDF + LogisticRegression multi-head world model
→ clean-prefix task-injection pair selection
→ real AgentDojo replay
→ ASR / utility evaluation
```

## 2. Dataset and baseline

- Clean trajectories: {audit['clean_trajectories']}
- Attack trajectories: {audit['attack_trajectories']}
- Total standardized steps: {audit['total_steps']}
- Attack positives: {audit['attack_positive_labels']}
- Attack negatives: {audit['attack_negative_labels']}
- Full attack ASR: {_pct(audit['attack_success_rate'])}
- Full attack utility/BUP: {_pct(audit['utility_preservation_rate_under_attack'])}

The train/val/test split is trajectory-level. Test contains
{split_summary['splits']['test']['trajectories']} trajectories and
{split_summary['splits']['test']['steps']} steps.

## 3. Current world model backbone

The current world model is implemented in `src/wmagentattack/world_model.py`.
It is a lightweight multi-head sklearn model:

```text
StepRecord
→ step_to_text()
→ TfidfVectorizer(ngram_range=(1, 2), max_features=12000, min_df=2)
→ LogisticRegression skill_head
→ LogisticRegression risk_head
→ LogisticRegression utility_head
```

The input text includes the user goal, agent history, current observation,
previous skills, candidate skills, hypothetical attack action, and target skill.

Prediction heads:

- `skill_head`: next selected skill.
- `risk_head`: probability of attack success.
- `utility_head`: probability of task success / utility preservation.

The clean-prefix one-step selection score is:

```text
selection_score = risk_score + 0.5 * utility_score + 0.3 * target_skill_probability
```

The rollout selector performs a lightweight horizon-3 imagined skill rollout by
recursively appending predicted skills to the state and aggregating risk,
utility, and target-skill probability. It does not generate new attack text.

## 4. World model held-out metrics

| Metric | Value |
|---|---:|
| Next-skill accuracy | {_pct(world_metrics['next_skill_accuracy'])} |
| Next-skill top-3 accuracy | {_pct(world_metrics['next_skill_top3_accuracy'])} |
| Risk AUC | {world_metrics['risk_auc']:.4f} |
| Risk F1 | {world_metrics['risk_f1']:.4f} |
| Utility AUC | {world_metrics['utility_auc']:.4f} |
| Risk Brier score | {world_metrics['calibration_brier_score']:.4f} |

## 5. Main clean-prefix replay results

| K | Method | ASR mean | ASR values | Utility mean | Utility values |
|---:|---|---:|---|---:|---|
"""
    for row in main_rows:
        report += (
            f"| {row['top_k']} | {row['method']} | "
            f"{_pct(row['targeted_asr_mean'])} | {row['targeted_asr_values']} | "
            f"{_pct(row['utility_rate_mean'])} | {row['utility_rate_values']} |\n"
        )

    report += """
## 6. K=24 ablation replay

| Method | ASR mean | ASR values | Utility mean | Utility values |
|---|---:|---|---:|---|
"""
    for row in ablation_rows:
        report += (
            f"| {row['method']} | {_pct(row['targeted_asr_mean'])} | "
            f"{row['targeted_asr_values']} | {_pct(row['utility_rate_mean'])} | "
            f"{row['utility_rate_values']} |\n"
        )

    report += """
## 7. Rollout selector result

| K | Method | ASR mean | ASR values | Utility mean | Utility values |
|---:|---|---:|---|---:|---|
"""
    for row in rollout_rows:
        report += (
            f"| {row['top_k']} | {row['method']} | "
            f"{_pct(row['targeted_asr_mean'])} | {row['targeted_asr_values']} | "
            f"{_pct(row['utility_rate_mean'])} | {row['utility_rate_values']} |\n"
        )

    report += """
## 8. Current interpretation

The clean-prefix world-model selector consistently improves over disjoint random
and low-score baselines. K=24 gives the strongest utility-preserving attack
selection, while K=32 remains stable at larger budget. The K=24 ablation shows
that using risk, utility, or target-skill probability alone is weaker than the
combined multi-head world-model score. The horizon-3 rollout selector further
improves K=32 ASR over the one-step selector while preserving utility.

## 9. Next engineering step

Replace the current TF-IDF + LogisticRegression backbone with a DreamerV3-style
world model using SheepRL. The immediate goal should be to preserve the same
input/output contract first:

```text
trajectory/step dataset
→ latent recurrent world model
→ skill, risk, utility, target-reachability heads
→ clean-prefix selection / rollout
→ AgentDojo replay
```

This allows direct comparison against the current sklearn backbone.
"""

    (out_dir / "experiment_report.md").write_text(report, encoding="utf-8")
    print(json.dumps({"out_dir": str(out_dir), "files": sorted(p.name for p in out_dir.iterdir())}, indent=2))


if __name__ == "__main__":
    main()
