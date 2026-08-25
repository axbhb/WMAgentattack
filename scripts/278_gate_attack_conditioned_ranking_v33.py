"""Apply the frozen task-level gate to v33 attack ranking predictions."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wmagentattack.attack_conditioned_ranker import ranking_metrics
from wmagentattack.joint_outcome_auxiliary import JOINT_OUTCOME_CLASSES


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _task_macro_probability_metrics(rows: list[dict], score_prefix: str) -> dict[str, float]:
    by_task: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_task[str(row["task_name"])].append(row)
    briers = []
    p11_briers = []
    cross_entropies = []
    for candidates in by_task.values():
        target = np.asarray([row["target"] for row in candidates], dtype=float)
        probability = np.asarray(
            [
                [row[f"{score_prefix}{name}"] for name in JOINT_OUTCOME_CLASSES]
                for row in candidates
            ],
            dtype=float,
        )
        probability = np.clip(probability, 1e-9, 1.0)
        probability /= probability.sum(axis=1, keepdims=True)
        briers.append(float(np.mean((probability - target) ** 2)))
        p11_briers.append(float(np.mean((probability[:, 3] - target[:, 3]) ** 2)))
        cross_entropies.append(float(np.mean(-(target * np.log(probability)).sum(axis=1))))
    return {
        "four_cell_brier": float(np.mean(briers)),
        "p11_brier": float(np.mean(p11_briers)),
        "four_cell_cross_entropy": float(np.mean(cross_entropies)),
    }


def _aggregate(rows: list[dict], arm: str) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row["arm"] == arm:
            grouped[str(row["row_id"])].append(row)
    output = []
    for values in grouped.values():
        first = values[0]
        row = {
            key: first[key]
            for key in ("row_id", "task_name", "fold", "attack_family", "target", "target_p11", "v5_p11")
        }
        for name in JOINT_OUTCOME_CLASSES:
            row[f"pred_{name}"] = float(np.mean([value[f"prob_{name}"] for value in values]))
        row["predicted_p11"] = row["pred_attack1_utility1"]
        output.append(row)
    return output


def _v5_distribution_rows(reference: list[dict]) -> list[dict]:
    target_mean = np.mean(np.asarray([row["target"] for row in reference], dtype=float), axis=0)
    other = target_mean[:3] / target_mean[:3].sum()
    output = []
    for source in reference:
        p11 = float(np.clip(source["v5_p11"], 1e-9, 1 - 1e-9))
        row = dict(source)
        for index, name in enumerate(JOINT_OUTCOME_CLASSES[:3]):
            row[f"pred_{name}"] = float((1 - p11) * other[index])
        row["pred_attack1_utility1"] = p11
        row["predicted_p11"] = p11
        output.append(row)
    return output


def _per_task_selection(rows: list[dict], key: str) -> dict[str, float]:
    by_task: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_task[row["task_name"]].append(row)
    return {
        task: float(max(values, key=lambda row: (float(row[key]), row["row_id"]))["target_p11"])
        for task, values in by_task.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--run-metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    rows = _read_jsonl(args.predictions)
    runtime = json.loads(args.run_metrics.read_text(encoding="utf-8"))
    arms = {
        arm: _aggregate(rows, arm)
        for arm in (
            "structured_attack_residual",
            "world_attack_residual",
            "world_family_diagnostic",
        )
    }
    reference = arms["world_attack_residual"]
    v5 = _v5_distribution_rows(reference)
    all_rows = {**arms, "frozen_v5": v5}
    metrics = {}
    for arm, candidates in all_rows.items():
        rank = ranking_metrics(candidates, "predicted_p11" if arm != "frozen_v5" else "v5_p11")
        probability = _task_macro_probability_metrics(candidates, "pred_")
        metrics[arm] = {**rank, **probability}

    candidate = metrics["world_attack_residual"]
    baseline = metrics["frozen_v5"]
    structured = metrics["structured_attack_residual"]
    family = metrics["world_family_diagnostic"]
    candidate_tasks = _per_task_selection(arms["world_attack_residual"], "predicted_p11")
    baseline_tasks = _per_task_selection(v5, "v5_p11")
    deltas = {task: candidate_tasks[task] - baseline_tasks[task] for task in candidate_tasks}
    seed_top1 = []
    for seed in protocol["training"]["seeds"]:
        seed_rows = [row for row in rows if row["arm"] == "world_attack_residual" and int(row["seed"]) == int(seed)]
        seed_metric = ranking_metrics(seed_rows, "predicted_p11")
        seed_top1.append(
            {
                "seed": int(seed),
                "top1_target_p11": seed_metric["top1_target_p11"],
                "gain_over_v5": seed_metric["top1_target_p11"] - baseline["top1_target_p11"],
            }
        )
    gates = protocol["acceptance_gate"]
    checks = {
        "complete_budget": runtime["model_fits"] == protocol["fixed_budget"]["model_fits"] and runtime["runtime_failures"] == 0,
        "candidate_rows_complete": len(reference) == 400,
        "task_count_complete": candidate["task_count"] == 20,
        "top1_gain_vs_v5": candidate["top1_target_p11"] - baseline["top1_target_p11"] >= gates["minimum_top1_p11_gain_vs_v5"],
        "pairwise_gain_vs_v5": candidate["pairwise_accuracy"] - baseline["pairwise_accuracy"] >= gates["minimum_pairwise_accuracy_gain_vs_v5"],
        "positive_task_fraction": sum(value > 0 for value in deltas.values()) / len(deltas) >= gates["minimum_positive_task_fraction"],
        "seed_replication": sum(row["gain_over_v5"] >= gates["minimum_top1_p11_gain_vs_v5"] for row in seed_top1) >= gates["minimum_positive_seeds"],
        "p11_brier_noninferiority": candidate["p11_brier"] - baseline["p11_brier"] <= gates["maximum_p11_brier_degradation"],
        "four_cell_ce_gain_vs_structured": structured["four_cell_cross_entropy"] - candidate["four_cell_cross_entropy"] >= gates["minimum_four_cell_ce_gain_vs_structured"],
        "beats_random_expectation": candidate["top1_target_p11"] - candidate["random_expected_target_p11"] >= gates["minimum_top1_gain_vs_random"],
        "family_shortcut_not_required": family["top1_target_p11"] - candidate["top1_target_p11"] <= gates["maximum_family_diagnostic_advantage"],
        "zero_new_attack_execution": all(runtime[key] == 0 for key in ("attack_examples_generated", "victim_llm_calls", "sandbox_tool_calls", "real_external_endpoint_calls")),
    }
    decision = "GO_SHORT_HORIZON_ATTACK_PLANNING_V33" if all(checks.values()) else "NO_GO_ATTACK_CONDITIONED_RANKING_V33"
    output = {
        "decision": decision,
        "passed_clauses": sum(checks.values()),
        "total_clauses": len(checks),
        "checks": checks,
        "metrics": metrics,
        "task_deltas_vs_v5": deltas,
        "seed_top1": seed_top1,
        "authorization": {
            "short_horizon_existing_sandbox_pilot": decision.startswith("GO_"),
            "new_attack_generation": False,
            "large_world_model_training": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
