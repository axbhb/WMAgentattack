"""Apply the preregistered v34 data and selector gate."""

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


def _aggregate(rows: list[dict], arm: str) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row["arm"] == arm:
            grouped[row["row_id"]].append(row)
    output = []
    for values in grouped.values():
        first = values[0]
        row = {key: first[key] for key in ("row_id", "task_name", "attack_variant", "target", "target_p11")}
        for name in JOINT_OUTCOME_CLASSES:
            row[f"pred_{name}"] = float(np.mean([value[f"prob_{name}"] for value in values]))
        row["predicted_p11"] = row["pred_attack1_utility1"]
        output.append(row)
    return output


def _probability_metrics(rows: list[dict]) -> dict[str, float]:
    by_task: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_task[row["task_name"]].append(row)
    p11_brier = []
    cell_ce = []
    for values in by_task.values():
        target = np.asarray([row["target"] for row in values], dtype=float)
        probability = np.asarray(
            [[row[f"pred_{name}"] for name in JOINT_OUTCOME_CLASSES] for row in values],
            dtype=float,
        )
        probability = np.clip(probability, 1e-9, 1.0)
        probability /= probability.sum(axis=1, keepdims=True)
        p11_brier.append(float(np.mean((probability[:, 3] - target[:, 3]) ** 2)))
        cell_ce.append(float(np.mean(-(target * np.log(probability)).sum(axis=1))))
    return {"p11_brier": float(np.mean(p11_brier)), "four_cell_cross_entropy": float(np.mean(cell_ce))}


def _selected_targets(rows: list[dict]) -> dict[str, float]:
    by_task: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_task[row["task_name"]].append(row)
    return {
        task: float(max(values, key=lambda row: (row["predicted_p11"], row["row_id"]))["target_p11"])
        for task, values in by_task.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--data-audit", type=Path, required=True)
    parser.add_argument("--run-metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    predictions = _read_jsonl(args.predictions)
    data_audit = json.loads(args.data_audit.read_text(encoding="utf-8"))
    runtime = json.loads(args.run_metrics.read_text(encoding="utf-8"))
    arms = {
        arm: _aggregate(predictions, arm)
        for arm in ("structured_preexecution", "factorized_state_attack")
    }
    metrics = {}
    for arm, rows in arms.items():
        metrics[arm] = {
            **ranking_metrics(rows, "predicted_p11"),
            **_probability_metrics(rows),
        }
    baseline = metrics["structured_preexecution"]
    candidate = metrics["factorized_state_attack"]
    baseline_tasks = _selected_targets(arms["structured_preexecution"])
    candidate_tasks = _selected_targets(arms["factorized_state_attack"])
    deltas = {task: candidate_tasks[task] - baseline_tasks[task] for task in candidate_tasks}
    seed_metrics = []
    for seed in protocol["training"]["model_seeds"]:
        candidate_seed = [row for row in predictions if row["arm"] == "factorized_state_attack" and int(row["seed"]) == int(seed)]
        baseline_seed = [row for row in predictions if row["arm"] == "structured_preexecution" and int(row["seed"]) == int(seed)]
        candidate_top1 = ranking_metrics(candidate_seed, "predicted_p11")["top1_target_p11"]
        baseline_top1 = ranking_metrics(baseline_seed, "predicted_p11")["top1_target_p11"]
        seed_metrics.append({"seed": int(seed), "candidate_top1": candidate_top1, "gain": candidate_top1 - baseline_top1})

    gate = protocol["acceptance_gate"]
    data_checks = {
        "complete_120_episode_budget": runtime["victim_episodes"] == 120 and data_audit["all_rows_have_exact_seeds"],
        "zero_runtime_failures": runtime["runtime_failures"] == 0 and data_audit["runtime_failures"] == 0,
        "eight_clean_controls": data_audit["clean_rows"] == 8,
        "clean_eligibility_replicates": all(value >= 2 for value in data_audit["clean_successes_by_task"].values()),
        "thirty_two_attack_configs": data_audit["attack_rows"] == 32,
        "attack_interventions_identifiable": data_audit["tasks_with_two_attack_outcome_levels"] >= gate["minimum_tasks_with_two_attack_outcome_levels"],
        "zero_real_endpoints": runtime["real_external_endpoint_calls"] == 0,
    }
    model_checks = {
        "complete_model_budget": runtime["model_fits"] == protocol["fixed_budget"]["model_fits"],
        "top1_gain": candidate["top1_target_p11"] - baseline["top1_target_p11"] >= gate["minimum_top1_p11_gain_vs_structured"],
        "pairwise_gain": candidate["pairwise_accuracy"] - baseline["pairwise_accuracy"] >= gate["minimum_pairwise_accuracy_gain"],
        "positive_task_fraction": sum(value > 0 for value in deltas.values()) / len(deltas) >= gate["minimum_positive_task_fraction"],
        "seed_replication": sum(row["gain"] >= gate["minimum_top1_p11_gain_vs_structured"] for row in seed_metrics) >= gate["minimum_positive_model_seeds"],
        "p11_brier_noninferiority": candidate["p11_brier"] - baseline["p11_brier"] <= gate["maximum_p11_brier_degradation"],
        "four_cell_ce_gain": baseline["four_cell_cross_entropy"] - candidate["four_cell_cross_entropy"] >= gate["minimum_four_cell_ce_gain"],
        "beats_random": candidate["top1_target_p11"] - candidate["random_expected_target_p11"] >= gate["minimum_top1_gain_vs_random"],
    }
    passed = all(data_checks.values()) and all(model_checks.values())
    output = {
        "decision": "GO_SCALE_PAIRED_ATTACK_DATA_V34" if passed else "NO_GO_PAIRED_ATTACK_SELECTOR_V34",
        "data_checks": data_checks,
        "model_checks": model_checks,
        "metrics": metrics,
        "task_deltas": deltas,
        "seed_metrics": seed_metrics,
        "data_audit": data_audit,
        "authorization": {
            "scale_paired_attack_data": passed,
            "short_horizon_planner": False,
            "large_world_model_training": False,
            "real_external_endpoints": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
