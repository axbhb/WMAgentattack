"""Apply the frozen task-level v35 data and policy gates."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wmagentattack.comparison_reward_policy import build_preference_pairs, preference_metrics


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _aggregate(rows: list[dict], arm: str) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row["arm"] == arm:
            grouped[str(row["row_id"])].append(row)
    output = []
    for row_id in sorted(grouped):
        values = grouped[row_id]
        first = values[0]
        output.append(
            {
                key: first[key]
                for key in (
                    "row_id",
                    "task_name",
                    "fold",
                    "attack_family",
                    "counts",
                    "target",
                    "target_p11",
                    "target_utility",
                    "target_reward",
                )
            }
            | {
                "predicted_score": float(np.mean([value["predicted_score"] for value in values])),
                "predicted_p11": float(np.mean([value["predicted_p11"] for value in values])),
                "predicted_utility": float(np.mean([value["predicted_utility"] for value in values])),
            }
        )
    return output


def _validate_predictions(rows: list[dict], protocol: dict) -> None:
    expected_arms = {"absolute_four_cell", "comparison_outcome_anchored", "family_comparison_diagnostic"}
    expected_seeds = set(protocol["training"]["model_seeds"])
    keys = [(row["arm"], int(row["seed"]), row["row_id"]) for row in rows]
    if len(keys) != 400 * len(expected_arms) * len(expected_seeds) or len(set(keys)) != len(keys):
        raise ValueError("incomplete or duplicate prediction matrix")
    common_ids = None
    targets = {}
    for arm in expected_arms:
        for seed in expected_seeds:
            cell = [row for row in rows if row["arm"] == arm and int(row["seed"]) == seed]
            ids = {row["row_id"] for row in cell}
            if len(ids) != 400 or len({row["task_name"] for row in cell}) != 20:
                raise ValueError("incomplete arm/seed task set")
            if common_ids is not None and common_ids != ids:
                raise ValueError("arm/seed row sets differ")
            common_ids = ids
            for row in cell:
                if not np.isfinite([row["predicted_score"], row["predicted_p11"], row["predicted_utility"]]).all():
                    raise ValueError("nonfinite prediction")
                target = (row["task_name"], row["fold"], tuple(row["counts"]), tuple(row["target"]))
                if row["row_id"] in targets and targets[row["row_id"]] != target:
                    raise ValueError("cross-arm target metadata mismatch")
                targets[row["row_id"]] = target


def _selection_by_task(rows: list[dict], key: str) -> dict[str, dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row["task_name"])].append(row)
    return {
        task: max(values, key=lambda row: (float(row[key]), str(row["row_id"])))
        for task, values in grouped.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--support-audit", type=Path, required=True)
    parser.add_argument("--run-metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    support = json.loads(args.support_audit.read_text(encoding="utf-8"))
    runtime = json.loads(args.run_metrics.read_text(encoding="utf-8"))
    if runtime["runtime_failures"] != 0 or any(
        runtime[key] != 0 for key in ("attack_examples_generated", "victim_llm_calls", "sandbox_tool_calls", "real_external_endpoint_calls")
    ):
        raise ValueError("runtime/scope integrity failure is not a scientific gate result")
    if not support["passed"]:
        output = {
            "decision": "NO_GO_COMPARISON_DATA_SUPPORT_V35",
            "data_support": support,
            "checks": {"data_support_passed": False, "zero_runtime_failures": runtime["runtime_failures"] == 0},
            "authorization": {
                "preregister_high_contrast_pilot": True,
                "attack_execution_before_independent_clean_gate": False,
                "online_attack_policy_training": False,
                "large_world_model_training": False,
            },
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(output, ensure_ascii=False, sort_keys=True, indent=2))
        return

    raw = _read_jsonl(args.predictions)
    _validate_predictions(raw, protocol)
    arms = {
        arm: _aggregate(raw, arm)
        for arm in ("absolute_four_cell", "comparison_outcome_anchored", "family_comparison_diagnostic")
    }
    reference = arms["comparison_outcome_anchored"]
    pairs, _ = build_preference_pairs(
        reference,
        draws=int(protocol["training"]["posterior_draws"]),
        posterior_seed=int(protocol["training"]["posterior_seed"]),
        minimum_confidence_gap=float(protocol["training"]["minimum_confidence_gap"]),
    )
    metrics = {
        arm: preference_metrics(rows, score_key="predicted_score", pairs=pairs)
        for arm, rows in arms.items()
    }
    candidate = metrics["comparison_outcome_anchored"]
    baseline = metrics["absolute_four_cell"]
    family = metrics["family_comparison_diagnostic"]
    candidate_selected = _selection_by_task(arms["comparison_outcome_anchored"], "predicted_score")
    baseline_selected = _selection_by_task(arms["absolute_four_cell"], "predicted_score")
    task_deltas = {
        task: float(candidate_selected[task]["target_reward"] - baseline_selected[task]["target_reward"])
        for task in candidate_selected
    }
    seed_results = []
    for seed in protocol["training"]["model_seeds"]:
        candidate_seed = [
            row for row in raw if row["arm"] == "comparison_outcome_anchored" and int(row["seed"]) == int(seed)
        ]
        baseline_seed = [
            row for row in raw if row["arm"] == "absolute_four_cell" and int(row["seed"]) == int(seed)
        ]
        candidate_seed.sort(key=lambda row: row["row_id"])
        baseline_seed.sort(key=lambda row: row["row_id"])
        candidate_metric = preference_metrics(candidate_seed, score_key="predicted_score", pairs=pairs)
        baseline_metric = preference_metrics(baseline_seed, score_key="predicted_score", pairs=pairs)
        seed_results.append(
            {
                "seed": int(seed),
                "top1_reward_gain": candidate_metric["top1_target_reward"] - baseline_metric["top1_target_reward"],
                "top1_p11_gain": candidate_metric["top1_target_p11"] - baseline_metric["top1_target_p11"],
            }
        )
    gate = protocol["acceptance_gate"]
    checks = {
        "data_support_passed": True,
        "complete_budget": runtime["model_fits"] == protocol["fixed_budget"]["model_fits"] and runtime["runtime_failures"] == 0,
        "candidate_rows_complete": len(reference) == 400,
        "task_count_complete": candidate["task_count"] == 20,
        "top1_reward_gain": candidate["top1_target_reward"] - baseline["top1_target_reward"] >= gate["minimum_top1_reward_gain"],
        "top1_p11_gain": candidate["top1_target_p11"] - baseline["top1_target_p11"] >= gate["minimum_top1_p11_gain"],
        "pairwise_gain": candidate["posterior_pairwise_accuracy"] - baseline["posterior_pairwise_accuracy"] >= gate["minimum_pairwise_accuracy_gain"],
        "positive_task_fraction": sum(value > 0 for value in task_deltas.values()) / len(task_deltas) >= gate["minimum_positive_task_fraction"],
        "seed_replication": sum(
            row["top1_reward_gain"] >= gate["minimum_top1_reward_gain"] for row in seed_results
        ) >= gate["minimum_positive_seeds"],
        "utility_noninferiority": baseline["top1_target_utility"] - candidate["top1_target_utility"] <= gate["maximum_utility_degradation"],
        "beats_random_p11": candidate["top1_target_p11"] - candidate["random_expected_p11"] >= gate["minimum_p11_gain_vs_random"],
        "family_shortcut_not_required": family["top1_target_reward"] - candidate["top1_target_reward"] <= gate["maximum_family_diagnostic_advantage"],
        "zero_new_execution": all(
            runtime[key] == 0
            for key in ("attack_examples_generated", "victim_llm_calls", "sandbox_tool_calls", "real_external_endpoint_calls")
        ),
    }
    decision = "GO_COMPARISON_REWARD_ATTACK_PILOT_V35" if all(checks.values()) else "NO_GO_COMPARISON_REWARD_POLICY_V35"
    output = {
        "decision": decision,
        "passed_clauses": sum(checks.values()),
        "total_clauses": len(checks),
        "checks": checks,
        "data_support": support,
        "metrics": metrics,
        "task_reward_deltas": task_deltas,
        "seed_results": seed_results,
        "authorization": {
            "preregister_high_contrast_pilot": True,
            "attack_execution_before_independent_clean_gate": False,
            "online_comparison_attack_pilot": decision.startswith("GO_"),
            "large_world_model_training": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
