"""Aggregate the fixed 15-run v45 budget and compare retained v5/v22 controls."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def mean(values) -> float:
    values = list(values)
    if not values:
        raise ValueError("empty metric surface")
    return float(np.mean(values))


def task_seed_surface(rows, *, metric: str, record_type: str, horizons: set[int] | None = None):
    grouped = defaultdict(list)
    for row in rows:
        if row.get("record_type") != record_type or row.get(metric) is None:
            continue
        if horizons is not None and int(row.get("horizon", -1)) not in horizons:
            continue
        horizon = int(row["horizon"]) if horizons is not None else 1
        grouped[(int(row["seed"]), horizon, str(row["task_name"]))].append(float(row[metric]))
    return {key: mean(values) for key, values in grouped.items()}


def v5_surface(rows, *, metric: str):
    grouped = defaultdict(list)
    for row in rows:
        if row.get("arm") != "structured_joint_aux" or row.get(metric) is None:
            continue
        if metric.startswith("action_") and not row.get("has_next_action", False):
            continue
        grouped[(int(row["training_seed"]), 1, str(row["task_name"]))].append(float(row[metric]))
    return {key: mean(values) for key, values in grouped.items()}


def normalized_joint_surface(rows, *, source: str):
    """Match the retained v5 trajectory -> attack-group -> task aggregation."""
    trajectory = defaultdict(list)
    trajectory_group = {}
    trajectory_task = {}
    for row in rows:
        if source == "v5" and row.get("arm") != "structured_joint_aux":
            continue
        metric = row.get("joint_cross_entropy")
        if metric is None or not row.get("joint_trainable", False):
            continue
        seed = int(row.get("training_seed", row.get("seed")))
        key = (seed, str(row["trajectory_id"]))
        trajectory[key].append(float(metric))
        trajectory_group[key] = str(row["joint_group_id"])
        trajectory_task[key] = str(row["task_name"])
    group = defaultdict(list)
    group_task = {}
    for key, values in trajectory.items():
        seed = key[0]
        group_key = (seed, trajectory_group[key])
        group[group_key].append(mean(values))
        group_task[group_key] = trajectory_task[key]
    task = defaultdict(list)
    for group_key, values in group.items():
        task[(group_key[0], 1, group_task[group_key])].append(mean(values))
    return {key: mean(values) for key, values in task.items()}


def normalized_joint_brier_surface(rows, *, source: str):
    remapped = []
    for row in rows:
        copy = dict(row)
        copy["joint_cross_entropy"] = row.get("joint_brier")
        remapped.append(copy)
    return normalized_joint_surface(remapped, source=source)


def v22_surface(rows):
    grouped = defaultdict(list)
    for row in rows:
        control = row.get("control", row.get("arm"))
        if control != "free_latent_residual" or int(row.get("horizon", -1)) not in {2, 3, 4, 5}:
            continue
        grouped[(int(row["training_seed"]), int(row["horizon"]), str(row["task_name"]))].append(
            float(row["action_nll"])
        )
    return {key: mean(values) for key, values in grouped.items()}


def paired_gain(baseline: dict, candidate: dict):
    keys = sorted(set(baseline) & set(candidate))
    if keys != sorted(baseline) or keys != sorted(candidate):
        raise ValueError("paired control surface mismatch")
    return {key: float(baseline[key] - candidate[key]) for key in keys}


def weighted_multistep_gain(gains: dict, horizon_weights: dict[str, float]) -> float:
    by_unit = defaultdict(list)
    for (seed, horizon, task), value in gains.items():
        by_unit[(seed, task)].append((int(horizon), float(value)))
    unit_values = []
    for rows in by_unit.values():
        numerator = sum(horizon_weights[str(h)] * value for h, value in rows)
        denominator = sum(horizon_weights[str(h)] for h, _ in rows)
        unit_values.append(numerator / denominator)
    return mean(unit_values)


def infer_agentdojo_suite(event: dict) -> str:
    """Recover the four frozen AgentDojo suites without using test labels."""
    known = ("banking", "slack", "travel", "workspace")
    task_name = str(event["task_name"]).lower()
    matches = [suite for suite in known if suite in task_name]
    source = str(event["causal_model_input"].get("source", "")).lower()
    if source in known and source not in matches:
        matches.append(source)
    if len(matches) != 1:
        raise ValueError(f"cannot infer exactly one AgentDojo suite for {event['task_name']!r}")
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--v5-predictions", type=Path, required=True)
    parser.add_argument("--v22-predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    expected = {
        (fold, seed)
        for fold in range(int(protocol["scope"]["fold_count"]))
        for seed in map(int, protocol["scope"]["seeds"])
    }
    run_files = sorted(args.runs_root.glob("fold*_seed*/metrics.json"))
    completed = set()
    runtime_failures = 0
    rows = []
    for metrics_path in run_files:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        key = (int(metrics["fold"]), int(metrics["seed"]))
        if key in completed:
            raise ValueError(f"duplicate formal fit: {key}")
        completed.add(key)
        runtime_failures += int(metrics.get("runtime_failures", 0))
        rows.extend(read_jsonl(metrics_path.with_name("predictions.jsonl")))
    if completed != expected:
        raise ValueError(f"incomplete fixed budget: got {sorted(completed)}, expected {sorted(expected)}")

    v5_rows = read_jsonl(args.v5_predictions)
    v22_rows = read_jsonl(args.v22_predictions)
    teacher_nll = task_seed_surface(rows, metric="teacher_action_nll", record_type="one_step")
    teacher_accuracy = task_seed_surface(rows, metric="teacher_action_correct", record_type="one_step")
    residual_h1 = task_seed_surface(rows, metric="residual_h1_nll", record_type="one_step")
    joint_ce = normalized_joint_surface(rows, source="v45")
    joint_brier = normalized_joint_brier_surface(rows, source="v45")
    v45_rollout = task_seed_surface(
        rows, metric="action_nll", record_type="rollout", horizons={2, 3, 4, 5}
    )
    v5_nll = v5_surface(v5_rows, metric="action_nll")
    v5_accuracy = v5_surface(v5_rows, metric="action_correct")
    v5_joint_ce = normalized_joint_surface(v5_rows, source="v5")
    v5_joint_brier = normalized_joint_brier_surface(v5_rows, source="v5")
    v22_rollout = v22_surface(v22_rows)

    action_nll_gain = paired_gain(v5_nll, teacher_nll)
    accuracy_gain = paired_gain(teacher_accuracy, v5_accuracy)
    joint_ce_gain = paired_gain(v5_joint_ce, joint_ce)
    joint_brier_gain = paired_gain(v5_joint_brier, joint_brier)
    h1_degradation = paired_gain(residual_h1, teacher_nll)
    rollout_gain = paired_gain(v22_rollout, v45_rollout)
    horizon_weights = protocol["training"]["residual_stage"]["horizon_weights"]
    weighted_gain = weighted_multistep_gain(rollout_gain, horizon_weights)

    by_task = defaultdict(list)
    by_seed = defaultdict(list)
    for (seed, _horizon, task), value in rollout_gain.items():
        by_task[task].append(value)
        by_seed[seed].append(value)
    positive_task_fraction = mean(mean(values) > 0 for values in by_task.values())
    positive_seed_count = sum(mean(values) > 0 for values in by_seed.values())

    task_suite = {}
    for event in dataset["events"]:
        task = str(event["task_name"])
        suite = infer_agentdojo_suite(event)
        if task in task_suite and task_suite[task] != suite:
            raise ValueError(f"task has multiple suites: {task}")
        task_suite[task] = suite
    by_suite = defaultdict(list)
    for (_seed, _horizon, task), value in rollout_gain.items():
        by_suite[task_suite[task]].append(value)
    worst_suite_gain = min(mean(values) for values in by_suite.values())

    legal_values = [
        float(row[key])
        for row in rows
        for key in ("legal_teacher_prediction", "legal_residual_prediction", "legal_prediction")
        if row.get(key) is not None
    ]
    legal_rate = mean(legal_values)
    gate = protocol["frozen_gate"]
    clauses = {
        "all_fifteen_fits_complete": completed == expected,
        "zero_runtime_failures": runtime_failures == gate["data_and_runtime"]["runtime_failures"],
        "legal_prediction_rate": legal_rate == gate["data_and_runtime"]["legal_prediction_rate"],
        "teacher_action_nll_gain": mean(action_nll_gain.values()) >= gate["teacher"]["action_nll_gain_over_v5_min"],
        "teacher_action_accuracy_noninferiority": -mean(accuracy_gain.values()) <= gate["teacher"]["action_accuracy_drop_over_v5_max"],
        "joint_cross_entropy_gain": mean(joint_ce_gain.values()) >= gate["teacher"]["joint_cross_entropy_gain_over_v5_min"],
        "joint_brier_gain": mean(joint_brier_gain.values()) >= gate["teacher"]["joint_brier_gain_over_v5_min"],
        "h1_noninferiority": mean(h1_degradation.values()) <= gate["residual"]["h1_action_nll_degradation_vs_teacher_max"],
        "h2_h5_weighted_nll_gain": weighted_gain >= gate["residual"]["horizon_weighted_nll_gain_over_v22_min"],
        "positive_task_fraction": positive_task_fraction >= gate["residual"]["positive_task_fraction_min"],
        "positive_seed_count": positive_seed_count >= gate["residual"]["positive_seed_count_min"],
        "suite_noninferiority": worst_suite_gain >= -gate["residual"]["suite_macro_drop_max"],
    }
    decision = "GO_LARGE_HYBRID_WORLD_MODEL_V45" if all(clauses.values()) else "NO_GO_LARGE_HYBRID_WORLD_MODEL_V45"
    payload = {
        "schema_version": "wmagentattack.large_hybrid_gate.v45",
        "decision": decision,
        "clauses": clauses,
        "passed": sum(clauses.values()),
        "total": len(clauses),
        "completed_fits": len(completed),
        "runtime_failures": runtime_failures,
        "metrics": {
            "teacher_action_nll_gain_over_v5": mean(action_nll_gain.values()),
            "teacher_action_accuracy_gain_over_v5": mean(accuracy_gain.values()),
            "joint_cross_entropy_gain_over_v5": mean(joint_ce_gain.values()),
            "joint_brier_gain_over_v5": mean(joint_brier_gain.values()),
            "residual_h1_nll_degradation": mean(h1_degradation.values()),
            "h2_h5_weighted_nll_gain_over_v22": weighted_gain,
            "h2_h5_positive_task_fraction": positive_task_fraction,
            "h2_h5_positive_seed_count": positive_seed_count,
            "worst_suite_nll_gain": worst_suite_gain,
            "legal_prediction_rate": legal_rate,
        },
        "counterevidence_retained": [
            "e5_open_vocabulary_v23_failed_recall_gate",
            "medium_capacity_v32_task_disjoint_overfit",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Large Hybrid World Model v45 formal result",
        "",
        f"Decision: `{decision}`",
        "",
        "| Frozen clause | Result |",
        "|---|---|",
        *[f"| {name} | {'PASS' if value else 'FAIL'} |" for name, value in clauses.items()],
        "",
        "The report retains all failed clauses and does not authorize post-result threshold changes.",
    ]
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"decision": decision, "passed": sum(clauses.values()), "total": len(clauses)}))


if __name__ == "__main__":
    main()
