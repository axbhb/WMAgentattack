"""Apply the frozen uncertainty-ensemble E2 gate at task level."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from wmagentattack.multisource_suitability import file_sha256
from wmagentattack.multisource_suitability_experiment import paired_bootstrap, exact_sign_test


def read(path): return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def task_map(rows, metric):
    values = defaultdict(list)
    for row in rows: values[row["task_name"]].append(float(row[metric]))
    return {key: float(np.mean(rows)) for key, rows in values.items()}


def effect(left, right, metric, higher=False):
    by_group = {}
    for group in (7, 37, 67):
        a = task_map([row for row in left if int(row["ensemble_group"]) == group], metric)
        b = task_map([row for row in right if int(row["ensemble_group"]) == group], metric)
        if set(a) != set(b): raise ValueError("paired ensemble task mismatch")
        by_group[group] = {task: (b[task] - a[task] if higher else a[task] - b[task]) for task in a}
    tasks = {task: float(np.mean([by_group[group][task] for group in by_group])) for task in by_group[7]}
    groups = {str(group): float(np.mean(list(values.values()))) for group, values in by_group.items()}
    return {
        "mean": float(np.mean(list(groups.values()))), "groups": groups, "tasks": tasks,
        "positive": sum(value > 0 for value in tasks.values()) / len(tasks),
        "bootstrap": paired_bootstrap(list(tasks.values()), draws=10000, seed=818),
        "sign": exact_sign_test(list(tasks.values())),
    }


def macro(rows, metric):
    values = defaultdict(list)
    for row in rows: values[(row["ensemble_group"], row["task_name"])].append(float(row[metric]))
    return float(np.mean([np.mean(rows) for rows in values.values()]))


def epistemic_error_gap(rows):
    task_gaps = {}; group_gaps = {}
    for group in (7, 37, 67):
        grouped = defaultdict(list)
        for row in rows:
            if int(row["ensemble_group"]) == group:
                grouped[row["task_name"]].append(row)
        values = {}
        for task, task_rows in grouped.items():
            errors = [row["epistemic_mi"] for row in task_rows if not row["action_correct"]]
            correct = [row["epistemic_mi"] for row in task_rows if row["action_correct"]]
            if errors and correct: values[task] = float(np.mean(errors) - np.mean(correct))
        group_gaps[str(group)] = float(np.mean(list(values.values())))
        for task, value in values.items(): task_gaps.setdefault(task, []).append(value)
    task_gaps = {task: float(np.mean(values)) for task, values in task_gaps.items()}
    return {"mean": float(np.mean(list(group_gaps.values()))), "groups": group_gaps, "tasks": task_gaps, "positive": sum(value > 0 for value in task_gaps.values()) / len(task_gaps)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--run-metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args(); protocol = json.loads(args.protocol.read_text())
    rows = read(args.predictions); metrics = json.loads(args.run_metrics.read_text())
    baseline = [row for row in rows if row["arm"] == "mean_member_v6"]
    ensemble = [row for row in rows if row["arm"] == "uncertainty_ensemble_e2"]
    h1b = [row for row in baseline if row["horizon"] == 1]; h1e = [row for row in ensemble if row["horizon"] == 1]
    mb = [row for row in baseline if row["horizon"] >= 2]; me = [row for row in ensemble if row["horizon"] >= 2]
    jb = [row for row in mb if row["joint_trainable"]]; je = [row for row in me if row["joint_trainable"]]
    effects = {
        "h1_nll": effect(h1b, h1e, "action_nll"),
        "h1_accuracy": effect(h1b, h1e, "action_correct", True),
        "h2_h5_nll": effect(mb, me, "action_nll"),
        "future_joint_ce": effect(jb, je, "joint_ce"),
        "error_epistemic_gap": epistemic_error_gap(ensemble),
    }
    gate = protocol["stage_e2_direction_switch"]["gate"]
    checks = {
        "h1_nll_gain": effects["h1_nll"]["mean"] >= gate["minimum_h1_nll_gain_vs_mean_member"],
        "h1_accuracy_noninferiority": effects["h1_accuracy"]["mean"] >= -gate["maximum_h1_accuracy_degradation_vs_mean_member"],
        "h2_h5_nll_gain": effects["h2_h5_nll"]["mean"] >= gate["minimum_h2_h5_nll_gain_vs_mean_member"],
        "future_joint_ce_gain": effects["future_joint_ce"]["mean"] >= gate["minimum_future_joint_ce_gain_vs_mean_member"],
        "task_breadth": effects["h2_h5_nll"]["positive"] >= gate["minimum_positive_task_fraction"],
        "group_replication": sum(value >= gate["minimum_h2_h5_nll_gain_vs_mean_member"] for value in effects["h2_h5_nll"]["groups"].values()) >= gate["minimum_positive_ensemble_groups"],
        "epistemic_error_separation": effects["error_epistemic_gap"]["mean"] >= gate["minimum_error_epistemic_gap"],
        "uniform_label_blind_mixture": metrics["uniform_member_weights"] and metrics["confirmation_tuned_parameters"] == 0,
        "all_legal": all(row["legal_prediction"] == 1 for row in ensemble),
        "complete_budget": metrics["teacher_fits"] == metrics["residual_fits"] == 45 and metrics["ensemble_evaluations"] == 15 and metrics["runtime_failures"] == 0,
    }
    decision = "GO_UNCERTAINTY_ENSEMBLE_E2" if all(checks.values()) else "NO_GO_UNCERTAINTY_ENSEMBLE_E2"
    summary = {
        "protocol_id": protocol["protocol_id"], "decision": decision,
        "checks": checks, "effects": effects,
        "absolute": {
            "member_h1_nll": macro(h1b, "action_nll"), "ensemble_h1_nll": macro(h1e, "action_nll"),
            "member_h1_accuracy": macro(h1b, "action_correct"), "ensemble_h1_accuracy": macro(h1e, "action_correct"),
            "member_h2_h5_nll": macro(mb, "action_nll"), "ensemble_h2_h5_nll": macro(me, "action_nll"),
            "member_future_joint_ce": macro(jb, "joint_ce"), "ensemble_future_joint_ce": macro(je, "joint_ce"),
            "mean_epistemic_mi": float(np.mean([row["epistemic_mi"] for row in ensemble])),
        },
        "predictions_sha256": file_sha256(args.predictions),
        "run_metrics_sha256": file_sha256(args.run_metrics),
    }
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    args.markdown.write_text("# Uncertainty ensemble E2\n\nDecision: `" + decision + "`\n\n" + "\n".join(f"- {key}: {'PASS' if value else 'FAIL'}" for key, value in checks.items()) + "\n")


if __name__ == "__main__":
    main()
