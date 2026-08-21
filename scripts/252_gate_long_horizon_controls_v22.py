"""Apply the frozen H1-H5 long-horizon gate and retain H10 counterevidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def task_surface(rows: list[dict], control: str, horizons: set[int]) -> dict[tuple[int, int, str], dict[str, float]]:
    grouped = defaultdict(lambda: defaultdict(list))
    for row in rows:
        row_control = row.get("control", row.get("arm"))
        if row_control != control or int(row["horizon"]) not in horizons:
            continue
        key = (int(row["training_seed"]), int(row["horizon"]), str(row["task_name"]))
        grouped[key]["nll"].append(float(row["action_nll"]))
        grouped[key]["accuracy"].append(float(row["action_correct"]))
        grouped[key]["legal"].append(float(row["legal_prediction"]))
    return {
        key: {metric: float(np.mean(values)) for metric, values in metrics.items()}
        for key, metrics in grouped.items()
    }


def paired_gain(left, right, metric: str) -> tuple[list[float], dict[int, list[float]], dict[str, list[float]]]:
    keys = sorted(set(left) & set(right))
    values = [left[key][metric] - right[key][metric] for key in keys]
    by_seed = defaultdict(list)
    by_task = defaultdict(list)
    for key in keys:
        gain = left[key][metric] - right[key][metric]
        by_seed[key[0]].append(gain)
        by_task[key[2]].append(gain)
    return values, by_seed, by_task


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--data-gate", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--v4-predictions", type=Path, required=True)
    parser.add_argument("--open-gate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    cfg = protocol["long_horizon_gate"]
    data_gate = json.loads(args.data_gate.read_text(encoding="utf-8"))
    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    open_gate = json.loads(args.open_gate.read_text(encoding="utf-8"))
    if data_gate["decision"] != "GO_LONG_HORIZON_MODEL_GATE_V22":
        raise ValueError("long-horizon model run was not authorized by data")
    if sha256(args.predictions) != metrics["predictions_sha256"]:
        raise ValueError("long-horizon prediction hash mismatch")
    if sha256(args.v4_predictions) != cfg["external_typed_v4_control"]["sha256"]:
        raise ValueError("typed v4 external control hash mismatch")
    rows = read_jsonl(args.predictions)
    v4_rows = read_jsonl(args.v4_predictions)
    repeated_h1 = task_surface(rows, "one_step_repeated", {1})
    candidate_h1 = task_surface(rows, "free_latent_residual", {1})
    free_h2_h5 = task_surface(rows, "free_latent_residual", {2, 3, 4, 5})
    teacher_h2_h5 = task_surface(rows, "teacher_forced_residual", {2, 3, 4, 5})
    v4_h2_h5 = task_surface(v4_rows, cfg["external_typed_v4_control"]["arm"], {2, 3, 4, 5})
    h1_nll, _, _ = paired_gain(repeated_h1, candidate_h1, "nll")
    h1_accuracy, _, _ = paired_gain(candidate_h1, repeated_h1, "accuracy")
    multistep, by_seed, by_task = paired_gain(v4_h2_h5, free_h2_h5, "nll")
    repeated_all = task_surface(rows, "one_step_repeated", {1, 2, 3, 5, 10})
    teacher_all = task_surface(rows, "teacher_forced_residual", {1, 2, 3, 5, 10})
    free_all = task_surface(rows, "free_latent_residual", {1, 2, 3, 5, 10})
    diagnostics = {}
    for horizon in (1, 2, 3, 5, 10):
        rep = {key: value for key, value in repeated_all.items() if key[1] == horizon}
        teacher = {key: value for key, value in teacher_all.items() if key[1] == horizon}
        free = {key: value for key, value in free_all.items() if key[1] == horizon}
        accumulation, _, _ = paired_gain(free, rep, "nll")
        teacher_gap, _, _ = paired_gain(free, teacher, "nll")
        diagnostics[str(horizon)] = {
            "task_seed_units": len(free),
            "free_minus_repeated_nll": float(np.mean(accumulation)) if accumulation else None,
            "free_minus_teacher_forced_nll": float(np.mean(teacher_gap)) if teacher_gap else None,
        }
    thresholds = cfg["model_gate_if_data_go"]
    legal = [float(row["legal_prediction"]) for row in rows if row["control"] == "free_latent_residual" and int(row["horizon"]) <= 5]
    clauses = {
        "complete_paired_fit_budget": metrics["completed_paired_fit_units"] == cfg["fixed_budget_if_data_go"]["agentdojo_residual_fits"],
        "zero_runtime_failures": metrics["runtime_failures"] == 0,
        "h1_nll_noninferiority": float(np.mean(h1_nll)) >= -thresholds["maximum_h1_nll_degradation_vs_teacher"],
        "h1_accuracy_noninferiority": float(np.mean(h1_accuracy)) >= -thresholds["maximum_h1_accuracy_degradation_vs_teacher"],
        "h2_h5_nll_gain_over_typed_v4": float(np.mean(multistep)) >= thresholds["minimum_h2_h5_nll_gain_over_typed_v4"],
        "h2_h5_positive_task_fraction": float(np.mean([np.mean(values) > 0 for values in by_task.values()])) >= thresholds["minimum_h2_h5_positive_task_fraction"],
        "positive_seeds": sum(np.mean(values) > 0 for values in by_seed.values()) >= thresholds["minimum_positive_seeds"],
        "all_predictions_legal": bool(legal) and min(legal) == 1.0,
        "v19_effect_rollout_noninferiority": open_gate["clauses"]["rollout_bce_noninferiority"],
        "query_read_recall_noninferiority": open_gate["clauses"]["query_read_recall_noninferiority"],
    }
    decision = "GO_LONG_HORIZON_H1_H5_V22" if all(clauses.values()) else "NO_GO_LONG_HORIZON_H1_H5_V22"
    payload = {
        "schema_version": "wmagentattack.long_horizon_gate.v22",
        "decision": decision,
        "clauses": clauses,
        "passed": sum(clauses.values()),
        "total": len(clauses),
        "metrics": {
            "h1_nll_gain_vs_teacher": float(np.mean(h1_nll)),
            "h1_accuracy_gain_vs_teacher": float(np.mean(h1_accuracy)),
            "h2_h5_nll_gain_over_typed_v4": float(np.mean(multistep)),
            "h2_h5_positive_task_fraction": float(np.mean([np.mean(values) > 0 for values in by_task.values()])),
            "positive_seeds": sum(np.mean(values) > 0 for values in by_seed.values()),
            "rollout_diagnostics": diagnostics,
        },
        "counterevidence": {
            "h10_is_not_a_gate": true,
            "h10_reason": "support spans only 9/20 tasks and one frozen fold has zero windows",
            "teacher_forced_is_not_free_rollout": true,
        },
        "hashes": {
            "protocol": sha256(args.protocol),
            "data_gate": sha256(args.data_gate),
            "metrics": sha256(args.metrics),
            "predictions": sha256(args.predictions),
            "v4_predictions": sha256(args.v4_predictions),
            "open_gate": sha256(args.open_gate),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"decision": decision, "passed": sum(clauses.values()), "total": len(clauses)}))


if __name__ == "__main__":
    main()
