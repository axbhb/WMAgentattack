"""Apply the frozen v20 model acceptance gate without result-dependent tuning."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def means(rows: list[dict], key: str) -> dict[str, float]:
    grouped = defaultdict(list)
    for row in rows:
        if row[key] is not None:
            grouped[row["arm"]].append(float(row[key]))
    return {arm: float(np.mean(values)) for arm, values in sorted(grouped.items())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    cfg = protocol["model_comparison"]
    gate_cfg = protocol["model_gate"]
    rows = metrics["runs"]
    aggregate = {
        key: means(rows, key)
        for key in (
            "one_step_task_macro_bce",
            "one_step_micro_f1",
            "execution_brier",
            "execution_accuracy",
            "pair_assignment_accuracy",
            "v19_rollout_task_macro_bce",
            "parameter_count",
        )
    }
    candidate = "intervention_modular_v20"
    baselines = ["structured_markov_v3", "structured_residual_v6"]
    best_one_step = min(aggregate["one_step_task_macro_bce"][arm] for arm in baselines)
    best_execution = min(aggregate["execution_brier"][arm] for arm in baselines)
    best_pair = max(aggregate["pair_assignment_accuracy"][arm] for arm in baselines)
    v6_rollout = aggregate["v19_rollout_task_macro_bce"]["structured_residual_v6"]
    checks = {
        "complete_fixed_budget": (
            metrics["completed_runs"] == protocol["fixed_budget"]["maximum_model_fits_if_union_go"]
            and metrics["runtime_failures"] == 0
        ),
        "tiny_parameter_budget": max(aggregate["parameter_count"].values()) <= gate_cfg["maximum_parameters"],
        "one_step_noninferiority": aggregate["one_step_task_macro_bce"][candidate] <= best_one_step + gate_cfg["one_step_bce_noninferiority_margin"],
        "execution_noninferiority": aggregate["execution_brier"][candidate] <= best_execution + gate_cfg["execution_brier_noninferiority_margin"],
        "pair_absolute_floor": aggregate["pair_assignment_accuracy"][candidate] >= gate_cfg["pair_assignment_absolute_floor"],
        "pair_noninferiority": aggregate["pair_assignment_accuracy"][candidate] >= best_pair - gate_cfg["pair_assignment_noninferiority_margin"],
        "v19_rollout_improvement": aggregate["v19_rollout_task_macro_bce"][candidate] <= v6_rollout - gate_cfg["minimum_v19_rollout_bce_gain"],
    }
    fold_wins = 0
    for fold in range(protocol["split_contract"]["folds"]):
        values = defaultdict(list)
        for row in rows:
            if row["fold"] == fold:
                values[row["arm"]].append(row["v19_rollout_task_macro_bce"])
        if np.mean(values[candidate]) < np.mean(values["structured_residual_v6"]):
            fold_wins += 1
    seed_wins = 0
    for seed in cfg["training_seeds"]:
        values = defaultdict(lambda: defaultdict(list))
        for row in rows:
            if row["seed"] == seed:
                values[row["arm"]]["rollout"].append(row["v19_rollout_task_macro_bce"])
                values[row["arm"]]["one_step"].append(row["one_step_task_macro_bce"])
        candidate_rollout = np.mean(values[candidate]["rollout"])
        v6_seed_rollout = np.mean(values["structured_residual_v6"]["rollout"])
        best_seed_one = min(np.mean(values[arm]["one_step"]) for arm in baselines)
        candidate_one = np.mean(values[candidate]["one_step"])
        if candidate_rollout < v6_seed_rollout and candidate_one <= best_seed_one + gate_cfg["one_step_bce_noninferiority_margin"]:
            seed_wins += 1
    checks["fold_robustness"] = fold_wins >= gate_cfg["minimum_positive_folds"]
    checks["seed_robustness"] = seed_wins >= gate_cfg["minimum_positive_seeds"]
    passed = all(checks.values())
    payload = {
        "schema_version": "wmagentattack.intervention_model_gate.v20",
        "decision": "GO_INTERVENTION_MODULAR_V20" if passed else "NO_GO_INTERVENTION_MODULAR_V20",
        "passed": passed,
        "checks": checks,
        "aggregate": aggregate,
        "fold_wins": fold_wins,
        "seed_wins": seed_wins,
        "metrics_sha256": sha256(args.metrics),
        "counterevidence_retained": True,
        "post_result_reruns_authorized": False,
    }
    write(args.output, payload)
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
