"""Apply the frozen v21 hard-label replacement and simplicity gate."""

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


def aggregate(rows: list[dict], suite: str, key: str) -> dict[str, float]:
    values = defaultdict(list)
    for row in rows:
        if row["split_suite"] == suite and row[key] is not None:
            values[row["arm"]].append(float(row[key]))
    return {arm: float(np.mean(items)) for arm, items in sorted(values.items())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    rows = metrics["runs"]
    gate = protocol["model_gate"]
    keys = (
        "hard_task_macro_bce",
        "hard_positive_task_macro_nll",
        "hard_positive_task_macro_recall",
        "unseen_positive_nll",
        "unseen_positive_recall",
        "execution_brier",
        "error_effect_positive_nll",
        "pair_assignment_accuracy",
        "v19_rollout_hard_bce",
        "v19_rollout_positive_nll",
        "parameter_count",
    )
    aggregates = {
        suite: {key: aggregate(rows, suite, key) for key in keys}
        for suite in ("task_disjoint", "tool_family_heldout", "source_heldout")
    }
    baseline = "structured_residual_v6"
    candidates = [
        "intervention_no_execution_experts_v21",
        "intervention_no_pair_v21",
        "intervention_modular_v20",
    ]
    candidate_checks = {}
    for candidate in candidates:
        primary = aggregates["task_disjoint"]
        tool = aggregates["tool_family_heldout"]
        source = aggregates["source_heldout"]
        checks = {
            "hard_bce_gain": primary["hard_task_macro_bce"][candidate] <= primary["hard_task_macro_bce"][baseline] - gate["minimum_hard_bce_gain"],
            "positive_nll_gain": primary["hard_positive_task_macro_nll"][candidate] <= primary["hard_positive_task_macro_nll"][baseline] - gate["minimum_positive_nll_gain"],
            "positive_recall_noninferiority": primary["hard_positive_task_macro_recall"][candidate] >= primary["hard_positive_task_macro_recall"][baseline] - gate["positive_recall_noninferiority_margin"],
            "rollout_bce_gain": primary["v19_rollout_hard_bce"][candidate] <= primary["v19_rollout_hard_bce"][baseline] - gate["minimum_rollout_bce_gain"],
            "execution_noninferiority": primary["execution_brier"][candidate] <= primary["execution_brier"][baseline] + gate["execution_brier_noninferiority_margin"],
            "tool_positive_nll_noninferiority": tool["hard_positive_task_macro_nll"][candidate] <= tool["hard_positive_task_macro_nll"][baseline] + gate["diagnostic_positive_nll_noninferiority_margin"],
            "tool_recall_noninferiority": tool["hard_positive_task_macro_recall"][candidate] >= tool["hard_positive_task_macro_recall"][baseline] - gate["diagnostic_recall_noninferiority_margin"],
            "source_positive_nll_noninferiority": source["hard_positive_task_macro_nll"][candidate] <= source["hard_positive_task_macro_nll"][baseline] + gate["diagnostic_positive_nll_noninferiority_margin"],
            "source_recall_noninferiority": source["hard_positive_task_macro_recall"][candidate] >= source["hard_positive_task_macro_recall"][baseline] - gate["diagnostic_recall_noninferiority_margin"],
            "tiny_parameter_budget": primary["parameter_count"][candidate] <= gate["maximum_parameters"],
        }
        fold_wins = 0
        for fold_marker in range(3):
            values = defaultdict(lambda: defaultdict(list))
            for row in rows:
                if row["split_suite"] == "task_disjoint" and row["fold_marker"] == fold_marker:
                    values[row["arm"]]["positive"].append(row["hard_positive_task_macro_nll"])
                    values[row["arm"]]["rollout"].append(row["v19_rollout_hard_bce"])
            if (
                np.mean(values[candidate]["positive"]) < np.mean(values[baseline]["positive"])
                and np.mean(values[candidate]["rollout"]) < np.mean(values[baseline]["rollout"])
            ):
                fold_wins += 1
        seed_wins = 0
        for seed in protocol["model_comparison"]["task_training_seeds"]:
            values = defaultdict(lambda: defaultdict(list))
            for row in rows:
                if row["split_suite"] == "task_disjoint" and row["seed"] == seed:
                    values[row["arm"]]["positive"].append(row["hard_positive_task_macro_nll"])
                    values[row["arm"]]["rollout"].append(row["v19_rollout_hard_bce"])
            if (
                np.mean(values[candidate]["positive"]) < np.mean(values[baseline]["positive"])
                and np.mean(values[candidate]["rollout"]) < np.mean(values[baseline]["rollout"])
            ):
                seed_wins += 1
        checks["fold_robustness"] = fold_wins >= gate["minimum_positive_folds"]
        checks["seed_robustness"] = seed_wins >= gate["minimum_positive_seeds"]
        candidate_checks[candidate] = {
            "checks": checks,
            "passed": all(checks.values()),
            "fold_wins": fold_wins,
            "seed_wins": seed_wins,
        }
    passing = [candidate for candidate in candidates if candidate_checks[candidate]["passed"]]
    selected = passing[0] if passing else None
    integrity = {
        "complete_fixed_budget": metrics["completed_runs"] == protocol["fixed_budget"]["maximum_model_fits_if_view_go"],
        "zero_runtime_failures": metrics["runtime_failures"] == 0,
    }
    passed = selected is not None and all(integrity.values())
    payload = {
        "schema_version": "wmagentattack.hard_label_gate.v21",
        "decision": f"GO_REPLACE_WITH_{selected.upper()}" if passed else "NO_GO_REPLACE_STRUCTURED_MARKOV",
        "passed": passed,
        "selected_architecture": selected,
        "simplicity_order": candidates,
        "integrity": integrity,
        "candidate_checks": candidate_checks,
        "aggregate": aggregates,
        "metrics_sha256": sha256(args.metrics),
        "counterevidence_retained": True,
        "post_result_reruns_authorized": False,
    }
    write(args.output, payload)
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
