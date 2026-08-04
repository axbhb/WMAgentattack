"""Evaluate the frozen headwise method on seven tasks with fresh outcomes."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def _load_base_module():
    path = ROOT / "scripts" / "49_evaluate_grouped_train_hybrid.py"
    spec = importlib.util.spec_from_file_location("grouped_hybrid_base", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = _load_base_module()


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _task_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row["suite"]), str(row["user_task_id"])


def _pair_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (*_task_key(row), str(row["injection_task_id"]))


def _trajectory_tasks(path: Path) -> set[tuple[str, str]]:
    tasks = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                tasks.add((str(row["domain"]), str(row["task_id"])))
    return tasks


def _prediction_arrays(
    rows: list[dict[str, Any]], model: str
) -> dict[str, np.ndarray]:
    return {
        field: np.asarray(
            [row["confirmation_models"][model][field] for row in rows],
            dtype=float,
        )
        for field in (
            "attack_rank",
            "utility_rank",
            "attack_probability",
            "utility_probability",
        )
    }


def _outcome_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    attempts = sum(int(row["replay_attempt_count"]) for row in rows)
    attack = sum(int(row["attack_success_count"]) for row in rows)
    utility = sum(int(row["utility_success_count"]) for row in rows)
    return {
        "attempt_count": attempts,
        "observed_asr": attack / attempts,
        "observed_bup": utility / attempts,
        "variable_attack_pair_count": sum(
            0 < int(row["attack_success_count"]) < int(row["replay_attempt_count"])
            for row in rows
        ),
        "variable_utility_pair_count": sum(
            0 < int(row["utility_success_count"]) < int(row["replay_attempt_count"])
            for row in rows
        ),
    }


def _gate(
    comparisons: dict[str, dict[str, Any]],
    results: dict[str, dict[str, Any]],
    selected: str,
    thresholds: dict[str, Any],
) -> dict[str, Any]:
    checks = {}
    for reference in ("clean_raw", "text_pointwise"):
        comparison = comparisons[f"{selected}__minus__{reference}"]
        checks[f"pairwise_gain_vs_{reference}"] = (
            comparison["pairwise_accuracy_difference"]
            >= float(thresholds["pairwise_point_difference_min"])
        )
        checks[f"pairwise_ci_lower_vs_{reference}"] = (
            comparison["pairwise_accuracy_difference_95ci"][0]
            >= float(thresholds["pairwise_ci_lower_min"])
        )
        checks[f"brier_vs_{reference}"] = (
            comparison["brier_difference"]
            <= float(thresholds["brier_difference_max"])
        )
        checks[f"informative_tasks_vs_{reference}"] = (
            comparison["informative_pairwise_task_count"]
            >= int(thresholds["informative_task_count_min"])
        )
    attack_selected = results[selected]["attack"]["within_task"][
        "pairwise_accuracy"
    ]
    attack_text = results["text_pointwise"]["attack"]["within_task"][
        "pairwise_accuracy"
    ]
    utility_selected = results[selected]["utility"]["within_task"][
        "pairwise_accuracy"
    ]
    utility_clean = results["clean_raw"]["utility"]["within_task"][
        "pairwise_accuracy"
    ]
    checks["attack_head_not_worse_than_text"] = attack_selected >= attack_text
    checks["utility_head_matches_clean"] = abs(utility_selected - utility_clean) < 1e-12
    passed = all(checks.values())
    return {
        "decision": "GO" if passed else "NO_GO",
        "all_checks_pass": passed,
        "checks": checks,
        "thresholds": thresholds,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--train-trajectories", type=Path, required=True)
    parser.add_argument("--test-trajectories", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=20000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260719)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    dataset = _load(args.dataset)
    selection = _load(args.selection)
    protocol = _load(args.protocol)
    rows = dataset.get("pairs")
    selected_rows = selection.get("selections", {}).get(
        "headwise_remaining_confirmation"
    )
    if not isinstance(rows, list) or len(rows) != 28:
        raise ValueError("Fresh dataset must contain 28 pairs")
    if not isinstance(selected_rows, list) or len(selected_rows) != 28:
        raise ValueError("Frozen selection must contain 28 pairs")
    if {_pair_key(row) for row in rows} != {
        _pair_key(row) for row in selected_rows
    }:
        raise ValueError("Fresh dataset does not match frozen selection")
    if len({_task_key(row) for row in rows}) != 7:
        raise ValueError("Fresh dataset must contain seven tasks")
    if set(Counter(_task_key(row) for row in rows).values()) != {4}:
        raise ValueError("Every task must contain four pairs")
    if any(int(row["replay_attempt_count"]) != 5 for row in rows):
        raise ValueError("Every pair must contain five fresh outcomes")
    if _sha256(args.selection) != protocol["selection"]["sha256"]:
        raise ValueError("Selection hash does not match frozen protocol")
    train_tasks = _trajectory_tasks(args.train_trajectories)
    test_tasks = _trajectory_tasks(args.test_trajectories)
    evaluation_tasks = {_task_key(row) for row in rows}
    if evaluation_tasks & train_tasks or not evaluation_tasks <= test_tasks:
        raise ValueError("Grouped task audit failed")

    selection_mapping = {_pair_key(row): row for row in selected_rows}
    aligned = [
        {**row, "confirmation_models": selection_mapping[_pair_key(row)]["confirmation_models"]}
        for row in rows
    ]
    attack_rates = np.asarray(
        [row["observed_attack_probability"] for row in aligned], dtype=float
    )
    utility_rates = np.asarray(
        [row["observed_utility_probability"] for row in aligned], dtype=float
    )
    model_names = tuple(selection["fixed_models"])
    methods = {
        name: _prediction_arrays(aligned, name) for name in model_names
    }
    results = {
        name: BASE._method_metrics(
            aligned, attack_rates, utility_rates, method
        )
        for name, method in methods.items()
    }
    selected_model = str(selection["selected_model"])
    comparisons = {}
    for reference in ("clean_raw", "text_pointwise"):
        comparisons[f"{selected_model}__minus__{reference}"] = (
            BASE._bootstrap_difference(
                aligned,
                methods[selected_model],
                methods[reference],
                attack_rates,
                utility_rates,
                samples=args.bootstrap_samples,
                seed=args.bootstrap_seed,
            )
        )
    comparisons[
        "world_attack_clean_utility_text_probability__minus__text_pointwise"
    ] = BASE._bootstrap_difference(
        aligned,
        methods["world_attack_clean_utility_text_probability"],
        methods["text_pointwise"],
        attack_rates,
        utility_rates,
        samples=args.bootstrap_samples,
        seed=args.bootstrap_seed + 1,
    )
    gate = _gate(
        comparisons,
        results,
        selected_model,
        protocol["decision_gate"],
    )
    output = {
        "scope": "fresh_headwise_remaining_grouped_task_confirmation",
        "status": "fresh_outcomes_frozen_method",
        "selection_sha256": protocol["selection"]["sha256"],
        "selection_uses_observed_labels": False,
        "fresh_outcome_count": 140,
        "world_model_training_task_overlap": 0,
        "prior_0713_fresh_task_overlap": 0,
        "selected_model": selected_model,
        "outcomes": _outcome_summary(aligned),
        "results": results,
        "comparisons": comparisons,
        "decision_gate": gate,
        "protocol": protocol,
        "interpretation_constraints": [
            "Only the seven remaining grouped-test tasks provide new stochastic outcomes; architecture design used the prior eight-task replay.",
            "The task set is suite-imbalanced with four workspace tasks and one task from each other suite.",
            "The selected four-pair coverage per task is a ranking stress test, not an AgentDojo prevalence estimate.",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.joinpath("headwise_remaining_summary.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    prediction_rows = []
    for index, row in enumerate(aligned):
        prediction_rows.append(
            {
                "suite": row["suite"],
                "user_task_id": row["user_task_id"],
                "injection_task_id": row["injection_task_id"],
                "observed_attack_probability": float(attack_rates[index]),
                "observed_utility_probability": float(utility_rates[index]),
                "models": {
                    name: {
                        field: float(values[field][index])
                        for field in values
                    }
                    for name, values in methods.items()
                },
            }
        )
    args.output_dir.joinpath("headwise_remaining_predictions.json").write_text(
        json.dumps({"pairs": prediction_rows}, indent=2), encoding="utf-8"
    )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
