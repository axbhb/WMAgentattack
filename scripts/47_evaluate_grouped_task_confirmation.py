"""Evaluate frozen Dreamer views on fresh grouped-task replay outcomes."""

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
FIXED_MODELS = (
    "clean_view",
    "injection_view",
    "dual_view",
    "symmetric_shrinkage_alpha_0_5",
)


def _load_metrics_module():
    path = ROOT / "scripts" / "38_evaluate_hierarchical_contrast_models.py"
    spec = importlib.util.spec_from_file_location("contrast_metrics", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import metrics from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


METRICS = _load_metrics_module()


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
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            tasks.add((str(row["domain"]), str(row["task_id"])))
    if not tasks:
        raise ValueError(f"No trajectory tasks in {path}")
    return tasks


def _fixed_predictions(
    rows: list[dict[str, Any]], *, seed: int | None = None
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    container = (
        "confirmation_predictions"
        if seed is None
        else "confirmation_seed_predictions"
    )
    predictions = {}
    for model in FIXED_MODELS:
        values = []
        for row in rows:
            source = row[container]
            if seed is not None:
                source = source[str(seed)]
            values.append(source[model])
        predictions[model] = (
            np.asarray(
                [value["attack_probability"] for value in values], dtype=float
            ),
            np.asarray(
                [value["utility_probability"] for value in values], dtype=float
            ),
        )
    return predictions


def _text_counterbaseline(
    development_rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    validation_tasks = {_task_key(row) for row in validation_rows}
    filtered = [
        row for row in development_rows if _task_key(row) not in validation_tasks
    ]
    if len({_task_key(row) for row in filtered}) < 8:
        raise ValueError("Too few task-disjoint development tasks for text baseline")
    combined = [*filtered, *validation_rows]
    _, _, _, attempts = METRICS._outcome_arrays(combined)
    matrix, _ = METRICS._feature_matrix(combined, "context_text")
    train = np.arange(len(filtered), dtype=int)
    valid = np.arange(len(filtered), len(combined), dtype=int)
    attack, utility, _ = METRICS._joint_predict(matrix, attempts, train, valid)
    return attack, utility, {
        "source_pair_count": len(development_rows),
        "task_disjoint_fit_pair_count": len(filtered),
        "excluded_overlap_pair_count": len(development_rows) - len(filtered),
        "fit_task_count": len({_task_key(row) for row in filtered}),
        "validation_task_overlap": 0,
    }


def _evaluate(
    rows: list[dict[str, Any]],
    predictions: dict[str, tuple[np.ndarray, np.ndarray]],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    attack_count, utility_count, trials, attempts = METRICS._outcome_arrays(rows)
    attack_rates = attack_count / trials
    utility_rates = utility_count / trials
    results = {
        name: METRICS._evaluate_model(
            rows,
            attempts,
            attack_rates,
            utility_rates,
            values[0],
            values[1],
        )
        for name, values in predictions.items()
    }

    def compare(left: str, right: str) -> dict[str, Any]:
        left_values = predictions[left]
        right_values = predictions[right]
        return METRICS._task_bootstrap_difference(
            rows,
            left_values[0],
            left_values[1],
            right_values[0],
            right_values[1],
            attack_rates,
            utility_rates,
            samples=bootstrap_samples,
            seed=bootstrap_seed,
        )

    comparisons = {
        "dual_view__minus__clean_view": compare("dual_view", "clean_view"),
        "injection_view__minus__clean_view": compare(
            "injection_view", "clean_view"
        ),
        "symmetric_shrinkage_alpha_0_5__minus__clean_view": compare(
            "symmetric_shrinkage_alpha_0_5", "clean_view"
        ),
    }
    if "task_disjoint_text_context" in predictions:
        comparisons["dual_view__minus__task_disjoint_text_context"] = compare(
            "dual_view", "task_disjoint_text_context"
        )
    return {"results": results, "comparisons": comparisons}


def _gate(
    comparison: dict[str, Any],
    thresholds: dict[str, Any],
    per_seed: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    point = float(comparison["pairwise_accuracy_difference"])
    lower = float(comparison["pairwise_accuracy_difference_95ci"][0])
    brier = float(comparison["brier_difference"])
    informative = int(comparison["informative_pairwise_task_count"])
    checks = {
        "pairwise_point": point
        >= float(thresholds["pairwise_point_difference_min"]),
        "pairwise_ci_lower": lower
        >= float(thresholds["pairwise_ci_lower_min"]),
        "brier_non_degradation": brier
        <= float(thresholds["brier_difference_max"]),
        "informative_tasks": informative
        >= int(thresholds["informative_task_count_min"]),
    }
    positive_seeds = sum(
        float(
            result["comparisons"]["dual_view__minus__clean_view"]
            ["pairwise_accuracy_difference"]
        )
        > 0.0
        for result in per_seed.values()
    )
    primary_pass = all(checks.values())
    strong_pass = (
        primary_pass
        and lower > 0.0
        and informative >= int(thresholds["strong_informative_task_count_min"])
    )
    return {
        "decision": (
            "CONFIRMED_STRONG"
            if strong_pass
            else "PARTIAL_GO"
            if primary_pass
            else "NO_GO"
        ),
        "primary_gate_pass": primary_pass,
        "strong_confirmation_pass": strong_pass,
        "checks": checks,
        "positive_seed_count": positive_seeds,
        "seed_count": len(per_seed),
        "seed_robustness_secondary_pass": positive_seeds
        >= int(thresholds["positive_seed_count_min_secondary"]),
        "thresholds": thresholds,
    }


def _outcome_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    attempts = sum(int(row["replay_attempt_count"]) for row in rows)
    attack = sum(int(row["attack_success_count"]) for row in rows)
    utility = sum(int(row["utility_success_count"]) for row in rows)
    task_attack_rates: dict[tuple[str, str], set[float]] = {}
    task_utility_rates: dict[tuple[str, str], set[float]] = {}
    for task in {_task_key(row) for row in rows}:
        task_rows = [row for row in rows if _task_key(row) == task]
        task_attack_rates[task] = {
            float(row["observed_attack_probability"]) for row in task_rows
        }
        task_utility_rates[task] = {
            float(row["observed_utility_probability"]) for row in task_rows
        }
    return {
        "attempt_count": attempts,
        "observed_asr": attack / attempts,
        "observed_bup": utility / attempts,
        "variable_attack_pair_count": sum(
            0 < int(row["attack_success_count"])
            < int(row["replay_attempt_count"])
            for row in rows
        ),
        "variable_utility_pair_count": sum(
            0 < int(row["utility_success_count"])
            < int(row["replay_attempt_count"])
            for row in rows
        ),
        "attack_informative_task_count": sum(
            len(values) > 1 for values in task_attack_rates.values()
        ),
        "utility_informative_task_count": sum(
            len(values) > 1 for values in task_utility_rates.values()
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--development-dataset", type=Path, required=True)
    parser.add_argument("--train-trajectories", type=Path, required=True)
    parser.add_argument("--test-trajectories", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=20000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260716)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    dataset = _load(args.dataset)
    selection = _load(args.selection)
    protocol = _load(args.protocol)
    rows = dataset.get("pairs")
    selected = selection.get("selections", {}).get(
        "grouped_task_confirmation"
    )
    if not isinstance(rows, list) or len(rows) != 32:
        raise ValueError("Confirmation dataset must contain 32 pairs")
    if not isinstance(selected, list) or len(selected) != 32:
        raise ValueError("Frozen selection must contain 32 pairs")
    if {_pair_key(row) for row in rows} != {_pair_key(row) for row in selected}:
        raise ValueError("Dataset pairs do not match frozen selection")
    if len({_task_key(row) for row in rows}) != 8:
        raise ValueError("Confirmation must contain eight tasks")
    if set(Counter(_task_key(row) for row in rows).values()) != {4}:
        raise ValueError("Each confirmation task must contain four pairs")
    if any(int(row["replay_attempt_count"]) != 5 for row in rows):
        raise ValueError("Every pair must contain five fresh outcomes")
    expected_sha = str(protocol["selection"]["sha256"])
    if _sha256(args.selection) != expected_sha:
        raise ValueError("Frozen selection SHA-256 does not match protocol")

    train_tasks = _trajectory_tasks(args.train_trajectories)
    test_tasks = _trajectory_tasks(args.test_trajectories)
    confirmation_tasks = {_task_key(row) for row in rows}
    if confirmation_tasks & train_tasks:
        raise ValueError("Confirmation user task leaked into grouped training")
    if not confirmation_tasks <= test_tasks:
        raise ValueError("Confirmation contains a non-test user task")

    ensemble_predictions = _fixed_predictions(rows)
    development_rows = _load(args.development_dataset).get("pairs")
    if not isinstance(development_rows, list) or not development_rows:
        raise ValueError("Development probability pairs are missing")
    text_attack, text_utility, text_metadata = _text_counterbaseline(
        development_rows, rows
    )
    ensemble_predictions["task_disjoint_text_context"] = (
        text_attack,
        text_utility,
    )
    ensemble = _evaluate(
        rows,
        ensemble_predictions,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    per_seed = {
        str(seed): _evaluate(
            rows,
            _fixed_predictions(rows, seed=seed),
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.bootstrap_seed + seed,
        )
        for seed in (7, 13, 21)
    }
    primary = ensemble["comparisons"]["dual_view__minus__clean_view"]
    gate = _gate(primary, protocol["decision_gate"], per_seed)
    output = {
        "scope": "fresh_grouped_unseen_task_confirmation",
        "status": "confirmatory_fresh_outcomes_fixed_models",
        "selection_sha256": expected_sha,
        "selection_uses_observed_labels": False,
        "fresh_outcome_count": 160,
        "world_model_training_task_overlap": 0,
        "confirmation_tasks_subset_of_grouped_test": True,
        "models_frozen_before_fresh_outcomes": list(FIXED_MODELS),
        "outcomes": _outcome_summary(rows),
        "ensemble": ensemble,
        "per_seed": per_seed,
        "task_disjoint_text_counterbaseline": text_metadata,
        "decision_gate": gate,
        "interpretation_constraints": [
            "This confirms fresh stochastic outcomes for fixed models, not a pristine task split: the architecture was developed after earlier exploratory labels on the grouped test split were inspected.",
            "The selected score-span/disagreement stress set estimates contrast ordering, not benchmark-wide prevalence.",
            "A general claim requires another untouched task group or cross-fitted outer folds if available.",
        ],
        "protocol": protocol,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.joinpath(
        "grouped_task_confirmation_summary.json"
    ).write_text(json.dumps(output, indent=2), encoding="utf-8")
    prediction_rows = []
    attack_count, utility_count, trials, _ = METRICS._outcome_arrays(rows)
    for index, row in enumerate(rows):
        prediction_rows.append(
            {
                "suite": row["suite"],
                "user_task_id": row["user_task_id"],
                "injection_task_id": row["injection_task_id"],
                "observed_attack_probability": float(attack_count[index] / trials[index]),
                "observed_utility_probability": float(utility_count[index] / trials[index]),
                "models": {
                    name: {
                        "attack_probability": float(values[0][index]),
                        "utility_probability": float(values[1][index]),
                    }
                    for name, values in ensemble_predictions.items()
                },
            }
        )
    args.output_dir.joinpath(
        "grouped_task_confirmation_predictions.json"
    ).write_text(
        json.dumps({"pairs": prediction_rows}, indent=2), encoding="utf-8"
    )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
