"""Evaluate a fixed shrinkage model on frozen pair-heldout replay outcomes."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def _load_model_module():
    path = ROOT / "scripts" / "38_evaluate_hierarchical_contrast_models.py"
    spec = importlib.util.spec_from_file_location("contrast_models", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import model evaluator from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MODELS = _load_model_module()


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _blend(clean: np.ndarray, injection: np.ndarray, alpha: float) -> np.ndarray:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    return np.clip(
        (1.0 - alpha) * clean + alpha * injection,
        MODELS.EPSILON,
        1.0 - MODELS.EPSILON,
    )


def _predict_model(
    model_name: str,
    rows: list[dict[str, Any]],
    attempts: list[list[tuple[int, int]]],
    train: np.ndarray,
    valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    model_type, feature_set = MODELS.MODEL_SPECS[model_name]
    matrix, _ = MODELS._feature_matrix(rows, feature_set)
    if model_type == "binary_ridge":
        return (
            MODELS._binary_ridge_predict(matrix, attempts, train, valid, 0),
            MODELS._binary_ridge_predict(matrix, attempts, train, valid, 1),
        )
    if model_type == "joint":
        attack, utility, _ = MODELS._joint_predict(
            matrix, attempts, train, valid
        )
        return attack, utility
    raise ValueError(f"Unsupported validation model {model_name}: {model_type}")


def _replication_status(comparison: dict[str, Any]) -> dict[str, Any]:
    pairwise = float(comparison["pairwise_accuracy_difference"])
    lower = float(comparison["pairwise_accuracy_difference_95ci"][0])
    brier = float(comparison["brier_difference"])
    brier_ok = brier <= 0.01
    return {
        "directional_replication": pairwise > 0.0 and brier_ok,
        "strong_replication": lower > 0.0 and brier_ok,
        "pairwise_direction_positive": pairwise > 0.0,
        "brier_non_degradation": brier_ok,
        "rule": (
            "directional: pairwise difference > 0 and Brier difference <= 0.01; "
            "strong: pairwise 95% CI lower > 0 and Brier difference <= 0.01"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-dataset", type=Path, required=True)
    parser.add_argument("--validation-dataset", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=20000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260714)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    training = _load(args.training_dataset).get("pairs")
    validation = _load(args.validation_dataset).get("pairs")
    protocol = _load(args.protocol)
    if not isinstance(training, list) or len(training) != 96:
        raise ValueError("Development dataset must contain 96 pairs")
    if not isinstance(validation, list) or len(validation) != 16:
        raise ValueError("Pair-heldout validation must contain 16 pairs")
    training_tasks = {MODELS._task_key(row) for row in training}
    validation_tasks = {MODELS._task_key(row) for row in validation}
    if len(training_tasks) != 24 or len(validation_tasks) != 8:
        raise ValueError("Unexpected development/validation task counts")
    if training_tasks & validation_tasks:
        raise ValueError("Validation user tasks overlap development tasks")
    training_pairs = {MODELS._pair_key(row) for row in training}
    validation_pairs = {MODELS._pair_key(row) for row in validation}
    if training_pairs & validation_pairs:
        raise ValueError("Validation injection pairs overlap development pairs")
    if any(int(row["replay_attempt_count"]) != 5 for row in validation):
        raise ValueError("Every validation pair must have five outcomes")

    alpha = float(protocol["development_disclosure"]["chosen_alpha"])
    if alpha != 0.5:
        raise ValueError("Frozen validation protocol requires alpha=0.5")
    combined = [*training, *validation]
    _, _, _, attempts = MODELS._outcome_arrays(combined)
    train = np.arange(len(training), dtype=int)
    valid = np.arange(len(training), len(combined), dtype=int)
    predictions = {
        name: _predict_model(name, combined, attempts, train, valid)
        for name in (
            "clean_world_ridge",
            "injection_world_ridge",
            "text_context_multinomial",
            "joint_text_multinomial",
        )
    }
    clean = predictions["clean_world_ridge"]
    injection = predictions["injection_world_ridge"]
    shrinkage_name = "shrinkage_injection_world_alpha_0_5"
    predictions[shrinkage_name] = (
        _blend(clean[0], injection[0], alpha),
        _blend(clean[1], injection[1], alpha),
    )
    predictions.update(MODELS._raw_models(validation))

    attack_count, utility_count, trials, validation_attempts = (
        MODELS._outcome_arrays(validation)
    )
    attack_rates = attack_count / trials
    utility_rates = utility_count / trials
    results = {
        name: MODELS._evaluate_model(
            validation,
            validation_attempts,
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
        return MODELS._task_bootstrap_difference(
            validation,
            left_values[0],
            left_values[1],
            right_values[0],
            right_values[1],
            attack_rates,
            utility_rates,
            samples=args.bootstrap_samples,
            seed=args.bootstrap_seed,
        )

    comparisons = {
        f"{shrinkage_name}__minus__clean_world_ridge": compare(
            shrinkage_name, "clean_world_ridge"
        ),
        f"{shrinkage_name}__minus__text_context_multinomial": compare(
            shrinkage_name, "text_context_multinomial"
        ),
        "injection_world_ridge__minus__clean_world_ridge": compare(
            "injection_world_ridge", "clean_world_ridge"
        ),
        "joint_text_multinomial__minus__text_context_multinomial": compare(
            "joint_text_multinomial", "text_context_multinomial"
        ),
    }
    primary_comparison = comparisons[
        f"{shrinkage_name}__minus__clean_world_ridge"
    ]
    replication = _replication_status(primary_comparison)
    prediction_rows = []
    for index, row in enumerate(validation):
        prediction_rows.append(
            {
                "suite": row["suite"],
                "user_task_id": row["user_task_id"],
                "injection_task_id": row["injection_task_id"],
                "observed_attack_probability": float(attack_rates[index]),
                "observed_utility_probability": float(utility_rates[index]),
                "models": {
                    name: {
                        "attack_probability": float(values[0][index]),
                        "utility_probability": float(values[1][index]),
                    }
                    for name, values in predictions.items()
                },
            }
        )
    output = {
        "scope": "pair_heldout_fixed_shrinkage_validation",
        "fixed_model": shrinkage_name,
        "alpha": alpha,
        "development_task_count": len(training_tasks),
        "validation_task_count": len(validation_tasks),
        "development_validation_task_overlap": 0,
        "development_validation_pair_overlap": 0,
        "validation_labels_used_for_fit_or_selection": False,
        "world_model_user_task_held_out": False,
        "protocol": protocol,
        "results": results,
        "comparisons": comparisons,
        "replication": replication,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.joinpath("pair_heldout_shrinkage_summary.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    args.output_dir.joinpath("pair_heldout_shrinkage_predictions.json").write_text(
        json.dumps({"pairs": prediction_rows}, indent=2), encoding="utf-8"
    )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
