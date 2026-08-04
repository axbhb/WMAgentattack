"""Refit the OOF-selected model and evaluate it on frozen task holdouts."""

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


def _fit_predict(
    model_name: str,
    rows: list[dict[str, Any]],
    attempts: list[list[tuple[int, int]]],
    attack_count: np.ndarray,
    utility_count: np.ndarray,
    trials: np.ndarray,
    train: np.ndarray,
    valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], list[str]]:
    model_type, feature_set = MODELS.MODEL_SPECS[model_name]
    matrix, feature_names = MODELS._feature_matrix(rows, feature_set)
    diagnostics: dict[str, Any] = {}
    if model_type == "binary_ridge":
        attack = MODELS._binary_ridge_predict(
            matrix, attempts, train, valid, 0
        )
        utility = MODELS._binary_ridge_predict(
            matrix, attempts, train, valid, 1
        )
    elif model_type == "hierarchical":
        attack, attack_diagnostics = MODELS._hierarchical_predict_compatible(
            matrix, attack_count, trials, rows, train, valid
        )
        utility, utility_diagnostics = MODELS._hierarchical_predict_compatible(
            matrix, utility_count, trials, rows, train, valid
        )
        diagnostics = {
            "attack": attack_diagnostics,
            "utility": utility_diagnostics,
        }
    elif model_type == "joint":
        attack, utility, _ = MODELS._joint_predict(
            matrix, attempts, train, valid
        )
    else:
        raise ValueError(f"Unknown model type {model_type!r}")
    return attack, utility, diagnostics, feature_names


def _replication_status(comparison: dict[str, Any]) -> dict[str, Any]:
    pairwise = float(comparison["pairwise_accuracy_difference"])
    lower = float(comparison["pairwise_accuracy_difference_95ci"][0])
    brier = float(comparison["brier_difference"])
    brier_ok = brier <= 0.01
    directional = pairwise > 0.0 and brier_ok
    strong = lower > 0.0 and brier_ok
    return {
        "directional_replication": directional,
        "strong_replication": strong,
        "claim_supported": directional,
        "rule": (
            "directional: pairwise difference > 0 and Brier difference <= 0.01; "
            "strong: pairwise 95% CI lower > 0 and Brier difference <= 0.01"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-dataset", type=Path, required=True)
    parser.add_argument("--confirmation-dataset", type=Path, required=True)
    parser.add_argument("--main-summary", type=Path, required=True)
    parser.add_argument("--gate-decision", type=Path, required=True)
    parser.add_argument("--analysis-protocol", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=20000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260713)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    training = _load(args.training_dataset).get("pairs")
    confirmation = _load(args.confirmation_dataset).get("pairs")
    summary = _load(args.main_summary)
    gate = _load(args.gate_decision)
    protocol = _load(args.analysis_protocol)
    if gate.get("go") is not True:
        raise ValueError("Confirmation evaluation requires a GO gate decision")
    if not isinstance(training, list) or len(training) != 96:
        raise ValueError("Training dataset must contain the frozen 96 pairs")
    if not isinstance(confirmation, list) or len(confirmation) != 32:
        raise ValueError("Confirmation dataset must contain the frozen 32 pairs")
    training_tasks = {MODELS._task_key(row) for row in training}
    confirmation_tasks = {MODELS._task_key(row) for row in confirmation}
    if len(training_tasks) != 24 or len(confirmation_tasks) != 8:
        raise ValueError("Unexpected task counts in training/confirmation data")
    if training_tasks & confirmation_tasks:
        raise ValueError("Confirmation user tasks overlap model-fitting tasks")
    if any(int(row["replay_attempt_count"]) != 5 for row in confirmation):
        raise ValueError("Every confirmation pair must have five outcomes")

    selected = str(summary["selected_model"])
    if selected not in MODELS.DEPLOYABLE_MODELS:
        raise ValueError(f"Main selected model is not deployable: {selected}")
    combined = [*training, *confirmation]
    attack_count, utility_count, trials, attempts = MODELS._outcome_arrays(
        combined
    )
    train = np.arange(len(training), dtype=int)
    valid = np.arange(len(training), len(combined), dtype=int)

    predictions = {}
    diagnostics = {}
    feature_names = {}
    for model_name in dict.fromkeys(
        (selected, "clean_world_ridge", "text_context_multinomial")
    ):
        attack, utility, model_diagnostics, names = _fit_predict(
            model_name,
            combined,
            attempts,
            attack_count,
            utility_count,
            trials,
            train,
            valid,
        )
        predictions[model_name] = (attack, utility)
        diagnostics[model_name] = model_diagnostics
        feature_names[model_name] = names
    predictions.update(MODELS._raw_models(confirmation))

    confirmation_attack, confirmation_utility, confirmation_trials, confirmation_attempts = (
        MODELS._outcome_arrays(confirmation)
    )
    attack_rates = confirmation_attack / confirmation_trials
    utility_rates = confirmation_utility / confirmation_trials
    results = {
        name: MODELS._evaluate_model(
            confirmation,
            confirmation_attempts,
            attack_rates,
            utility_rates,
            values[0],
            values[1],
        )
        for name, values in predictions.items()
    }
    selected_values = predictions[selected]
    comparator_values = predictions["clean_world_ridge"]
    comparison = MODELS._task_bootstrap_difference(
        confirmation,
        selected_values[0],
        selected_values[1],
        comparator_values[0],
        comparator_values[1],
        attack_rates,
        utility_rates,
        samples=args.bootstrap_samples,
        seed=args.bootstrap_seed,
    )
    text_values = predictions["text_context_multinomial"]
    text_comparison = MODELS._task_bootstrap_difference(
        confirmation,
        selected_values[0],
        selected_values[1],
        text_values[0],
        text_values[1],
        attack_rates,
        utility_rates,
        samples=args.bootstrap_samples,
        seed=args.bootstrap_seed,
    )
    replication = _replication_status(comparison)
    prediction_rows = []
    for index, row in enumerate(confirmation):
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
        "scope": "frozen_within_task_confirmation_evaluation",
        "selected_model_from_main_oof": selected,
        "comparator": "clean_world_ridge",
        "fit_task_count": len(training_tasks),
        "confirmation_task_count": len(confirmation_tasks),
        "task_overlap": 0,
        "confirmation_labels_used_for_fit_or_selection": False,
        "analysis_protocol": protocol,
        "results": results,
        "selected_minus_comparator": comparison,
        "selected_minus_text_context": text_comparison,
        "world_model_incremental_evidence": (
            "For a selected joint_text_multinomial model, improvement over "
            "text_context_multinomial isolates the incremental contribution "
            "of clean and injection-conditioned world-model features."
        ),
        "replication": replication,
        "fit_diagnostics": diagnostics,
        "feature_names": feature_names,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.joinpath("within_task_confirmation_summary.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    args.output_dir.joinpath("within_task_confirmation_predictions.json").write_text(
        json.dumps({"pairs": prediction_rows}, indent=2), encoding="utf-8"
    )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
