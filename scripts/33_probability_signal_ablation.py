"""Diagnose whether replay calibration learns world scores or suite priors.

This is a validation-only counterevidence audit. It reuses the exact grouped
repeated-OOF folds and ridge logistic estimator from script 32, then compares:

* context prior: suite for attack; clean solvability plus suite for utility;
* world scores: Dreamer/candidate scores without suite identifiers;
* full model: the frozen feature set used by the deployed calibrator.

The audit never changes the already frozen test selection.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def _load_base():
    path = ROOT / "scripts" / "32_fit_replay_probability_calibrators.py"
    spec = importlib.util.spec_from_file_location("replay_probability_base", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import probability calibrator from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = _load_base()
VARIANTS = ("context_prior", "world_scores", "full")


def _reconstruct_attempts(
    rows: list[dict[str, Any]],
) -> tuple[list[list[tuple[int, int]]], np.ndarray, np.ndarray]:
    attempts = []
    security_rates = []
    utility_rates = []
    for row in rows:
        count = int(row["replay_attempt_count"])
        security_rate = float(row["replay_observed_attack_rate"])
        utility_rate = float(row["replay_observed_utility_rate"])
        security_count = int(round(security_rate * count))
        utility_count = int(round(utility_rate * count))
        attempts.append(
            [
                (int(index < security_count), int(index < utility_count))
                for index in range(count)
            ]
        )
        security_rates.append(security_count / count)
        utility_rates.append(utility_count / count)
    return (
        attempts,
        np.asarray(security_rates, dtype=float),
        np.asarray(utility_rates, dtype=float),
    )


def _variant_matrix(
    rows: list[dict[str, Any]],
    clean_rates: dict[tuple[str, str], float],
    *,
    head: str,
    variant: str,
) -> tuple[np.ndarray, list[str]]:
    full, names = BASE._feature_matrix(
        rows, clean_rates, head=head, model_name="ridge_logistic"
    )
    suite_start = next(
        index for index, name in enumerate(names) if name.startswith("suite_")
    )
    if variant == "full":
        indices = list(range(len(names)))
    elif variant == "context_prior":
        if head == "attack":
            indices = list(range(suite_start, len(names)))
        else:
            clean_index = names.index("logit_clean_solvability")
            indices = [clean_index, *range(suite_start, len(names))]
    elif variant == "world_scores":
        indices = list(range(suite_start))
        if head == "utility":
            indices.remove(names.index("logit_clean_solvability"))
    else:
        raise ValueError(f"Unknown signal-ablation variant: {variant}")
    return full[:, indices], [names[index] for index in indices]


def _crossfit_variant(
    rows: list[dict[str, Any]],
    clean_rates: dict[tuple[str, str], float],
    attempts: list[list[tuple[int, int]]],
    security_rates: np.ndarray,
    utility_rates: np.ndarray,
    *,
    variant: str,
    cv_seeds: list[int],
    n_splits: int,
) -> dict[str, Any]:
    groups = np.asarray(
        [f"{row['suite']}::{row['user_task_id']}" for row in rows]
    )
    output: dict[str, Any] = {"variant": variant, "heads": {}}
    for head_index, (head, rates) in enumerate(
        (("attack", security_rates), ("utility", utility_rates))
    ):
        matrix, names = _variant_matrix(
            rows, clean_rates, head=head, variant=variant
        )
        repeats = np.full((len(cv_seeds), len(rows)), np.nan)
        for repeat_index, seed in enumerate(cv_seeds):
            folds = BASE._make_folds(
                security_rates,
                utility_rates,
                groups,
                n_splits=n_splits,
                random_state=seed,
            )
            for train_index, valid_index in folds:
                train_x, train_y = BASE._expanded_training_data(
                    matrix, attempts, train_index, head_index
                )
                repeats[repeat_index, valid_index] = BASE._fit_predict_head(
                    "ridge_logistic", train_x, train_y, matrix[valid_index]
                )
        if np.isnan(repeats).any():
            raise AssertionError(f"Missing OOF prediction for {variant}/{head}")
        prediction = repeats.mean(axis=0)
        output["heads"][head] = {
            "feature_names": names,
            "metrics": BASE._probability_metrics(
                rates, prediction, attempts, head_index
            ),
            "mean_oof_uncertainty": float(repeats.std(axis=0).mean()),
            "prediction": prediction,
        }
    output["mean_pair_soft_brier"] = float(
        0.5
        * (
            output["heads"]["attack"]["metrics"]["pair_soft_brier"]
            + output["heads"]["utility"]["metrics"]["pair_soft_brier"]
        )
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-candidates", type=Path, required=True)
    parser.add_argument("--clean-solvability-json", type=Path, required=True)
    parser.add_argument("--cv-seeds", default="101,211,307,401,503")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    payload = json.loads(args.pilot_candidates.read_text(encoding="utf-8"))
    rows = payload.get("candidates")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"No pilot candidates found in {args.pilot_candidates}")
    attempts, security_rates, utility_rates = _reconstruct_attempts(rows)
    clean_rates = BASE._load_clean_rates(args.clean_solvability_json)
    cv_seeds = BASE._parse_ints(args.cv_seeds)

    results = {
        variant: _crossfit_variant(
            rows,
            clean_rates,
            attempts,
            security_rates,
            utility_rates,
            variant=variant,
            cv_seeds=cv_seeds,
            n_splits=args.folds,
        )
        for variant in VARIANTS
    }
    for index, row in enumerate(rows):
        for variant, result in results.items():
            for head in ("attack", "utility"):
                row[f"signal_ablation_{variant}_{head}_oof"] = float(
                    result["heads"][head]["prediction"][index]
                )

    full_brier = results["full"]["mean_pair_soft_brier"]
    summary = {
        "scope": "validation_only_probability_signal_ablation",
        "changes_frozen_test_selection": False,
        "pilot_candidates": str(args.pilot_candidates.resolve()),
        "grouping_unit": "suite_and_user_task_id",
        "cv_seeds": cv_seeds,
        "folds": args.folds,
        "results": {
            variant: {
                "mean_pair_soft_brier": result["mean_pair_soft_brier"],
                "delta_vs_full": result["mean_pair_soft_brier"] - full_brier,
                "attack": {
                    key: value
                    for key, value in result["heads"]["attack"].items()
                    if key != "prediction"
                },
                "utility": {
                    key: value
                    for key, value in result["heads"]["utility"].items()
                    if key != "prediction"
                },
            }
            for variant, result in results.items()
        },
        "candidates": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    printable = {**summary, "candidates": f"{len(rows)} rows omitted"}
    print(json.dumps(printable, indent=2))


if __name__ == "__main__":
    main()
