"""Fit train-only monotonic utility calibration and freeze it on validation."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wmagentattack.full_dreamer_v3 import (
    FullSheepRLDreamerV3,
    evaluate_full_dreamer_predictions,
)
from wmagentattack.risk_calibration import (
    GroupedRiskData,
    MonotonicAffineRiskCalibrator,
    fit_monotonic_affine_group_calibrator,
    grouped_risk_brier,
)


BASE_PATH = ROOT / "scripts" / "70_evaluate_v2_downstream_selection.py"
SPEC = importlib.util.spec_from_file_location("v2_downstream", BASE_PATH)
BASE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(BASE)

REGULARIZATIONS = (0.0, 0.001, 0.01, 0.1, 1.0)


def _grouped_utility_data(
    steps: list[Any], utility_scores: np.ndarray, *, decision_step: str = "first"
) -> GroupedRiskData:
    scores = np.asarray(utility_scores, dtype=np.float64)
    if len(scores) != len(steps):
        raise ValueError("utility scores must align with steps")
    indices = BASE._decision_indices(steps, decision_step)
    grouped: dict[str, list[int]] = {}
    for index in indices.values():
        step = steps[index]
        group_id = str(step.multiseed_group_id or "")
        if group_id.startswith("attack::") and step.utility_probability_target is not None:
            grouped.setdefault(group_id, []).append(index)
    if not grouped:
        raise ValueError("No complete attack groups with utility targets")
    group_ids = []
    score_groups = []
    targets = []
    for group_id in sorted(grouped):
        members = grouped[group_id]
        expected = {int(steps[index].multiseed_trials or 0) for index in members}
        if len(expected) != 1 or len(members) != next(iter(expected)):
            raise ValueError(f"Incomplete utility group: {group_id}")
        target_values = {
            float(steps[index].utility_probability_target) for index in members
        }
        if len(target_values) != 1:
            raise ValueError(f"Inconsistent utility target: {group_id}")
        group_ids.append(group_id)
        score_groups.append(scores[members].copy())
        targets.append(next(iter(target_values)))
    return GroupedRiskData(
        group_ids=tuple(group_ids),
        score_groups=tuple(score_groups),
        targets=np.asarray(targets, dtype=np.float64),
    )


def _calibrated_predictions(
    predictions: dict[str, Any], calibrator: MonotonicAffineRiskCalibrator
) -> dict[str, Any]:
    output = dict(predictions)
    output["utility_score"] = calibrator.transform(predictions["utility_score"])
    return output


def _downstream(
    *,
    val_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    risk_key: str,
    budgets: tuple[int, ...],
) -> dict[str, Any]:
    result = {}
    for budget in budgets:
        validation = BASE._validation_select(
            {7: val_rows}, risk_key=risk_key, budget=budget
        )
        recipe = validation["selected_recipe"]
        selected = BASE._select(
            test_rows,
            risk_key=risk_key,
            recipe=recipe,
            budget_per_task=budget,
        )
        result[str(budget)] = {
            "validation": validation,
            "frozen_validation_recipe": recipe,
            "test": BASE._metrics(selected),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--decision-step", choices=["first", "final"], default="first")
    parser.add_argument("--budgets", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    steps = {
        split: BASE._steps(args.data_root / f"{split}_steps.jsonl")
        for split in ("train", "val", "test")
    }
    model = FullSheepRLDreamerV3.load(args.model)
    raw_predictions = {split: model.predict(values) for split, values in steps.items()}

    train_data = _grouped_utility_data(
        steps["train"],
        raw_predictions["train"]["utility_score"],
        decision_step=args.decision_step,
    )
    val_data = _grouped_utility_data(
        steps["val"],
        raw_predictions["val"]["utility_score"],
        decision_step=args.decision_step,
    )
    test_data = _grouped_utility_data(
        steps["test"],
        raw_predictions["test"]["utility_score"],
        decision_step=args.decision_step,
    )

    candidates: dict[str, MonotonicAffineRiskCalibrator] = {
        "identity": MonotonicAffineRiskCalibrator()
    }
    fit_stats = {}
    for regularization in REGULARIZATIONS:
        name = f"reg{regularization:g}"
        candidates[name], fit_stats[name] = fit_monotonic_affine_group_calibrator(
            train_data, regularization=regularization
        )
    validation_brier = {
        name: grouped_risk_brier(val_data, calibrator)
        for name, calibrator in candidates.items()
    }
    order = {name: index for index, name in enumerate(candidates)}
    selected_name = min(
        candidates, key=lambda name: (validation_brier[name], order[name])
    )
    selected = candidates[selected_name]

    calibrated_predictions = {
        split: _calibrated_predictions(predictions, selected)
        for split, predictions in raw_predictions.items()
    }
    risk_calibrator, risk_fit = BASE._fit_decision_time_calibrator(
        steps["train"], raw_predictions["train"]
    )
    risk_candidates = {
        "identity": MonotonicAffineRiskCalibrator(),
        "reg0": risk_calibrator,
    }
    risk_validation = {}
    raw_rows_by_risk = {}
    for name, risk_candidate in risk_candidates.items():
        rows = BASE._configuration_rows(
            steps["val"],
            raw_predictions["val"],
            risk_candidate,
            decision_step=args.decision_step,
        )
        raw_rows_by_risk[name] = rows
        risk_validation[name] = BASE._group_risk_brier(
            rows,
            "raw_risk_score" if name == "identity" else "calibrated_risk_score",
        )
    selected_risk_name = min(risk_validation, key=risk_validation.get)
    selected_risk = risk_candidates[selected_risk_name]
    risk_key = (
        "raw_risk_score"
        if selected_risk_name == "identity"
        else "calibrated_risk_score"
    )

    rows = {}
    for mode, predictions_by_split in (
        ("raw", raw_predictions),
        ("calibrated", calibrated_predictions),
    ):
        rows[mode] = {
            split: BASE._configuration_rows(
                steps[split],
                predictions_by_split[split],
                selected_risk,
                decision_step=args.decision_step,
            )
            for split in ("val", "test")
        }
    downstream = {
        mode: _downstream(
            val_rows=split_rows["val"],
            test_rows=split_rows["test"],
            risk_key=risk_key,
            budgets=tuple(args.budgets),
        )
        for mode, split_rows in rows.items()
    }

    result = {
        "scope": "train-fit validation-selected monotonic utility calibration",
        "protocol": {
            "decision_step": args.decision_step,
            "fit_split": "train",
            "selection_split": "validation",
            "test_retuning": False,
            "regularizations": list(REGULARIZATIONS),
            "strictly_monotonic": True,
        },
        "provenance": {
            "data_root": str(args.data_root.resolve()),
            "model": str(args.model.resolve()),
        },
        "fit_stats": fit_stats,
        "validation_grouped_utility_brier": validation_brier,
        "selected_utility_calibrator": selected_name,
        "utility_calibrator": selected.to_dict(),
        "grouped_utility_brier": {
            "train_raw": grouped_risk_brier(train_data),
            "train_calibrated": grouped_risk_brier(train_data, selected),
            "val_raw": grouped_risk_brier(val_data),
            "val_calibrated": grouped_risk_brier(val_data, selected),
            "test_raw": grouped_risk_brier(test_data),
            "test_calibrated": grouped_risk_brier(test_data, selected),
        },
        "risk_calibration": {
            "fit": risk_fit,
            "validation_brier": risk_validation,
            "selected": selected_risk_name,
        },
        "prediction_metrics": {
            mode: {
                split: evaluate_full_dreamer_predictions(
                    steps[split],
                    predictions_by_split[split],
                    validation_risk_mode=model.config.validation_risk_mode,
                    validation_utility_mode=model.config.validation_utility_mode,
                    validation_aggregation=model.config.validation_aggregation,
                    validation_group_step=args.decision_step,
                )
                for split in ("val", "test")
            }
            for mode, predictions_by_split in (
                ("raw", raw_predictions),
                ("calibrated", calibrated_predictions),
            )
        },
        "downstream": downstream,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
