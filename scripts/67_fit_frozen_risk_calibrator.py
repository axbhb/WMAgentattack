"""Fit monotonic risk calibrators on frozen DreamerV3 predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wmagentattack.full_dreamer_v3 import (
    FullSheepRLDreamerV3,
    evaluate_full_dreamer_predictions,
)
from wmagentattack.io_utils import read_jsonl
from wmagentattack.risk_calibration import (
    build_grouped_risk_data,
    fit_monotonic_affine_group_calibrator,
)
from wmagentattack.schema import StepRecord


REGULARIZATION_GRID = {
    "reg0": 0.0,
    "reg1e-4": 1e-4,
    "reg1e-3": 1e-3,
    "reg1e-2": 1e-2,
    "reg1e-1": 1e-1,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _steps(path: Path) -> list[StepRecord]:
    return [StepRecord.model_validate(row) for row in read_jsonl(path)]


def _metrics(model, steps, predictions):
    return evaluate_full_dreamer_predictions(
        steps,
        predictions,
        validation_risk_mode=model.config.validation_risk_mode,
        validation_utility_mode=model.config.validation_utility_mode,
        validation_aggregation=model.config.validation_aggregation,
    )


def _invariants(raw: dict, calibrated: dict) -> dict:
    non_risk_deltas = {
        key: abs(float(calibrated[key]) - float(value))
        for key, value in raw.items()
        if isinstance(value, (int, float))
        and calibrated.get(key) is not None
        and "risk" not in key
        and not key.startswith("validation_objective")
    }
    result = {
        "risk_auc_delta": float(calibrated["risk_auc"] - raw["risk_auc"]),
        "non_risk_max_abs_delta": max(non_risk_deltas.values(), default=0.0),
        "non_risk_deltas": non_risk_deltas,
    }
    if abs(result["risk_auc_delta"]) > 1e-12:
        raise RuntimeError("Strictly monotonic calibration changed binary risk AUC")
    if result["non_risk_max_abs_delta"] > 1e-12:
        raise RuntimeError("Frozen calibration changed a non-risk metric")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--val", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    train_steps = _steps(args.train)
    val_steps = _steps(args.val)
    model = FullSheepRLDreamerV3.load(args.model)
    train_predictions = model.predict(train_steps)
    val_predictions = model.predict(val_steps)
    raw_val_metrics = _metrics(model, val_steps, val_predictions)
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "identity_val_metrics.json").write_text(
        json.dumps({"metrics": raw_val_metrics}, indent=2), encoding="utf-8"
    )
    grouped_train = build_grouped_risk_data(
        train_steps, train_predictions["risk_score"]
    )

    candidates = {}
    for label, regularization in REGULARIZATION_GRID.items():
        calibrator, fit_stats = fit_monotonic_affine_group_calibrator(
            grouped_train, regularization=regularization
        )
        calibrated_predictions = dict(val_predictions)
        calibrated_predictions["risk_score"] = calibrator.transform(
            val_predictions["risk_score"]
        )
        calibrated_metrics = _metrics(
            model, val_steps, calibrated_predictions
        )
        invariants = _invariants(raw_val_metrics, calibrated_metrics)
        output = args.output_root / label
        output.mkdir(parents=True, exist_ok=True)
        calibrator_payload = {
            "calibrator": calibrator.to_dict(),
            "fit": fit_stats,
            "model": str(args.model.resolve()),
            "model_metadata_sha256": _sha256(args.model / "metadata.json"),
            "train_sha256": _sha256(args.train),
            "validation_sha256": _sha256(args.val),
        }
        (output / "calibrator.json").write_text(
            json.dumps(calibrator_payload, indent=2), encoding="utf-8"
        )
        (output / "val_metrics.json").write_text(
            json.dumps(
                {
                    "metrics": calibrated_metrics,
                    "raw_metrics": raw_val_metrics,
                    "invariants": invariants,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        candidates[label] = {
            "regularization": regularization,
            "calibrator": calibrator.to_dict(),
            "fit": fit_stats,
            "validation_objective": calibrated_metrics[
                "validation_objective"
            ],
            "grouped_risk_probability_brier_score": calibrated_metrics[
                "grouped_risk_probability_brier_score"
            ],
            "invariants": invariants,
        }

    result = {
        "scope": "frozen_monotonic_affine_risk_calibration_validation",
        "model": str(args.model.resolve()),
        "train_steps": len(train_steps),
        "validation_steps": len(val_steps),
        "identity_validation_objective": raw_val_metrics["validation_objective"],
        "identity_grouped_risk_probability_brier_score": raw_val_metrics[
            "grouped_risk_probability_brier_score"
        ],
        "candidates": candidates,
    }
    (args.output_root / "fit_summary.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
