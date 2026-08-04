"""Evaluate a validation-selected calibrator on a frozen Dreamer checkpoint."""

from __future__ import annotations

import argparse
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
from wmagentattack.risk_calibration import MonotonicAffineRiskCalibrator
from wmagentattack.schema import StepRecord


def _metrics(model, steps, predictions):
    return evaluate_full_dreamer_predictions(
        steps,
        predictions,
        validation_risk_mode=model.config.validation_risk_mode,
        validation_utility_mode=model.config.validation_utility_mode,
        validation_aggregation=model.config.validation_aggregation,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--calibrator", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    steps = [StepRecord.model_validate(row) for row in read_jsonl(args.test)]
    model = FullSheepRLDreamerV3.load(args.model)
    predictions = model.predict(steps)
    raw_metrics = _metrics(model, steps, predictions)
    if args.calibrator is None:
        calibrator = MonotonicAffineRiskCalibrator()
        calibrator_source = "identity"
    else:
        payload = json.loads(args.calibrator.read_text(encoding="utf-8"))
        calibrator = MonotonicAffineRiskCalibrator.from_dict(payload["calibrator"])
        calibrator_source = str(args.calibrator.resolve())
    calibrated_predictions = dict(predictions)
    calibrated_predictions["risk_score"] = calibrator.transform(
        predictions["risk_score"]
    )
    calibrated_metrics = _metrics(model, steps, calibrated_predictions)
    non_risk_deltas = {
        key: abs(float(calibrated_metrics[key]) - float(value))
        for key, value in raw_metrics.items()
        if isinstance(value, (int, float))
        and calibrated_metrics.get(key) is not None
        and "risk" not in key
        and not key.startswith("validation_objective")
    }
    invariants = {
        "risk_auc_delta": calibrated_metrics["risk_auc"] - raw_metrics["risk_auc"],
        "non_risk_max_abs_delta": max(non_risk_deltas.values(), default=0.0),
        "non_risk_deltas": non_risk_deltas,
    }
    if abs(invariants["risk_auc_delta"]) > 1e-12:
        raise RuntimeError("Monotonic calibration changed risk AUC")
    if invariants["non_risk_max_abs_delta"] > 1e-12:
        raise RuntimeError("Frozen calibration changed a non-risk metric")
    result = {
        "scope": "frozen_monotonic_affine_risk_calibration_test",
        "model": str(args.model.resolve()),
        "calibrator_source": calibrator_source,
        "calibrator": calibrator.to_dict(),
        "test_steps": len(steps),
        "raw_metrics": raw_metrics,
        "calibrated_metrics": calibrated_metrics,
        "invariants": invariants,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
