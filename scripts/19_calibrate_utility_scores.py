"""Calibrate cached utility scores with validation labels.

This script is deliberately inference-free.  It learns a small logistic
calibrator on validation candidates and appends calibrated score columns to a
target candidate cache.  The calibrated cache can then be consumed by
``18_pareto_utility_selection.py`` via ``--utility-keys``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _fit_logistic_1d(scores: np.ndarray, labels: np.ndarray, *, steps: int = 500, lr: float = 0.1) -> tuple[float, float]:
    """Fit a tiny 1-D logistic calibrator without requiring sklearn."""

    scale = float(np.std(scores))
    if scale <= 1e-8:
        scale = 1.0
    mean = float(np.mean(scores))
    x = (scores - mean) / scale
    weight = 0.0
    bias = float(np.log((labels.mean() + 1e-3) / (1.0 - labels.mean() + 1e-3)))
    for _ in range(steps):
        logits = weight * x + bias
        probs = 1.0 / (1.0 + np.exp(-logits))
        error = probs - labels
        weight -= lr * float(np.mean(error * x))
        bias -= lr * float(np.mean(error))
    return float(weight / scale), float(bias - weight * mean / scale)


def _apply(scores: np.ndarray, weight: float, bias: float) -> np.ndarray:
    logits = weight * scores + bias
    return 1.0 / (1.0 + np.exp(-logits))


def _brier(labels: np.ndarray, scores: np.ndarray) -> float:
    return float(np.mean((labels - scores) ** 2))


def _binary_auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    labels = labels.astype(int)
    positives = int(labels.sum())
    negatives = int(len(labels) - positives)
    if positives == 0 or negatives == 0:
        return None
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    pos_rank_sum = ranks[labels == 1].sum()
    return float((pos_rank_sum - positives * (positives + 1) / 2) / (positives * negatives))


def _scores(payload: dict[str, Any], key: str) -> np.ndarray:
    return np.array([float(row.get(key, row.get("utility_score", 0.0))) for row in payload["candidates"]])


def _labels(payload: dict[str, Any]) -> np.ndarray:
    return np.array([bool(row["observed_utility"]) for row in payload["candidates"]], dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration-candidates", type=Path, required=True)
    parser.add_argument("--target-candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--score-keys",
        default="utility_score,final_utility_score,min_utility_score",
    )
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    args = parser.parse_args()

    calibration = json.loads(args.calibration_candidates.read_text(encoding="utf-8"))
    target = json.loads(args.target_candidates.read_text(encoding="utf-8"))
    calibration_labels = _labels(calibration)
    target_labels = _labels(target)
    score_keys = [item.strip() for item in args.score_keys.split(",") if item.strip()]

    report = {}
    for key in score_keys:
        calibration_scores = _scores(calibration, key)
        target_scores = _scores(target, key)
        weight, bias = _fit_logistic_1d(
            calibration_scores,
            calibration_labels,
            steps=args.steps,
            lr=args.learning_rate,
        )
        calibrated_calibration = _apply(calibration_scores, weight, bias)
        calibrated_target = _apply(target_scores, weight, bias)
        output_key = f"calibrated_{key}"
        for row, score in zip(target["candidates"], calibrated_target, strict=True):
            row[output_key] = float(score)
        report[key] = {
            "output_key": output_key,
            "weight": weight,
            "bias": bias,
            "calibration_brier_before": _brier(calibration_labels, calibration_scores),
            "calibration_brier_after": _brier(calibration_labels, calibrated_calibration),
            "calibration_auc_before": _binary_auc(calibration_labels, calibration_scores),
            "calibration_auc_after": _binary_auc(calibration_labels, calibrated_calibration),
            "target_brier_before": _brier(target_labels, target_scores),
            "target_brier_after": _brier(target_labels, calibrated_target),
            "target_auc_before": _binary_auc(target_labels, target_scores),
            "target_auc_after": _binary_auc(target_labels, calibrated_target),
        }

    target["utility_calibration"] = {
        "scope": "utility_score_logistic_calibration",
        "calibration_candidates": str(args.calibration_candidates.resolve()),
        "score_keys": score_keys,
        "report": report,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(target, indent=2), encoding="utf-8")
    print(json.dumps(target["utility_calibration"], indent=2))


if __name__ == "__main__":
    main()
