"""Strict validation selection and frozen test summary for risk calibration."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any


CANDIDATES = ("identity", "reg0", "reg1e-4", "reg1e-3", "reg1e-2", "reg1e-1")
METRICS = (
    "validation_objective",
    "grouped_risk_probability_brier_score",
    "grouped_risk_probability_mae",
    "risk_probability_brier_score",
    "risk_probability_mae",
    "risk_auc",
    "grouped_utility_probability_brier_score",
    "grouped_preservation_probability_brier_score",
    "next_skill_accuracy",
)


def _mean_std(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_validation(root: Path, seed: int, candidate: str) -> dict[str, Any]:
    if candidate == "identity":
        path = root / f"seed{seed}" / "identity_val_metrics.json"
    else:
        path = root / f"seed{seed}" / candidate / "val_metrics.json"
    return json.loads(path.read_text(encoding="utf-8"))["metrics"]


def _aggregate(runs: dict[str, dict[int, dict[str, Any]]]):
    result = {}
    for candidate, by_seed in runs.items():
        result[candidate] = {}
        for metric in METRICS:
            values = [row.get(metric) for row in by_seed.values()]
            numeric = [float(value) for value in values if value is not None]
            result[candidate][metric] = _mean_std(numeric) if numeric else None
    return result


def _paired_deltas(
    runs: dict[str, dict[int, dict[str, Any]]], baseline: str = "identity"
):
    result = {}
    for candidate in runs:
        if candidate == baseline:
            continue
        result[candidate] = {}
        for metric in METRICS:
            values = []
            by_seed = {}
            for seed in sorted(runs[candidate]):
                left = runs[baseline][seed].get(metric)
                right = runs[candidate][seed].get(metric)
                if left is None or right is None:
                    continue
                delta = float(right) - float(left)
                values.append(delta)
                by_seed[str(seed)] = delta
            result[candidate][metric] = (
                {"candidate_minus_identity_by_seed": by_seed, **_mean_std(values)}
                if values
                else None
            )
    return result


def select(root: Path, seeds: list[int]) -> dict[str, Any]:
    runs = {
        candidate: {
            seed: _load_validation(root, seed, candidate) for seed in seeds
        }
        for candidate in CANDIDATES
    }
    aggregate = _aggregate(runs)
    selected = min(
        CANDIDATES,
        key=lambda candidate: aggregate[candidate]["validation_objective"]["mean"],
    )
    return {
        "scope": "frozen_monotonic_affine_risk_calibration_selection",
        "selection_protocol": (
            "Choose identity or regularization by mean group-aware validation "
            "objective across fixed checkpoint seeds; do not read test."
        ),
        "root": str(root.resolve()),
        "seeds": seeds,
        "candidates": list(CANDIDATES),
        "selected_candidate_by_validation": selected,
        "validation_aggregate": aggregate,
        "validation_paired_deltas": _paired_deltas(runs),
        "validation_runs": runs,
    }


def finalize(root: Path, seeds: list[int], selection_path: Path) -> dict[str, Any]:
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selected = selection["selected_candidate_by_validation"]
    if selected not in CANDIDATES:
        raise ValueError(f"Unknown selected candidate: {selected}")
    raw_runs = {}
    calibrated_runs = {}
    invariants = {}
    for seed in seeds:
        path = root / f"seed{seed}" / selected / "test_result.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        raw_runs[seed] = payload["raw_metrics"]
        calibrated_runs[seed] = payload["calibrated_metrics"]
        invariants[seed] = payload["invariants"]
    runs = {"identity": raw_runs, selected: calibrated_runs}
    if selected == "identity":
        runs = {"identity": raw_runs}
    return {
        "scope": "frozen_monotonic_affine_risk_calibration_final",
        "selection_file": str(selection_path.resolve()),
        "selection_file_sha256": _sha256(selection_path),
        "selected_candidate_by_validation": selected,
        "test_aggregate": _aggregate(runs),
        "test_paired_deltas": _paired_deltas(runs),
        "test_runs": runs,
        "test_invariants": invariants,
        "selection": selection,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 13, 21])
    parser.add_argument("--stage", choices=["select", "final"], required=True)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.stage == "select":
        result = select(args.root, args.seeds)
    else:
        if args.selection is None:
            parser.error("--selection is required for final stage")
        result = finalize(args.root, args.seeds, args.selection)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
