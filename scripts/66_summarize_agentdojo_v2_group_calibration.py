"""Select and report group-final risk calibration without test leakage."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any


VARIANT_WEIGHTS = {
    "calib0": 0.0,
    "calib01": 0.1,
    "calib025": 0.25,
    "calib05": 0.5,
}
METRICS = (
    "validation_objective",
    "validation_objective_multiseed_group",
    "grouped_risk_probability_brier_score",
    "grouped_risk_probability_mae",
    "grouped_utility_probability_brier_score",
    "grouped_preservation_probability_brier_score",
    "risk_probability_brier_score",
    "risk_auc",
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
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _load_metric(root: Path, variant: str, seed: int, split: str) -> dict[str, Any]:
    path = root / variant / f"seed{seed}" / f"{split}_metrics.json"
    return json.loads(path.read_text(encoding="utf-8"))["metrics"]


def _aggregate(
    runs: dict[str, dict[int, dict[str, Any]]],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for variant, by_seed in runs.items():
        result[variant] = {}
        for metric in METRICS:
            values = [row.get(metric) for row in by_seed.values()]
            numeric = [float(value) for value in values if value is not None]
            result[variant][metric] = _mean_std(numeric) if numeric else None
    return result


def _paired_deltas(
    runs: dict[str, dict[int, dict[str, Any]]],
    control: str = "calib0",
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for variant in runs:
        if variant == control:
            continue
        result[variant] = {}
        for metric in METRICS:
            by_seed = {}
            values = []
            for seed in sorted(runs[variant]):
                left = runs[control][seed].get(metric)
                right = runs[variant][seed].get(metric)
                if left is None or right is None:
                    continue
                delta = float(right) - float(left)
                by_seed[str(seed)] = delta
                values.append(delta)
            result[variant][metric] = (
                {"variant_minus_control_by_seed": by_seed, **_mean_std(values)}
                if values
                else None
            )
    return result


def select(root: Path, seeds: list[int]) -> dict[str, Any]:
    runs = {
        variant: {
            seed: _load_metric(root, variant, seed, "val") for seed in seeds
        }
        for variant in VARIANT_WEIGHTS
    }
    aggregate = _aggregate(runs)
    selected = min(
        VARIANT_WEIGHTS,
        key=lambda variant: aggregate[variant]["validation_objective"]["mean"],
    )
    return {
        "scope": "agentdojo_v2_group_final_risk_calibration_selection",
        "selection_protocol": (
            "Choose calibration weight by mean group-aware validation objective "
            "across fixed seeds. No test file is read in this stage."
        ),
        "root": str(root.resolve()),
        "seeds": seeds,
        "variant_weights": VARIANT_WEIGHTS,
        "selected_variant_by_validation": selected,
        "selected_weight": VARIANT_WEIGHTS[selected],
        "validation_aggregate": aggregate,
        "validation_paired_deltas": _paired_deltas(runs),
        "validation_runs": runs,
    }


def finalize(root: Path, seeds: list[int], selection_path: Path) -> dict[str, Any]:
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selected = selection["selected_variant_by_validation"]
    if selected not in VARIANT_WEIGHTS:
        raise ValueError(f"Unknown selected variant: {selected}")
    variants = ["calib0"] if selected == "calib0" else ["calib0", selected]
    test_runs = {
        variant: {
            seed: _load_metric(root, variant, seed, "test") for seed in seeds
        }
        for variant in variants
    }
    return {
        "scope": "agentdojo_v2_group_final_risk_calibration_final",
        "selection_file": str(selection_path.resolve()),
        "selection_file_sha256": _sha256(selection_path),
        "selected_variant_by_validation": selected,
        "selected_weight": VARIANT_WEIGHTS[selected],
        "test_variants_evaluated": variants,
        "test_aggregate": _aggregate(test_runs),
        "test_paired_deltas": _paired_deltas(test_runs),
        "test_runs": test_runs,
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
            parser.error("--selection is required for --stage final")
        result = finalize(args.root, args.seeds, args.selection)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
