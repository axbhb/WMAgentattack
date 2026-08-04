"""Aggregate paired binary-risk and soft-risk DreamerV3 experiments."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


VARIANTS = ("binary_risk", "soft_risk")
METRICS = (
    "validation_objective",
    "validation_objective_multiseed_group",
    "grouped_risk_probability_brier_score",
    "grouped_risk_probability_mae",
    "grouped_utility_probability_brier_score",
    "grouped_utility_probability_mae",
    "grouped_preservation_probability_brier_score",
    "grouped_preservation_probability_mae",
    "risk_probability_brier_score",
    "utility_probability_brier_score",
    "preservation_probability_brier_score",
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


def _load(root: Path, variant: str, seed: int, split: str) -> dict[str, Any]:
    path = root / variant / f"seed{seed}" / f"{split}_metrics.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload["metrics"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 13, 21])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    runs = {
        variant: {
            seed: {
                split: _load(args.root, variant, seed, split)
                for split in ("val", "test")
            }
            for seed in args.seeds
        }
        for variant in VARIANTS
    }
    aggregate: dict[str, Any] = {}
    for variant in VARIANTS:
        aggregate[variant] = {}
        for split in ("val", "test"):
            aggregate[variant][split] = {}
            for metric in METRICS:
                values = [runs[variant][seed][split].get(metric) for seed in args.seeds]
                numeric = [float(value) for value in values if value is not None]
                aggregate[variant][split][metric] = (
                    _mean_std(numeric) if numeric else None
                )

    paired_deltas: dict[str, Any] = {}
    for split in ("val", "test"):
        paired_deltas[split] = {}
        for metric in METRICS:
            values = []
            by_seed = {}
            for seed in args.seeds:
                binary = runs["binary_risk"][seed][split].get(metric)
                soft = runs["soft_risk"][seed][split].get(metric)
                if binary is None or soft is None:
                    continue
                delta = float(soft) - float(binary)
                values.append(delta)
                by_seed[str(seed)] = delta
            paired_deltas[split][metric] = (
                {"soft_minus_binary_by_seed": by_seed, **_mean_std(values)}
                if values
                else None
            )

    selected_variant = min(
        VARIANTS,
        key=lambda variant: aggregate[variant]["val"]["validation_objective"][
            "mean"
        ],
    )
    result = {
        "scope": "agentdojo_v2_final_dreamer_soft_risk_ablation",
        "selection_protocol": (
            "Choose the variant by mean validation_objective across fixed seeds; "
            "test is reported only after this choice."
        ),
        "root": str(args.root.resolve()),
        "seeds": args.seeds,
        "variants": list(VARIANTS),
        "selected_variant_by_validation": selected_variant,
        "aggregate": aggregate,
        "paired_deltas": paired_deltas,
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
