"""Compare baseline and grouped-continuous-utility smoke runs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


VARIANTS = ("group_utility_detached", "group_utility_end_to_end")
MODEL_METRICS = (
    "validation_objective",
    "grouped_risk_probability_brier_score",
    "grouped_utility_probability_brier_score",
    "grouped_preservation_probability_brier_score",
    "binary_utility_auc",
    "risk_auc",
)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _model_metrics(path: Path) -> dict[str, Any]:
    metrics = _load(path)["metrics"]
    return {key: metrics.get(key) for key in MODEL_METRICS}


def _downstream(path: Path) -> dict[str, Any]:
    payload = _load(path)
    return {
        budget: {
            "selected_recipe": payload["test"][budget]["calibrated"][
                "frozen_validation_recipe"
            ],
            **{
                key: payload["test"][budget]["calibrated"]["ensemble"][key]
                for key in ("ASR", "BUP", "ASR_plus_BUP")
            },
        }
        for budget in ("1", "2", "4")
    }


def _training_activation(path: Path) -> dict[str, Any]:
    payload = _load(path)
    history = payload["training_history"]
    tracked = (
        "world",
        "group_utility_calibration",
        "group_utility_calibration_count",
        "group_utility_ranking",
        "group_utility_ranking_pair_count",
    )
    finite = all(
        math.isfinite(float(epoch[key]))
        for epoch in history
        for key in tracked
    )
    return {
        "epoch_count": len(history),
        "best_epoch": payload["best_epoch"],
        "all_tracked_losses_finite": finite,
        "mean_group_count_per_update": sum(
            float(epoch["group_utility_calibration_count"]) for epoch in history
        )
        / len(history),
        "mean_pair_count_per_update": sum(
            float(epoch["group_utility_ranking_pair_count"]) for epoch in history
        )
        / len(history),
        "final_epoch": {key: history[-1][key] for key in tracked},
    }


def summarize(
    root: Path,
    *,
    fold: int,
    seed: int,
    variants: tuple[str, ...] = VARIANTS,
) -> dict[str, Any]:
    if not variants:
        raise ValueError("At least one variant is required")
    baseline = {
        split: _model_metrics(root / "baseline" / f"fold{fold}" / f"{split}_metrics.json")
        for split in ("val", "test")
    }
    baseline["downstream"] = _downstream(root / "baseline_downstream.json")

    variant_names = variants
    variants = {}
    for variant in variant_names:
        run = root / "models" / f"fold{fold}" / variant / f"seed{seed}"
        variants[variant] = {
            "training": _training_activation(run / "training_stdout.json"),
            "val": _model_metrics(run / "val_metrics.json"),
            "test": _model_metrics(run / "test_metrics.json"),
            "downstream": _downstream(root / f"{variant}_downstream.json"),
        }
        for split in ("val", "test"):
            variants[variant][split]["grouped_utility_brier_improvement"] = (
                float(baseline[split]["grouped_utility_probability_brier_score"])
                - float(
                    variants[variant][split][
                        "grouped_utility_probability_brier_score"
                    ]
                )
            )
        for budget in ("1", "2", "4"):
            variants[variant]["downstream"][budget][
                "ASR_plus_BUP_delta_vs_baseline"
            ] = (
                float(variants[variant]["downstream"][budget]["ASR_plus_BUP"])
                - float(baseline["downstream"][budget]["ASR_plus_BUP"])
            )

    selected_variant = min(
        variants,
        key=lambda variant: float(
            variants[variant]["val"]["grouped_utility_probability_brier_score"]
        ),
    )
    activation_valid = all(
        row["training"]["all_tracked_losses_finite"]
        and row["training"]["mean_group_count_per_update"] > 0.0
        and row["training"]["mean_pair_count_per_update"] > 0.0
        for row in variants.values()
    )
    selected_val_brier = float(
        variants[selected_variant]["val"][
            "grouped_utility_probability_brier_score"
        ]
    )
    baseline_val_brier = float(
        baseline["val"]["grouped_utility_probability_brier_score"]
    )
    proceed = activation_valid and selected_val_brier <= baseline_val_brier + 0.01
    return {
        "scope": (
            f"fold{fold} seed{seed} grouped continuous utility smoke"
        ),
        "protocol": {
            "fold": fold,
            "seed": seed,
            "variants": list(variants),
            "variant_selection_uses": "validation grouped utility Brier only",
            "test_is_diagnostic_only": True,
            "formal_success_rule_not_evaluated_on_smoke": True,
        },
        "baseline": baseline,
        "variants": variants,
        "smoke_gate": {
            "activation_valid": activation_valid,
            "validation_selected_variant": selected_variant,
            "selected_validation_grouped_utility_brier": selected_val_brier,
            "baseline_validation_grouped_utility_brier": baseline_val_brier,
            "maximum_tolerated_validation_regression": 0.01,
            "proceed_to_formal_5fold": proceed,
            "reason": (
                "losses active and validation calibration is non-catastrophic"
                if proceed
                else "loss activation failed or validation calibration regressed"
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS)
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(
        args.archive_root,
        fold=args.fold,
        seed=args.seed,
        variants=tuple(args.variants),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
