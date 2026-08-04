"""Summarize same-fold checkpoint-seed replication for grouped utility."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any


MODEL_METRICS = (
    "validation_objective",
    "grouped_risk_probability_brier_score",
    "grouped_utility_probability_brier_score",
    "grouped_preservation_probability_brier_score",
    "binary_utility_auc",
    "risk_auc",
)
OUTCOME_METRICS = ("ASR", "BUP", "ASR_plus_BUP")


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _mean_std(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def _metrics(path: Path) -> dict[str, float]:
    payload = _load(path)["metrics"]
    return {key: float(payload[key]) for key in MODEL_METRICS}


def _model_runs(
    root: Path, *, fold: int, seeds: tuple[int, ...], variant: str | None
) -> dict[str, Any]:
    runs = {}
    for seed in seeds:
        if variant is None:
            run = root / "baseline" / f"fold{fold}" / f"seed{seed}"
        else:
            run = root / "models" / f"fold{fold}" / variant / f"seed{seed}"
        runs[str(seed)] = {
            split: _metrics(run / f"{split}_metrics.json")
            for split in ("val", "test")
        }
    return {
        "runs": runs,
        "aggregate": {
            split: {
                metric: _mean_std(
                    [runs[str(seed)][split][metric] for seed in seeds]
                )
                for metric in MODEL_METRICS
            }
            for split in ("val", "test")
        },
    }


def _training_activation(
    root: Path, *, fold: int, seeds: tuple[int, ...], variant: str
) -> dict[str, Any]:
    rows = {}
    for seed in seeds:
        payload = _load(
            root
            / "models"
            / f"fold{fold}"
            / variant
            / f"seed{seed}"
            / "training_stdout.json"
        )
        history = payload["training_history"]
        finite = all(
            math.isfinite(float(epoch[key]))
            for epoch in history
            for key in (
                "world",
                "group_utility_calibration",
                "group_utility_ranking",
                "group_utility_calibration_count",
                "group_utility_ranking_pair_count",
            )
        )
        rows[str(seed)] = {
            "best_epoch": int(payload["best_epoch"]),
            "epoch_count": len(history),
            "all_tracked_losses_finite": finite,
            "mean_group_count_per_update": statistics.fmean(
                float(epoch["group_utility_calibration_count"])
                for epoch in history
            ),
            "mean_pair_count_per_update": statistics.fmean(
                float(epoch["group_utility_ranking_pair_count"])
                for epoch in history
            ),
        }
    return {
        "per_seed": rows,
        "all_seeds_valid": all(
            row["all_tracked_losses_finite"]
            and row["mean_group_count_per_update"] > 0.0
            and row["mean_pair_count_per_update"] > 0.0
            for row in rows.values()
        ),
    }


def _downstream(path: Path) -> dict[str, Any]:
    payload = _load(path)
    output = {}
    for budget in ("1", "2", "4"):
        row = payload["test"][budget]["calibrated"]
        output[budget] = {
            "selected_recipe": row["frozen_validation_recipe"],
            "ensemble": {
                metric: float(row["ensemble"][metric])
                for metric in OUTCOME_METRICS
            },
            "per_seed": {
                str(seed): {
                    metric: float(metrics[metric])
                    for metric in OUTCOME_METRICS
                }
                for seed, metrics in row["per_seed"].items()
            },
        }
    return output


def summarize(
    root: Path,
    *,
    fold: int,
    seeds: tuple[int, ...],
    variant: str,
) -> dict[str, Any]:
    baseline_models = _model_runs(root, fold=fold, seeds=seeds, variant=None)
    variant_models = _model_runs(root, fold=fold, seeds=seeds, variant=variant)
    activation = _training_activation(
        root, fold=fold, seeds=seeds, variant=variant
    )
    model_contrast = {
        split: {
            metric: {
                "variant_minus_baseline": _mean_std(
                    [
                        variant_models["runs"][str(seed)][split][metric]
                        - baseline_models["runs"][str(seed)][split][metric]
                        for seed in seeds
                    ]
                ),
                "per_seed": {
                    str(seed): (
                        variant_models["runs"][str(seed)][split][metric]
                        - baseline_models["runs"][str(seed)][split][metric]
                    )
                    for seed in seeds
                },
            }
            for metric in MODEL_METRICS
        }
        for split in ("val", "test")
    }

    baseline_downstream = _downstream(root / "baseline_downstream.json")
    variant_downstream = _downstream(root / "variant_downstream.json")
    downstream_contrast = {}
    for budget in ("1", "2", "4"):
        downstream_contrast[budget] = {
            "ensemble_variant_minus_baseline": {
                metric: (
                    variant_downstream[budget]["ensemble"][metric]
                    - baseline_downstream[budget]["ensemble"][metric]
                )
                for metric in OUTCOME_METRICS
            },
            "per_seed_variant_minus_baseline": {
                str(seed): {
                    metric: (
                        variant_downstream[budget]["per_seed"][str(seed)][metric]
                        - baseline_downstream[budget]["per_seed"][str(seed)][metric]
                    )
                    for metric in OUTCOME_METRICS
                }
                for seed in seeds
            },
        }

    utility_brier_deltas = model_contrast["test"][
        "grouped_utility_probability_brier_score"
    ]["per_seed"]
    utility_brier_improved_count = sum(
        delta < 0.0 for delta in utility_brier_deltas.values()
    )
    top1 = downstream_contrast["1"]["ensemble_variant_minus_baseline"]
    mean_risk_auc_delta = model_contrast["test"]["risk_auc"][
        "variant_minus_baseline"
    ]["mean"]
    gate_checks = {
        "loss_activation_valid": activation["all_seeds_valid"],
        "top1_asr_plus_bup_delta_at_least_0_05": (
            top1["ASR_plus_BUP"] >= 0.05 - 1e-12
        ),
        "top1_bup_delta_nonnegative": top1["BUP"] >= -1e-12,
        "mean_test_risk_auc_delta_at_least_minus_0_05": (
            mean_risk_auc_delta >= -0.05 - 1e-12
        ),
        "test_utility_brier_improves_at_least_two_seeds": (
            utility_brier_improved_count >= 2
        ),
    }
    return {
        "scope": "same-fold three-checkpoint-seed grouped utility replication",
        "protocol": {
            "fold": fold,
            "seeds": list(seeds),
            "method_frozen_before_replication": True,
            "test_is_replication_diagnostic_not_final_claim": True,
        },
        "training_activation": activation,
        "baseline_models": baseline_models,
        "variant_models": variant_models,
        "model_contrast": model_contrast,
        "baseline_downstream": baseline_downstream,
        "variant_downstream": variant_downstream,
        "downstream_contrast": downstream_contrast,
        "replication_gate": {
            "checks": gate_checks,
            "test_utility_brier_improved_seed_count": utility_brier_improved_count,
            "mean_test_risk_auc_delta": mean_risk_auc_delta,
            "proceed_to_broader_folds": all(gate_checks.values()),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 13, 21])
    parser.add_argument("--variant", default="group_utility_head_only")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(
        args.archive_root,
        fold=args.fold,
        seeds=tuple(args.seeds),
        variant=args.variant,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
