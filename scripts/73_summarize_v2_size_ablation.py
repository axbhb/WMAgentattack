"""Aggregate the nested AgentDojo-v2 data-scale ablation."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

import numpy as np


SEEDS = (7, 13, 21)
SIZES = ("pct25", "pct50", "pct100")
METRICS = (
    "validation_objective",
    "grouped_risk_probability_brier_score",
    "grouped_risk_probability_mae",
    "grouped_utility_probability_brier_score",
    "grouped_preservation_probability_brier_score",
    "risk_auc",
    "next_skill_accuracy",
)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _mean_std(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def _aggregate_model_metrics(root: Path) -> dict[str, Any]:
    output = {}
    for split in ("val", "test"):
        runs = {
            seed: _load(root / f"seed{seed}" / f"{split}_metrics.json")[
                "metrics"
            ]
            for seed in SEEDS
        }
        output[split] = {
            "aggregate": {
                metric: _mean_std([float(runs[seed][metric]) for seed in SEEDS])
                for metric in METRICS
            },
            "per_seed": runs,
        }
    return output


def _calibration_summary(path: Path) -> dict[str, Any]:
    payload = _load(path)
    selected = payload["selected_candidate_by_validation"]
    aggregate = payload["test_aggregate"]
    selected_metrics = aggregate[selected]
    return {
        "selected_candidate": selected,
        "identity": aggregate["identity"],
        "selected": selected_metrics,
        "selected_minus_identity": (
            payload["test_paired_deltas"].get(selected, {})
            if selected != "identity"
            else {}
        ),
        "source": str(path.resolve()),
    }


def _prospective_summary(
    result_path: Path, randomization_path: Path
) -> dict[str, Any]:
    result = _load(result_path)
    randomization = _load(randomization_path)
    budgets = {}
    for budget in ("1", "2", "4"):
        budgets[budget] = {}
        for variant in ("raw", "calibrated"):
            validation = result["validation_selection"][budget][variant]
            test = result["test"][budget][variant]
            random_test = randomization["tests"][budget][variant]["metrics"]
            budgets[budget][variant] = {
                "selected_recipe": validation["selected_recipe"],
                "validation_ASR_plus_BUP": validation["candidates"][
                    validation["selected_recipe"]
                ]["aggregate"]["ASR_plus_BUP"]["mean"],
                "test_checkpoint_aggregate": test["aggregate"],
                "test_ensemble": test["ensemble"],
                "test_task_bootstrap": test["ensemble_task_bootstrap"],
                "randomization": random_test,
            }
    return {
        "selected_first_step_calibration": result["selected_calibration"],
        "first_step_calibration_validation": result["calibration_validation"],
        "budgets": budgets,
        "result_source": str(result_path.resolve()),
        "randomization_source": str(randomization_path.resolve()),
    }


def _paired_task_bootstrap(
    left: dict[str, Any],
    right: dict[str, Any],
    *,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    metrics = ("ASR", "BUP", "ASR_plus_BUP")
    left_tasks = left["by_task"]
    right_tasks = right["by_task"]
    if set(left_tasks) != set(right_tasks):
        raise ValueError("Prospective size runs cover different test tasks")
    task_keys = sorted(left_tasks)
    deltas = {
        metric: np.asarray(
            [
                float(right_tasks[key][metric])
                - float(left_tasks[key][metric])
                for key in task_keys
            ],
            dtype=np.float64,
        )
        for metric in metrics
    }
    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0, len(task_keys), size=(draws, len(task_keys))
    )
    output = {}
    for metric, values in deltas.items():
        samples = values[indices].mean(axis=1)
        output[metric] = {
            "observed_delta": float(np.mean(values)),
            "ci95_low": float(np.quantile(samples, 0.025)),
            "ci95_high": float(np.quantile(samples, 0.975)),
            "bootstrap_probability_gt_zero": float(np.mean(samples > 0.0)),
            "per_task_deltas": {
                key: float(value)
                for key, value in zip(task_keys, values, strict=True)
            },
        }
    return {
        "unit": "held-out user task",
        "draws": draws,
        "metrics": output,
    }


def _metric_deltas(
    model_metrics: dict[str, Any], left: str, right: str
) -> dict[str, Any]:
    output = {}
    for split in ("val", "test"):
        output[split] = {}
        for metric in METRICS:
            left_runs = model_metrics[left][split]["per_seed"]
            right_runs = model_metrics[right][split]["per_seed"]
            values = [
                float(right_runs[str(seed) if str(seed) in right_runs else seed][metric])
                - float(left_runs[str(seed) if str(seed) in left_runs else seed][metric])
                for seed in SEEDS
            ]
            output[split][metric] = {
                "right_minus_left": _mean_std(values),
                "per_seed": {
                    str(seed): value
                    for seed, value in zip(SEEDS, values, strict=True)
                },
                "all_three_positive": all(value > 0.0 for value in values),
                "all_three_negative": all(value < 0.0 for value in values),
            }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation-root", type=Path, required=True)
    parser.add_argument("--full-data-root", type=Path, required=True)
    parser.add_argument("--full-model-root", type=Path, required=True)
    parser.add_argument("--full-calibration-final", type=Path, required=True)
    parser.add_argument("--full-prospective-result", type=Path, required=True)
    parser.add_argument("--full-randomization", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=100000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260715)
    args = parser.parse_args()

    model_roots = {
        "pct25": args.ablation_root / "models" / "pct25",
        "pct50": args.ablation_root / "models" / "pct50",
        "pct100": args.full_model_root,
    }
    model_metrics = {
        size: _aggregate_model_metrics(root)
        for size, root in model_roots.items()
    }
    calibration_paths = {
        "pct25": args.ablation_root
        / "calibration"
        / "pct25"
        / "final_summary.json",
        "pct50": args.ablation_root
        / "calibration"
        / "pct50"
        / "final_summary.json",
        "pct100": args.full_calibration_final,
    }
    calibration = {
        size: _calibration_summary(path)
        for size, path in calibration_paths.items()
    }
    prospective_paths = {
        "pct25": (
            args.ablation_root / "downstream" / "pct25" / "result.json",
            args.ablation_root
            / "downstream"
            / "pct25"
            / "randomization.json",
        ),
        "pct50": (
            args.ablation_root / "downstream" / "pct50" / "result.json",
            args.ablation_root
            / "downstream"
            / "pct50"
            / "randomization.json",
        ),
        "pct100": (
            args.full_prospective_result,
            args.full_randomization,
        ),
    }
    prospective = {
        size: _prospective_summary(*paths)
        for size, paths in prospective_paths.items()
    }

    size_contrasts = {}
    for left, right in (("pct25", "pct50"), ("pct50", "pct100"), ("pct25", "pct100")):
        key = f"{right}_minus_{left}"
        prospective_contrasts = {}
        for budget in ("1", "2", "4"):
            left_result = prospective[left]["budgets"][budget]["calibrated"][
                "test_ensemble"
            ]
            right_result = prospective[right]["budgets"][budget]["calibrated"][
                "test_ensemble"
            ]
            prospective_contrasts[budget] = _paired_task_bootstrap(
                left_result,
                right_result,
                draws=args.bootstrap_draws,
                seed=args.bootstrap_seed + int(budget) + len(size_contrasts) * 100,
            )
        size_contrasts[key] = {
            "raw_model_metrics": _metric_deltas(
                model_metrics, left, right
            ),
            "prospective_calibrated_selection": prospective_contrasts,
        }

    subset_summary = _load(args.ablation_root / "datasets" / "summary.json")
    full_audit = _load(args.full_data_root / "audit.json")
    result = {
        "scope": "nested AgentDojo-v2 data-scale ablation",
        "primary_decision_metric": (
            "first-step calibrated ASR+BUP with one selected configuration "
            "per held-out user task"
        ),
        "secondary_decision_budgets": [2, 4],
        "protocol": {
            "attack_configuration_fractions": {
                "pct25": 0.25,
                "pct50": 0.5,
                "pct100": 1.0,
            },
            "checkpoint_seeds": list(SEEDS),
            "train_subsets_nested": subset_summary["nested_checks"],
            "sampling_is_label_blind": True,
            "validation_and_test_are_identical_across_sizes": True,
            "test_is_never_used_for_training_or_recipe_selection": True,
        },
        "dataset": {
            "pct25": subset_summary["subsets"]["pct25"],
            "pct50": subset_summary["subsets"]["pct50"],
            "pct100": {
                "attack_group_count": full_audit["split_group_audit"]["train"][
                    "attack_groups"
                ],
                "attack_trajectory_count": full_audit["split_group_audit"]["train"][
                    "attack_trajectories"
                ],
                "clean_group_count": 12,
                "step_count": sum(
                    1
                    for _ in (args.full_data_root / "train_steps.jsonl").open(
                        encoding="utf-8"
                    )
                ),
            },
        },
        "model_metrics": model_metrics,
        "frozen_final_step_calibration": calibration,
        "prospective_first_step_selection": prospective,
        "size_contrasts": size_contrasts,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
