"""Strict validation-selected downstream selection for AgentDojo-v2.

The unit of decision is one repeated attack configuration.  Each configuration
is represented by its final-step prediction averaged over victim-model seeds.
One shared scoring recipe is selected on validation across checkpoint seeds and
then frozen for test.  Test labels never participate in recipe selection.
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wmagentattack.full_dreamer_v3 import FullSheepRLDreamerV3
from wmagentattack.io_utils import read_jsonl
from wmagentattack.risk_calibration import MonotonicAffineRiskCalibrator
from wmagentattack.risk_calibration import (
    build_grouped_risk_data,
    fit_monotonic_affine_group_calibrator,
)
from wmagentattack.schema import StepRecord


# These four recipes are fixed before looking at validation or test outcomes.
# Utility weights sum to one in every composite recipe, keeping their scale
# directly comparable to the probability-valued risk score.
RECIPES: dict[str, tuple[float, float]] = {
    "risk_only": (0.0, 0.0),
    "risk_plus_utility": (1.0, 0.0),
    "risk_plus_preservation": (0.0, 1.0),
    "risk_plus_blended_utility": (0.5, 0.5),
}
RISK_VARIANTS = {
    "raw": "raw_risk_score",
    "calibrated": "calibrated_risk_score",
}
PREDICTION_KEYS = (
    "raw_risk_score",
    "calibrated_risk_score",
    "utility_score",
    "preservation_score",
    "critic_value_score",
    "configuration_value_score",
)


def _steps(path: Path) -> list[StepRecord]:
    return [StepRecord.model_validate(row) for row in read_jsonl(path)]


def _load_calibrator(path: Path | None) -> MonotonicAffineRiskCalibrator:
    if path is None:
        return MonotonicAffineRiskCalibrator()
    payload = json.loads(path.read_text(encoding="utf-8"))
    return MonotonicAffineRiskCalibrator.from_dict(payload["calibrator"])


def _decision_indices(
    steps: list[StepRecord], position: str
) -> dict[str, int]:
    if position not in {"first", "final"}:
        raise ValueError("position must be 'first' or 'final'")
    selected: dict[str, int] = {}
    for index, step in enumerate(steps):
        previous = selected.get(step.trajectory_id)
        if previous is None:
            selected[step.trajectory_id] = index
        elif position == "first" and step.step_id < steps[previous].step_id:
            selected[step.trajectory_id] = index
        elif position == "final" and step.step_id > steps[previous].step_id:
            selected[step.trajectory_id] = index
    return selected


def _fit_decision_time_calibrator(
    steps: list[StepRecord], predictions: dict[str, Any]
) -> tuple[MonotonicAffineRiskCalibrator, dict[str, Any]]:
    indices = sorted(_decision_indices(steps, "first").values())
    decision_steps = [steps[index] for index in indices]
    decision_scores = np.asarray(
        predictions["risk_score"], dtype=np.float64
    )[indices]
    grouped = build_grouped_risk_data(decision_steps, decision_scores)
    return fit_monotonic_affine_group_calibrator(
        grouped, regularization=0.0
    )


def _configuration_rows(
    steps: list[StepRecord],
    predictions: dict[str, Any],
    calibrator: MonotonicAffineRiskCalibrator,
    *,
    decision_step: str,
) -> list[dict[str, Any]]:
    raw_risk = np.asarray(predictions["risk_score"], dtype=np.float64)
    calibrated_risk = calibrator.transform(raw_risk)
    utility = np.asarray(predictions["utility_score"], dtype=np.float64)
    preservation = np.asarray(
        predictions["preservation_score"], dtype=np.float64
    )
    critic_value = np.asarray(
        predictions.get("value_score", np.zeros_like(raw_risk)),
        dtype=np.float64,
    )
    configuration_value = np.asarray(
        predictions.get("configuration_value_score", np.zeros_like(raw_risk)),
        dtype=np.float64,
    )
    for values in (
        raw_risk,
        utility,
        preservation,
        critic_value,
        configuration_value,
    ):
        if len(values) != len(steps):
            raise ValueError("Prediction length does not match step count")

    grouped: dict[str, list[int]] = defaultdict(list)
    for index in _decision_indices(steps, decision_step).values():
        step = steps[index]
        if (
            step.multiseed_group_id is not None
            and step.attack_probability_target is not None
        ):
            grouped[str(step.multiseed_group_id)].append(index)
    if not grouped:
        raise ValueError("No repeated attack configurations found")

    rows = []
    for group_id in sorted(grouped):
        indices = grouped[group_id]
        records = [steps[index] for index in indices]
        task_keys = {(record.domain, record.task_id) for record in records}
        if len(task_keys) != 1:
            raise ValueError(f"Configuration spans tasks: {group_id}")
        expected_trials = {
            int(record.multiseed_trials)
            for record in records
            if record.multiseed_trials is not None
        }
        if len(expected_trials) != 1 or len(indices) != next(
            iter(expected_trials)
        ):
            raise ValueError(f"Incomplete repeated configuration: {group_id}")
        attack_targets = {
            round(float(record.attack_probability_target), 12)
            for record in records
        }
        utility_targets = {
            round(float(record.utility_probability_target), 12)
            for record in records
            if record.utility_probability_target is not None
        }
        if len(attack_targets) != 1 or len(utility_targets) != 1:
            raise ValueError(f"Inconsistent probability target: {group_id}")
        domain, task_id = next(iter(task_keys))
        attack_outcomes = [int(record.attack_success) for record in records]
        utility_outcomes = [int(record.task_success) for record in records]
        rows.append(
            {
                "group_id": group_id,
                "task_key": f"{domain}|{task_id}",
                "domain": domain,
                "task_id": task_id,
                "trials": len(indices),
                "attack_outcomes": attack_outcomes,
                "utility_outcomes": utility_outcomes,
                "observed_asr": float(np.mean(attack_outcomes)),
                "observed_bup": float(np.mean(utility_outcomes)),
                "target_asr": next(iter(attack_targets)),
                "target_bup": next(iter(utility_targets)),
                "raw_risk_score": float(np.mean(raw_risk[indices])),
                "calibrated_risk_score": float(
                    np.mean(calibrated_risk[indices])
                ),
                "utility_score": float(np.mean(utility[indices])),
                "preservation_score": float(
                    np.mean(preservation[indices])
                ),
                "critic_value_score": float(
                    np.mean(critic_value[indices])
                ),
                "configuration_value_score": float(
                    np.mean(configuration_value[indices])
                ),
            }
        )
    return rows


def _group_risk_brier(
    rows: list[dict[str, Any]], score_key: str
) -> float:
    return float(
        np.mean(
            [
                (float(row[score_key]) - float(row["target_asr"])) ** 2
                for row in rows
            ]
        )
    )


def _decision_score(row: dict[str, Any], risk_key: str, recipe: str) -> float:
    utility_weight, preservation_weight = RECIPES[recipe]
    return float(
        row[risk_key]
        + utility_weight * row["utility_score"]
        + preservation_weight * row["preservation_score"]
    )


def _select(
    rows: list[dict[str, Any]],
    *,
    risk_key: str,
    recipe: str,
    budget_per_task: int,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["task_key"]].append(row)
    selected = []
    for task_key in sorted(grouped):
        candidates = grouped[task_key]
        if len(candidates) < budget_per_task:
            raise ValueError(
                f"Task {task_key} has {len(candidates)} candidates, "
                f"below budget {budget_per_task}"
            )
        ranked = sorted(
            candidates,
            key=lambda row: (
                -_decision_score(row, risk_key, recipe),
                row["group_id"],
            ),
        )
        for row in ranked[:budget_per_task]:
            selected.append(
                {
                    **row,
                    "decision_score": _decision_score(
                        row, risk_key, recipe
                    ),
                }
            )
    return selected


def _metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("Cannot summarize an empty selection")
    task_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        task_rows[row["task_key"]].append(row)
    asr = float(np.mean([row["observed_asr"] for row in rows]))
    bup = float(np.mean([row["observed_bup"] for row in rows]))
    return {
        "selected_configuration_count": len(rows),
        "selected_episode_count": sum(int(row["trials"]) for row in rows),
        "task_count": len(task_rows),
        "ASR": asr,
        "BUP": bup,
        "ASR_plus_BUP": asr + bup,
        "mean_target_ASR": float(np.mean([row["target_asr"] for row in rows])),
        "mean_target_BUP": float(np.mean([row["target_bup"] for row in rows])),
        "selected_group_ids": [row["group_id"] for row in rows],
        "by_task": {
            task_key: {
                "configuration_count": len(values),
                "ASR": float(
                    np.mean([row["observed_asr"] for row in values])
                ),
                "BUP": float(
                    np.mean([row["observed_bup"] for row in values])
                ),
                "ASR_plus_BUP": float(
                    np.mean(
                        [
                            row["observed_asr"] + row["observed_bup"]
                            for row in values
                        ]
                    )
                ),
            }
            for task_key, values in sorted(task_rows.items())
        },
    }


def _mean_std(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def _aggregate_seed_metrics(by_seed: dict[int, dict[str, Any]]) -> dict[str, Any]:
    return {
        key: _mean_std([float(row[key]) for row in by_seed.values()])
        for key in ("ASR", "BUP", "ASR_plus_BUP")
    }


def _ensemble(rows_by_seed: dict[int, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    seeds = sorted(rows_by_seed)
    maps = {
        seed: {row["group_id"]: row for row in rows_by_seed[seed]}
        for seed in seeds
    }
    group_ids = set(maps[seeds[0]])
    if any(set(maps[seed]) != group_ids for seed in seeds[1:]):
        raise ValueError("Checkpoint seeds produced different configuration ids")
    output = []
    for group_id in sorted(group_ids):
        reference = maps[seeds[0]][group_id]
        row = dict(reference)
        for key in PREDICTION_KEYS:
            row[key] = float(
                np.mean([maps[seed][group_id][key] for seed in seeds])
            )
        output.append(row)
    return output


def _validation_select(
    rows_by_seed: dict[int, list[dict[str, Any]]],
    *,
    risk_key: str,
    budget: int,
) -> dict[str, Any]:
    candidates = {}
    for recipe in RECIPES:
        by_seed = {
            seed: _metrics(
                _select(
                    rows,
                    risk_key=risk_key,
                    recipe=recipe,
                    budget_per_task=budget,
                )
            )
            for seed, rows in rows_by_seed.items()
        }
        candidates[recipe] = {
            "per_seed": by_seed,
            "aggregate": _aggregate_seed_metrics(by_seed),
        }
    recipe_order = {recipe: index for index, recipe in enumerate(RECIPES)}
    selected = max(
        RECIPES,
        key=lambda recipe: (
            candidates[recipe]["aggregate"]["ASR_plus_BUP"]["mean"],
            candidates[recipe]["aggregate"]["BUP"]["mean"],
            candidates[recipe]["aggregate"]["ASR"]["mean"],
            -recipe_order[recipe],
        ),
    )
    return {
        "selection_rule": (
            "maximize mean validation ASR+BUP across checkpoint seeds; "
            "tie-break by BUP, ASR, then fixed recipe order"
        ),
        "selected_recipe": selected,
        "candidates": candidates,
    }


def _task_bootstrap(
    rows: list[dict[str, Any]], *, draws: int, seed: int
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["task_key"]].append(row)
    task_keys = sorted(grouped)
    task_values = np.asarray(
        [
            [
                np.mean([row["observed_asr"] for row in grouped[key]]),
                np.mean([row["observed_bup"] for row in grouped[key]]),
            ]
            for key in task_keys
        ],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(task_keys), size=(draws, len(task_keys)))
    samples = task_values[indices].mean(axis=1)
    joint = samples[:, 0] + samples[:, 1]

    def interval(values: np.ndarray) -> dict[str, float]:
        return {
            "estimate": float(np.mean(values)),
            "ci95_low": float(np.quantile(values, 0.025)),
            "ci95_high": float(np.quantile(values, 0.975)),
        }

    return {
        "unit": "held-out user task",
        "draws": draws,
        "ASR": interval(samples[:, 0]),
        "BUP": interval(samples[:, 1]),
        "ASR_plus_BUP": interval(joint),
    }


def _paired_task_delta_bootstrap(
    left: list[dict[str, Any]],
    right: list[dict[str, Any]],
    *,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    def by_task(rows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[row["task_key"]].append(row)
        return {
            key: np.asarray(
                [
                    np.mean([row["observed_asr"] for row in values]),
                    np.mean([row["observed_bup"] for row in values]),
                ],
                dtype=np.float64,
            )
            for key, values in grouped.items()
        }

    left_tasks = by_task(left)
    right_tasks = by_task(right)
    if set(left_tasks) != set(right_tasks):
        raise ValueError("Paired selections cover different task sets")
    keys = sorted(left_tasks)
    deltas = np.stack([right_tasks[key] - left_tasks[key] for key in keys])
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(keys), size=(draws, len(keys)))
    samples = deltas[indices].mean(axis=1)
    joint = samples[:, 0] + samples[:, 1]

    def interval(values: np.ndarray) -> dict[str, float]:
        return {
            "estimate": float(np.mean(values)),
            "ci95_low": float(np.quantile(values, 0.025)),
            "ci95_high": float(np.quantile(values, 0.975)),
            "probability_gt_zero": float(np.mean(values > 0.0)),
        }

    return {
        "contrast": "calibrated_minus_raw",
        "unit": "held-out user task",
        "draws": draws,
        "ASR": interval(samples[:, 0]),
        "BUP": interval(samples[:, 1]),
        "ASR_plus_BUP": interval(joint),
    }


def _pool_expected(rows: list[dict[str, Any]], budget: int) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["task_key"]].append(row)
    asr = float(
        np.mean(
            [
                np.mean([row["observed_asr"] for row in values])
                for values in grouped.values()
            ]
        )
    )
    bup = float(
        np.mean(
            [
                np.mean([row["observed_bup"] for row in values])
                for values in grouped.values()
            ]
        )
    )
    return {
        "meaning": "expected metric for uniform task-balanced random selection",
        "selected_configuration_count": budget * len(grouped),
        "ASR": asr,
        "BUP": bup,
        "ASR_plus_BUP": asr + bup,
    }


def _oracle(rows: list[dict[str, Any]], budget: int) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["task_key"]].append(row)
    selected = []
    for task_key in sorted(grouped):
        selected.extend(
            sorted(
                grouped[task_key],
                key=lambda row: (
                    -(row["observed_asr"] + row["observed_bup"]),
                    -row["observed_bup"],
                    -row["observed_asr"],
                    row["group_id"],
                ),
            )[:budget]
        )
    return _metrics(selected)


def _overlap(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> dict[str, Any]:
    left_ids = {row["group_id"] for row in left}
    right_ids = {row["group_id"] for row in right}
    union = left_ids | right_ids
    return {
        "intersection": len(left_ids & right_ids),
        "union": len(union),
        "jaccard": len(left_ids & right_ids) / len(union) if union else 1.0,
        "identical": left_ids == right_ids,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--calibration-root", type=Path)
    parser.add_argument("--calibration-selection", type=Path)
    parser.add_argument(
        "--calibration-mode",
        choices=["external", "fit_decision_time"],
        default="external",
    )
    parser.add_argument("--calibration-train", type=Path)
    parser.add_argument(
        "--decision-step", choices=["first", "final"], default="final"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 13, 21])
    parser.add_argument("--budgets", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260715)
    args = parser.parse_args()

    if any(budget < 1 for budget in args.budgets):
        parser.error("--budgets must contain positive integers")
    if args.calibration_mode == "external":
        if args.calibration_root is None or args.calibration_selection is None:
            parser.error(
                "external calibration requires --calibration-root and "
                "--calibration-selection"
            )
        selection = json.loads(
            args.calibration_selection.read_text(encoding="utf-8")
        )
        selected_calibration = selection[
            "selected_candidate_by_validation"
        ]
    else:
        if args.calibration_train is None:
            parser.error(
                "fit_decision_time calibration requires --calibration-train"
            )
        if args.decision_step != "first":
            parser.error(
                "fit_decision_time calibration is defined for first-step "
                "prospective decisions"
            )
        selected_calibration = "pending_validation_selection"
    split_steps = {
        split: _steps(args.data_root / f"{split}_steps.jsonl")
        for split in ("val", "test")
    }
    calibration_train_steps = (
        _steps(args.calibration_train)
        if args.calibration_mode == "fit_decision_time"
        else None
    )
    rows_by_split_seed: dict[str, dict[int, list[dict[str, Any]]]] = {
        "val": {},
        "test": {},
    }
    calibrators: dict[str, Any] = {}
    calibration_fit: dict[str, Any] = {}
    for seed in args.seeds:
        if args.calibration_mode == "external":
            calibrator_path = None
            if selected_calibration != "identity":
                calibrator_path = (
                    args.calibration_root
                    / f"seed{seed}"
                    / selected_calibration
                    / "calibrator.json"
                )
            calibrator = _load_calibrator(calibrator_path)
        else:
            calibrator = None
        if calibrator is not None:
            calibrators[str(seed)] = calibrator.to_dict()
        model = FullSheepRLDreamerV3.load(
            args.model_root / f"seed{seed}" / "model"
        )
        if args.calibration_mode == "fit_decision_time":
            if calibration_train_steps is None:
                raise RuntimeError("Missing decision-time calibration data")
            train_predictions = model.predict(calibration_train_steps)
            calibrator, fit_stats = _fit_decision_time_calibrator(
                calibration_train_steps, train_predictions
            )
            calibrators[str(seed)] = calibrator.to_dict()
            calibration_fit[str(seed)] = fit_stats
        for split, steps in split_steps.items():
            rows_by_split_seed[split][seed] = _configuration_rows(
                steps,
                model.predict(steps),
                calibrator,
                decision_step=args.decision_step,
            )
        del model
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    calibration_validation = None
    if args.calibration_mode == "fit_decision_time":
        by_candidate = {
            "identity": {
                seed: _group_risk_brier(rows, "raw_risk_score")
                for seed, rows in rows_by_split_seed["val"].items()
            },
            "reg0": {
                seed: _group_risk_brier(rows, "calibrated_risk_score")
                for seed, rows in rows_by_split_seed["val"].items()
            },
        }
        aggregate = {
            candidate: _mean_std(list(values.values()))
            for candidate, values in by_candidate.items()
        }
        selected_calibration = min(
            aggregate,
            key=lambda candidate: aggregate[candidate]["mean"],
        )
        calibration_validation = {
            "selection_rule": (
                "fit two affine-logit parameters on train first-step groups; "
                "choose identity or reg0 by mean validation first-step group "
                "risk Brier across checkpoint seeds"
            ),
            "per_seed_group_risk_brier": by_candidate,
            "aggregate_group_risk_brier": aggregate,
            "selected_candidate": selected_calibration,
        }
        if selected_calibration == "identity":
            for split in ("val", "test"):
                for rows in rows_by_split_seed[split].values():
                    for row in rows:
                        row["calibrated_risk_score"] = row[
                            "raw_risk_score"
                        ]

    ensemble_rows = {
        split: _ensemble(rows_by_split_seed[split])
        for split in ("val", "test")
    }
    validation_selection: dict[str, Any] = {}
    test_results: dict[str, Any] = {}
    for budget in args.budgets:
        budget_key = str(budget)
        validation_selection[budget_key] = {}
        test_results[budget_key] = {}
        selected_ensemble_rows = {}
        for variant, risk_key in RISK_VARIANTS.items():
            validation = _validation_select(
                rows_by_split_seed["val"],
                risk_key=risk_key,
                budget=budget,
            )
            validation_selection[budget_key][variant] = validation
            recipe = validation["selected_recipe"]
            per_seed_rows = {
                seed: _select(
                    rows,
                    risk_key=risk_key,
                    recipe=recipe,
                    budget_per_task=budget,
                )
                for seed, rows in rows_by_split_seed["test"].items()
            }
            per_seed_metrics = {
                seed: _metrics(rows) for seed, rows in per_seed_rows.items()
            }
            selected_ensemble_rows[variant] = _select(
                ensemble_rows["test"],
                risk_key=risk_key,
                recipe=recipe,
                budget_per_task=budget,
            )
            ensemble_metrics = _metrics(selected_ensemble_rows[variant])
            test_results[budget_key][variant] = {
                "frozen_validation_recipe": recipe,
                "per_seed": per_seed_metrics,
                "aggregate": _aggregate_seed_metrics(per_seed_metrics),
                "ensemble": ensemble_metrics,
                "ensemble_task_bootstrap": _task_bootstrap(
                    selected_ensemble_rows[variant],
                    draws=args.bootstrap_draws,
                    seed=args.bootstrap_seed + budget,
                ),
            }

        fixed_recipe = "risk_plus_utility"
        fixed_rows = {
            variant: _select(
                ensemble_rows["test"],
                risk_key=risk_key,
                recipe=fixed_recipe,
                budget_per_task=budget,
            )
            for variant, risk_key in RISK_VARIANTS.items()
        }
        risk_only_rows = {
            variant: _select(
                ensemble_rows["test"],
                risk_key=risk_key,
                recipe="risk_only",
                budget_per_task=budget,
            )
            for variant, risk_key in RISK_VARIANTS.items()
        }
        test_results[budget_key]["baselines"] = {
            "random_expected": _pool_expected(
                ensemble_rows["test"], budget
            ),
            "test_oracle_diagnostic_only": _oracle(
                ensemble_rows["test"], budget
            ),
        }
        test_results[budget_key]["calibrated_minus_raw"] = (
            _paired_task_delta_bootstrap(
                selected_ensemble_rows["raw"],
                selected_ensemble_rows["calibrated"],
                draws=args.bootstrap_draws,
                seed=args.bootstrap_seed + 100 + budget,
            )
        )
        test_results[budget_key]["fixed_recipe_diagnostic"] = {
            "recipe": fixed_recipe,
            "raw": _metrics(fixed_rows["raw"]),
            "calibrated": _metrics(fixed_rows["calibrated"]),
            "selection_overlap": _overlap(
                fixed_rows["raw"], fixed_rows["calibrated"]
            ),
        }
        test_results[budget_key]["risk_only_rank_diagnostic"] = {
            "note": (
                "Calibration is monotonic per trajectory, but averaging after "
                "the nonlinear transform can still change group ranks."
            ),
            "raw": _metrics(risk_only_rows["raw"]),
            "calibrated": _metrics(risk_only_rows["calibrated"]),
            "selection_overlap": _overlap(
                risk_only_rows["raw"], risk_only_rows["calibrated"]
            ),
        }

    result = {
        "scope": "AgentDojo-v2 repeated-configuration downstream selection",
        "protocol": {
            "split_unit": "suite plus user_task_id",
            "decision_unit": (
                f"one attack configuration, {args.decision_step}-step "
                "prediction averaged over five victim-model seeds"
            ),
            "decision_step": args.decision_step,
            "calibration_mode": args.calibration_mode,
            "validation_rule": (
                "one shared recipe selected by mean validation performance "
                "across checkpoint seeds"
            ),
            "test_rule": "frozen recipe; no test retuning",
            "outcome_metrics": (
                "empirical ASR and BUP from the five repeated victim runs"
            ),
            "recipes_fixed_before_evaluation": RECIPES,
            "budgets_per_user_task": args.budgets,
            "checkpoint_seeds": args.seeds,
        },
        "provenance": {
            "data_root": str(args.data_root.resolve()),
            "model_root": str(args.model_root.resolve()),
            "calibration_root": (
                str(args.calibration_root.resolve())
                if args.calibration_root is not None
                else None
            ),
            "calibration_selection": (
                str(args.calibration_selection.resolve())
                if args.calibration_selection is not None
                else None
            ),
            "calibration_train": (
                str(args.calibration_train.resolve())
                if args.calibration_train is not None
                else None
            ),
        },
        "selected_calibration": selected_calibration,
        "calibrators": calibrators,
        "calibration_fit": calibration_fit,
        "calibration_validation": calibration_validation,
        "counts": {
            split: {
                "steps": len(split_steps[split]),
                "configurations": len(ensemble_rows[split]),
                "user_tasks": len(
                    {row["task_key"] for row in ensemble_rows[split]}
                ),
                "victim_episodes": sum(
                    int(row["trials"]) for row in ensemble_rows[split]
                ),
            }
            for split in ("val", "test")
        },
        "validation_selection": validation_selection,
        "test": test_results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
