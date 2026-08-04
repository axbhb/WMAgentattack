"""Summarize 20-task grouped OOF AgentDojo-v2 scale experiments."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


FOLDS = tuple(range(5))
SIZES = ("pct25", "pct100")
SEEDS = (7, 13, 21)
BUDGETS = (1, 2, 4)
METRICS = (
    "validation_objective",
    "grouped_risk_probability_brier_score",
    "grouped_risk_probability_mae",
    "grouped_utility_probability_brier_score",
    "grouped_utility_probability_mae",
    "grouped_preservation_probability_brier_score",
    "grouped_preservation_probability_mae",
    "risk_auc",
    "next_skill_accuracy",
)
OUTCOME_METRICS = ("ASR", "BUP", "ASR_plus_BUP")
SIGN_FLIP_ATOL = 1e-12


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _mean_std(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def _configuration_outcomes(path: Path) -> dict[str, dict[str, Any]]:
    steps = _read_jsonl(path)
    first: dict[str, dict[str, Any]] = {}
    for step in steps:
        trajectory_id = str(step["trajectory_id"])
        previous = first.get(trajectory_id)
        if previous is None or int(step["step_id"]) < int(previous["step_id"]):
            first[trajectory_id] = step
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for step in first.values():
        if (
            step.get("multiseed_group_id") is not None
            and step.get("attack_probability_target") is not None
        ):
            grouped[str(step["multiseed_group_id"])].append(step)
    output = {}
    for group_id, rows in grouped.items():
        task_keys = {f"{row['domain']}|{row['task_id']}" for row in rows}
        expected = {
            int(row["multiseed_trials"])
            for row in rows
            if row.get("multiseed_trials") is not None
        }
        if len(task_keys) != 1 or expected != {len(rows)}:
            raise ValueError(f"Invalid repeated group: {group_id}")
        asr = float(np.mean([bool(row["attack_success"]) for row in rows]))
        bup = float(np.mean([bool(row["task_success"]) for row in rows]))
        output[group_id] = {
            "group_id": group_id,
            "task_key": next(iter(task_keys)),
            "ASR": asr,
            "BUP": bup,
            "ASR_plus_BUP": asr + bup,
            "trials": len(rows),
        }
    if len(output) != 80:
        raise ValueError(f"Expected 80 test configurations in {path}")
    return output


def _task_selected_values(
    selected_ids: list[str], outcomes: dict[str, dict[str, Any]]
) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for group_id in selected_ids:
        grouped[outcomes[group_id]["task_key"]].append(outcomes[group_id])
    return {
        task_key: {
            metric: float(np.mean([row[metric] for row in rows]))
            for metric in OUTCOME_METRICS
        }
        for task_key, rows in grouped.items()
    }


def _task_bootstrap(
    by_task: dict[str, dict[str, float]], *, draws: int, seed: int
) -> dict[str, Any]:
    task_keys = sorted(by_task)
    values = np.asarray(
        [[by_task[key][metric] for metric in OUTCOME_METRICS] for key in task_keys],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(task_keys), size=(draws, len(task_keys)))
    samples = values[indices].mean(axis=1)
    return {
        metric: {
            "estimate": float(values[:, index].mean()),
            "ci95_low": float(np.quantile(samples[:, index], 0.025)),
            "ci95_high": float(np.quantile(samples[:, index], 0.975)),
        }
        for index, metric in enumerate(OUTCOME_METRICS)
    }


def _random_null(
    outcomes: dict[str, dict[str, Any]],
    *,
    budget: int,
    draws: int,
    seed: int,
) -> dict[str, np.ndarray]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in outcomes.values():
        grouped[row["task_key"]].append(row)
    samples = {
        metric: np.zeros(draws, dtype=np.float64)
        for metric in OUTCOME_METRICS
    }
    rng = np.random.default_rng(seed)
    for task_key in sorted(grouped):
        rows = grouped[task_key]
        if len(rows) != 20:
            raise ValueError(f"Expected 20 candidates for {task_key}")
        indices = np.argpartition(
            rng.random((draws, len(rows))), budget - 1, axis=1
        )[:, :budget]
        for metric in OUTCOME_METRICS:
            values = np.asarray([row[metric] for row in rows], dtype=np.float64)
            samples[metric] += values[indices].mean(axis=1)
    for metric in OUTCOME_METRICS:
        samples[metric] /= len(grouped)
    return samples


def _random_comparison(
    observed: float, random: np.ndarray
) -> dict[str, float]:
    return {
        "observed": observed,
        "random_mean": float(random.mean()),
        "random_std": float(random.std()),
        "random_ci95_low": float(np.quantile(random, 0.025)),
        "random_ci95_high": float(np.quantile(random, 0.975)),
        "random_percentile": float(np.mean(random <= observed)),
        "one_sided_p_random_at_least_observed": float(
            (1 + np.sum(random >= observed)) / (len(random) + 1)
        ),
    }


def _paired_contrast(
    left: dict[str, dict[str, float]],
    right: dict[str, dict[str, float]],
    *,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    if set(left) != set(right):
        raise ValueError("Paired methods cover different OOF tasks")
    task_keys = sorted(left)
    deltas = np.asarray(
        [
            [right[key][metric] - left[key][metric] for metric in OUTCOME_METRICS]
            for key in task_keys
        ],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(task_keys), size=(draws, len(task_keys)))
    boot = deltas[indices].mean(axis=1)
    signs = rng.choice(np.asarray([-1.0, 1.0]), size=(draws, len(task_keys)))
    sign_flip = (signs[:, :, None] * deltas[None, :, :]).mean(axis=1)
    output = {}
    for index, metric in enumerate(OUTCOME_METRICS):
        observed = float(deltas[:, index].mean())
        exact_p, exact_assignments = _exact_sign_flip_one_sided_p(
            deltas[:, index], observed
        )
        output[metric] = {
            "observed_mean_delta": observed,
            "observed_delta_pct100_minus_pct25": observed,
            "bootstrap_ci95_low": float(np.quantile(boot[:, index], 0.025)),
            "bootstrap_ci95_high": float(np.quantile(boot[:, index], 0.975)),
            "bootstrap_probability_gt_zero": float(np.mean(boot[:, index] > 0.0)),
            "sign_flip_one_sided_p": float(
                (
                    1
                    +
                    np.sum(
                        sign_flip[:, index]
                        >= observed - SIGN_FLIP_ATOL
                    )
                )
                / (draws + 1)
            ),
            "exact_sign_flip_one_sided_p": exact_p,
            "exact_sign_flip_effective_assignment_count": exact_assignments,
            "positive_task_count": int(np.sum(deltas[:, index] > 0.0)),
            "tie_task_count": int(np.sum(deltas[:, index] == 0.0)),
            "negative_task_count": int(np.sum(deltas[:, index] < 0.0)),
            "per_task_delta": {
                key: float(value)
                for key, value in zip(
                    task_keys, deltas[:, index], strict=True
                )
            },
        }
    return {
        "unit": "OOF held-out user task",
        "task_count": len(task_keys),
        "draws": draws,
        "metrics": output,
    }


def _exact_sign_flip_one_sided_p(
    deltas: np.ndarray,
    observed: float,
    *,
    max_effective_tasks: int = 24,
    chunk_size: int = 65536,
) -> tuple[float | None, int | None]:
    """Enumerate non-zero paired signs, ignoring duplicated zero assignments."""
    active = np.asarray(deltas, dtype=np.float64)
    active = active[np.abs(active) > SIGN_FLIP_ATOL]
    if len(active) > max_effective_tasks:
        return None, None
    assignment_count = 1 << len(active)
    extreme_count = 0
    bits = np.arange(len(active), dtype=np.uint64)
    for start in range(0, assignment_count, chunk_size):
        stop = min(start + chunk_size, assignment_count)
        assignments = np.arange(start, stop, dtype=np.uint64)[:, None]
        signs = 1.0 - 2.0 * ((assignments >> bits) & 1)
        statistics = (signs * active).sum(axis=1) / len(deltas)
        extreme_count += int(
            np.count_nonzero(statistics >= observed - SIGN_FLIP_ATOL)
        )
    return extreme_count / assignment_count, assignment_count


def _aggregate_model_metrics(root: Path, size: str) -> dict[str, Any]:
    runs = {}
    per_fold = {}
    for fold in FOLDS:
        fold_runs = {}
        for seed in SEEDS:
            metrics = _load(
                root
                / "models"
                / f"fold{fold}"
                / size
                / f"seed{seed}"
                / "test_metrics.json"
            )["metrics"]
            fold_runs[str(seed)] = metrics
            runs[f"fold{fold}_seed{seed}"] = metrics
        per_fold[str(fold)] = {
            metric: _mean_std(
                [float(fold_runs[str(seed)][metric]) for seed in SEEDS]
            )
            for metric in METRICS
        }
    return {
        "overall_15_fold_seed_runs": {
            metric: _mean_std([float(row[metric]) for row in runs.values()])
            for metric in METRICS
        },
        "per_fold_three_seed_mean": per_fold,
        "runs": runs,
    }


def _model_contrast(
    left: dict[str, Any], right: dict[str, Any]
) -> dict[str, Any]:
    if set(left["runs"]) != set(right["runs"]):
        raise ValueError("Model scale runs are not paired")
    output = {}
    for metric in METRICS:
        values = {
            run: float(right["runs"][run][metric])
            - float(left["runs"][run][metric])
            for run in sorted(left["runs"])
        }
        output[metric] = {
            "pct100_minus_pct25": _mean_std(list(values.values())),
            "positive_run_count": sum(value > 0.0 for value in values.values()),
            "tie_run_count": sum(value == 0.0 for value in values.values()),
            "negative_run_count": sum(value < 0.0 for value in values.values()),
            "per_fold_seed": values,
        }
    return output


def _holm(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values, key=p_values.get)
    adjusted = {}
    running = 0.0
    count = len(ordered)
    for rank, key in enumerate(ordered):
        value = min(1.0, (count - rank) * p_values[key])
        running = max(running, value)
        adjusted[key] = running
    return adjusted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=20260715)
    args = parser.parse_args()
    if args.draws < 10000:
        parser.error("--draws must be at least 10000")

    data_summary = _load(args.archive_root / "data" / "summary.json")
    all_outcomes = {}
    fold_outcomes = {}
    results: dict[str, dict[int, dict[str, Any]]] = {
        size: {} for size in SIZES
    }
    for fold in FOLDS:
        outcomes = _configuration_outcomes(
            args.archive_root
            / "data"
            / f"fold{fold}"
            / "full"
            / "test_steps.jsonl"
        )
        if set(all_outcomes) & set(outcomes):
            raise ValueError("OOF test configuration overlap across folds")
        all_outcomes.update(outcomes)
        fold_outcomes[fold] = outcomes
        for size in SIZES:
            results[size][fold] = _load(
                args.archive_root
                / "prospective"
                / f"fold{fold}"
                / size
                / "result.json"
            )
    if len(all_outcomes) != 400:
        raise ValueError("OOF test folds do not cover all 400 configurations")
    if len({row["task_key"] for row in all_outcomes.values()}) != 20:
        raise ValueError("OOF outcomes do not cover all 20 tasks")

    random_nulls = {
        budget: _random_null(
            all_outcomes,
            budget=budget,
            draws=args.draws,
            seed=args.seed + budget,
        )
        for budget in BUDGETS
    }
    prospective: dict[str, dict[str, Any]] = {size: {} for size in SIZES}
    selected_by_size_budget: dict[str, dict[int, dict[str, dict[str, float]]]] = {
        size: {} for size in SIZES
    }
    for size in SIZES:
        for budget in BUDGETS:
            selected_ids = []
            per_fold = {}
            checkpoint_runs = []
            recipes = Counter()
            calibration_choices = Counter()
            for fold in FOLDS:
                result = results[size][fold]
                row = result["test"][str(budget)]["calibrated"]
                ids = row["ensemble"]["selected_group_ids"]
                selected_ids.extend(ids)
                per_fold[str(fold)] = {
                    "test_cohort": data_summary["folds"][f"fold{fold}"][
                        "test_cohort"
                    ],
                    "selected_recipe": row["frozen_validation_recipe"],
                    "ASR": row["ensemble"]["ASR"],
                    "BUP": row["ensemble"]["BUP"],
                    "ASR_plus_BUP": row["ensemble"]["ASR_plus_BUP"],
                    "selected_group_ids": ids,
                }
                recipes[row["frozen_validation_recipe"]] += 1
                calibration_choices[result["selected_calibration"]] += 1
                checkpoint_runs.extend(
                    [
                        run["ASR_plus_BUP"]
                        for run in row["per_seed"].values()
                    ]
                )
            if len(selected_ids) != budget * 20:
                raise ValueError("Unexpected OOF selected configuration count")
            by_task = _task_selected_values(selected_ids, all_outcomes)
            if len(by_task) != 20:
                raise ValueError("OOF selection does not cover 20 tasks")
            selected_by_size_budget[size][budget] = by_task
            bootstrap = _task_bootstrap(
                by_task, draws=args.draws, seed=args.seed + 1000 + budget
            )
            comparisons = {
                metric: _random_comparison(
                    bootstrap[metric]["estimate"], random_nulls[budget][metric]
                )
                for metric in OUTCOME_METRICS
            }
            prospective[size][str(budget)] = {
                "selected_configuration_count": len(selected_ids),
                "selected_episode_count": len(selected_ids) * 5,
                "task_count": len(by_task),
                "metrics": bootstrap,
                "randomization": comparisons,
                "checkpoint_fold_seed_ASR_plus_BUP": _mean_std(
                    [float(value) for value in checkpoint_runs]
                ),
                "selected_recipe_fold_counts": dict(recipes),
                "selected_calibration_fold_counts": dict(calibration_choices),
                "per_fold": per_fold,
                "by_task": by_task,
                "selected_group_ids": selected_ids,
            }

    for size in SIZES:
        p_values = {
            str(budget): prospective[size][str(budget)]["randomization"][
                "ASR_plus_BUP"
            ]["one_sided_p_random_at_least_observed"]
            for budget in BUDGETS
        }
        adjusted = _holm(p_values)
        for budget in BUDGETS:
            prospective[size][str(budget)]["randomization"][
                "ASR_plus_BUP"
            ]["holm_p_across_three_budgets"] = adjusted[str(budget)]

    contrasts = {
        str(budget): _paired_contrast(
            selected_by_size_budget["pct25"][budget],
            selected_by_size_budget["pct100"][budget],
            draws=args.draws,
            seed=args.seed + 2000 + budget,
        )
        for budget in BUDGETS
    }
    model_metrics = {
        size: _aggregate_model_metrics(args.archive_root, size)
        for size in SIZES
    }
    calibration = {}
    for size in SIZES:
        per_fold = {}
        for fold in FOLDS:
            payload = results[size][fold]["calibration_validation"]
            per_fold[str(fold)] = {
                "selected": payload["selected_candidate"],
                "identity_brier": payload["aggregate_group_risk_brier"][
                    "identity"
                ]["mean"],
                "reg0_brier": payload["aggregate_group_risk_brier"]["reg0"][
                    "mean"
                ],
            }
        calibration[size] = {
            "identity_brier": _mean_std(
                [row["identity_brier"] for row in per_fold.values()]
            ),
            "reg0_brier": _mean_std(
                [row["reg0_brier"] for row in per_fold.values()]
            ),
            "selected_fold_counts": dict(
                Counter(row["selected"] for row in per_fold.values())
            ),
            "per_fold": per_fold,
        }

    summary = {
        "scope": "20-task grouped five-fold OOF AgentDojo-v2 scale test",
        "primary_endpoint": (
            "pct100 minus pct25 first-step calibrated ensemble ASR+BUP, "
            "one selected configuration per OOF user task"
        ),
        "protocol": {
            "task_count": 20,
            "fold_count": 5,
            "tasks_per_test_fold": 4,
            "tasks_per_validation_fold": 4,
            "tasks_per_train_fold": 12,
            "checkpoint_seeds": list(SEEDS),
            "primary_budget_per_task": 1,
            "secondary_budgets_per_task": [2, 4],
            "decision_step": "first",
            "recipe_selection": "fold validation only",
            "calibration_selection": "fold validation only",
            "randomization_draws": args.draws,
            "sampling_is_label_blind": True,
        },
        "data_summary": data_summary,
        "model_metrics": model_metrics,
        "model_metric_contrast": _model_contrast(
            model_metrics["pct25"], model_metrics["pct100"]
        ),
        "first_step_calibration": calibration,
        "prospective_oof": prospective,
        "paired_scale_contrast": contrasts,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
