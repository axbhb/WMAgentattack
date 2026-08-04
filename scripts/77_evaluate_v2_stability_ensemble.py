"""Evaluate frozen stability-aware seed aggregators on AgentDojo-v2 OOF folds."""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _load_base():
    path = ROOT / "scripts" / "70_evaluate_v2_downstream_selection.py"
    spec = importlib.util.spec_from_file_location("v2_downstream_selection", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = _load_base()

AGGREGATOR_ORDER = (
    "mean_score",
    "mean_borda",
    "rank_lcb_0p5",
    "consensus_borda_0p25",
)
RANK_LCB_PENALTY = 0.5
CONSENSUS_BONUS = 0.25
EXPERIMENT_SCOPE = "frozen stability-aware ensemble evaluation"
AGGREGATOR_PARAMETERS = {
    "rank_lcb_penalty": RANK_LCB_PENALTY,
    "consensus_bonus": CONSENSUS_BONUS,
}


def _aggregate_rows(
    rows_by_seed: dict[int, list[dict[str, Any]]],
    *,
    risk_key: str,
    recipe: str,
    budget: int,
) -> dict[str, list[dict[str, Any]]]:
    """Build one aligned candidate table for every frozen aggregator."""
    seeds = sorted(rows_by_seed)
    if len(seeds) < 2:
        raise ValueError("Stability aggregation requires at least two seeds")
    maps = {
        seed: {row["group_id"]: row for row in rows_by_seed[seed]}
        for seed in seeds
    }
    group_ids = set(maps[seeds[0]])
    if any(set(maps[seed]) != group_ids for seed in seeds[1:]):
        raise ValueError("Checkpoint seeds produced different configuration ids")

    reference = maps[seeds[0]]
    by_task: dict[str, list[str]] = defaultdict(list)
    for group_id in group_ids:
        by_task[reference[group_id]["task_key"]].append(group_id)
    if any(len(group_ids) < budget for group_ids in by_task.values()):
        raise ValueError("A task has fewer candidates than the requested budget")

    decision_scores = {
        seed: {
            group_id: BASE._decision_score(row, risk_key, recipe)
            for group_id, row in maps[seed].items()
        }
        for seed in seeds
    }
    normalized_ranks: dict[int, dict[str, float]] = {
        seed: {} for seed in seeds
    }
    top_budget: dict[int, set[str]] = {seed: set() for seed in seeds}
    for seed in seeds:
        for task_key in sorted(by_task):
            ordered = sorted(
                by_task[task_key],
                key=lambda group_id: (
                    -decision_scores[seed][group_id],
                    group_id,
                ),
            )
            denominator = max(1, len(ordered) - 1)
            for rank, group_id in enumerate(ordered):
                normalized_ranks[seed][group_id] = 1.0 - rank / denominator
            top_budget[seed].update(ordered[:budget])

    output = {aggregator: [] for aggregator in AGGREGATOR_ORDER}
    for group_id in sorted(group_ids):
        seed_scores = [decision_scores[seed][group_id] for seed in seeds]
        seed_ranks = [normalized_ranks[seed][group_id] for seed in seeds]
        mean_rank = statistics.fmean(seed_ranks)
        rank_std = statistics.pstdev(seed_ranks)
        vote_fraction = statistics.fmean(
            float(group_id in top_budget[seed]) for seed in seeds
        )
        aggregate_scores = {
            "mean_score": statistics.fmean(seed_scores),
            "mean_borda": mean_rank,
            "rank_lcb_0p5": mean_rank - RANK_LCB_PENALTY * rank_std,
            "consensus_borda_0p25": (
                mean_rank + CONSENSUS_BONUS * vote_fraction
            ),
        }
        for aggregator, aggregate_score in aggregate_scores.items():
            output[aggregator].append(
                {
                    **reference[group_id],
                    "decision_score": float(aggregate_score),
                    "aggregation_score": float(aggregate_score),
                    "seed_decision_scores": {
                        str(seed): float(decision_scores[seed][group_id])
                        for seed in seeds
                    },
                    "seed_normalized_ranks": {
                        str(seed): float(normalized_ranks[seed][group_id])
                        for seed in seeds
                    },
                    "rank_std": float(rank_std),
                    "top_budget_vote_fraction": float(vote_fraction),
                }
            )
    return output


def _select_aggregated(
    rows: list[dict[str, Any]], budget: int
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["task_key"]].append(row)
    selected = []
    for task_key in sorted(grouped):
        selected.extend(
            sorted(
                grouped[task_key],
                key=lambda row: (-row["aggregation_score"], row["group_id"]),
            )[:budget]
        )
    return selected


def _choose_aggregator(metrics: dict[str, dict[str, Any]]) -> str:
    order = {name: index for index, name in enumerate(AGGREGATOR_ORDER)}
    return max(
        AGGREGATOR_ORDER,
        key=lambda name: (
            float(metrics[name]["ASR_plus_BUP"]),
            float(metrics[name]["BUP"]),
            float(metrics[name]["ASR"]),
            -order[name],
        ),
    )


def _ids(rows: list[dict[str, Any]]) -> list[str]:
    return [str(row["group_id"]) for row in rows]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--calibration-train", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 13, 21])
    parser.add_argument("--budgets", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260715)
    args = parser.parse_args()
    if any(budget < 1 for budget in args.budgets):
        parser.error("--budgets must contain positive integers")

    split_steps = {
        split: BASE._steps(args.data_root / f"{split}_steps.jsonl")
        for split in ("val", "test")
    }
    calibration_train_steps = BASE._steps(args.calibration_train)
    rows_by_split_seed: dict[str, dict[int, list[dict[str, Any]]]] = {
        "val": {},
        "test": {},
    }
    calibrators = {}
    calibration_fit = {}
    for seed in args.seeds:
        model = BASE.FullSheepRLDreamerV3.load(
            args.model_root / f"seed{seed}" / "model"
        )
        train_predictions = model.predict(calibration_train_steps)
        calibrator, fit_stats = BASE._fit_decision_time_calibrator(
            calibration_train_steps, train_predictions
        )
        calibrators[str(seed)] = calibrator.to_dict()
        calibration_fit[str(seed)] = fit_stats
        for split, steps in split_steps.items():
            rows_by_split_seed[split][seed] = BASE._configuration_rows(
                steps,
                model.predict(steps),
                calibrator,
                decision_step="first",
            )
        del model
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    by_candidate = {
        "identity": {
            seed: BASE._group_risk_brier(rows, "raw_risk_score")
            for seed, rows in rows_by_split_seed["val"].items()
        },
        "reg0": {
            seed: BASE._group_risk_brier(rows, "calibrated_risk_score")
            for seed, rows in rows_by_split_seed["val"].items()
        },
    }
    aggregate_calibration = {
        candidate: BASE._mean_std(list(values.values()))
        for candidate, values in by_candidate.items()
    }
    selected_calibration = min(
        aggregate_calibration,
        key=lambda candidate: aggregate_calibration[candidate]["mean"],
    )
    if selected_calibration == "identity":
        for split in ("val", "test"):
            for rows in rows_by_split_seed[split].values():
                for row in rows:
                    row["calibrated_risk_score"] = row["raw_risk_score"]

    results = {}
    for budget in args.budgets:
        validation_recipe = BASE._validation_select(
            rows_by_split_seed["val"],
            risk_key="calibrated_risk_score",
            budget=budget,
        )
        recipe = validation_recipe["selected_recipe"]
        aggregated = {
            split: _aggregate_rows(
                rows_by_split_seed[split],
                risk_key="calibrated_risk_score",
                recipe=recipe,
                budget=budget,
            )
            for split in ("val", "test")
        }
        selections = {
            split: {
                aggregator: _select_aggregated(rows, budget)
                for aggregator, rows in by_aggregator.items()
            }
            for split, by_aggregator in aggregated.items()
        }
        validation_metrics = {
            aggregator: BASE._metrics(rows)
            for aggregator, rows in selections["val"].items()
        }
        selected_aggregator = _choose_aggregator(validation_metrics)

        baseline_reference = BASE._select(
            BASE._ensemble(rows_by_split_seed["test"]),
            risk_key="calibrated_risk_score",
            recipe=recipe,
            budget_per_task=budget,
        )
        if _ids(baseline_reference) != _ids(selections["test"]["mean_score"]):
            raise RuntimeError("mean_score failed to reproduce the existing ensemble")

        test_metrics = {}
        for aggregator, rows in selections["test"].items():
            test_metrics[aggregator] = {
                **BASE._metrics(rows),
                "task_bootstrap": BASE._task_bootstrap(
                    rows,
                    draws=args.bootstrap_draws,
                    seed=args.bootstrap_seed + budget,
                ),
            }
        results[str(budget)] = {
            "frozen_validation_recipe": recipe,
            "validation_recipe_selection": validation_recipe,
            "validation_aggregators": validation_metrics,
            "selected_aggregator": selected_aggregator,
            "test_aggregators": test_metrics,
            "validation_selected_test": {
                "selected_aggregator": selected_aggregator,
                **test_metrics[selected_aggregator],
            },
            "baseline_reproduction": {
                "identical_selected_group_ids": True,
                "selected_group_ids": _ids(baseline_reference),
            },
        }

    payload = {
        "scope": EXPERIMENT_SCOPE,
        "protocol": {
            "decision_step": "first",
            "risk_variant": "validation-selected identity or reg0",
            "recipe_selection": (
                "existing mean per-seed validation ASR+BUP rule"
            ),
            "aggregators": list(AGGREGATOR_ORDER),
            "aggregator_parameters": AGGREGATOR_PARAMETERS,
            "aggregator_selection": (
                "validation ASR+BUP, then BUP, ASR, fixed order"
            ),
            "budgets_per_task": args.budgets,
            "checkpoint_seeds": args.seeds,
        },
        "provenance": {
            "data_root": str(args.data_root),
            "model_root": str(args.model_root),
            "calibration_train": str(args.calibration_train),
        },
        "counts": {
            split: {
                "configurations": len(rows_by_split_seed[split][args.seeds[0]]),
                "tasks": len(
                    {
                        row["task_key"]
                        for row in rows_by_split_seed[split][args.seeds[0]]
                    }
                ),
            }
            for split in ("val", "test")
        },
        "calibrators": calibrators,
        "calibration_fit": calibration_fit,
        "calibration_validation": {
            "per_seed_group_risk_brier": by_candidate,
            "aggregate_group_risk_brier": aggregate_calibration,
            "selected_candidate": selected_calibration,
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
