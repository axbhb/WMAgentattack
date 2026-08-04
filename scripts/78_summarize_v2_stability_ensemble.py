"""Summarize frozen stability-aware aggregators over 20 grouped OOF tasks."""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
FIXED_METHODS = (
    "mean_score",
    "mean_borda",
    "rank_lcb_0p5",
    "consensus_borda_0p25",
)
METHODS = FIXED_METHODS + ("validation_selected",)
SIZES = ("pct25", "pct100")
BUDGETS = (1, 2, 4)
FOLDS = tuple(range(5))
OUTCOME_METRICS = ("ASR", "BUP", "ASR_plus_BUP")
PRIMARY_METHOD = "rank_lcb_0p5"
SUMMARY_SCOPE = "20-task grouped OOF stability-aware ensemble comparison"


def _load_base():
    path = ROOT / "scripts" / "75_summarize_v2_grouped_oof.py"
    spec = importlib.util.spec_from_file_location("v2_oof_summary", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = _load_base()


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _ids_by_task(
    group_ids: list[str], outcomes: dict[str, dict[str, Any]]
) -> dict[str, set[str]]:
    output: dict[str, set[str]] = defaultdict(set)
    for group_id in group_ids:
        output[outcomes[group_id]["task_key"]].add(group_id)
    return dict(output)


def _selection_overlap(
    baseline_ids: list[str],
    method_ids: list[str],
    outcomes: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    baseline = _ids_by_task(baseline_ids, outcomes)
    method = _ids_by_task(method_ids, outcomes)
    if set(baseline) != set(method):
        raise ValueError("Selection methods cover different tasks")
    per_task = {}
    for task in sorted(baseline):
        union = baseline[task] | method[task]
        per_task[task] = len(baseline[task] & method[task]) / len(union)
    return {
        "mean_task_jaccard": statistics.fmean(per_task.values()),
        "exact_task_set_match_count": sum(value == 1.0 for value in per_task.values()),
        "zero_overlap_task_count": sum(value == 0.0 for value in per_task.values()),
        "per_task_jaccard": per_task,
    }


def _domain_contrast(paired: dict[str, Any]) -> dict[str, Any]:
    metrics = paired["metrics"]
    tasks = metrics["ASR_plus_BUP"]["per_task_delta"]
    output = {}
    for domain in sorted({task.split("|", 1)[0] for task in tasks}):
        domain_tasks = [task for task in tasks if task.startswith(f"{domain}|")]
        output[domain] = {
            "task_count": len(domain_tasks),
            **{
                metric: statistics.fmean(
                    float(metrics[metric]["per_task_delta"][task])
                    for task in domain_tasks
                )
                for metric in OUTCOME_METRICS
            },
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--source-oof-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=20260715)
    args = parser.parse_args()
    if args.draws < 10000:
        parser.error("--draws must be at least 10000")

    source_summary = _load(args.source_oof_root / "final_summary.json")
    outcomes = {}
    for fold in FOLDS:
        fold_outcomes = BASE._configuration_outcomes(
            args.source_oof_root
            / "data"
            / f"fold{fold}"
            / "full"
            / "test_steps.jsonl"
        )
        if set(outcomes) & set(fold_outcomes):
            raise ValueError("OOF test configuration overlap across folds")
        outcomes.update(fold_outcomes)
    if len(outcomes) != 400:
        raise ValueError("Expected 400 OOF configurations")

    fold_results = {
        size: {
            fold: _load(
                args.archive_root / f"fold{fold}" / size / "result.json"
            )
            for fold in FOLDS
        }
        for size in SIZES
    }
    random_nulls = {
        budget: BASE._random_null(
            outcomes,
            budget=budget,
            draws=args.draws,
            seed=args.seed + budget,
        )
        for budget in BUDGETS
    }

    pooled: dict[str, dict[str, dict[str, Any]]] = {
        size: {} for size in SIZES
    }
    by_task: dict[str, dict[int, dict[str, dict[str, dict[str, float]]]]] = {
        size: {} for size in SIZES
    }
    selected_ids: dict[str, dict[int, dict[str, list[str]]]] = {
        size: {} for size in SIZES
    }
    for size in SIZES:
        for budget in BUDGETS:
            selected_ids[size][budget] = {method: [] for method in METHODS}
            recipe_counts = Counter()
            selected_aggregator_counts = Counter()
            for fold in FOLDS:
                row = fold_results[size][fold]["results"][str(budget)]
                recipe_counts[row["frozen_validation_recipe"]] += 1
                selected_aggregator_counts[row["selected_aggregator"]] += 1
                for method in FIXED_METHODS:
                    selected_ids[size][budget][method].extend(
                        row["test_aggregators"][method]["selected_group_ids"]
                    )
                selected_ids[size][budget]["validation_selected"].extend(
                    row["validation_selected_test"]["selected_group_ids"]
                )
            expected_count = budget * 20
            if any(
                len(ids) != expected_count
                for ids in selected_ids[size][budget].values()
            ):
                raise ValueError("Unexpected pooled selection count")

            source_ids = set(
                source_summary["prospective_oof"][size][str(budget)][
                    "selected_group_ids"
                ]
            )
            if set(selected_ids[size][budget]["mean_score"]) != source_ids:
                raise ValueError("mean_score does not reproduce source OOF selection")

            by_task[size][budget] = {}
            method_results = {}
            for method in METHODS:
                method_by_task = BASE._task_selected_values(
                    selected_ids[size][budget][method], outcomes
                )
                by_task[size][budget][method] = method_by_task
                bootstrap = BASE._task_bootstrap(
                    method_by_task,
                    draws=args.draws,
                    seed=args.seed + 1000 + budget,
                )
                method_results[method] = {
                    "metrics": bootstrap,
                    "randomization": {
                        metric: BASE._random_comparison(
                            bootstrap[metric]["estimate"],
                            random_nulls[budget][metric],
                        )
                        for metric in OUTCOME_METRICS
                    },
                    "selected_configuration_count": len(
                        selected_ids[size][budget][method]
                    ),
                    "selected_group_ids": selected_ids[size][budget][method],
                    "by_task": method_by_task,
                    "overlap_with_mean_score": _selection_overlap(
                        selected_ids[size][budget]["mean_score"],
                        selected_ids[size][budget][method],
                        outcomes,
                    ),
                }
            pooled[size][str(budget)] = {
                "methods": method_results,
                "frozen_recipe_fold_counts": dict(recipe_counts),
                "validation_selected_aggregator_fold_counts": dict(
                    selected_aggregator_counts
                ),
                "mean_score_exactly_reproduces_source": True,
            }

    contrasts = {size: {} for size in SIZES}
    for size in SIZES:
        for budget in BUDGETS:
            contrasts[size][str(budget)] = {
                method: BASE._paired_contrast(
                    by_task[size][budget]["mean_score"],
                    by_task[size][budget][method],
                    draws=args.draws,
                    seed=args.seed + 2000 + budget,
                )
                for method in METHODS
                if method != "mean_score"
            }

    fixed_primary_p = {
        method: contrasts["pct100"]["1"][method]["metrics"][
            "ASR_plus_BUP"
        ]["exact_sign_flip_one_sided_p"]
        for method in FIXED_METHODS
        if method != "mean_score"
    }
    fixed_primary_holm = BASE._holm(fixed_primary_p)
    primary = contrasts["pct100"]["1"][PRIMARY_METHOD]

    payload = {
        "scope": SUMMARY_SCOPE,
        "protocol": {
            "primary_contrast": (
                f"pct100 {PRIMARY_METHOD} minus pct100 mean_score at Top-1"
            ),
            "fixed_methods": list(FIXED_METHODS),
            "exploratory_method": "validation_selected",
            "budgets_per_task": list(BUDGETS),
            "task_count": 20,
            "draws": args.draws,
            "test_labels_used_for_method_selection": False,
            "adaptive_to_prior_oof_diagnostic": True,
        },
        "baseline_reproduction": {
            "all_size_budget_selections_exact": True,
            "source_summary": str(args.source_oof_root / "final_summary.json"),
        },
        "pooled_oof": pooled,
        "paired_contrast_vs_mean_score": contrasts,
        "primary_result": primary,
        "fixed_method_primary_exact_p": fixed_primary_p,
        "fixed_method_primary_holm_p": fixed_primary_holm,
        "primary_domain_contrast": _domain_contrast(primary),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
