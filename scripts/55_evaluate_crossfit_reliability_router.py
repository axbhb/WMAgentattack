"""Test whether a task-level world/text router transfers across suites.

The router search is deliberately small: six predeclared Dreamer span or
disagreement statistics, two threshold directions, and leave-one-suite-out
evaluation.  All 15 tasks already have inspected outcomes, so this remains a
post-hoc cross-fitted diagnostic rather than a fresh confirmation.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FEATURES = (
    "injection_attack_span",
    "injection_utility_span",
    "joint_score_span",
    "mean_absolute_view_gap",
    "mean_seed_disagreement",
    "disagreement_score",
)
SUITES = ("banking", "slack", "travel", "workspace")


def _load_module(filename: str, name: str):
    path = ROOT / "scripts" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = _load_module("49_evaluate_grouped_train_hybrid.py", "hybrid_base")
SELECT = _load_module("46_select_grouped_task_confirmation.py", "grouped_select")
AGG = _load_module("54_analyze_all_grouped_fresh_outcomes.py", "fresh_aggregate")


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _task_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row["suite"]), str(row["user_task_id"])


def _task_primary(
    rows: list[dict[str, Any]],
    method: dict[str, np.ndarray],
    attack: np.ndarray,
    utility: np.ndarray,
    task: tuple[str, str],
) -> float | None:
    indices = np.asarray(
        [index for index, row in enumerate(rows) if _task_key(row) == task]
    )
    task_rows = [rows[index] for index in indices]
    values = []
    for head, rates in (("attack", attack), ("utility", utility)):
        metric = BASE.METRICS._within_task_metrics(
            task_rows, rates[indices], method[f"{head}_rank"][indices]
        )["pairwise_accuracy"]
        if metric is not None:
            values.append(float(metric))
    return float(np.mean(values)) if values else None


def _candidate_thresholds(values: list[float]) -> list[float]:
    unique = sorted(set(float(value) for value in values))
    if not unique:
        raise ValueError("No finite threshold values")
    margin = max(unique[-1] - unique[0], 1.0)
    thresholds = [unique[0] - margin]
    thresholds.extend(
        0.5 * (left + right) for left, right in zip(unique, unique[1:])
    )
    thresholds.append(unique[-1] + margin)
    return thresholds


def _route(value: float, direction: str, threshold: float) -> bool:
    if direction == "high":
        return value >= threshold
    if direction == "low":
        return value <= threshold
    raise ValueError(f"Unknown direction: {direction}")


def _select_stump(
    train_tasks: list[tuple[str, str]],
    statistics: dict[tuple[str, str], dict[str, float]],
    hybrid_scores: dict[tuple[str, str], float | None],
    text_scores: dict[tuple[str, str], float | None],
) -> dict[str, Any]:
    informative = [
        task
        for task in train_tasks
        if hybrid_scores[task] is not None and text_scores[task] is not None
    ]
    if len(informative) < 4:
        raise ValueError("Too few informative training tasks for router selection")
    candidates = []
    for feature in FEATURES:
        thresholds = _candidate_thresholds(
            [statistics[task][feature] for task in informative]
        )
        for direction in ("high", "low"):
            for threshold in thresholds:
                routes = {
                    task: _route(statistics[task][feature], direction, threshold)
                    for task in informative
                }
                scores = [
                    hybrid_scores[task] if routes[task] else text_scores[task]
                    for task in informative
                ]
                candidates.append(
                    {
                        "feature": feature,
                        "direction": direction,
                        "threshold": threshold,
                        "train_mean_primary": float(np.mean(scores)),
                        "train_world_route_count": sum(routes.values()),
                        "train_informative_task_count": len(informative),
                    }
                )
    selected = max(
        candidates,
        key=lambda row: (
            row["train_mean_primary"],
            -row["train_world_route_count"],
            row["feature"],
            row["direction"],
            -row["threshold"],
        ),
    )
    return {**selected, "candidate_count": len(candidates)}


def _routed_method(
    rows: list[dict[str, Any]],
    hybrid: dict[str, np.ndarray],
    text: dict[str, np.ndarray],
    routes: dict[tuple[str, str], bool],
) -> dict[str, np.ndarray]:
    mask = np.asarray([routes[_task_key(row)] for row in rows], dtype=bool)
    return {
        field: np.where(mask, hybrid[field], text[field])
        for field in hybrid
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-predictions", type=Path, required=True)
    parser.add_argument("--enriched-fresh-predictions", type=Path, required=True)
    parser.add_argument("--remaining-fresh-predictions", type=Path, required=True)
    parser.add_argument("--source", action="append", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=20000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260721)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    candidate_rows = _load(args.candidate_predictions).get("pairs")
    enriched = _load(args.enriched_fresh_predictions).get("pairs")
    remaining = _load(args.remaining_fresh_predictions).get("pairs")
    if not all(isinstance(rows, list) for rows in (candidate_rows, enriched, remaining)):
        raise ValueError("A required pair list is missing")
    fresh = [*enriched, *remaining]
    tasks = sorted({_task_key(row) for row in fresh})
    if len(tasks) != 15:
        raise ValueError("Reliability diagnostic expects 15 grouped-test tasks")

    sources = SELECT._parse_sources(args.source)
    reference, mappings = SELECT._align_sources(sources)
    annotated = SELECT._annotate_candidates(reference, mappings)
    grouped = {}
    for row in annotated:
        grouped.setdefault(_task_key(row), []).append(row)
    statistics = {
        task: SELECT._task_statistics(rows) for task, rows in grouped.items()
    }
    if set(statistics) != set(tasks):
        raise ValueError("Task statistics do not align with all fresh tasks")

    candidate_mapping = {AGG._key(row): row for row in candidate_rows}
    rank_mapping = AGG._candidate_task_ranks(candidate_rows)
    enriched_tasks = {_task_key(row) for row in enriched}
    methods = AGG._cohort_methods(
        fresh, candidate_mapping, rank_mapping, enriched_tasks
    )
    evaluation_rows = [
        {
            "suite": row["suite"],
            "user_task_id": row["user_task_id"],
            "injection_task_id": row["injection_task_id"],
        }
        for row in fresh
    ]
    attack = np.asarray([row["observed_attack_probability"] for row in fresh])
    utility = np.asarray([row["observed_utility_probability"] for row in fresh])
    hybrid_name = "headwise_alpha_0p75_clean_utility_text_probability"
    hybrid = methods[hybrid_name]
    text = methods["text_pointwise"]
    hybrid_scores = {
        task: _task_primary(evaluation_rows, hybrid, attack, utility, task)
        for task in tasks
    }
    text_scores = {
        task: _task_primary(evaluation_rows, text, attack, utility, task)
        for task in tasks
    }

    routes = {}
    folds = []
    for held_suite in SUITES:
        train_tasks = [task for task in tasks if task[0] != held_suite]
        held_tasks = [task for task in tasks if task[0] == held_suite]
        stump = _select_stump(
            train_tasks, statistics, hybrid_scores, text_scores
        )
        held_routes = {
            task: _route(
                statistics[task][stump["feature"]],
                stump["direction"],
                stump["threshold"],
            )
            for task in held_tasks
        }
        routes.update(held_routes)
        folds.append(
            {
                "held_suite": held_suite,
                **stump,
                "held_routes": {
                    "::".join(task): bool(route)
                    for task, route in held_routes.items()
                },
            }
        )
    if set(routes) != set(tasks):
        raise AssertionError("Cross-fitted routes do not cover all tasks")
    crossfit = _routed_method(evaluation_rows, hybrid, text, routes)
    oracle_routes = {
        task: (
            hybrid_scores[task] is not None
            and text_scores[task] is not None
            and hybrid_scores[task] > text_scores[task]
        )
        for task in tasks
    }
    oracle = _routed_method(evaluation_rows, hybrid, text, oracle_routes)
    evaluated = {
        "clean_raw": methods["clean_raw"],
        "text_pointwise": text,
        "always_world_hybrid": hybrid,
        "prior_stratum_router": methods["stratum_gated_world_or_text"],
        "leave_one_suite_out_stump_router": crossfit,
        "task_oracle_router": oracle,
    }
    results = {
        name: BASE._method_metrics(
            evaluation_rows, attack, utility, method
        )
        for name, method in evaluated.items()
    }
    comparisons = {}
    for index, name in enumerate(
        (
            "prior_stratum_router",
            "leave_one_suite_out_stump_router",
            "task_oracle_router",
        )
    ):
        comparisons[f"{name}__minus__text_pointwise"] = (
            BASE._bootstrap_difference(
                evaluation_rows,
                evaluated[name],
                text,
                attack,
                utility,
                samples=args.bootstrap_samples,
                seed=args.bootstrap_seed + index,
            )
        )
    informative = [
        task
        for task in tasks
        if hybrid_scores[task] is not None and text_scores[task] is not None
    ]
    route_accuracy = float(
        np.mean([routes[task] == oracle_routes[task] for task in informative])
    )
    output = {
        "scope": "posthoc_leave_one_suite_out_world_text_reliability_router",
        "fresh_confirmation_claim": False,
        "fixed_search_budget": {
            "features": list(FEATURES),
            "directions": ["high", "low"],
            "thresholds": "midpoints of training-suite task values plus all/none",
            "outer_folds": "leave one AgentDojo suite out",
        },
        "results": results,
        "comparisons": comparisons,
        "folds": folds,
        "crossfit_routes": {
            "::".join(task): bool(route) for task, route in sorted(routes.items())
        },
        "oracle_routes": {
            "::".join(task): bool(route)
            for task, route in sorted(oracle_routes.items())
        },
        "crossfit_route_accuracy_on_informative_tasks": route_accuracy,
        "interpretation_rule": (
            "A deployable reliability claim requires the leave-one-suite-out router, "
            "not the prior-stratum or task-oracle descriptive upper bounds, to beat text."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
