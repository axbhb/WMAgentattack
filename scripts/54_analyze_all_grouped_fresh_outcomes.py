"""Aggregate all 15 grouped-test tasks after both fresh replay rounds.

This is a post-hoc heterogeneity analysis.  It contrasts the original enriched
eight-task stress cohort with the seven remaining tasks and must not be reported
as another confirmation test.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ALPHA = 0.75


def _load_base_module():
    path = ROOT / "scripts" / "49_evaluate_grouped_train_hybrid.py"
    spec = importlib.util.spec_from_file_location("grouped_hybrid_base", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = _load_base_module()


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row["suite"]),
        str(row["user_task_id"]),
        str(row["injection_task_id"]),
    )


def _task_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row["suite"]), str(row["user_task_id"])


def _candidate_task_ranks(
    candidate_rows: list[dict[str, Any]],
) -> dict[tuple[str, str, str], dict[str, float]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in candidate_rows:
        grouped.setdefault(_task_key(row), []).append(row)
    output = {}
    for task_rows in grouped.values():
        world = np.asarray(
            [
                row["all_methods"]["world_pairwise_c0p03"]["attack_rank"]
                for row in task_rows
            ],
            dtype=float,
        )
        text = np.asarray(
            [row["all_methods"]["text_pointwise"]["attack_rank"] for row in task_rows],
            dtype=float,
        )
        world_rank = BASE._within_task_rank(task_rows, world)
        text_rank = BASE._within_task_rank(task_rows, text)
        for index, row in enumerate(task_rows):
            output[_key(row)] = {
                "world_attack_rank": float(world[index]),
                "world_attack_percentile": float(world_rank[index]),
                "text_attack_percentile": float(text_rank[index]),
                "hybrid_attack_rank": float(
                    ALPHA * world_rank[index] + (1.0 - ALPHA) * text_rank[index]
                ),
            }
    return output


def _cohort_methods(
    rows: list[dict[str, Any]],
    candidate_mapping: dict[tuple[str, str, str], dict[str, Any]],
    rank_mapping: dict[tuple[str, str, str], dict[str, float]],
    world_enabled_tasks: set[tuple[str, str]],
) -> dict[str, dict[str, np.ndarray]]:
    candidates = [candidate_mapping[_key(row)] for row in rows]

    def array(method: str, field: str) -> np.ndarray:
        return np.asarray(
            [candidate["all_methods"][method][field] for candidate in candidates],
            dtype=float,
        )

    clean = BASE._method(
        array("clean_raw", "attack_rank"),
        array("clean_raw", "utility_rank"),
        array("clean_raw", "attack_probability"),
        array("clean_raw", "utility_probability"),
    )
    text = BASE._method(
        array("text_pointwise", "attack_rank"),
        array("text_pointwise", "utility_rank"),
        array("text_pointwise", "attack_probability"),
        array("text_pointwise", "utility_probability"),
    )
    text_attack_probability = array("text_pointwise", "attack_probability")
    text_utility_probability = array("text_pointwise", "utility_probability")
    clean_utility = array("clean_raw", "utility_rank")
    world_attack = np.asarray(
        [rank_mapping[_key(row)]["world_attack_rank"] for row in rows], dtype=float
    )
    hybrid_attack = np.asarray(
        [rank_mapping[_key(row)]["hybrid_attack_rank"] for row in rows], dtype=float
    )
    methods = {
        "clean_raw": clean,
        "text_pointwise": text,
        "world_attack_clean_utility_text_probability": BASE._method(
            world_attack,
            clean_utility,
            text_attack_probability,
            text_utility_probability,
        ),
        "headwise_alpha_0p75_clean_utility_text_probability": BASE._method(
            hybrid_attack,
            clean_utility,
            text_attack_probability,
            text_utility_probability,
        ),
    }
    hybrid = methods["headwise_alpha_0p75_clean_utility_text_probability"]
    route_world = np.asarray(
        [_task_key(row) in world_enabled_tasks for row in rows], dtype=bool
    )
    methods["stratum_gated_world_or_text"] = {
        field: np.where(route_world, hybrid[field], text[field])
        for field in hybrid
    }
    return methods


def _evaluate_cohort(
    rows: list[dict[str, Any]],
    methods: dict[str, dict[str, np.ndarray]],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    evaluation_rows = [
        {
            "suite": row["suite"],
            "user_task_id": row["user_task_id"],
            "injection_task_id": row["injection_task_id"],
        }
        for row in rows
    ]
    attack = np.asarray([row["observed_attack_probability"] for row in rows])
    utility = np.asarray([row["observed_utility_probability"] for row in rows])
    results = {
        name: BASE._method_metrics(evaluation_rows, attack, utility, method)
        for name, method in methods.items()
    }
    selected = "headwise_alpha_0p75_clean_utility_text_probability"
    comparisons = {}
    for method_index, method_name in enumerate(
        (selected, "stratum_gated_world_or_text")
    ):
        for reference_index, reference in enumerate(
            ("clean_raw", "text_pointwise")
        ):
            comparisons[f"{method_name}__minus__{reference}"] = (
                BASE._bootstrap_difference(
                    evaluation_rows,
                    methods[method_name],
                    methods[reference],
                    attack,
                    utility,
                    samples=bootstrap_samples,
                    seed=bootstrap_seed + 10 * method_index + reference_index,
                )
            )
    task_effects = {}
    for task in sorted({_task_key(row) for row in evaluation_rows}):
        indices = [
            index
            for index, row in enumerate(evaluation_rows)
            if _task_key(row) == task
        ]
        task_rows = [evaluation_rows[index] for index in indices]
        task_attack = attack[indices]
        task_utility = utility[indices]

        def task_primary(method: dict[str, np.ndarray]) -> float | None:
            values = []
            for head, rates in (("attack", task_attack), ("utility", task_utility)):
                metrics = BASE.METRICS._within_task_metrics(
                    task_rows, rates, method[f"{head}_rank"][indices]
                )
                if metrics["pairwise_accuracy"] is not None:
                    values.append(float(metrics["pairwise_accuracy"]))
            return float(np.mean(values)) if values else None

        task_effects["::".join(task)] = {
            name: task_primary(method)
            for name, method in methods.items()
        }
    return {
        "pair_count": len(rows),
        "task_count": len({_task_key(row) for row in rows}),
        "attempt_count": 5 * len(rows),
        "observed_asr": float(attack.mean()),
        "observed_bup": float(utility.mean()),
        "results": results,
        "comparisons": comparisons,
        "task_effects": task_effects,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-predictions", type=Path, required=True)
    parser.add_argument("--enriched-fresh-predictions", type=Path, required=True)
    parser.add_argument("--remaining-fresh-predictions", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=20000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260720)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    candidate_rows = _load(args.candidate_predictions).get("pairs")
    enriched = _load(args.enriched_fresh_predictions).get("pairs")
    remaining = _load(args.remaining_fresh_predictions).get("pairs")
    if not all(isinstance(rows, list) for rows in (candidate_rows, enriched, remaining)):
        raise ValueError("A required pair list is missing")
    if len(enriched) != 32 or len(remaining) != 28:
        raise ValueError("Expected 32 enriched and 28 remaining fresh pairs")
    enriched_tasks = {_task_key(row) for row in enriched}
    remaining_tasks = {_task_key(row) for row in remaining}
    if enriched_tasks & remaining_tasks:
        raise ValueError("Fresh cohorts overlap by user task")
    if len(enriched_tasks | remaining_tasks) != 15:
        raise ValueError("Combined fresh cohorts must cover 15 grouped-test tasks")
    candidate_mapping = {_key(row): row for row in candidate_rows}
    rank_mapping = _candidate_task_ranks(candidate_rows)
    for row in [*enriched, *remaining]:
        if _key(row) not in candidate_mapping:
            raise ValueError(f"Fresh pair is absent from candidates: {_key(row)}")

    cohorts = {
        "enriched_prior_8_tasks": enriched,
        "remaining_new_7_tasks": remaining,
        "all_15_grouped_test_tasks": [*enriched, *remaining],
    }
    output = {
        "scope": "posthoc_all_grouped_test_fresh_outcome_heterogeneity",
        "fresh_confirmation_claim": False,
        "total_task_count": 15,
        "total_pair_count": 60,
        "total_fresh_outcome_count": 300,
        "method": (
            "0.75 world-pairwise plus 0.25 text attack percentile rank, "
            "clean utility ordering, text probability reporting"
        ),
        "posthoc_router": {
            "world_route": (
                "tasks selected before the first fresh replay as per-suite high "
                "prediction-span or high view/seed-disagreement stress strata"
            ),
            "fallback_route": "text-pointwise for every other grouped-test task",
            "fresh_confirmation_claim": False,
            "limitation": (
                "The world expert was designed after inspecting outcomes in the "
                "world-routed cohort; this router is a descriptive hypothesis."
            ),
        },
        "cohorts": {
            name: _evaluate_cohort(
                rows,
                _cohort_methods(
                    rows, candidate_mapping, rank_mapping, enriched_tasks
                ),
                bootstrap_samples=args.bootstrap_samples,
                bootstrap_seed=args.bootstrap_seed + 100 * index,
            )
            for index, (name, rows) in enumerate(cohorts.items())
        },
        "interpretation_constraint": (
            "The enriched cohort informed the method and the remaining cohort is "
            "the only new confirmation evidence; the pooled result is descriptive."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
