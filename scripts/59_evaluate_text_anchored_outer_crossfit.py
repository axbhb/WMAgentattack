"""Evaluate a conservative text-anchored use of outer-crossfit world scores.

The world-assisted method may change only attack ordering.  Utility ordering
and both reported probabilities are copied from the strongest text baseline.
If the attack correction does not pass a validation-only gain/non-inferiority
gate, deployment falls back exactly to the text baseline.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


POST = _load_module(
    "outer_crossfit_fresh_diagnostic",
    ROOT / "scripts" / "58_diagnose_outer_crossfit_on_fresh_outcomes.py",
)
BASE = POST.BASE
SAFE_NAME = "outer_crossfit_attack_text_utility_probability"


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _safe_methods(
    candidates: list[dict[str, Any]], selected_method: str, text_method: str
) -> dict[str, dict[str, np.ndarray]]:
    clean = POST._method_from_candidates(candidates, "clean_raw")
    text = POST._method_from_candidates(candidates, text_method)
    selected = POST._method_from_candidates(candidates, selected_method)
    safe = BASE._method(
        selected["attack_rank"],
        text["utility_rank"],
        text["attack_probability"],
        text["utility_probability"],
    )
    return {
        "clean_raw": clean,
        text_method: text,
        selected_method: selected,
        SAFE_NAME: safe,
    }


def _head_task_difference(
    candidate_result: dict[str, Any],
    reference_result: dict[str, Any],
    *,
    head: str,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    candidate_tasks = candidate_result[head]["within_task"]["per_task"]
    reference_tasks = reference_result[head]["within_task"]["per_task"]
    task_names = sorted(set(candidate_tasks) & set(reference_tasks))
    differences = []
    used_tasks = []
    for name in task_names:
        candidate = candidate_tasks[name]["pairwise_accuracy"]
        reference = reference_tasks[name]["pairwise_accuracy"]
        if candidate is None or reference is None:
            continue
        differences.append(float(candidate) - float(reference))
        used_tasks.append(name)
    if not differences:
        raise ValueError(f"No informative {head} tasks")
    values = np.asarray(differences, dtype=float)
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(samples, len(values)), replace=True).mean(axis=1)
    return {
        "mean_task_difference": float(values.mean()),
        "mean_task_difference_95ci": [
            float(np.quantile(draws, 0.025)),
            float(np.quantile(draws, 0.975)),
        ],
        "informative_task_count": len(values),
        "per_task_difference": dict(zip(used_tasks, differences)),
    }


def _evaluate(
    rows: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    attack: np.ndarray,
    utility: np.ndarray,
    *,
    selected_method: str,
    text_method: str,
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
    methods = _safe_methods(candidates, selected_method, text_method)
    results = {
        name: BASE._method_metrics(evaluation_rows, attack, utility, method)
        for name, method in methods.items()
    }
    comparisons = {}
    for index, reference in enumerate(("clean_raw", text_method)):
        comparisons[f"{SAFE_NAME}__minus__{reference}"] = (
            BASE._bootstrap_difference(
                evaluation_rows,
                methods[SAFE_NAME],
                methods[reference],
                attack,
                utility,
                samples=bootstrap_samples,
                seed=bootstrap_seed + index,
            )
        )
    return {
        "pair_count": len(rows),
        "task_count": len({POST._task_key(row) for row in rows}),
        "results": results,
        "comparisons": comparisons,
        "safe_attack_minus_text": _head_task_difference(
            results[SAFE_NAME],
            results[text_method],
            head="attack",
            samples=bootstrap_samples,
            seed=bootstrap_seed + 20,
        ),
        "utility_rank_exactly_text": bool(
            np.array_equal(
                methods[SAFE_NAME]["utility_rank"], methods[text_method]["utility_rank"]
            )
        ),
        "probabilities_exactly_text": bool(
            np.array_equal(
                methods[SAFE_NAME]["attack_probability"],
                methods[text_method]["attack_probability"],
            )
            and np.array_equal(
                methods[SAFE_NAME]["utility_probability"],
                methods[text_method]["utility_probability"],
            )
        ),
    }


def _fresh_evaluation(
    fresh_rows: list[dict[str, Any]],
    candidate_mapping: dict[tuple[str, str, str], dict[str, Any]],
    **kwargs,
) -> dict[str, Any]:
    candidates = []
    for row in fresh_rows:
        candidate = candidate_mapping.get(POST._key(row))
        if candidate is None:
            raise ValueError(f"Fresh pair missing from predictions: {POST._key(row)}")
        candidates.append(candidate)
    output = _evaluate(
        fresh_rows,
        candidates,
        np.asarray(
            [row["observed_attack_probability"] for row in fresh_rows], dtype=float
        ),
        np.asarray(
            [row["observed_utility_probability"] for row in fresh_rows], dtype=float
        ),
        **kwargs,
    )
    output["fresh_outcome_count"] = 5 * len(fresh_rows)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-summary", type=Path, required=True)
    parser.add_argument("--validation-predictions", type=Path, required=True)
    parser.add_argument("--test-predictions", type=Path, required=True)
    parser.add_argument("--enriched-fresh-predictions", type=Path, required=True)
    parser.add_argument("--remaining-fresh-predictions", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=20000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260725)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    summary = _load(args.validation_summary)
    validation_payload = _load(args.validation_predictions)
    test_payload = _load(args.test_predictions)
    selected_method = str(summary["selected_world_method"])
    text_method = str(summary["selected_text_counterbaseline"])
    for payload, labels_expected in (
        (validation_payload, True),
        (test_payload, False),
    ):
        if payload.get("selected_world_method") != selected_method:
            raise ValueError("Selected method provenance mismatch")
        if payload.get("selected_text_counterbaseline") != text_method:
            raise ValueError("Text baseline provenance mismatch")
        if payload.get("labels_included") is not labels_expected:
            raise ValueError("Prediction label provenance mismatch")

    validation_rows = validation_payload.get("pairs")
    test_rows = test_payload.get("pairs")
    enriched = _load(args.enriched_fresh_predictions).get("pairs")
    remaining = _load(args.remaining_fresh_predictions).get("pairs")
    if not all(
        isinstance(rows, list) and rows
        for rows in (validation_rows, test_rows, enriched, remaining)
    ):
        raise ValueError("A required pair list is missing")
    test_mapping = {POST._key(row): row for row in test_rows}
    if len(test_mapping) != len(test_rows):
        raise ValueError("Duplicate test candidate key")
    if {POST._task_key(row) for row in enriched} & {
        POST._task_key(row) for row in remaining
    }:
        raise ValueError("Fresh cohorts overlap")

    common = {
        "selected_method": selected_method,
        "text_method": text_method,
        "bootstrap_samples": args.bootstrap_samples,
    }
    validation = _evaluate(
        validation_rows,
        validation_rows,
        np.asarray(
            [row["observed_attack_target"] for row in validation_rows], dtype=float
        ),
        np.asarray(
            [row["observed_utility_target"] for row in validation_rows], dtype=float
        ),
        bootstrap_seed=args.bootstrap_seed,
        **common,
    )
    attack_comparison = validation["safe_attack_minus_text"]
    primary_comparison = validation["comparisons"][f"{SAFE_NAME}__minus__{text_method}"]
    gate_checks = {
        "attack_task_mean_gain_at_least_0p03": (
            attack_comparison["mean_task_difference"] >= 0.03
        ),
        "attack_ci_lower_at_least_minus_0p02": (
            attack_comparison["mean_task_difference_95ci"][0] >= -0.02
        ),
        "primary_task_mean_noninferior": (
            primary_comparison["pairwise_accuracy_difference"] >= 0.0
        ),
        "at_least_6_informative_attack_tasks": (
            attack_comparison["informative_task_count"] >= 6
        ),
        "utility_rank_exactly_text": validation["utility_rank_exactly_text"],
        "probabilities_exactly_text": validation["probabilities_exactly_text"],
    }
    gate_status = "GO" if all(gate_checks.values()) else "NO-GO"

    fresh_cohorts = {
        "enriched_prior_8_tasks": enriched,
        "remaining_new_7_tasks": remaining,
        "all_15_grouped_test_tasks": [*enriched, *remaining],
    }
    output = {
        "scope": "text_anchored_outer_crossfit_attack_correction",
        "mechanism_change": (
            "Only the validation-frozen outer-crossfit method supplies attack "
            "ordering; utility ordering and both probabilities remain text-only."
        ),
        "selected_attack_source": selected_method,
        "text_anchor": text_method,
        "safe_method_name": SAFE_NAME,
        "validation_gate_status": gate_status,
        "validation_gate_checks": gate_checks,
        "deployed_method_under_gate": SAFE_NAME if gate_status == "GO" else text_method,
        "validation": validation,
        "fresh_confirmation_claim": False,
        "fresh_reason": "All 15 fresh outcomes were inspected in prior rounds.",
        "fresh_posthoc": {
            name: _fresh_evaluation(
                rows,
                test_mapping,
                bootstrap_seed=args.bootstrap_seed + 100 * (index + 1),
                **common,
            )
            for index, (name, rows) in enumerate(fresh_cohorts.items())
        },
        "interpretation": (
            "A failed gate means the implemented deployment policy is exactly the "
            "text baseline. Post-hoc fresh gains can nominate, but not validate, a "
            "future external confirmation candidate."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
