"""Post-hoc audit of outer-crossfit predictions on the 15 fresh test tasks.

The grouped-test outcomes were inspected in earlier rounds.  This script does
not select a new method from those labels: it evaluates the method frozen by
the grouped validation run and reports all-method rankings only as a clearly
marked diagnostic ceiling.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


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


def _method_from_candidates(
    candidates: list[dict[str, Any]], method_name: str
) -> dict[str, np.ndarray]:
    fields = (
        "attack_rank",
        "utility_rank",
        "attack_probability",
        "utility_probability",
    )
    return BASE._method(
        *(
            np.asarray(
                [row["all_methods"][method_name][field] for row in candidates],
                dtype=float,
            )
            for field in fields
        )
    )


def _task_primary(result: dict[str, Any], task_name: str) -> float | None:
    values = []
    for head in ("attack", "utility"):
        value = result[head]["within_task"]["per_task"][task_name][
            "pairwise_accuracy"
        ]
        if value is not None:
            values.append(float(value))
    return float(np.mean(values)) if values else None


def _evaluate_cohort(
    fresh_rows: list[dict[str, Any]],
    candidate_mapping: dict[tuple[str, str, str], dict[str, Any]],
    *,
    selected_method: str,
    text_method: str,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    candidates = []
    for row in fresh_rows:
        candidate = candidate_mapping.get(_key(row))
        if candidate is None:
            raise ValueError(f"Fresh pair missing from predictions: {_key(row)}")
        candidates.append(candidate)
    method_names = sorted(candidates[0]["all_methods"])
    for candidate in candidates:
        if sorted(candidate["all_methods"]) != method_names:
            raise ValueError("Candidate method sets differ")
    required = {"clean_raw", text_method, selected_method}
    if not required.issubset(method_names):
        raise ValueError(f"Required methods missing: {sorted(required - set(method_names))}")

    rows = [
        {
            "suite": row["suite"],
            "user_task_id": row["user_task_id"],
            "injection_task_id": row["injection_task_id"],
        }
        for row in fresh_rows
    ]
    attack = np.asarray(
        [row["observed_attack_probability"] for row in fresh_rows], dtype=float
    )
    utility = np.asarray(
        [row["observed_utility_probability"] for row in fresh_rows], dtype=float
    )
    methods = {
        name: _method_from_candidates(candidates, name) for name in method_names
    }
    results = {
        name: BASE._method_metrics(rows, attack, utility, method)
        for name, method in methods.items()
    }
    comparisons = {}
    for index, reference in enumerate(("clean_raw", text_method)):
        comparisons[f"{selected_method}__minus__{reference}"] = (
            BASE._bootstrap_difference(
                rows,
                methods[selected_method],
                methods[reference],
                attack,
                utility,
                samples=bootstrap_samples,
                seed=bootstrap_seed + index,
            )
        )

    task_effects = {}
    for task in sorted({_task_key(row) for row in rows}):
        task_name = "::".join(task)
        task_effects[task_name] = {
            name: _task_primary(results[name], task_name)
            for name in ("clean_raw", text_method, selected_method)
        }
        selected = task_effects[task_name][selected_method]
        text = task_effects[task_name][text_method]
        task_effects[task_name]["selected_minus_text"] = (
            None if selected is None or text is None else selected - text
        )

    diagnostic_order = sorted(
        (
            {
                "method": name,
                "primary_mean_within_task_pairwise_accuracy": result[
                    "primary_mean_within_task_pairwise_accuracy"
                ],
                "attack_pairwise_accuracy": result["attack"]["within_task"][
                    "pairwise_accuracy"
                ],
                "utility_pairwise_accuracy": result["utility"]["within_task"][
                    "pairwise_accuracy"
                ],
                "mean_pair_soft_brier": result["mean_pair_soft_brier"],
            }
            for name, result in results.items()
        ),
        key=lambda row: (
            -row["primary_mean_within_task_pairwise_accuracy"], row["method"]
        ),
    )
    selected_minus_text = comparisons[f"{selected_method}__minus__{text_method}"]
    return {
        "pair_count": len(rows),
        "task_count": len({_task_key(row) for row in rows}),
        "fresh_outcome_count": 5 * len(rows),
        "observed_asr": float(attack.mean()),
        "observed_bup": float(utility.mean()),
        "frozen_results": {
            name: results[name]
            for name in ("clean_raw", text_method, selected_method)
        },
        "frozen_comparisons": comparisons,
        "selected_beats_text_point_estimate": (
            selected_minus_text["pairwise_accuracy_difference"] > 0
        ),
        "task_effects": task_effects,
        "posthoc_all_method_order": diagnostic_order,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-summary", type=Path, required=True)
    parser.add_argument("--candidate-predictions", type=Path, required=True)
    parser.add_argument("--enriched-fresh-predictions", type=Path, required=True)
    parser.add_argument("--remaining-fresh-predictions", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=20000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260724)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    validation = _load(args.validation_summary)
    candidate_payload = _load(args.candidate_predictions)
    selected_method = str(validation["selected_world_method"])
    text_method = str(validation["selected_text_counterbaseline"])
    if candidate_payload.get("selected_world_method") != selected_method:
        raise ValueError("Validation and candidate selected methods differ")
    if candidate_payload.get("selected_text_counterbaseline") != text_method:
        raise ValueError("Validation and candidate text baselines differ")
    if candidate_payload.get("labels_included") is not False:
        raise ValueError("Candidate prediction file is not label blind")

    candidate_rows = candidate_payload.get("pairs")
    enriched = _load(args.enriched_fresh_predictions).get("pairs")
    remaining = _load(args.remaining_fresh_predictions).get("pairs")
    if not all(isinstance(rows, list) and rows for rows in (candidate_rows, enriched, remaining)):
        raise ValueError("A required pair list is missing")
    candidate_mapping = {_key(row): row for row in candidate_rows}
    if len(candidate_mapping) != len(candidate_rows):
        raise ValueError("Duplicate candidate prediction key")
    enriched_tasks = {_task_key(row) for row in enriched}
    remaining_tasks = {_task_key(row) for row in remaining}
    if enriched_tasks & remaining_tasks:
        raise ValueError("Fresh cohorts overlap by task")
    if len(enriched) != 32 or len(remaining) != 28:
        raise ValueError("Expected 32 enriched and 28 remaining fresh pairs")
    if len(enriched_tasks | remaining_tasks) != 15:
        raise ValueError("Fresh cohorts must cover all 15 grouped-test tasks")

    cohorts = {
        "enriched_prior_8_tasks": enriched,
        "remaining_new_7_tasks": remaining,
        "all_15_grouped_test_tasks": [*enriched, *remaining],
    }
    output = {
        "scope": "posthoc_outer_crossfit_on_all_grouped_fresh_outcomes",
        "fresh_confirmation_claim": False,
        "reason": "All 15 grouped-test outcomes were inspected before this audit.",
        "upstream_leakage_removed": True,
        "validation_gate_status": validation["gate_status"],
        "frozen_selected_method": selected_method,
        "frozen_text_reference": text_method,
        "candidate_protocol_sha256": candidate_payload["protocol_sha256"],
        "cohorts": {
            name: _evaluate_cohort(
                rows,
                candidate_mapping,
                selected_method=selected_method,
                text_method=text_method,
                bootstrap_samples=args.bootstrap_samples,
                bootstrap_seed=args.bootstrap_seed + 100 * index,
            )
            for index, (name, rows) in enumerate(cohorts.items())
        },
        "interpretation_constraint": (
            "Only the validation-frozen method versus clean/text comparisons are "
            "decision evidence. The all-method order is post-hoc diagnostics and "
            "cannot select a replacement method."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
