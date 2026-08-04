"""Aggregate frozen dual-component value OOF folds and compare with Dreamer."""

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
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


PROBE = _load_module(
    "semantic_probe", ROOT / "scripts" / "85_probe_v2_semantic_configuration_value.py"
)
SEMANTIC_SUMMARY = _load_module(
    "semantic_fold_summary", ROOT / "scripts" / "86_summarize_v2_semantic_value_folds.py"
)


def summarize(
    paths: list[Path],
    *,
    baseline_summary: Path,
    bootstrap_samples: int = 200_000,
    bootstrap_seed: int = 20260716,
) -> dict[str, Any]:
    fold_results = []
    candidate_rows = []
    task_to_fold: dict[str, int] = {}
    frozen_methods = set()
    for fold, path in enumerate(paths):
        payload = json.loads(path.read_text(encoding="utf-8"))
        protocol = payload["protocol"]
        method = {
            "representation": protocol["representation"],
            "attack_estimator": protocol["estimator"],
            "attack_alpha": protocol["ridge_alpha"],
            "utility_estimator": protocol["utility_estimator"],
            "utility_alpha": protocol["utility_alpha"],
            "recipe": payload["applied_test_recipe"],
        }
        frozen_methods.add(json.dumps(method, sort_keys=True))
        fold_results.append(
            {"fold": fold, "path": str(path.resolve()), "test": payload["test"]}
        )
        for source in payload["test_candidate_scores"]:
            row = {**source, "prediction": source["joint_prediction"]}
            task = str(row["task_key"])
            previous = task_to_fold.setdefault(task, fold)
            if previous != fold:
                raise ValueError(f"Held-out task appears in multiple folds: {task}")
            candidate_rows.append(row)
    if len(frozen_methods) != 1:
        raise ValueError("Fold results used different frozen dual-component methods")
    expected_tasks = sum(item["test"]["task_count"] for item in fold_results)
    if len(task_to_fold) != expected_tasks:
        raise ValueError("Fold task counts do not form a disjoint OOF partition")
    aggregate = PROBE._evaluate(
        candidate_rows,
        rank_scores=np.asarray([float(row["rank_score"]) for row in candidate_rows]),
        predictions=np.asarray([float(row["prediction"]) for row in candidate_rows]),
    )
    baseline_payload = json.loads(baseline_summary.read_text(encoding="utf-8"))
    comparison = SEMANTIC_SUMMARY._paired_comparison(
        aggregate,
        baseline_payload,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    return {
        "scope": "five-fold frozen dual-component E5+structured value OOF",
        "protocol": {
            "method_frozen_before_fold_replication": True,
            "test_retuning": False,
            "frozen_method": json.loads(next(iter(frozen_methods))),
        },
        "counts": {
            "tasks": len(task_to_fold),
            "configurations": len(candidate_rows),
        },
        "per_fold": fold_results,
        "oof_aggregate": aggregate,
        "paired_baseline_comparison": comparison,
        "baseline_summary_path": str(baseline_summary.resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold-results", type=Path, nargs="+", required=True)
    parser.add_argument("--baseline-summary", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=200_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260716)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(
        args.fold_results,
        baseline_summary=args.baseline_summary,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
