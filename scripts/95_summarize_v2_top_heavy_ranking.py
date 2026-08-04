"""Aggregate frozen top-heavy semantic ranking folds and paired baselines."""

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
    "semantic_probe_top_heavy_summary",
    ROOT / "scripts" / "85_probe_v2_semantic_configuration_value.py",
)
SEMANTIC_SUMMARY = _load_module(
    "semantic_fold_top_heavy_summary",
    ROOT / "scripts" / "86_summarize_v2_semantic_value_folds.py",
)


def _as_baseline(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "prospective_oof": {
            "pct100": {
                "1": {
                    "by_task": {
                        task: {
                            "ASR": float(row["selected_ASR"]),
                            "BUP": float(row["selected_BUP"]),
                            "ASR_plus_BUP": float(row["selected_observed"]),
                        }
                        for task, row in summary["oof_aggregate"]["per_task"].items()
                    }
                }
            }
        }
    }


def summarize(
    paths: list[Path],
    *,
    dreamer_summary: Path,
    e5_summary: Path,
    bootstrap_samples: int = 200_000,
    bootstrap_seed: int = 20260716,
) -> dict[str, Any]:
    rows = []
    folds = []
    candidates = set()
    task_to_fold: dict[str, int] = {}
    for fold, path in enumerate(paths):
        payload = json.loads(path.read_text(encoding="utf-8"))
        candidate = payload["protocol"]["frozen_candidate"]
        candidates.add(json.dumps(candidate, sort_keys=True))
        folds.append({"fold": fold, "path": str(path.resolve()), "test": payload["test"]})
        for row in payload["test_candidate_scores"]:
            task = str(row["task_key"])
            previous = task_to_fold.setdefault(task, fold)
            if previous != fold:
                raise ValueError(f"Held-out task appears in multiple folds: {task}")
            rows.append(row)
    if len(candidates) != 1:
        raise ValueError("Fold results use different frozen candidates")
    if len(task_to_fold) != sum(fold["test"]["task_count"] for fold in folds):
        raise ValueError("Held-out folds do not form a disjoint task partition")

    aggregate = PROBE._evaluate(
        rows,
        rank_scores=np.asarray([float(row["rank_score"]) for row in rows]),
        predictions=np.asarray([float(row["prediction"]) for row in rows]),
    )
    dreamer = json.loads(dreamer_summary.read_text(encoding="utf-8"))
    e5 = json.loads(e5_summary.read_text(encoding="utf-8"))
    comparisons = {
        "dreamer": SEMANTIC_SUMMARY._paired_comparison(
            aggregate,
            dreamer,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed,
        ),
        "e5_largest_gap": SEMANTIC_SUMMARY._paired_comparison(
            aggregate,
            _as_baseline(e5),
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed + 1,
        ),
    }
    dreamer_gate = comparisons["dreamer"]["integration_gate"]
    e5_metrics = comparisons["e5_largest_gap"]["metrics"]
    e5_checks = {
        "joint_delta_at_least_0.05": e5_metrics["ASR_plus_BUP"]["mean_delta"]
        >= 0.05 - 1e-12,
        "BUP_delta_nonnegative": e5_metrics["BUP"]["mean_delta"] >= -1e-12,
    }
    decision = (
        "GO_TO_DREAMER_INTEGRATION"
        if dreamer_gate["decision"] == "GO" and all(e5_checks.values())
        else "NO_GO"
    )
    return {
        "scope": "five-fold frozen top-heavy semantic ranking OOF",
        "protocol": {
            "fold_count": len(paths),
            "method_fixed_before_fold_tests": True,
            "test_retuning": False,
            "frozen_candidate": json.loads(next(iter(candidates))),
        },
        "counts": {"tasks": len(task_to_fold), "configurations": len(rows)},
        "per_fold": folds,
        "oof_aggregate": aggregate,
        "paired_comparisons": comparisons,
        "decision_gate": {"dreamer_gate": dreamer_gate, "e5_checks": e5_checks, "decision": decision},
        "provenance": {
            "dreamer_summary": str(dreamer_summary.resolve()),
            "e5_summary": str(e5_summary.resolve()),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold-results", type=Path, nargs="+", required=True)
    parser.add_argument("--dreamer-summary", type=Path, required=True)
    parser.add_argument("--e5-summary", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=200_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260716)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(
        args.fold_results,
        dreamer_summary=args.dreamer_summary,
        e5_summary=args.e5_summary,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
