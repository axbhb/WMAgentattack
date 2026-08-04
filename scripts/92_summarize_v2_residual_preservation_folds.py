"""Aggregate frozen residual-preservation OOF folds and paired baselines."""

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
    "semantic_probe_residual_summary",
    ROOT / "scripts" / "85_probe_v2_semantic_configuration_value.py",
)
SEMANTIC_SUMMARY = _load_module(
    "semantic_fold_summary_residual",
    ROOT / "scripts" / "86_summarize_v2_semantic_value_folds.py",
)


def _candidate_baseline_payload(summary: dict[str, Any]) -> dict[str, Any]:
    per_task = summary["oof_aggregate"]["per_task"]
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
                        for task, row in per_task.items()
                    }
                }
            }
        }
    }


def _comparison(
    aggregate: dict[str, Any],
    baseline_payload: dict[str, Any],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    return SEMANTIC_SUMMARY._paired_comparison(
        aggregate,
        baseline_payload,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )


def summarize(
    paths: list[Path],
    *,
    dreamer_summary: Path,
    e5_summary: Path,
    dual_summary: Path | None = None,
    bootstrap_samples: int = 200_000,
    bootstrap_seed: int = 20260716,
) -> dict[str, Any]:
    fold_results = []
    candidate_rows = []
    task_to_fold: dict[str, int] = {}
    frozen_candidates = set()
    for fold, path in enumerate(paths):
        payload = json.loads(path.read_text(encoding="utf-8"))
        candidate = payload["protocol"]["frozen_candidate"]
        frozen_candidates.add(json.dumps(candidate, sort_keys=True))
        fold_results.append(
            {"fold": fold, "path": str(path.resolve()), "test": payload["test"]}
        )
        for row in payload["test_candidate_scores"]:
            task = str(row["task_key"])
            previous = task_to_fold.setdefault(task, fold)
            if previous != fold:
                raise ValueError(f"Held-out task appears in multiple folds: {task}")
            candidate_rows.append(row)
    if len(frozen_candidates) != 1:
        raise ValueError("Fold results used different frozen candidates")
    expected_tasks = sum(item["test"]["task_count"] for item in fold_results)
    if len(task_to_fold) != expected_tasks:
        raise ValueError("Fold task counts do not form a disjoint OOF partition")

    aggregate = PROBE._evaluate(
        candidate_rows,
        rank_scores=np.asarray([float(row["rank_score"]) for row in candidate_rows]),
        predictions=np.asarray([float(row["prediction"]) for row in candidate_rows]),
    )
    dreamer_payload = json.loads(dreamer_summary.read_text(encoding="utf-8"))
    e5_payload = json.loads(e5_summary.read_text(encoding="utf-8"))
    comparisons = {
        "dreamer": _comparison(
            aggregate,
            dreamer_payload,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed,
        ),
        "e5_joint_probe": _comparison(
            aggregate,
            _candidate_baseline_payload(e5_payload),
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed + 1,
        ),
    }
    provenance = {
        "dreamer_summary": str(dreamer_summary.resolve()),
        "e5_summary": str(e5_summary.resolve()),
    }
    if dual_summary is not None:
        dual_payload = json.loads(dual_summary.read_text(encoding="utf-8"))
        comparisons["dual_component_probe"] = _comparison(
            aggregate,
            _candidate_baseline_payload(dual_payload),
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed + 2,
        )
        provenance["dual_summary"] = str(dual_summary.resolve())

    dreamer_gate = comparisons["dreamer"]["integration_gate"]
    e5_metrics = comparisons["e5_joint_probe"]["metrics"]
    e5_checks = {
        "joint_delta_nonnegative": e5_metrics["ASR_plus_BUP"]["mean_delta"]
        >= -1e-12,
        "BUP_delta_nonnegative": e5_metrics["BUP"]["mean_delta"] >= -1e-12,
    }
    integration_pass = dreamer_gate["decision"] == "GO" and all(
        e5_checks.values()
    )
    if integration_pass:
        decision = "GO_TO_DREAMER_INTEGRATION"
    elif dreamer_gate["engineering_pass"] and all(e5_checks.values()):
        decision = "PILOT_ONLY_UNCONFIRMED"
    else:
        decision = "NO_GO"

    return {
        "scope": "five-fold frozen clean-conditioned residual-preservation OOF",
        "protocol": {
            "fold_count": len(paths),
            "method_frozen_before_fold_tests": True,
            "test_retuning": False,
            "frozen_candidate": json.loads(next(iter(frozen_candidates))),
        },
        "counts": {
            "tasks": len(task_to_fold),
            "configurations": len(candidate_rows),
        },
        "per_fold": fold_results,
        "oof_aggregate": aggregate,
        "paired_comparisons": comparisons,
        "decision_gate": {
            "dreamer_gate": dreamer_gate,
            "e5_checks": e5_checks,
            "decision": decision,
        },
        "provenance": provenance,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold-results", type=Path, nargs="+", required=True)
    parser.add_argument("--dreamer-summary", type=Path, required=True)
    parser.add_argument("--e5-summary", type=Path, required=True)
    parser.add_argument("--dual-summary", type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=200_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260716)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(
        args.fold_results,
        dreamer_summary=args.dreamer_summary,
        e5_summary=args.e5_summary,
        dual_summary=args.dual_summary,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
