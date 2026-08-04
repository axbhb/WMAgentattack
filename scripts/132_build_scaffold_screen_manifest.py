"""Build a deterministic, screening-only stock-task scaffold manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


FRESH_CLEAN_SCOPE = "AgentDojo sandbox only; clean-task solvability screen"
SUITES = ("banking", "slack", "travel", "workspace")


def _existing_tasks(protocol: Mapping[str, Any]) -> set[tuple[str, str]]:
    return {
        (suite, task_id)
        for suite, split_tasks in protocol["task_selection"].items()
        for task_ids in split_tasks.values()
        for task_id in task_ids
    }


def build_manifest(
    existing_protocol: Mapping[str, Any],
    suite_tasks: Mapping[str, list[str]],
    *,
    benchmark_version: str,
    tasks_per_suite: int,
) -> dict[str, Any]:
    excluded = _existing_tasks(existing_protocol)
    rows = []
    pool_counts = {}
    for suite in SUITES:
        candidates = [
            task_id
            for task_id in suite_tasks[suite]
            if (suite, task_id) not in excluded
        ]
        pool_counts[suite] = len(candidates)
        ordered = sorted(
            candidates,
            key=lambda task_id: (
                hashlib.sha256(
                    f"0727-scaffold-screen::{suite}::{task_id}".encode()
                ).hexdigest(),
                task_id,
            ),
        )
        if len(ordered) < tasks_per_suite:
            raise ValueError(f"not enough screening tasks in {suite}")
        for task_id in ordered[:tasks_per_suite]:
            row_id = f"clean::{suite}::{task_id}"
            rows.append(
                {
                    "row_id": row_id,
                    "suite": suite,
                    "user_task_id": task_id,
                    "selection_hash": hashlib.sha256(
                        f"0727-scaffold-screen::{suite}::{task_id}".encode()
                    ).hexdigest(),
                    "screening_only": True,
                    "eligible_for_future_confirmation": False,
                }
            )
    rows.sort(key=lambda row: (row["suite"], row["selection_hash"]))
    return {
        "scope": FRESH_CLEAN_SCOPE,
        "manifest_id": "0727_scaffold_screen_stock_tasks_fixed_v1",
        "benchmark_version": benchmark_version,
        "safety_contract": {
            "synthetic_environment_only": True,
            "allow_real_network_endpoints": False,
            "external_side_effects": False,
            "clean_tasks_only": True,
        },
        "independence_contract": {
            "stock_tasks_have_prior_clean_development_exposure": True,
            "screening_only": True,
            "eligible_for_model_confirmation": False,
            "eligible_for_attack_confirmation": False,
            "eligible_for_future_custom_task_panel": False,
        },
        "selection": {
            "rule": "first four SHA-256 ordered tasks per suite outside the historical 20-task set",
            "salt": "0727-scaffold-screen",
            "outcome_labels_read": False,
            "tasks_per_suite": tasks_per_suite,
            "excluded_historical_tasks": len(excluded),
            "candidate_pool_counts": pool_counts,
        },
        "summary": {
            "rows": len(rows),
            "rows_by_suite": {
                suite: sum(row["suite"] == suite for row in rows)
                for suite in SUITES
            },
        },
        "rows": rows,
    }


def main() -> None:
    from agentdojo.task_suite.load_suites import get_suite

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--existing-protocol",
        type=Path,
        default=Path("configs/0714_agentdojo_v2_protocol.json"),
    )
    parser.add_argument("--benchmark-version", default="v1.2.2")
    parser.add_argument("--tasks-per-suite", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    existing = json.loads(args.existing_protocol.read_text(encoding="utf-8"))
    suite_tasks = {
        suite: sorted(get_suite(args.benchmark_version, suite).user_tasks)
        for suite in SUITES
    }
    manifest = build_manifest(
        existing,
        suite_tasks,
        benchmark_version=args.benchmark_version,
        tasks_per_suite=args.tasks_per_suite,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest["summary"], indent=2))


if __name__ == "__main__":
    main()
