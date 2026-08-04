"""Build a clean-only manifest from AgentDojo tasks unseen in v2."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

FRESH_CLEAN_SCOPE = "AgentDojo sandbox only; clean-task solvability screen"
SUITES = ("banking", "slack", "travel", "workspace")


def _existing_tasks(protocol: dict[str, Any]) -> set[tuple[str, str]]:
    return {
        (suite, task)
        for suite, split_tasks in protocol["task_selection"].items()
        for tasks in split_tasks.values()
        for task in tasks
    }


def build_manifest(
    existing_protocol: dict[str, Any],
    suite_tasks: dict[str, list[str]],
    *,
    benchmark_version: str,
) -> dict[str, Any]:
    excluded = _existing_tasks(existing_protocol)
    rows = []
    for suite in SUITES:
        for task in sorted(suite_tasks[suite]):
            if (suite, task) in excluded:
                continue
            row_id = f"clean::{suite}::{task}"
            rows.append(
                {
                    "row_id": row_id,
                    "suite": suite,
                    "user_task_id": task,
                    "stable_order": hashlib.sha256(row_id.encode()).hexdigest(),
                }
            )
    rows.sort(key=lambda row: (row["stable_order"], row["row_id"]))
    by_suite = {
        suite: sum(row["suite"] == suite for row in rows) for suite in SUITES
    }
    return {
        "scope": FRESH_CLEAN_SCOPE,
        "benchmark_version": benchmark_version,
        "safety_contract": {
            "synthetic_environment_only": True,
            "allow_real_network_endpoints": False,
            "external_side_effects": False,
            "clean_tasks_only": True,
        },
        "selection": {
            "rule": "all tasks outside the existing 20-task v2 set",
            "outcome_labels_read": False,
            "excluded_task_count": len(excluded),
        },
        "summary": {
            "rows": len(rows),
            "rows_by_suite": by_suite,
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
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.existing_protocol.read_text(encoding="utf-8"))
    suite_tasks = {
        suite: sorted(get_suite(args.benchmark_version, suite).user_tasks)
        for suite in SUITES
    }
    result = build_manifest(
        protocol, suite_tasks, benchmark_version=args.benchmark_version
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
