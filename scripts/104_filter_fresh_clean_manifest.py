"""Create a deterministic clean-only subset of a fresh-task manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


EXPECTED_SCOPE = "AgentDojo sandbox only; clean-task solvability screen"


def filter_manifest(
    manifest: dict[str, Any],
    *,
    suite: str,
    task_ids: set[str] | None = None,
) -> dict[str, Any]:
    if manifest.get("scope") != EXPECTED_SCOPE:
        raise ValueError("Unexpected source manifest scope")
    safety = manifest.get("safety_contract", {})
    if safety.get("clean_tasks_only") is not True or safety.get(
        "allow_real_network_endpoints"
    ) is not False:
        raise ValueError("Source manifest does not enforce clean sandbox execution")
    rows = [
        row
        for row in manifest["rows"]
        if row["suite"] == suite
        and (task_ids is None or str(row["user_task_id"]) in task_ids)
    ]
    if not rows:
        raise ValueError("Filter selected no tasks")
    if task_ids is not None:
        selected = {str(row["user_task_id"]) for row in rows}
        missing = task_ids - selected
        if missing:
            raise ValueError(f"Unknown task IDs: {sorted(missing)}")
    result = dict(manifest)
    result["rows"] = rows
    result["selection"] = {
        "source_rows": len(manifest["rows"]),
        "suite": suite,
        "task_ids": sorted(task_ids) if task_ids is not None else "all",
        "selected_rows": len(rows),
        "attack_outcomes_used": False,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--task-id", action="append")
    args = parser.parse_args()
    manifest = json.loads(args.input.read_text(encoding="utf-8"))
    result = filter_manifest(
        manifest,
        suite=args.suite,
        task_ids=set(args.task_id) if args.task_id else None,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["selection"], indent=2))


if __name__ == "__main__":
    main()
