"""Estimate per-task clean solvability from one or more AgentDojo clean runs."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _is_clean_benchmark_trace(raw: dict[str, Any]) -> bool:
    return (
        str(raw.get("user_task_id", "")).startswith("user_task_")
        and raw.get("injection_task_id") in (None, "none")
        and raw.get("attack_type") in (None, "none")
        and "utility" in raw
        and "suite_name" in raw
    )


def _infer_seed(path: Path) -> str:
    for part in reversed(path.parts):
        match = re.search(r"seed[-_]?(\d+)", part)
        if match:
            return match.group(1)
    return "unknown"


def _trace_paths(roots: list[Path]) -> list[Path]:
    paths: list[Path] = []
    for root in roots:
        if root.is_file():
            paths.append(root)
        elif root.exists():
            paths.extend(root.rglob("*.json"))
    return sorted(set(paths))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--clean-root",
        type=Path,
        action="append",
        required=True,
        help="Clean AgentDojo log root. Repeat for multiple seed directories.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    skipped = Counter()
    for path in _trace_paths(args.clean_root):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            skipped["unreadable_or_not_json"] += 1
            continue
        if not _is_clean_benchmark_trace(raw):
            skipped["not_clean_benchmark_trace"] += 1
            continue
        grouped[(raw["suite_name"], raw["user_task_id"])].append(
            {
                "utility": bool(raw.get("utility")),
                "path": str(path),
                "seed": _infer_seed(path),
                "pipeline_name": raw.get("pipeline_name"),
            }
        )

    tasks = []
    for (suite, user_task_id), rows in sorted(grouped.items()):
        successes = sum(row["utility"] for row in rows)
        attempts = len(rows)
        tasks.append(
            {
                "suite": suite,
                "user_task_id": user_task_id,
                "attempts": attempts,
                "successes": successes,
                "base_success_rate": successes / attempts if attempts else 0.0,
                "seeds": sorted({row["seed"] for row in rows}),
                "pipeline_names": sorted(
                    {str(row["pipeline_name"]) for row in rows if row["pipeline_name"]}
                ),
                "trace_paths": [row["path"] for row in rows],
            }
        )

    output = {
        "scope": "agentdojo_clean_solvability",
        "clean_roots": [str(root.resolve()) for root in args.clean_root],
        "task_count": len(tasks),
        "trace_count": sum(task["attempts"] for task in tasks),
        "mean_base_success_rate": (
            sum(task["base_success_rate"] for task in tasks) / len(tasks)
            if tasks
            else 0.0
        ),
        "attempt_count_distribution": dict(Counter(task["attempts"] for task in tasks)),
        "skipped": dict(skipped),
        "tasks": tasks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "task_count": output["task_count"],
                "trace_count": output["trace_count"],
                "mean_base_success_rate": output["mean_base_success_rate"],
                "attempt_count_distribution": output["attempt_count_distribution"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
