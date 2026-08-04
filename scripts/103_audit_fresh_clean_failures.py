"""Audit whether fresh clean failures reached AgentDojo tools."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def _assistant_tool_calls(trace: dict[str, Any]) -> int:
    return sum(
        len(message.get("tool_calls") or [])
        for message in trace.get("messages", [])
        if message.get("role") == "assistant"
    )


def audit(
    archive_root: Path,
    *,
    seeds: tuple[int, ...] = (101, 103, 107),
    chunks: int = 4,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        for chunk in range(chunks):
            chunk_path = archive_root / f"seed{seed}" / f"chunk{chunk}.json"
            payload = json.loads(chunk_path.read_text(encoding="utf-8"))
            if int(payload["run_seed"]) != seed:
                raise ValueError(f"Seed mismatch in {chunk_path}")
            for result in payload["results"]:
                if result.get("status") != "completed":
                    raise ValueError(f"Incomplete result in {chunk_path}")
                trace_path = Path(result["raw_trace"])
                trace = json.loads(trace_path.read_text(encoding="utf-8"))
                tool_calls = _assistant_tool_calls(trace)
                rows.append(
                    {
                        "seed": seed,
                        "suite": result["suite"],
                        "user_task_id": result["user_task_id"],
                        "utility": bool(result["utility"]),
                        "tool_calls": tool_calls,
                        "messages": len(trace.get("messages", [])),
                        "raw_trace": str(trace_path.resolve()),
                    }
                )

    by_suite: dict[str, Any] = {}
    for suite in sorted({row["suite"] for row in rows}):
        suite_rows = [row for row in rows if row["suite"] == suite]
        failures = [row for row in suite_rows if not row["utility"]]
        successes = [row for row in suite_rows if row["utility"]]
        by_suite[suite] = {
            "episodes": len(suite_rows),
            "utility_successes": len(successes),
            "utility_failures": len(failures),
            "failures_without_tool_call": sum(
                row["tool_calls"] == 0 for row in failures
            ),
            "failures_with_tool_call": sum(
                row["tool_calls"] > 0 for row in failures
            ),
            "successes_without_tool_call": sum(
                row["tool_calls"] == 0 for row in successes
            ),
            "failure_tool_call_histogram": dict(
                sorted(Counter(row["tool_calls"] for row in failures).items())
            ),
        }
    failures = [row for row in rows if not row["utility"]]
    return {
        "scope": "fresh clean failure/tool-use audit",
        "attack_outcomes_read": False,
        "protocol": {"seeds": list(seeds), "chunks_per_seed": chunks},
        "counts": {
            "episodes": len(rows),
            "utility_failures": len(failures),
            "failures_without_tool_call": sum(
                row["tool_calls"] == 0 for row in failures
            ),
        },
        "by_suite": by_suite,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[101, 103, 107])
    parser.add_argument("--chunks", type=int, default=4)
    args = parser.parse_args()
    result = audit(
        args.archive_root,
        seeds=tuple(args.seeds),
        chunks=args.chunks,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"counts": result["counts"], "by_suite": result["by_suite"]}, indent=2))


if __name__ == "__main__":
    main()
