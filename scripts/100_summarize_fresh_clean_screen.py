"""Merge clean-screen chunks and audit the fresh task pool."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def summarize(manifest_path: Path, chunk_paths: list[Path]) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {str(row["row_id"]): row for row in manifest["rows"]}
    results: dict[str, dict[str, Any]] = {}
    for path in chunk_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for row in payload["results"]:
            row_id = str(row["row_id"])
            if row_id in results:
                raise ValueError(f"Duplicate clean result: {row_id}")
            results[row_id] = row
    if set(results) != set(expected):
        raise ValueError(
            f"Result coverage differs from manifest: missing={sorted(set(expected)-set(results))} "
            f"extra={sorted(set(results)-set(expected))}"
        )
    failed = [row for row in results.values() if row.get("status") != "completed"]
    if failed:
        raise ValueError(f"Fresh clean screen contains {len(failed)} failures")
    ordered = [results[str(row["row_id"])] for row in manifest["rows"]]
    by_suite = {}
    for suite in sorted({str(row["suite"]) for row in ordered}):
        rows = [row for row in ordered if row["suite"] == suite]
        successes = [row for row in rows if bool(row["utility"])]
        by_suite[suite] = {
            "tasks": len(rows),
            "clean_successes": len(successes),
            "clean_success_rate": len(successes) / len(rows),
            "successful_task_ids": sorted(row["user_task_id"] for row in successes),
        }
    durations = [float(row["elapsed_seconds"]) for row in ordered]
    return {
        "scope": "fresh task seed-101 clean solvability screen",
        "attack_outcomes_read": False,
        "counts": {
            "tasks": len(ordered),
            "clean_successes": sum(bool(row["utility"]) for row in ordered),
            "clean_failures": sum(not bool(row["utility"]) for row in ordered),
            "by_suite": dict(Counter(row["suite"] for row in ordered)),
        },
        "by_suite": by_suite,
        "timing_seconds": {
            "aggregate": sum(durations),
            "mean": sum(durations) / len(durations),
            "max": max(durations),
        },
        "results": ordered,
        "provenance": {
            "manifest": str(manifest_path.resolve()),
            "chunks": [str(path.resolve()) for path in chunk_paths],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--chunks", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.manifest, args.chunks)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"counts": result["counts"], "by_suite": result["by_suite"]}, indent=2))


if __name__ == "__main__":
    main()
