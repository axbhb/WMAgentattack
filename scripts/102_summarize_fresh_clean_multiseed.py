"""Summarize the complete three-seed clean census over fresh tasks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def summarize(
    manifest_path: Path,
    archive_root: Path,
    *,
    seeds: tuple[int, ...] = (101, 103, 107),
    chunks: int = 4,
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {str(row["row_id"]): row for row in manifest["rows"]}
    evidence: dict[str, dict[int, dict[str, Any]]] = {
        row_id: {} for row_id in expected
    }
    provenance = []
    for seed in seeds:
        seed_ids = set()
        for chunk in range(chunks):
            path = archive_root / f"seed{seed}" / f"chunk{chunk}.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            if int(payload["run_seed"]) != seed:
                raise ValueError(f"Seed mismatch in {path}")
            provenance.append(str(path.resolve()))
            for result in payload["results"]:
                row_id = str(result["row_id"])
                if row_id not in expected:
                    raise ValueError(f"Unexpected task result: {row_id}")
                if row_id in seed_ids:
                    raise ValueError(f"Duplicate seed result: seed={seed} row={row_id}")
                if result.get("status") != "completed":
                    raise ValueError(f"Incomplete task: seed={seed} row={row_id}")
                seed_ids.add(row_id)
                evidence[row_id][seed] = result
        if seed_ids != set(expected):
            raise ValueError(f"Seed {seed} does not cover the complete manifest")

    tasks = []
    for row in manifest["rows"]:
        row_id = str(row["row_id"])
        outcomes = evidence[row_id]
        successes = sum(bool(outcomes[seed]["utility"]) for seed in seeds)
        tasks.append(
            {
                "row_id": row_id,
                "suite": row["suite"],
                "user_task_id": row["user_task_id"],
                "attempts": len(seeds),
                "successes": successes,
                "base_success_rate": successes / len(seeds),
                "seeds": list(seeds),
                "outcomes": {
                    str(seed): bool(outcomes[seed]["utility"]) for seed in seeds
                },
                "retained": successes >= 2,
            }
        )
    by_suite = {}
    for suite in sorted({row["suite"] for row in tasks}):
        rows = [row for row in tasks if row["suite"] == suite]
        retained = [row for row in rows if row["retained"]]
        by_suite[suite] = {
            "tasks": len(rows),
            "retained_at_least_2_of_3": len(retained),
            "retained_task_ids": sorted(row["user_task_id"] for row in retained),
            "success_count_distribution": {
                str(count): sum(row["successes"] == count for row in rows)
                for count in range(4)
            },
        }
    retained = [row for row in tasks if row["retained"]]
    return {
        "scope": "complete fresh-task three-seed clean solvability census",
        "attack_outcomes_read": False,
        "protocol": {
            "seeds": list(seeds),
            "retention_rule": "at least two clean successes out of three",
            "all_fresh_tasks_repeated": True,
        },
        "counts": {
            "tasks": len(tasks),
            "clean_episodes": len(tasks) * len(seeds),
            "retained_tasks": len(retained),
        },
        "by_suite": by_suite,
        "tasks": tasks,
        "provenance": {
            "manifest": str(manifest_path.resolve()),
            "chunk_results": provenance,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[101, 103, 107])
    parser.add_argument("--chunks", type=int, default=4)
    args = parser.parse_args()
    result = summarize(
        args.manifest,
        args.archive_root,
        seeds=tuple(args.seeds),
        chunks=args.chunks,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"counts": result["counts"], "by_suite": result["by_suite"]}, indent=2))


if __name__ == "__main__":
    main()
