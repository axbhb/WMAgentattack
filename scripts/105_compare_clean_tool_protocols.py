"""Compare two clean AgentDojo tool protocols on identical task/seed pairs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def _load_results(
    archive: Path,
    *,
    seeds: tuple[int, ...],
    chunks: int,
) -> dict[tuple[int, str, str], dict[str, Any]]:
    rows: dict[tuple[int, str, str], dict[str, Any]] = {}
    for seed in seeds:
        for chunk in range(chunks):
            path = archive / f"seed{seed}" / f"chunk{chunk}.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            if int(payload["run_seed"]) != seed:
                raise ValueError(f"Seed mismatch in {path}")
            for result in payload["results"]:
                if result.get("status") != "completed":
                    raise ValueError(f"Incomplete result in {path}")
                key = (seed, str(result["suite"]), str(result["user_task_id"]))
                if key in rows:
                    raise ValueError(f"Duplicate result: {key}")
                trace = json.loads(Path(result["raw_trace"]).read_text(encoding="utf-8"))
                tool_calls = sum(
                    len(message.get("tool_calls") or [])
                    for message in trace.get("messages", [])
                    if message.get("role") == "assistant"
                )
                rows[key] = {
                    "utility": bool(result["utility"]),
                    "tool_calls": tool_calls,
                    "raw_trace": result["raw_trace"],
                }
    return rows


def _exact_two_sided_sign_p(wins: int, losses: int) -> float:
    discordant = wins + losses
    if discordant == 0:
        return 1.0
    tail = min(wins, losses)
    probability = sum(
        math.comb(discordant, k) for k in range(tail + 1)
    ) / (2**discordant)
    return min(1.0, 2.0 * probability)


def compare(
    baseline: dict[tuple[int, str, str], dict[str, Any]],
    candidate: dict[tuple[int, str, str], dict[str, Any]],
    *,
    retention_successes: int = 2,
) -> dict[str, Any]:
    if not candidate:
        raise ValueError("Candidate has no results")
    missing = set(candidate) - set(baseline)
    if missing:
        raise ValueError(f"Baseline is missing {len(missing)} candidate pairs")
    keys = sorted(candidate)
    pairs = []
    for key in keys:
        base = baseline[key]
        cand = candidate[key]
        pairs.append(
            {
                "seed": key[0],
                "suite": key[1],
                "user_task_id": key[2],
                "baseline_utility": base["utility"],
                "candidate_utility": cand["utility"],
                "baseline_tool_calls": base["tool_calls"],
                "candidate_tool_calls": cand["tool_calls"],
            }
        )
    wins = sum(not row["baseline_utility"] and row["candidate_utility"] for row in pairs)
    losses = sum(row["baseline_utility"] and not row["candidate_utility"] for row in pairs)

    task_keys = sorted({(row["suite"], row["user_task_id"]) for row in pairs})
    task_rows = []
    for suite, task_id in task_keys:
        task_pairs = [
            row
            for row in pairs
            if row["suite"] == suite and row["user_task_id"] == task_id
        ]
        base_successes = sum(row["baseline_utility"] for row in task_pairs)
        candidate_successes = sum(row["candidate_utility"] for row in task_pairs)
        task_rows.append(
            {
                "suite": suite,
                "user_task_id": task_id,
                "pairs": len(task_pairs),
                "baseline_successes": base_successes,
                "candidate_successes": candidate_successes,
                "baseline_retained": base_successes >= retention_successes,
                "candidate_retained": candidate_successes >= retention_successes,
            }
        )
    baseline_retained = [
        row for row in task_rows if row["baseline_retained"]
    ]
    candidate_retained = [
        row for row in task_rows if row["candidate_retained"]
    ]
    baseline_failures = [row for row in pairs if not row["baseline_utility"]]
    candidate_failures = [row for row in pairs if not row["candidate_utility"]]
    return {
        "scope": "paired clean-only tool protocol comparison",
        "attack_outcomes_read": False,
        "episodes": {
            "pairs": len(pairs),
            "baseline_successes": sum(row["baseline_utility"] for row in pairs),
            "candidate_successes": sum(row["candidate_utility"] for row in pairs),
            "candidate_wins": wins,
            "candidate_losses": losses,
            "ties": len(pairs) - wins - losses,
            "paired_success_delta": (
                sum(row["candidate_utility"] for row in pairs)
                - sum(row["baseline_utility"] for row in pairs)
            ) / len(pairs),
            "exact_two_sided_sign_p": _exact_two_sided_sign_p(wins, losses),
        },
        "tool_execution": {
            "baseline_failures_without_tool_call": sum(
                row["baseline_tool_calls"] == 0 for row in baseline_failures
            ),
            "candidate_failures_without_tool_call": sum(
                row["candidate_tool_calls"] == 0 for row in candidate_failures
            ),
        },
        "tasks": {
            "count": len(task_rows),
            "retention_successes": retention_successes,
            "baseline_retained": len(baseline_retained),
            "candidate_retained": len(candidate_retained),
            "baseline_retained_ids": [
                f'{row["suite"]}::{row["user_task_id"]}' for row in baseline_retained
            ],
            "candidate_retained_ids": [
                f'{row["suite"]}::{row["user_task_id"]}' for row in candidate_retained
            ],
        },
        "task_rows": task_rows,
        "pairs": pairs,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-archive", type=Path, required=True)
    parser.add_argument("--candidate-archive", type=Path, required=True)
    parser.add_argument("--baseline-chunks", type=int, default=4)
    parser.add_argument("--candidate-chunks", type=int, default=2)
    parser.add_argument("--seeds", type=int, nargs="+", default=[101, 103, 107])
    parser.add_argument("--retention-successes", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    seeds = tuple(args.seeds)
    baseline = _load_results(
        args.baseline_archive,
        seeds=seeds,
        chunks=args.baseline_chunks,
    )
    candidate = _load_results(
        args.candidate_archive,
        seeds=seeds,
        chunks=args.candidate_chunks,
    )
    result = compare(
        baseline,
        candidate,
        retention_successes=args.retention_successes,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "episodes": result["episodes"],
                "tool_execution": result["tool_execution"],
                "tasks": result["tasks"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
