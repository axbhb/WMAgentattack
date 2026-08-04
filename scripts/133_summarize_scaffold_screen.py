"""Evaluate a frozen, clean-only victim scaffold screening experiment."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping


def _exact_two_sided_sign_p(wins: int, losses: int) -> float:
    discordant = wins + losses
    if discordant == 0:
        return 1.0
    tail = min(wins, losses)
    probability = sum(math.comb(discordant, k) for k in range(tail + 1))
    return min(1.0, 2.0 * probability / (2**discordant))


def _tool_call_count(path: Path) -> int:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return sum(
        len(message.get("tool_calls") or [])
        for message in payload.get("messages", [])
        if message.get("role") == "assistant"
    )


def _load_candidate(
    archive_root: Path,
    candidate_id: str,
    *,
    expected_rows: Mapping[str, Mapping[str, Any]],
    seeds: tuple[int, ...],
    chunks: int,
) -> dict[str, Any]:
    records: dict[tuple[int, str], dict[str, Any]] = {}
    source_errors: list[dict[str, Any]] = []
    provenance: list[str] = []
    for seed in seeds:
        seed_ids: set[str] = set()
        for chunk in range(chunks):
            path = archive_root / candidate_id / f"seed{seed}" / f"chunk{chunk}.json"
            provenance.append(str(path.resolve()))
            if not path.is_file():
                source_errors.append(
                    {"type": "missing_chunk", "seed": seed, "chunk": chunk, "path": str(path)}
                )
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception as error:  # keep a frozen negative result inspectable
                source_errors.append(
                    {
                        "type": "unreadable_chunk",
                        "seed": seed,
                        "chunk": chunk,
                        "path": str(path),
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
                continue
            if int(payload.get("run_seed", -1)) != seed:
                source_errors.append(
                    {"type": "seed_mismatch", "seed": seed, "chunk": chunk, "path": str(path)}
                )
            if int(payload.get("chunk_index", -1)) != chunk:
                source_errors.append(
                    {"type": "chunk_index_mismatch", "seed": seed, "chunk": chunk, "path": str(path)}
                )
            if int(payload.get("num_chunks", -1)) != chunks:
                source_errors.append(
                    {"type": "num_chunks_mismatch", "seed": seed, "chunk": chunk, "path": str(path)}
                )
            for result in payload.get("results", []):
                row_id = str(result.get("row_id"))
                key = (seed, row_id)
                if row_id not in expected_rows:
                    source_errors.append(
                        {"type": "unexpected_row", "seed": seed, "row_id": row_id, "path": str(path)}
                    )
                    continue
                if row_id in seed_ids or key in records:
                    source_errors.append(
                        {"type": "duplicate_row", "seed": seed, "row_id": row_id, "path": str(path)}
                    )
                    continue
                seed_ids.add(row_id)
                enriched = dict(result)
                if result.get("status") == "completed":
                    trace_path = Path(str(result.get("raw_trace", "")))
                    if not trace_path.is_file():
                        enriched["trace_error"] = "missing_raw_trace"
                    else:
                        try:
                            enriched["tool_calls"] = _tool_call_count(trace_path)
                        except Exception as error:
                            enriched["trace_error"] = f"{type(error).__name__}: {error}"
                records[key] = enriched

    expected_pairs = {(seed, row_id) for seed in seeds for row_id in expected_rows}
    missing_pairs = sorted(expected_pairs - set(records))
    completed = {
        key: row
        for key, row in records.items()
        if row.get("status") == "completed" and "trace_error" not in row
    }
    failed_rows = [
        {"seed": key[0], "row_id": key[1], **row}
        for key, row in records.items()
        if row.get("status") != "completed"
    ]
    trace_failures = [
        {"seed": key[0], "row_id": key[1], "trace_error": row["trace_error"]}
        for key, row in records.items()
        if "trace_error" in row
    ]
    task_rows = []
    for row_id, manifest_row in expected_rows.items():
        outcomes = [
            completed[(seed, row_id)]
            for seed in seeds
            if (seed, row_id) in completed
        ]
        successes = sum(bool(row["utility"]) for row in outcomes)
        task_rows.append(
            {
                "row_id": row_id,
                "suite": str(manifest_row["suite"]),
                "user_task_id": str(manifest_row["user_task_id"]),
                "attempts": len(outcomes),
                "successes": successes,
                "retained": len(outcomes) == len(seeds) and successes >= 2,
            }
        )
    suites = sorted({str(row["suite"]) for row in expected_rows.values()})
    by_suite = {}
    for suite in suites:
        suite_tasks = [row for row in task_rows if row["suite"] == suite]
        suite_completed = [
            row
            for (seed, row_id), row in completed.items()
            if str(expected_rows[row_id]["suite"]) == suite
        ]
        by_suite[suite] = {
            "tasks": len(suite_tasks),
            "retained_tasks": sum(row["retained"] for row in suite_tasks),
            "retained_task_ids": sorted(
                row["user_task_id"] for row in suite_tasks if row["retained"]
            ),
            "clean_successes": sum(bool(row["utility"]) for row in suite_completed),
            "completed_episodes": len(suite_completed),
        }
    failed_clean = [row for row in completed.values() if not bool(row["utility"])]
    return {
        "candidate_id": candidate_id,
        "coverage": {
            "expected_episodes": len(expected_pairs),
            "recorded_rows": len(records),
            "completed_with_trace": len(completed),
            "missing_pairs": len(missing_pairs),
            "failed_rows": len(failed_rows),
            "trace_failures": len(trace_failures),
            "source_errors": len(source_errors),
            "complete": (
                len(completed) == len(expected_pairs)
                and not missing_pairs
                and not failed_rows
                and not trace_failures
                and not source_errors
            ),
        },
        "episodes": {
            "clean_successes": sum(bool(row["utility"]) for row in completed.values()),
            "clean_failures": sum(not bool(row["utility"]) for row in completed.values()),
            "failures_without_tool_call": sum(row["tool_calls"] == 0 for row in failed_clean),
            "failures_with_tool_call": sum(row["tool_calls"] > 0 for row in failed_clean),
        },
        "tasks": {
            "count": len(task_rows),
            "retained": sum(row["retained"] for row in task_rows),
            "retained_ids": sorted(
                f'{row["suite"]}::{row["user_task_id"]}' for row in task_rows if row["retained"]
            ),
        },
        "by_suite": by_suite,
        "task_rows": task_rows,
        "diagnostics": {
            "missing_pairs": [list(item) for item in missing_pairs],
            "failed_rows": failed_rows,
            "trace_failures": trace_failures,
            "source_errors": source_errors,
        },
        "records": completed,
        "provenance": provenance,
    }


def _paired_comparison(
    baseline: Mapping[tuple[int, str], Mapping[str, Any]],
    candidate: Mapping[tuple[int, str], Mapping[str, Any]],
    *,
    expected_pairs: int,
) -> dict[str, Any]:
    keys = sorted(set(baseline) & set(candidate))
    wins = sum(not bool(baseline[key]["utility"]) and bool(candidate[key]["utility"]) for key in keys)
    losses = sum(bool(baseline[key]["utility"]) and not bool(candidate[key]["utility"]) for key in keys)
    return {
        "expected_pairs": expected_pairs,
        "paired_rows": len(keys),
        "candidate_wins": wins,
        "candidate_losses": losses,
        "ties": len(keys) - wins - losses,
        "clean_success_delta": sum(bool(candidate[key]["utility"]) for key in keys)
        - sum(bool(baseline[key]["utility"]) for key in keys),
        "exact_two_sided_sign_p": _exact_two_sided_sign_p(wins, losses),
    }


def summarize(protocol_path: Path, manifest_path: Path, archive_root: Path) -> dict[str, Any]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    screen = protocol["screening"]
    gate = protocol["frozen_selection_gate"]
    candidate_specs = screen["candidates"]
    candidate_ids = [str(row["id"]) for row in candidate_specs]
    baseline_id = str(screen["baseline_candidate"])
    if baseline_id not in candidate_ids:
        raise ValueError("Frozen baseline is not one of the candidate IDs")
    expected_rows = {str(row["row_id"]): row for row in manifest["rows"]}
    if len(expected_rows) != int(screen["tasks"]):
        raise ValueError("Manifest task count differs from the frozen protocol")
    if any(not row.get("screening_only") for row in manifest["rows"]):
        raise ValueError("Manifest contains a row that is not screening-only")
    seeds = tuple(int(seed) for seed in screen["seeds"])
    chunks = int(screen["chunks_per_seed"])
    loaded = {
        candidate_id: _load_candidate(
            archive_root,
            candidate_id,
            expected_rows=expected_rows,
            seeds=seeds,
            chunks=chunks,
        )
        for candidate_id in candidate_ids
    }
    expected_pairs = len(expected_rows) * len(seeds)
    baseline = loaded[baseline_id]
    comparisons = {}
    eligible = []
    for candidate_id in candidate_ids:
        candidate = loaded[candidate_id]
        comparison = _paired_comparison(
            baseline["records"], candidate["records"], expected_pairs=expected_pairs
        )
        retained_gain = candidate["tasks"]["retained"] - baseline["tasks"]["retained"]
        success_gain = (
            candidate["episodes"]["clean_successes"]
            - baseline["episodes"]["clean_successes"]
        )
        no_domain_regression = all(
            candidate["by_suite"][suite]["retained_tasks"]
            >= baseline["by_suite"][suite]["retained_tasks"]
            for suite in baseline["by_suite"]
        )
        conditions = {
            "complete": bool(candidate["coverage"]["complete"]),
            "retained_task_gain_at_least_threshold": retained_gain
            >= int(gate["minimum_retained_task_gain_over_baseline"]),
            "clean_success_gain_at_least_threshold": success_gain
            >= int(gate["minimum_clean_success_gain_over_baseline"]),
            "no_domain_retained_task_regression": no_domain_regression,
        }
        comparison.update(
            {
                "retained_task_gain": retained_gain,
                "material_gain_conditions": conditions,
                "eligible_to_replace_baseline": (
                    candidate_id != baseline_id and all(conditions.values())
                ),
            }
        )
        comparisons[candidate_id] = comparison
        if comparison["eligible_to_replace_baseline"]:
            eligible.append(candidate_id)

    all_complete = all(row["coverage"]["complete"] for row in loaded.values())
    tie_order = {
        candidate_id: index
        for index, candidate_id in enumerate(screen["candidate_order_for_exact_ties"])
    }
    if all_complete and eligible:
        selected = sorted(
            eligible,
            key=lambda candidate_id: (
                -loaded[candidate_id]["tasks"]["retained"],
                -loaded[candidate_id]["episodes"]["clean_successes"],
                loaded[candidate_id]["episodes"]["failures_without_tool_call"],
                tie_order.get(candidate_id, len(tie_order)),
            ),
        )[0]
        decision = f"SCAFFOLD_SCREEN_SELECT_{selected.upper()}"
    elif all_complete:
        selected = baseline_id
        decision = "SCAFFOLD_SCREEN_RETAIN_BASE_SAMPLED_NO_MATERIAL_IMPROVEMENT"
    else:
        selected = None
        decision = "SCAFFOLD_SCREEN_INCOMPLETE_NO_SELECTION"

    serializable_candidates = {}
    for candidate_id, row in loaded.items():
        serializable_candidates[candidate_id] = {
            key: value for key, value in row.items() if key != "records"
        }
    return {
        "scope": "frozen clean-only victim scaffold screen",
        "attack_outcomes_read": False,
        "screening_only": True,
        "eligible_for_final_confirmation_claim": False,
        "protocol": str(protocol_path.resolve()),
        "manifest": str(manifest_path.resolve()),
        "archive_root": str(archive_root.resolve()),
        "expected": {
            "candidates": len(candidate_ids),
            "tasks_per_candidate": len(expected_rows),
            "seeds": list(seeds),
            "episodes_per_candidate": expected_pairs,
            "total_episodes": expected_pairs * len(candidate_ids),
        },
        "candidates": serializable_candidates,
        "paired_vs_baseline": comparisons,
        "gate": {
            "all_candidates_complete": all_complete,
            "eligible_candidates": eligible,
            "selected_candidate": selected,
            "decision": decision,
            "attack_data_permitted": False,
            "dreamer_training_permitted": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.protocol, args.manifest, args.archive_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["gate"], indent=2))


if __name__ == "__main__":
    main()
