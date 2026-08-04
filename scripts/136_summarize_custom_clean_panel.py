"""Summarize frozen custom-panel clean replays and apply data gates."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping


def _tool_call_count(path: Path) -> int:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return sum(
        len(message.get("tool_calls") or [])
        for message in payload.get("messages", [])
        if message.get("role") == "assistant"
    )


def _load_results(
    manifest: Mapping[str, Any],
    archive_root: Path,
    *,
    seeds: tuple[int, ...],
    chunks: int,
) -> tuple[dict[tuple[int, str], dict[str, Any]], list[dict[str, Any]]]:
    expected = {str(row["row_id"]): row for row in manifest["rows"]}
    results: dict[tuple[int, str], dict[str, Any]] = {}
    errors: list[dict[str, Any]] = []
    for seed in seeds:
        for chunk in range(chunks):
            path = archive_root / f"seed{seed}" / f"chunk{chunk}.json"
            if not path.is_file():
                errors.append({"type": "missing_chunk", "seed": seed, "chunk": chunk, "path": str(path)})
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception as error:
                errors.append({"type": "unreadable_chunk", "seed": seed, "chunk": chunk, "error": f"{type(error).__name__}: {error}"})
                continue
            if int(payload.get("run_seed", -1)) != seed:
                errors.append({"type": "seed_mismatch", "seed": seed, "chunk": chunk})
            if int(payload.get("chunk_index", -1)) != chunk or int(payload.get("num_chunks", -1)) != chunks:
                errors.append({"type": "chunk_metadata_mismatch", "seed": seed, "chunk": chunk})
            if payload.get("custom_task_module") != manifest.get("custom_task_module"):
                errors.append({"type": "custom_module_mismatch", "seed": seed, "chunk": chunk})
            for result in payload.get("results", []):
                row_id = str(result.get("row_id"))
                key = (seed, row_id)
                if row_id not in expected:
                    errors.append({"type": "unexpected_row", "seed": seed, "row_id": row_id})
                    continue
                if key in results:
                    errors.append({"type": "duplicate_row", "seed": seed, "row_id": row_id})
                    continue
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
                results[key] = enriched
    return results, errors


def summarize(protocol_path: Path, manifest_path: Path, archive_root: Path) -> dict[str, Any]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    replay = protocol["clean_replay"]
    dev_seeds = tuple(int(seed) for seed in replay["development_seeds"])
    conf_seeds = tuple(int(seed) for seed in replay["confirmation_seeds"])
    all_seeds = dev_seeds + conf_seeds
    chunks = int(replay["chunks_per_seed"])
    expected_rows = {str(row["row_id"]): row for row in manifest["rows"]}
    results, source_errors = _load_results(
        manifest, archive_root, seeds=all_seeds, chunks=chunks
    )
    expected_pairs = {(seed, row_id) for seed in all_seeds for row_id in expected_rows}
    missing_pairs = sorted(expected_pairs - set(results))
    failed_rows = [
        {"seed": seed, "row_id": row_id, "status": row.get("status"), "error": row.get("error")}
        for (seed, row_id), row in results.items()
        if row.get("status") != "completed"
    ]
    trace_failures = [
        {"seed": seed, "row_id": row_id, "trace_error": row["trace_error"]}
        for (seed, row_id), row in results.items()
        if "trace_error" in row
    ]
    completed = {
        key: row
        for key, row in results.items()
        if row.get("status") == "completed" and "trace_error" not in row
    }
    task_rows = []
    for row_id, manifest_row in expected_rows.items():
        dev = [completed.get((seed, row_id)) for seed in dev_seeds]
        conf = [completed.get((seed, row_id)) for seed in conf_seeds]
        dev_outcomes = [bool(row["utility"]) for row in dev if row is not None]
        conf_outcomes = [bool(row["utility"]) for row in conf if row is not None]
        dev_successes = sum(dev_outcomes)
        conf_successes = sum(conf_outcomes)
        dev_retained = len(dev_outcomes) == len(dev_seeds) and dev_successes >= 2
        conf_retained = len(conf_outcomes) == len(conf_seeds) and conf_successes >= 2
        all_outcomes = dev_outcomes + conf_outcomes
        task_rows.append(
            {
                "row_id": row_id,
                "suite": manifest_row["suite"],
                "split": manifest_row["split"],
                "template_family": manifest_row["template_family"],
                "development_successes": dev_successes,
                "confirmation_successes": conf_successes,
                "development_retained": dev_retained,
                "confirmation_retained": conf_retained,
                "durable": dev_retained and conf_retained,
                "all_six_failure": len(all_outcomes) == 6 and not any(all_outcomes),
                "seed_variant": len(all_outcomes) == 6 and len(set(all_outcomes)) > 1,
                "outcomes": {
                    str(seed): bool(completed[(seed, row_id)]["utility"])
                    for seed in all_seeds
                    if (seed, row_id) in completed
                },
            }
        )
    by_split = {}
    for split in ("training", "calibration", "confirmation"):
        rows = [row for row in task_rows if row["split"] == split]
        by_split[split] = {
            "tasks": len(rows),
            "development_retained": sum(row["development_retained"] for row in rows),
            "confirmation_retained": sum(row["confirmation_retained"] for row in rows),
            "durable": sum(row["durable"] for row in rows),
            "all_six_failure": sum(row["all_six_failure"] for row in rows),
            "durable_ids": sorted(row["row_id"] for row in rows if row["durable"]),
        }
    by_suite = {}
    for suite in ("banking", "slack", "travel", "workspace"):
        rows = [row for row in task_rows if row["suite"] == suite]
        by_suite[suite] = {
            "tasks": len(rows),
            "durable": sum(row["durable"] for row in rows),
            "all_six_failure": sum(row["all_six_failure"] for row in rows),
            "durable_ids": sorted(row["row_id"] for row in rows if row["durable"]),
        }
    complete = (
        len(completed) == len(expected_pairs)
        and not missing_pairs
        and not failed_rows
        and not trace_failures
        and not source_errors
    )
    gate_spec = protocol["frozen_data_sufficiency_gate"]
    durable_total = sum(row["durable"] for row in task_rows)
    core_requirements = gate_spec["minimum_durable_tasks_in_each_core_suite"]
    domains_with_two = sum(values["durable"] >= 2 for values in by_suite.values())
    data_conditions = {
        "complete_144_episode_panel": complete and len(completed) == 144,
        "minimum_total_durable_tasks": durable_total
        >= int(gate_spec["minimum_total_durable_tasks"]),
        "minimum_durable_training_tasks": by_split["training"]["durable"]
        >= int(gate_spec["minimum_durable_tasks_per_split"]["training"]),
        "minimum_durable_calibration_tasks": by_split["calibration"]["durable"]
        >= int(gate_spec["minimum_durable_tasks_per_split"]["calibration"]),
        "minimum_durable_confirmation_tasks": by_split["confirmation"]["durable"]
        >= int(gate_spec["minimum_durable_tasks_per_split"]["confirmation"]),
        "minimum_each_core_suite": all(
            by_suite[suite]["durable"] >= int(minimum)
            for suite, minimum in core_requirements.items()
        ),
        "minimum_domains_with_two_durable_tasks": domains_with_two
        >= int(gate_spec["minimum_domains_with_two_durable_tasks"]),
    }
    dynamics_ready = all(data_conditions.values())
    balance = protocol["completion_head_balance_gate"]
    completion_conditions = {
        "data_sufficiency_gate_passed": dynamics_ready,
        "training_durable_success_tasks": by_split["training"]["durable"]
        >= int(balance["minimum_training_durable_success_tasks"]),
        "training_all_six_failure_tasks": by_split["training"]["all_six_failure"]
        >= int(balance["minimum_training_all_six_failure_tasks"]),
        "confirmation_durable_success_tasks": by_split["confirmation"]["durable"]
        >= int(balance["minimum_confirmation_durable_success_tasks"]),
        "confirmation_all_six_failure_tasks": by_split["confirmation"]["all_six_failure"]
        >= int(balance["minimum_confirmation_all_six_failure_tasks"]),
    }
    completion_ready = all(completion_conditions.values())
    if completion_ready:
        decision = "CUSTOM_PANEL_READY_FOR_FROZEN_LEDGER_AND_COMPLETION_ABLATION"
    elif dynamics_ready:
        decision = "CUSTOM_PANEL_DYNAMICS_READY_COMPLETION_HEAD_BLOCKED"
    else:
        decision = "CUSTOM_PANEL_DATA_SUFFICIENCY_NO_GO"
    failed_utility_rows = [row for row in completed.values() if not bool(row["utility"])]
    return {
        "scope": "template-disjoint custom AgentDojo clean panel data gate",
        "attack_outcomes_read": False,
        "protocol": str(protocol_path.resolve()),
        "manifest": str(manifest_path.resolve()),
        "archive_root": str(archive_root.resolve()),
        "coverage": {
            "expected_episodes": len(expected_pairs),
            "completed_with_trace": len(completed),
            "missing_pairs": len(missing_pairs),
            "failed_rows": len(failed_rows),
            "trace_failures": len(trace_failures),
            "source_errors": len(source_errors),
            "complete": complete,
        },
        "episodes": {
            "clean_successes": sum(bool(row["utility"]) for row in completed.values()),
            "clean_failures": len(failed_utility_rows),
            "failures_without_tool_call": sum(row["tool_calls"] == 0 for row in failed_utility_rows),
            "failures_with_tool_call": sum(row["tool_calls"] > 0 for row in failed_utility_rows),
        },
        "task_counts": {
            "tasks": len(task_rows),
            "durable": durable_total,
            "all_six_failure": sum(row["all_six_failure"] for row in task_rows),
            "seed_variant": sum(row["seed_variant"] for row in task_rows),
            "seed_split_informative": any(row["seed_variant"] for row in task_rows),
        },
        "by_split": by_split,
        "by_suite": by_suite,
        "data_sufficiency_gate": {
            "conditions": data_conditions,
            "passed": dynamics_ready,
        },
        "completion_head_balance_gate": {
            "conditions": completion_conditions,
            "passed": completion_ready,
        },
        "decision": decision,
        "dynamics_progress_ablation_permitted": dynamics_ready,
        "completion_value_head_permitted": completion_ready,
        "attack_data_permitted": False,
        "dreamer_training_permitted": False,
        "tasks": task_rows,
        "diagnostics": {
            "missing_pairs": [list(item) for item in missing_pairs],
            "failed_rows": failed_rows,
            "trace_failures": trace_failures,
            "source_errors": source_errors,
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
    print(json.dumps({"coverage": result["coverage"], "task_counts": result["task_counts"], "data_sufficiency_gate": result["data_sufficiency_gate"], "completion_head_balance_gate": result["completion_head_balance_gate"], "decision": result["decision"]}, indent=2))


if __name__ == "__main__":
    main()
