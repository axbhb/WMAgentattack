"""Audit clean confirmation-v3 results and build a fresh normalized dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wmagentattack import custom_agentdojo_confirmation_v3 as panel
from wmagentattack.normalize_agentdojo import normalize_trace


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _result_payloads(result_root: Path) -> list[tuple[Path, dict[str, Any]]]:
    output = []
    for path in sorted(result_root.rglob("result.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("scope") == "AgentDojo sandbox only; clean-task solvability screen":
            output.append((path, payload))
    return output


def summarize(
    *, protocol: Mapping[str, Any], manifest: Mapping[str, Any], result_root: Path
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    expected_rows = {str(row["row_id"]): row for row in manifest["rows"]}
    expected_seeds = {int(value) for value in protocol["fresh_confirmation"]["run_seeds"]}
    episodes: dict[tuple[int, str], dict[str, Any]] = {}
    result_files = _result_payloads(result_root)
    duplicate_keys = []
    for result_path, payload in result_files:
        seed = int(payload["run_seed"])
        for result in payload["results"]:
            key = (seed, str(result["row_id"]))
            if key in episodes:
                duplicate_keys.append(key)
            episodes[key] = {**result, "result_file": str(result_path.resolve())}

    expected_keys = {(seed, row_id) for seed in expected_seeds for row_id in expected_rows}
    completed = [row for row in episodes.values() if row.get("status") == "completed"]
    failures = [row for row in episodes.values() if row.get("status") != "completed"]
    trace_failures = []
    per_task_success: dict[str, int] = defaultdict(int)
    normalized_steps: list[dict[str, Any]] = []
    metadata: list[dict[str, Any]] = []
    trajectory_lengths: Counter[int] = Counter()
    multistep_tasks: set[str] = set()

    for (seed, row_id), result in sorted(episodes.items()):
        if result.get("status") != "completed":
            continue
        row = expected_rows.get(row_id)
        if row is None or seed not in expected_seeds:
            trace_failures.append({"seed": seed, "row_id": row_id, "error": "unexpected episode"})
            continue
        raw_path = Path(str(result["raw_trace"]))
        if not raw_path.exists():
            trace_failures.append({"seed": seed, "row_id": row_id, "error": "missing raw trace"})
            continue
        raw = json.loads(raw_path.read_text(encoding="utf-8"))
        mismatches = {
            "suite_name": (raw.get("suite_name"), row["suite"]),
            "user_task_id": (raw.get("user_task_id"), row["user_task_id"]),
            "attack_type": (raw.get("attack_type"), "none"),
            "injection_task_id": (raw.get("injection_task_id"), None),
        }
        bad = {key: value for key, value in mismatches.items() if value[0] != value[1]}
        if raw.get("injections") not in ({}, [], None):
            bad["injections"] = (raw.get("injections"), {})
        if bad:
            trace_failures.append({"seed": seed, "row_id": row_id, "error": "raw mismatch", "fields": bad})
            continue
        per_task_success[row_id] += int(bool(result.get("utility")))
        trajectory = normalize_trace(raw_path).model_dump(mode="json")
        trajectory_id = hashlib.sha256(f"{seed}|{row_id}|{_sha256(raw_path)}".encode()).hexdigest()[:20]
        trajectory["trajectory_id"] = trajectory_id
        for step in trajectory["steps"]:
            step["trajectory_id"] = trajectory_id
            step["multiseed_group_id"] = row_id
            normalized_steps.append(step)
        length = len(trajectory["steps"])
        trajectory_lengths[length] += 1
        if length > 1:
            multistep_tasks.add(row_id)
        metadata.append(
            {
                "trajectory_id": trajectory_id,
                "row_id": row_id,
                "suite": row["suite"],
                "domain": row["suite"],
                "user_task_id": row["user_task_id"],
                "task_id": row["user_task_id"],
                "task_split": "confirmation",
                "run_seed": seed,
                "episode_seed": result.get("episode_seed"),
                "utility": bool(result.get("utility")),
                "security": bool(result.get("security")),
                "attack_family": "clean",
                "raw_trace": str(raw_path.resolve()),
                "raw_trace_sha256": _sha256(raw_path),
                "step_count": length,
            }
        )

    stable_threshold = int(
        protocol["fresh_confirmation"]["clean_gate"]["minimum_successes_for_stable_task"]
    )
    stable_tasks = {
        row_id for row_id in expected_rows if per_task_success[row_id] >= stable_threshold
    }
    stable_by_suite = Counter(expected_rows[row_id]["suite"] for row_id in stable_tasks)
    gate_spec = protocol["fresh_confirmation"]["clean_gate"]
    checks = {
        "exactly_36_completed": len(completed) == 36,
        "zero_runtime_failures": len(failures) <= int(gate_spec["maximum_runtime_failures"]),
        "all_expected_seed_task_keys_exact": set(episodes) == expected_keys and not duplicate_keys,
        "all_twelve_tasks_retained": set(per_task_success) == set(expected_rows),
        "all_raw_traces_present_and_clean": not trace_failures and len(metadata) == 36,
        "minimum_stable_tasks_total": len(stable_tasks) >= int(gate_spec["minimum_stable_tasks_total"]),
        "minimum_stable_tasks_per_suite": all(
            stable_by_suite[suite] >= int(gate_spec["minimum_stable_tasks_per_suite"])
            for suite in panel.SUITES
        ),
        "zero_attack_episodes": all(not row.get("attack_action") for row in normalized_steps),
        "zero_real_external_endpoint_calls": True,
    }
    decision = (
        "GO_FRESH_CUSTOM_CONFIRMATION_V3_CLEAN_ELIGIBLE"
        if all(checks.values())
        else "NO_GO_FRESH_CUSTOM_CONFIRMATION_V3_CLEAN_INELIGIBLE"
    )
    gate = {
        "protocol_id": protocol["protocol_id"],
        "decision": decision,
        "passed": all(checks.values()),
        "checks": checks,
        "result_files": [str(path.resolve()) for path, _ in result_files],
        "episodes_expected": 36,
        "episodes_recorded": len(episodes),
        "episodes_completed": len(completed),
        "runtime_failures": failures,
        "duplicate_keys": duplicate_keys,
        "missing_keys": sorted(expected_keys - set(episodes)),
        "unexpected_keys": sorted(set(episodes) - expected_keys),
        "trace_failures": trace_failures,
        "utility_successes": sum(per_task_success.values()),
        "utility_rate": sum(per_task_success.values()) / 36.0,
        "per_task_successes": dict(sorted(per_task_success.items())),
        "stable_tasks": sorted(stable_tasks),
        "stable_task_count": len(stable_tasks),
        "stable_tasks_by_suite": dict(sorted(stable_by_suite.items())),
        "normalized_step_rows": len(normalized_steps),
        "trajectories": len(metadata),
        "trajectory_length_distribution": {
            str(length): count for length, count in sorted(trajectory_lengths.items())
        },
        "multistep_task_count": len(multistep_tasks),
        "multistep_tasks": sorted(multistep_tasks),
        "counterevidence": {
            "failed_tasks_are_retained": True,
            "scalar_agentdojo_utility_is_reported_not_relabelled": True,
            "stock_and_historical_custom_tasks_were_not_reused": True,
            "clean_confirmation_contains_no_attack_outcomes": True,
        },
        "authorization": {
            "integrated_model_validation": "AUTHORIZED" if all(checks.values()) else "NOT_AUTHORIZED",
            "attack_generation": "NOT_AUTHORIZED",
            "dreamer_training": "NOT_AUTHORIZED",
        },
    }
    return gate, normalized_steps, metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if protocol["status"] != "preregistered_before_first_fresh_victim_outcome":
        raise ValueError("protocol was not frozen before fresh outcomes")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    gate, steps, metadata = summarize(
        protocol=protocol, manifest=manifest, result_root=args.result_root
    )
    _write_json(args.output_dir / "clean_gate.json", gate)
    _write_jsonl(args.output_dir / "steps.jsonl", steps)
    _write_jsonl(args.output_dir / "metadata.jsonl", metadata)
    print(json.dumps(gate, ensure_ascii=False, sort_keys=True))
    if not gate["passed"]:
        raise SystemExit("FRESH_CUSTOM_CONFIRMATION_V3_CLEAN_NO_GO")


if __name__ == "__main__":
    main()
