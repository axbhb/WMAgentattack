"""Validate and normalize the user-authorized complete dual-source corpus."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def tool_call_count(messages: list[dict[str, Any]]) -> int:
    return sum(len(message.get("tool_calls") or []) for message in messages if message.get("role") == "assistant")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--injecagent-manifest", type=Path, required=True)
    parser.add_argument("--agentdojo-root", type=Path, required=True)
    parser.add_argument("--injecagent-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    manifest = json.loads(args.injecagent_manifest.read_text(encoding="utf-8"))
    if selection["protocol_id"] != protocol["protocol_id"] or manifest["protocol_id"] != protocol["protocol_id"]:
        raise ValueError("protocol and manifest IDs differ")

    seeds = {int(seed) for seed in protocol["victim_model"]["decoding"]["seeds"]}
    attacks = set(protocol["agentdojo"]["published_attacks"])
    selected = {
        (suite, task)
        for suite, row in selection["suites"].items()
        for task in row["user_tasks"]
    }
    fold_by_task = {
        (suite, task): int(fold)
        for suite, row in selection["suites"].items()
        for task, fold in row["outer_fold_by_user_task"].items()
    }
    expected_injections = {
        suite: set(row["injection_tasks"]) for suite, row in selection["suites"].items()
    }
    expected_keys = set()
    for seed in seeds:
        for suite, row in selection["suites"].items():
            for task in row["user_tasks"]:
                expected_keys.add((suite, task, seed, "clean", "none"))
                for attack in attacks:
                    for injection in row["injection_tasks"]:
                        expected_keys.add((suite, task, seed, attack, injection))

    trajectories: list[dict[str, Any]] = []
    actual_keys: Counter[tuple[str, str, int, str, str]] = Counter()
    auxiliary_count = 0
    malformed_json = 0
    for path in sorted(args.agentdojo_root.glob("seed*/**/*.json")):
        if path.name.startswith("summary_"):
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            malformed_json += 1
            continue
        if not {"suite_name", "user_task_id", "messages", "utility"} <= set(payload):
            continue
        seed_part = next((part for part in path.parts if part.startswith("seed")), None)
        if seed_part is None:
            continue
        seed = int(seed_part.removeprefix("seed"))
        if seed not in seeds:
            continue
        suite = str(payload["suite_name"])
        task = str(payload["user_task_id"])
        attack = payload.get("attack_type")
        is_clean = attack is None
        if (suite, task) not in selected:
            if is_clean and task in expected_injections.get(suite, set()):
                auxiliary_count += 1
                continue
            continue
        key = (suite, task, seed, str(attack or "clean"), str(payload.get("injection_task_id") or "none"))
        actual_keys[key] += 1
        trajectories.append(
            {
                "schema_version": "wmagentattack.agentdojo_trajectory.v4",
                "source": "agentdojo",
                "suite": suite,
                "task_id": task,
                "task_key": f"{suite}::{task}",
                "outer_fold": fold_by_task[(suite, task)],
                "seed": seed,
                "condition": "clean" if is_clean else "attack",
                "attack": attack,
                "injection_task_id": payload.get("injection_task_id"),
                "task_success": bool(payload["utility"]),
                "attack_success": bool(payload.get("security", False)) if not is_clean else False,
                "joint_success": bool(payload["utility"] and payload.get("security", False)) if not is_clean else False,
                "tool_calls": tool_call_count(payload["messages"]),
                "messages": payload["messages"],
                "runtime_error": payload.get("error"),
            }
        )

    injec_records = []
    injec_duplicate_ids = Counter()
    for path in sorted(args.injecagent_root.glob("seed*/**/*.json")):
        if path.name.startswith("summary_") or path.suffix == ".tmp":
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if "row_id" not in payload:
            continue
        injec_duplicate_ids[payload["row_id"]] += 1
        injec_records.append(payload)
    expected_injec_ids = {row["row_id"] for row in manifest["records"]}
    actual_injec_ids = set(injec_duplicate_ids)

    clean = [row for row in trajectories if row["condition"] == "clean"]
    attacked = [row for row in trajectories if row["condition"] == "attack"]
    clean_successes = Counter((row["suite"], row["task_id"]) for row in clean if row["task_success"])
    stable = {key for key, count in clean_successes.items() if count >= 2}
    stable_per_suite = Counter(suite for suite, _ in stable)
    candidate_rates: dict[tuple[str, str], dict[tuple[str, str], list[bool]]] = defaultdict(lambda: defaultdict(list))
    for row in attacked:
        candidate_rates[(row["suite"], row["task_id"])][(str(row["attack"]), str(row["injection_task_id"]))].append(row["joint_success"])
    diverse = {
        task for task, candidates in candidate_rates.items()
        if len({sum(values) / len(values) for values in candidates.values()}) >= 2
    }
    multistep = {(row["suite"], row["task_id"]) for row in trajectories if row["tool_calls"] >= 2}

    checks = {
        "agentdojo_exact_clean_count": len(clean) == int(protocol["agentdojo"]["budget"]["clean_trajectories"]),
        "agentdojo_exact_attack_count": len(attacked) == int(protocol["agentdojo"]["budget"]["attack_trajectories"]),
        "agentdojo_exact_expected_keys": set(actual_keys) == expected_keys,
        "agentdojo_no_duplicate_keys": all(count == 1 for count in actual_keys.values()),
        "agentdojo_no_malformed_json": malformed_json == 0,
        "agentdojo_zero_runtime_failures": not any(row["runtime_error"] for row in trajectories),
        "injecagent_exact_count": len(injec_records) == int(protocol["injecagent"]["budget"]["decisions"]),
        "injecagent_exact_expected_ids": actual_injec_ids == expected_injec_ids,
        "injecagent_no_duplicate_ids": all(count == 1 for count in injec_duplicate_ids.values()),
        "injecagent_nonempty_completions": all(str(row.get("completion", "")).strip() for row in injec_records),
        "injecagent_zero_runtime_failures": not any(row.get("runtime_error") for row in injec_records),
        "one_model_contract": all(row.get("model_contract_id") == protocol["protocol_id"] + "::llama31-8b-sampled" for row in injec_records),
        "zero_real_external_endpoint_calls": sum(int(row.get("real_external_endpoint_calls", 0)) for row in injec_records) == 0,
    }
    complete = all(checks.values())
    summary = {
        "protocol_id": protocol["protocol_id"],
        "complete": complete,
        "decision": "COMPLETE_EXPLORATORY_FULL_CORPUS" if complete else "INVALID_INCOMPLETE_FULL_CORPUS",
        "checks": checks,
        "agentdojo": {
            "trajectories": len(trajectories),
            "clean": len(clean),
            "attack": len(attacked),
            "auxiliary_injection_utility_diagnostics": auxiliary_count,
            "stable_clean_tasks": len(stable),
            "stable_clean_tasks_per_suite": dict(stable_per_suite),
            "diverse_joint_outcome_tasks": len(diverse),
            "joint_success_trajectories": sum(row["joint_success"] for row in attacked),
            "multistep_tasks": len(multistep),
        },
        "injecagent": {
            "decisions": len(injec_records),
            "pairs": len({row["pair_id"] for row in injec_records}),
            "clean_attacker_tool_rate": sum(bool(row.get("selected_attacker_tool")) for row in injec_records if row["variant"] == "clean") / max(1, sum(row["variant"] == "clean" for row in injec_records)),
            "poisoned_attacker_tool_rate": sum(bool(row.get("selected_attacker_tool")) for row in injec_records if row["variant"] == "poisoned") / max(1, sum(row["variant"] == "poisoned" for row in injec_records)),
        },
        "scientific_context": "The 0903 pilot NO_SCALE conclusion remains unchanged; this corpus was built under explicit user authorization for exploratory analysis.",
        "content_checksums": "disabled",
    }
    write_jsonl(args.output_dir / "agentdojo_trajectories.jsonl", trajectories)
    write_jsonl(args.output_dir / "injecagent_interventions.jsonl", injec_records)
    write_json(args.output_dir / "full_collection_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True), flush=True)
    if not complete:
        raise SystemExit(summary["decision"])


if __name__ == "__main__":
    main()
