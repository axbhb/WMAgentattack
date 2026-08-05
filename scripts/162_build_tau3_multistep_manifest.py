"""Build the frozen, victim-outcome-blind tau3 multi-step pilot manifest."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wmagentattack.multisource_semantic_data import build_model_input
from wmagentattack.tau3_multistep import (
    MANIFEST_SCHEMA_VERSION,
    allocate_stratum,
    file_sha256,
    stable_hash,
    task_key,
)


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, default=str)
        + "\n",
        encoding="utf-8",
    )


def _goal(task: Any) -> str:
    instructions = task.user_scenario.model_dump(mode="json").get("instructions")
    if isinstance(instructions, dict):
        parts = [
            str(instructions[key]).strip()
            for key in (
                "reason_for_call",
                "known_info",
                "unknown_info",
                "task_instructions",
            )
            if instructions.get(key)
        ]
        return "\n".join(parts)
    return str(instructions)


def _reset(registry: Any, domain: str, task: Any) -> Any:
    environment = registry.get_env_constructor(domain)()
    initial = task.initial_state
    environment.set_state(
        initial.initialization_data if initial else None,
        initial.initialization_actions if initial else None,
        initial.message_history if initial and initial.message_history else [],
    )
    return environment


def _reference_structure(registry: Any, domain: str, task: Any) -> dict[str, Any]:
    actions = task.evaluation_criteria.actions if task.evaluation_criteria else []
    calls = [
        {"name": action.name, "arguments": action.arguments}
        for action in (actions or [])
        if action.requestor == "assistant"
    ]
    environment = _reset(registry, domain, task)
    mutations = 0
    errors = 0
    for call in calls:
        before = environment.get_db_hash()
        try:
            environment.make_tool_call(
                call["name"], requestor="assistant", **call["arguments"]
            )
        except Exception:
            errors += 1
        mutations += int(before != environment.get_db_hash())
    return {
        "assistant_action_count": len(calls),
        "mutation_count": mutations,
        "error_count": errors,
        "tool_names": sorted({call["name"] for call in calls}),
        "final_state_sha256": environment.get_db_hash(),
    }


def _split_counts(protocol: dict[str, Any], domain: str, size: int) -> dict[str, int]:
    selection = protocol["task_selection"]
    if domain == "telecom":
        return selection["experimental_split_counts_within_telecom_16_task_stratum"]
    if size == 12:
        return selection["experimental_split_counts_within_each_12_task_stratum"]
    if size == 4:
        return selection["experimental_split_counts_within_each_4_task_stratum"]
    raise ValueError("unsupported frozen stratum size")


def build(
    protocol: dict[str, Any], source_root: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    sys.path.insert(0, str(source_root / "src"))
    from loguru import logger
    from tau2.registry import registry
    from tau2.runner.helpers import load_tasks

    logger.remove()
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=source_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if commit != protocol["source"]["commit"]:
        raise ValueError("tau3 source commit differs from frozen protocol")

    structural: dict[str, list[dict[str, Any]]] = defaultdict(list)
    all_eligible = Counter()
    for domain in protocol["source"]["domains"]:
        for source_split in protocol["source"]["source_splits"]:
            for task in load_tasks(domain, source_split):
                summary = _reference_structure(registry, domain, task)
                if summary["assistant_action_count"] < int(
                    protocol["source"][
                        "minimum_reference_assistant_actions_for_structural_eligibility"
                    ]
                ):
                    continue
                stratum = (
                    "reference_mutating"
                    if summary["mutation_count"] > 0
                    else "reference_read_only"
                )
                key = task_key(domain, source_split, str(task.id))
                structural[f"{domain}::{stratum}"].append(
                    {
                        "domain": domain,
                        "source_split": source_split,
                        "task": task,
                        "task_key": key,
                        "stratum": stratum,
                        "reference_summary": summary,
                    }
                )
                all_eligible[f"{domain}::{stratum}"] += 1

    selected: list[dict[str, Any]] = []
    split_assignment: dict[str, str] = {}
    seed = str(protocol["task_selection"]["selection_seed"])
    for domain in protocol["source"]["domains"]:
        strata = (
            protocol["task_selection"]["telecom_strata"]
            if domain == "telecom"
            else protocol["task_selection"]["retail_and_airline_strata"]
        )
        for stratum, count_value in strata.items():
            count = int(count_value)
            if count == 0:
                continue
            pool = structural[f"{domain}::{stratum}"]
            ordered = sorted(
                pool,
                key=lambda row: stable_hash(
                    [seed, domain, stratum, row["task_key"]]
                ),
            )
            if len(ordered) < count:
                raise ValueError(f"insufficient {domain}/{stratum} tasks")
            chosen = ordered[:count]
            selected.extend(chosen)
            split_assignment.update(
                allocate_stratum(
                    [row["task_key"] for row in chosen],
                    seed=seed,
                    domain=domain,
                    stratum=stratum,
                    counts=_split_counts(protocol, domain, count),
                )
            )

    contract_hash = stable_hash(protocol["shared_llm_contract"])
    rows = []
    split_tasks: dict[str, set[str]] = defaultdict(set)
    selected_strata = Counter()
    selected_domains = Counter()
    for row in sorted(
        selected,
        key=lambda item: (item["domain"], item["source_split"], item["task_key"]),
    ):
        task = row["task"]
        domain = row["domain"]
        environment = _reset(registry, domain, task)
        schemas = [tool.openai_schema for tool in environment.get_tools()]
        model_input = build_model_input(
            trusted_goal=_goal(task),
            tool_schemas=schemas,
            policy=environment.get_policy(),
        )
        if set(model_input) != {"trusted_goal", "tool_schemas", "policy"}:
            raise ValueError("unexpected model input fields")
        experimental_split = split_assignment[row["task_key"]]
        split_tasks[experimental_split].add(row["task_key"])
        selected_strata[f"{domain}::{row['stratum']}"] += 1
        selected_domains[domain] += 1
        for llm_seed in protocol["trajectory_collection"]["llm_seeds"]:
            episode_id = (
                f"tau3-multistep::{domain}::{row['source_split']}::"
                f"{task.id}::seed{int(llm_seed)}"
            )
            rows.append(
                {
                    "episode_id": episode_id,
                    "task_key": row["task_key"],
                    "domain": domain,
                    "source_split": row["source_split"],
                    "task_id": str(task.id),
                    "experimental_split": experimental_split,
                    "structural_stratum": row["stratum"],
                    "llm_seed": int(llm_seed),
                    "llm_contract_sha256": contract_hash,
                    "model_input": model_input,
                    "simulator_audit_only": row["reference_summary"],
                }
            )

    split_names = ("training", "calibration", "confirmation")
    overlaps = {
        f"{left}::{right}": sorted(split_tasks[left] & split_tasks[right])
        for left_index, left in enumerate(split_names)
        for right in split_names[left_index + 1 :]
    }
    leaked = {
        row["episode_id"]: sorted(
            set(row["model_input"])
            & {
                "evaluation_criteria",
                "gold_actions",
                "reference_actions",
                "reward",
                "state_changed",
                "task_success",
            }
        )
        for row in rows
        if set(row["model_input"])
        & {
            "evaluation_criteria",
            "gold_actions",
            "reference_actions",
            "reward",
            "state_changed",
            "task_success",
        }
    }
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "protocol_id": protocol["protocol_id"],
        "source_commit": commit,
        "llm_contract_sha256": contract_hash,
        "real_external_endpoint_calls": 0,
        "rows": rows,
    }
    checks = {
        "expected_tasks": len(selected)
        == int(protocol["task_selection"]["expected_tasks"]),
        "expected_episodes": len(rows)
        == int(protocol["trajectory_collection"]["expected_episodes"]),
        "expected_domain_tasks": all(
            selected_domains[domain]
            == int(protocol["task_selection"]["per_domain"])
            for domain in protocol["source"]["domains"]
        ),
        "expected_split_tasks": {
            split: len(split_tasks[split])
            == int(protocol["task_selection"]["expected_task_counts"][split])
            for split in split_names
        },
        "zero_task_overlap": not any(overlaps.values()),
        "unique_episode_ids": len(rows)
        == len({row["episode_id"] for row in rows}),
        "one_llm_contract": all(
            row["llm_contract_sha256"] == contract_hash for row in rows
        ),
        "label_blind_model_inputs": not leaked,
        "zero_real_external_endpoints": manifest["real_external_endpoint_calls"] == 0,
    }
    flat_checks = [
        value
        for value in checks.values()
        if isinstance(value, bool)
    ] + list(checks["expected_split_tasks"].values())
    audit = {
        "checks": checks,
        "passed": all(flat_checks),
        "eligible_strata": dict(sorted(all_eligible.items())),
        "selected_strata": dict(sorted(selected_strata.items())),
        "selected_domain_tasks": dict(sorted(selected_domains.items())),
        "split_task_counts": {
            split: len(split_tasks[split]) for split in split_names
        },
        "split_task_overlaps": overlaps,
        "leaked_model_inputs": leaked,
        "manifest_content_sha256": stable_hash(manifest),
    }
    return manifest, audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if protocol["status"] != "preregistered_before_manifest_build_or_victim_outcomes":
        raise ValueError("tau3 multi-step protocol is not frozen before manifest build")
    manifest, audit = build(protocol, args.source_root)
    _write(args.output, manifest)
    audit.update(
        {
            "protocol_file_sha256": file_sha256(args.protocol),
            "output_file_sha256": file_sha256(args.output),
        }
    )
    _write(args.audit, audit)
    print(json.dumps(audit, ensure_ascii=False, sort_keys=True))
    if not audit["passed"]:
        raise SystemExit("TAU3_MULTISTEP_MANIFEST_NO_GO")


if __name__ == "__main__":
    main()
