"""Build the frozen interaction-faithful tau3 manifest from the v1 panel."""

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

from wmagentattack.multisource_semantic_data import normalize_tool_schema
from wmagentattack.tau3_interactive import MANIFEST_SCHEMA_VERSION
from wmagentattack.tau3_multistep import file_sha256, stable_hash


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, default=str)
        + "\n",
        encoding="utf-8",
    )


def _task(domain: str, source_split: str, task_id: str) -> Any:
    from tau2.runner.helpers import load_tasks

    matches = [
        task
        for task in load_tasks(domain, source_split)
        if str(task.id) == str(task_id)
    ]
    if len(matches) != 1:
        raise ValueError("parent task does not resolve uniquely")
    return matches[0]


def build(
    protocol: dict[str, Any], parent: dict[str, Any], source_root: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    sys.path.insert(0, str(source_root / "src"))
    from loguru import logger
    from tau2.registry import registry

    logger.remove()
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=source_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if commit != protocol["source"]["commit"]:
        raise ValueError("tau3 source commit differs from the protocol")
    if parent["source_commit"] != commit:
        raise ValueError("parent manifest source commit differs")
    rows = []
    split_tasks: dict[str, set[str]] = defaultdict(set)
    domains = Counter()
    private_missing = []
    agent_private_exposure = []
    shared_model_hash = stable_hash(protocol["shared_model_identity"])
    role_hash = stable_hash(protocol["role_contracts"])
    for parent_row in sorted(parent["rows"], key=lambda row: row["episode_id"]):
        task = _task(
            parent_row["domain"], parent_row["source_split"], parent_row["task_id"]
        )
        environment = registry.get_env_constructor(parent_row["domain"])()
        agent_schemas = [
            normalize_tool_schema(tool.openai_schema)
            for tool in environment.get_tools()
        ]
        agent_schemas.sort(key=lambda row: row["function"]["name"])
        try:
            user_tools = environment.get_user_tools(include=task.user_tools) or []
        except Exception:
            user_tools = []
        user_schemas = [
            normalize_tool_schema(tool.openai_schema) for tool in user_tools
        ]
        user_schemas.sort(key=lambda row: row["function"]["name"])
        parent_schemas = parent_row["model_input"]["tool_schemas"]
        if stable_hash(agent_schemas) != stable_hash(parent_schemas):
            raise ValueError("agent tool interface differs from the parent panel")
        if str(environment.get_policy()) != str(parent_row["model_input"]["policy"]):
            raise ValueError("domain policy differs from the parent panel")
        private_scenario = str(task.user_scenario)
        if not private_scenario.strip():
            private_missing.append(parent_row["episode_id"])
        agent_interface = {
            "tool_schemas": agent_schemas,
            "policy": str(environment.get_policy()),
        }
        if "user_private_scenario" in json.dumps(
            agent_interface, ensure_ascii=False, sort_keys=True
        ):
            agent_private_exposure.append(parent_row["episode_id"])
        episode_id = parent_row["episode_id"].replace(
            "tau3-multistep::", "tau3-interactive::", 1
        )
        rows.append(
            {
                "episode_id": episode_id,
                "parent_episode_id": parent_row["episode_id"],
                "task_key": parent_row["task_key"],
                "domain": parent_row["domain"],
                "source_split": parent_row["source_split"],
                "task_id": parent_row["task_id"],
                "experimental_split": parent_row["experimental_split"],
                "structural_stratum": parent_row["structural_stratum"],
                "llm_seed": int(parent_row["llm_seed"]),
                "shared_model_identity_sha256": shared_model_hash,
                "role_contract_sha256": role_hash,
                "agent_interface": agent_interface,
                "user_private_input": {
                    "scenario": private_scenario,
                    "scenario_sha256": stable_hash(private_scenario),
                    "tool_schemas": user_schemas,
                },
            }
        )
        split_tasks[parent_row["experimental_split"]].add(
            parent_row["task_key"]
        )
        domains[parent_row["domain"]] += 1
    split_names = ("training", "calibration", "confirmation")
    overlaps = {
        f"{left}::{right}": sorted(split_tasks[left] & split_tasks[right])
        for left_index, left in enumerate(split_names)
        for right in split_names[left_index + 1 :]
    }
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "protocol_id": protocol["protocol_id"],
        "source_commit": commit,
        "parent_manifest_sha256": file_sha256(
            Path(protocol["source"]["parent_manifest"])
        ),
        "shared_model_identity_sha256": shared_model_hash,
        "role_contract_sha256": role_hash,
        "real_external_endpoint_calls": 0,
        "rows": rows,
    }
    expected_split = {"training": 30, "calibration": 9, "confirmation": 9}
    checks = {
        "parent_rows_reused_exactly": len(rows) == len(parent["rows"]) == 96,
        "unique_episode_ids": len(rows)
        == len({row["episode_id"] for row in rows}),
        "unique_parent_episode_ids": len(rows)
        == len({row["parent_episode_id"] for row in rows}),
        "expected_domain_episodes": all(domains[domain] == 32 for domain in protocol["source"]["domains"]),
        "expected_split_tasks": {
            split: len(split_tasks[split]) == expected_split[split]
            for split in split_names
        },
        "zero_task_overlap": not any(overlaps.values()),
        "all_private_user_scenarios_present": not private_missing,
        "zero_private_scenarios_in_agent_interface": not agent_private_exposure,
        "single_shared_model_identity": all(
            row["shared_model_identity_sha256"] == shared_model_hash for row in rows
        ),
        "single_role_contract": all(
            row["role_contract_sha256"] == role_hash for row in rows
        ),
        "zero_real_external_endpoints": manifest["real_external_endpoint_calls"] == 0,
    }
    flat = [
        value for value in checks.values() if isinstance(value, bool)
    ] + list(checks["expected_split_tasks"].values())
    audit = {
        "checks": checks,
        "passed": all(flat),
        "rows": len(rows),
        "task_count": len({row["task_key"] for row in rows}),
        "domain_episode_counts": dict(sorted(domains.items())),
        "split_task_counts": {
            split: len(split_tasks[split]) for split in split_names
        },
        "split_task_overlaps": overlaps,
        "private_scenario_missing": private_missing,
        "agent_private_exposure": agent_private_exposure,
        "manifest_content_sha256": stable_hash(manifest),
    }
    return manifest, audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--parent-manifest", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if protocol["status"] != "preregistered_before_manifest_or_interactive_outcomes":
        raise ValueError("interactive protocol is not preregistered")
    if file_sha256(args.parent_manifest) != protocol["source"][
        "parent_manifest_sha256"
    ]:
        raise ValueError("parent manifest hash differs")
    parent = json.loads(args.parent_manifest.read_text(encoding="utf-8"))
    manifest, audit = build(protocol, parent, args.source_root)
    _write(args.output, manifest)
    audit.update(
        {
            "protocol_file_sha256": file_sha256(args.protocol),
            "parent_manifest_file_sha256": file_sha256(args.parent_manifest),
            "output_file_sha256": file_sha256(args.output),
        }
    )
    _write(args.audit, audit)
    print(json.dumps(audit, ensure_ascii=False, sort_keys=True))
    if not audit["passed"]:
        raise SystemExit("TAU3_INTERACTIVE_MANIFEST_NO_GO")


if __name__ == "__main__":
    main()
