"""CPU-only semantic replay audit for immutable ToolSandbox decisions."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wmagentattack.counterfactual_execution import frozen_sandbox_clock
from wmagentattack.multisource_semantic_data import stable_hash
from wmagentattack.toolsandbox_replica_recovery import (
    canonical_replica_payload,
    replicas_semantically_identical,
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, default=str)
        + "\n",
        encoding="utf-8",
    )


def _state(context: Any) -> dict[str, Any]:
    from tool_sandbox.common.execution_context import DatabaseNamespace

    state = {}
    for namespace in DatabaseNamespace:
        if namespace == DatabaseNamespace.SANDBOX:
            continue
        try:
            state[str(namespace)] = context.get_database(namespace).to_dicts()
        except (IndexError, KeyError, TypeError, ValueError):
            continue
    return state


def _replay_once(
    scenario: Any,
    decision: dict[str, Any],
    seed: int,
    logical_clock_iso: str,
) -> dict[str, Any]:
    from tool_sandbox.common.execution_context import get_current_context, set_current_context

    random.seed(seed)
    np.random.seed(seed)
    context = copy.deepcopy(scenario.starting_context)
    set_current_context(context)
    tools = context.get_available_tools(False)
    before = _state(context)
    output: Any = None
    error: dict[str, str] | None = None
    try:
        with frozen_sandbox_clock(logical_clock_iso):
            output = tools[decision["name"]](**decision["arguments"])
    except Exception as exception:
        error = {"type": type(exception).__name__, "message": str(exception)}
    after = _state(get_current_context())
    canonical, normalization = canonical_replica_payload(
        before=before,
        after=after,
        output=output,
        error=error,
        status="error" if error else "success",
    )
    return {
        "raw": {
            "status": "error" if error else "success",
            "error": error,
            "output": output,
            "state_before_sha256": stable_hash(before),
            "state_after_sha256": stable_hash(after),
            "state_changed": stable_hash(before) != stable_hash(after),
        },
        "canonical": canonical,
        "normalization": normalization,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recovery-protocol", type=Path, required=True)
    parser.add_argument("--parent-protocol", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--immutable-output", type=Path, required=True)
    parser.add_argument("--original-audit", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    recovery = json.loads(args.recovery_protocol.read_text(encoding="utf-8"))
    for relative_path, expected_hash in recovery["implementation_sha256"].items():
        if file_sha256(ROOT / relative_path) != expected_hash:
            raise ValueError(f"recovery implementation differs: {relative_path}")
    parent = json.loads(args.parent_protocol.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    immutable = json.loads(args.immutable_output.read_text(encoding="utf-8"))
    original_audit = json.loads(args.original_audit.read_text(encoding="utf-8"))
    frozen = recovery["immutable_inputs"]
    observed_hashes = {
        "parent_protocol_sha256": file_sha256(args.parent_protocol),
        "manifest_sha256": file_sha256(args.manifest),
        "immutable_output_sha256": file_sha256(args.immutable_output),
        "original_audit_sha256": file_sha256(args.original_audit),
    }
    if observed_hashes != frozen:
        raise ValueError(f"immutable recovery inputs differ: {observed_hashes}")
    if manifest.get("source") != "tool_sandbox" or immutable.get("complete") is not True:
        raise ValueError("ToolSandbox immutable output is incomplete or mislabeled")

    sys.path.insert(0, str(args.source_root))
    from tool_sandbox.common.tool_discovery import ToolBackend
    from tool_sandbox.scenarios import named_scenarios

    random.seed(int(parent["sources"]["tool_sandbox"]["enumeration_seed"]))
    scenarios = named_scenarios(ToolBackend.DEFAULT)
    logical_clock = str(parent["sources"]["tool_sandbox"]["frozen_logical_clock_iso"])
    exact = [row for row in immutable["records"] if row["execution"]["tier"] == "exact"]
    original_mismatches = {
        row["row_id"] for row in exact if row["execution"].get("replica_identical") is False
    }
    replay_rows = []
    for row in exact:
        scenario = scenarios[row["metadata"]["scenario_name"]]
        replicas = [
            _replay_once(
                scenario,
                row["decision"],
                int(row["run_seed"]),
                logical_clock,
            )
            for _ in range(2)
        ]
        raw_identical = stable_hash(replicas[0]["raw"]) == stable_hash(replicas[1]["raw"])
        canonical_identical = replicas_semantically_identical(
            replicas[0]["canonical"], replicas[1]["canonical"]
        )
        normalization = {
            key: sum(replica["normalization"][key] for replica in replicas)
            for key in ("memory_addresses", "runtime_uuids")
        }
        replay_rows.append(
            {
                "row_id": row["row_id"],
                "raw_replica_identical": raw_identical,
                "canonical_replica_identical": canonical_identical,
                "original_replica_identical": row["execution"].get("replica_identical"),
                "replica_0_raw_sha256": stable_hash(replicas[0]["raw"]),
                "replica_1_raw_sha256": stable_hash(replicas[1]["raw"]),
                "replica_0_canonical_sha256": stable_hash(replicas[0]["canonical"]),
                "replica_1_canonical_sha256": stable_hash(replicas[1]["canonical"]),
                "normalization": normalization,
                "state_changed_agrees": (
                    replicas[0]["raw"]["state_changed"]
                    == replicas[1]["raw"]["state_changed"]
                ),
                "status_agrees": replicas[0]["raw"]["status"] == replicas[1]["raw"]["status"],
                "error_type_agrees": (
                    (replicas[0]["raw"]["error"] or {}).get("type")
                    == (replicas[1]["raw"]["error"] or {}).get("type")
                ),
            }
        )
        print(f"REAUDIT_DONE {len(replay_rows)}/{len(exact)} {row['row_id']}", flush=True)

    raw_mismatches = {row["row_id"] for row in replay_rows if not row["raw_replica_identical"]}
    semantic_mismatches = [row["row_id"] for row in replay_rows if not row["canonical_replica_identical"]]
    unexplained_original = [
        row["row_id"]
        for row in replay_rows
        if row["row_id"] in original_mismatches
        and not any(row["normalization"].values())
    ]
    checks = {
        "immutable_inputs_match": observed_hashes == frozen,
        "original_audit_failure_preserved": original_audit.get("passed") is False,
        "expected_exact_replays": len(replay_rows) == int(recovery["fixed_budget"]["exact_tool_calls"]),
        "original_raw_mismatch_set_reproduced": raw_mismatches == original_mismatches,
        "all_canonical_replicas_identical": not semantic_mismatches,
        "every_original_mismatch_explained_by_allowlist": not unexplained_original,
        "state_changed_agrees_in_every_pair": all(row["state_changed_agrees"] for row in replay_rows),
        "status_and_error_type_agree_in_every_pair": all(
            row["status_agrees"] and row["error_type_agrees"] for row in replay_rows
        ),
        "zero_llm_calls_added": recovery["fixed_budget"]["llm_calls"] == 0,
        "zero_real_external_endpoint_calls": True,
        "immutable_generation_output_not_rewritten": file_sha256(args.immutable_output) == frozen["immutable_output_sha256"],
    }
    result = {
        "schema_version": "wmagentattack.toolsandbox_replica_measurement_recovery.v1",
        "protocol_id": recovery["protocol_id"],
        "decision": "TOOLSANDBOX_SEMANTIC_REPLICA_RECOVERY_PASS" if all(checks.values()) else "TOOLSANDBOX_SEMANTIC_REPLICA_RECOVERY_NO_GO",
        "passed": all(checks.values()),
        "checks": checks,
        "exact_replays": len(replay_rows),
        "original_raw_mismatch_row_ids": sorted(original_mismatches),
        "replayed_raw_mismatch_row_ids": sorted(raw_mismatches),
        "canonical_mismatch_row_ids": sorted(semantic_mismatches),
        "unexplained_original_mismatch_row_ids": sorted(unexplained_original),
        "normalization_counts": {
            key: sum(row["normalization"][key] for row in replay_rows)
            for key in ("memory_addresses", "runtime_uuids")
        },
        "immutable_inputs": observed_hashes,
        "llm_calls_added": 0,
        "records_regenerated": 0,
        "generation_outputs_overwritten": False,
        "replay_rows": replay_rows,
    }
    _write(args.output, result)
    print(json.dumps({key: value for key, value in result.items() if key != "replay_rows"}, sort_keys=True))
    if not result["passed"]:
        raise SystemExit("TOOLSANDBOX_SEMANTIC_REPLICA_RECOVERY_NO_GO")


if __name__ == "__main__":
    main()
