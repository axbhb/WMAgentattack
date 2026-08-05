"""Label-blind release audit for outputs produced by modulo-sharded workers."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

from wmagentattack.multisource_semantic_data import stable_hash


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_sharded_output(
    *,
    manifest: Mapping[str, Any],
    protocol: Mapping[str, Any],
    output_payload: Mapping[str, Any],
    original_audit: Mapping[str, Any],
    chunk_index: int,
    num_chunks: int,
) -> dict[str, Any]:
    """Audit one immutable shard while deferring cross-shard pairs to the merge gate."""

    if num_chunks <= 0 or not 0 <= chunk_index < num_chunks:
        raise ValueError("invalid chunk specification")
    expected = [
        row
        for index, row in enumerate(manifest["rows"])
        if index % num_chunks == chunk_index
    ]
    records = list(output_payload.get("records", ()))
    expected_ids = [str(row["row_id"]) for row in expected]
    record_ids = [str(row.get("row_id")) for row in records]

    failures = [row for row in records if row.get("runtime_error")]
    empty = [
        str(row.get("row_id"))
        for row in records
        if not str(row.get("completion", "")).strip()
    ]
    invalid_names = []
    for row in records:
        decision = row.get("decision", {})
        if decision.get("kind") != "tool_call":
            continue
        allowed = {
            tool["function"]["name"]
            for tool in row.get("model_input", {}).get("tool_schemas", ())
        }
        if decision.get("name") not in allowed:
            invalid_names.append(str(row.get("row_id")))
    nondeterministic = [
        str(row.get("row_id"))
        for row in records
        if row.get("execution", {}).get("tier") == "exact"
        and row.get("execution", {}).get("replica_identical") is not True
    ]
    endpoint_calls = sum(
        int(row.get("execution", {}).get("real_external_endpoint_calls", 0))
        for row in records
    )
    contract_hash = stable_hash(protocol["shared_llm_contract"])
    observed_contracts = {str(row.get("llm_contract_sha256")) for row in records}

    pair_groups: dict[str, set[str]] = {}
    for row in records:
        if row.get("source") == "injecagent":
            pair_groups.setdefault(str(row["group_id"]), set()).add(str(row["variant"]))
    incomplete_pairs = sorted(
        group for group, variants in pair_groups.items() if variants != {"clean", "poisoned"}
    )
    original_failed_checks = sorted(
        name for name, passed in original_audit.get("checks", {}).items() if not passed
    )
    pair_is_cross_shard = (
        manifest.get("source") == "injecagent" and num_chunks > 1
    )
    checks = {
        "immutable_output_marked_complete": output_payload.get("complete") is True,
        "exact_expected_row_ids_in_manifest_order": record_ids == expected_ids,
        "zero_runtime_failures": not failures,
        "nonempty_completions": not empty,
        "parsed_tool_names_in_schema": not invalid_names,
        "exact_replica_determinism": not nondeterministic,
        "single_frozen_llm_contract": observed_contracts == {contract_hash},
        "zero_real_external_endpoint_calls": endpoint_calls == 0,
        "original_failure_only_local_pair_scope": (
            original_failed_checks == ["injecagent_pair_completeness"]
            if pair_is_cross_shard
            else not original_failed_checks
        ),
        "pair_completeness_deferred_only_to_global_merge": pair_is_cross_shard,
    }
    return {
        "schema_version": "wmagentattack.multisource.shard_release_audit.v1",
        "source": manifest.get("source"),
        "chunk_index": chunk_index,
        "num_chunks": num_chunks,
        "rows": len(records),
        "checks": checks,
        "passed": all(checks.values()),
        "repair_class": "label_blind_orchestration_gate_scope_only",
        "llm_calls_added": 0,
        "records_regenerated": 0,
        "outputs_overwritten": False,
        "original_failed_checks": original_failed_checks,
        "local_incomplete_pair_groups": len(incomplete_pairs),
        "local_incomplete_pair_groups_sha256": stable_hash(incomplete_pairs),
        "pair_completeness_release_scope": "global_merged_dataset",
        "runtime_failures": len(failures),
        "empty_completions": len(empty),
        "invalid_tool_names": len(invalid_names),
        "nondeterministic_exact_executions": len(nondeterministic),
        "real_external_endpoint_calls": endpoint_calls,
        "llm_contract_sha256": contract_hash,
    }
