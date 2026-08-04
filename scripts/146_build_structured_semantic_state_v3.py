"""Build and audit Structured Semantic State v3 from a frozen clean dataset."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from wmagentattack.semantic_state_v3 import (
    SEMANTIC_STATE_V3_SCHEMA_VERSION,
    find_semantic_state_v3_leakage,
    semantic_state_v3_fingerprint,
    semantic_state_v3_payload,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def build(source_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    source = json.loads(source_path.read_text(encoding="utf-8"))
    output = copy.deepcopy(source)
    output["schema_version"] = "wmagentattack.panel_v2.semantic_state_v3_dataset"
    output["semantic_state_schema_version"] = SEMANTIC_STATE_V3_SCHEMA_VERSION
    fingerprints: list[str] = []
    leakage: list[dict[str, Any]] = []
    hidden_invariance_failures: list[str] = []
    runtime_invariance_failures: list[str] = []
    state_bytes = 0
    split_tasks: dict[str, set[str]] = {}
    suites = Counter()

    for episode in output["episodes"]:
        split_tasks.setdefault(str(episode["split"]), set()).add(str(episode["task_id"]))
        suites[str(episode["suite"])] += 1
        for prefix in episode["prefixes"]:
            features = prefix["features"]
            payload = semantic_state_v3_payload(features)
            fingerprint = semantic_state_v3_fingerprint(features)
            prefix["features"] = {"semantic_state_v3": payload}
            fingerprints.append(fingerprint)
            state_bytes += len(
                json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
            )
            findings = find_semantic_state_v3_leakage(payload)
            if findings:
                leakage.append(
                    {
                        "episode_id": episode["episode_id"],
                        "prefix_index": prefix["prefix_index"],
                        "paths": list(findings),
                    }
                )

            hidden_changed = copy.deepcopy(features)
            hidden_changed["causal_state_summary"] = {
                "synthetic_hidden_oracle": fingerprint,
                "last_state_changed": not bool(
                    features.get("causal_state_summary", {}).get(
                        "last_state_changed", False
                    )
                ),
            }
            if semantic_state_v3_fingerprint(hidden_changed) != fingerprint:
                hidden_invariance_failures.append(
                    f"{episode['episode_id']}::{prefix['prefix_index']}"
                )

            runtime_changed = copy.deepcopy(features)
            for index, record in enumerate(
                runtime_changed.get("ledger_v2", {}).get("records", ())
            ):
                record["record_id"] = f"synthetic::{fingerprint}::{index}"
                record["state_provenance"] = "mutating"
            for receipt in runtime_changed.get("ledger_v2", {}).get(
                "execution_receipts", ()
            ):
                receipt["episode_id"] = f"synthetic::{fingerprint}"
                receipt["observation_fingerprint"] = fingerprint
            if semantic_state_v3_fingerprint(runtime_changed) != fingerprint:
                runtime_invariance_failures.append(
                    f"{episode['episode_id']}::{prefix['prefix_index']}"
                )

    split_names = sorted(split_tasks)
    overlaps = {
        f"{left}::{right}": sorted(split_tasks[left] & split_tasks[right])
        for left_index, left in enumerate(split_names)
        for right in split_names[left_index + 1 :]
    }
    audit = {
        "schema_version": SEMANTIC_STATE_V3_SCHEMA_VERSION,
        "source_path": str(source_path),
        "source_sha256": _sha256(source_path),
        "episodes": len(output["episodes"]),
        "states": len(fingerprints),
        "unique_state_fingerprints": len(set(fingerprints)),
        "mean_state_bytes": state_bytes / max(1, len(fingerprints)),
        "suite_episode_counts": dict(sorted(suites.items())),
        "split_task_counts": {
            split: len(tasks) for split, tasks in sorted(split_tasks.items())
        },
        "split_task_overlaps": overlaps,
        "leakage_failures": leakage,
        "hidden_oracle_invariance_failures": hidden_invariance_failures,
        "runtime_identifier_invariance_failures": runtime_invariance_failures,
        "all_states_causal_and_label_blind": not (
            leakage or hidden_invariance_failures or runtime_invariance_failures
        ),
        "task_disjoint_splits": not any(overlaps.values()),
    }
    return output, audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()
    dataset, audit = build(args.source)
    if not audit["all_states_causal_and_label_blind"]:
        raise SystemExit("semantic state v3 causal/leakage audit failed")
    if not audit["task_disjoint_splits"]:
        raise SystemExit("semantic state v3 task-disjoint split audit failed")
    _write_json(args.output, dataset)
    audit["output_sha256"] = _sha256(args.output)
    _write_json(args.audit, audit)
    print(json.dumps(audit, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
