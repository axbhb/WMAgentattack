"""Build the deterministic grouped v20 intervention union."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wmagentattack.intervention_union import build_intervention_union


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--v17", type=Path, required=True)
    parser.add_argument("--v18", type=Path, required=True)
    parser.add_argument("--v19", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if protocol["status"] not in {"preregistered_before_union_build", "union_frozen_before_model_results"}:
        raise ValueError("v20 union protocol status is not buildable")
    for path, key in ((args.v17, "v17"), (args.v18, "v18"), (args.v19, "v19")):
        if sha256(path) != protocol["sources"][key]["sha256"]:
            raise ValueError(f"frozen {key} hash mismatch")
    values = [json.loads(path.read_text(encoding="utf-8")) for path in (args.v17, args.v18, args.v19)]
    dataset, audit = build_intervention_union(*values)
    gate = protocol["union_gate"]
    checks = {
        "exact_source_counts": audit["source_raw_transition_counts"] == gate["source_raw_transition_counts"],
        "exact_raw_count": audit["raw_transition_count"] == gate["raw_transition_count"],
        "canonical_count_range": gate["minimum_canonical_transition_count"] <= audit["canonical_transition_count"] <= gate["maximum_canonical_transition_count"],
        "exact_tasks": audit["tasks"] == gate["tasks"],
        "all_suites": audit["suites"] == gate["suites"],
        "all_difficulties": audit["difficulties"] == gate["difficulties"],
        "zero_target_conflicts": not audit["target_conflicts"],
        "zero_semantic_leakage": not audit["semantic_state_leakage"],
        "zero_model_input_group_keys": not audit["model_input_group_key_leakage"],
        "zero_task_cross_fold": not audit["task_cross_fold_groups"],
        "zero_root_cross_fold": not audit["root_cross_fold_groups"],
        "zero_pair_cross_fold": not audit["pair_cross_fold_groups"],
        "zero_sequence_cross_fold": not audit["sequence_cross_fold_groups"],
        "unique_transition_refs": audit["unique_transition_refs"],
        "all_sources_present": audit["all_sources_present"],
        "all_semantic_steps_adjacent": audit["all_semantic_steps_adjacent"],
    }
    audit["gate_checks"] = checks
    audit["passed"] = all(checks.values())
    write(args.output, dataset)
    audit["dataset_sha256"] = sha256(args.output)
    frozen = protocol.get("union", {}).get("sha256")
    if frozen is not None and frozen != audit["dataset_sha256"]:
        raise ValueError("frozen union hash mismatch")
    write(args.audit, audit)
    print(json.dumps(audit, sort_keys=True))
    if not audit["passed"]:
        raise SystemExit("v20 intervention union gate failed")


if __name__ == "__main__":
    main()
