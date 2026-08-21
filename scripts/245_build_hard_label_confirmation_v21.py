"""Build and gate the deterministic v21 hard-label evaluation view."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wmagentattack.hard_label_confirmation import build_hard_label_confirmation


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--union", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if protocol["status"] not in {"preregistered_before_view_build", "model_preregistered_before_results"}:
        raise ValueError("v21 protocol is not buildable")
    if sha256(args.union) != protocol["source_union"]["sha256"]:
        raise ValueError("frozen v20 union hash mismatch")
    union = json.loads(args.union.read_text(encoding="utf-8"))
    dataset, audit = build_hard_label_confirmation(union)
    gate = protocol["view_gate"]
    checks = {
        "exact_rows": audit["rows"] == gate["rows"],
        "exact_hard_vocabulary": audit["hard_vocabulary_size"] == gate["hard_vocabulary_size"],
        "exact_removed_source_occurrences": audit["removed_source_token_occurrences"] == gate["removed_source_token_occurrences"],
        "exact_tool_family_counts": audit["tool_family_counts"] == gate["tool_family_counts"],
        "exact_source_test_counts": audit["source_test_counts"] == gate["source_test_counts"],
        "zero_split_overlap": not audit["split_overlap"],
        "all_rows_retain_positive_targets": audit["all_rows_retain_positive_targets"],
        "no_source_tokens_remain": audit["no_source_tokens_remain"],
        "model_inputs_unchanged": audit["model_inputs_unchanged"],
    }
    audit["gate_checks"] = checks
    audit["passed"] = all(checks.values())
    write(args.output, dataset)
    audit["dataset_sha256"] = sha256(args.output)
    frozen = protocol.get("hard_view", {}).get("sha256")
    if frozen is not None and frozen != audit["dataset_sha256"]:
        raise ValueError("frozen v21 hard-view hash mismatch")
    write(args.audit, audit)
    print(json.dumps(audit, sort_keys=True))
    if not audit["passed"]:
        raise SystemExit("v21 hard-label view gate failed")


if __name__ == "__main__":
    main()
