"""Partition the frozen v12 event graph into exact protocol and stochastic evidence features."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wmagentattack.multisource_suitability import file_sha256


def _partition(feature: str, exact_prefixes, evidence_prefixes) -> str:
    exact = any(feature.startswith(prefix) for prefix in exact_prefixes)
    evidence = any(feature.startswith(prefix) for prefix in evidence_prefixes)
    if exact == evidence:
        raise ValueError(f"feature must match exactly one partition: {feature}")
    return "exact" if exact else "evidence"


def build(source, protocol):
    rule = protocol["partition"]
    exact_catalog = []
    evidence_catalog = []
    for feature in source["feature_catalog"]:
        target = _partition(feature, rule["exact_prefixes"], rule["evidence_prefixes"])
        (exact_catalog if target == "exact" else evidence_catalog).append(feature)
    exact_set = set(exact_catalog)
    rows = []
    for row in source["rows"]:
        exact = sorted(feature for feature in row["features"] if feature in exact_set)
        evidence = sorted(feature for feature in row["features"] if feature not in exact_set)
        rows.append(
            {
                "event_id": row["event_id"],
                "task_name": row["task_name"],
                "trajectory_id": row["trajectory_id"],
                "step_id": row["step_id"],
                "exact_features": exact,
                "evidence_features": evidence,
            }
        )
    return {
        "schema_version": "wmagentattack.event_graph_partition.v14",
        "source_event_graph_sha256": protocol["sources"]["event_graph_sha256"],
        "full_feature_catalog": source["feature_catalog"],
        "exact_feature_catalog": exact_catalog,
        "evidence_feature_catalog": evidence_catalog,
        "rows": rows,
    }


def audit(dataset, source, protocol):
    gate = protocol["data_gate"]
    full = set(dataset["full_feature_catalog"])
    exact = set(dataset["exact_feature_catalog"])
    evidence = set(dataset["evidence_feature_catalog"])
    source_rows = {row["event_id"]: row for row in source["rows"]}
    checks = {
        "expected_rows": len(dataset["rows"]) == gate["expected_rows"],
        "expected_tasks": len({row["task_name"] for row in dataset["rows"]}) == gate["expected_tasks"],
        "unique_events": len({row["event_id"] for row in dataset["rows"]}) == len(dataset["rows"]),
        "disjoint_partition": not (exact & evidence),
        "exact_union": exact | evidence == full,
        "both_partitions_nonempty": bool(exact) and bool(evidence),
        "forbidden_absent": not any(token in feature for token in gate["forbidden_tokens"] for feature in full),
        "row_reconstruction": all(
            sorted(row["exact_features"] + row["evidence_features"])
            == source_rows[row["event_id"]]["features"]
            for row in dataset["rows"]
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "rows": len(dataset["rows"]),
        "tasks": len({row["task_name"] for row in dataset["rows"]}),
        "full_features": len(full),
        "exact_features": len(exact),
        "evidence_features": len(evidence),
        "exact_active_fraction": sum(len(row["exact_features"]) for row in dataset["rows"])
        / sum(len(source_rows[row["event_id"]]["features"]) for row in dataset["rows"]),
    }


def _write(path: Path, value: object):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--event-graph", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    if protocol["status"] != "preregistered_partition_data_gate":
        raise ValueError("v14 partition protocol not frozen")
    if file_sha256(args.event_graph) != protocol["sources"]["event_graph_sha256"]:
        raise ValueError("event graph hash mismatch")
    source = json.loads(args.event_graph.read_text())
    dataset = build(source, protocol)
    result = audit(dataset, source, protocol)
    _write(args.output_dir / "dataset.json", dataset)
    result["dataset_sha256"] = file_sha256(args.output_dir / "dataset.json")
    _write(args.output_dir / "audit.json", result)
    if not result["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
