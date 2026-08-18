"""Deterministically build and audit factorized semantic-transition labels."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from wmagentattack.factorized_transition_labels import (
    FACTOR_CLASSES, audit_factorized_transition_rows, build_factorized_transition_rows,
)
from wmagentattack.multisource_suitability import file_sha256


def write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(); protocol = json.loads(args.protocol.read_text())
    if protocol["status"] != "preregistered_label_gate_before_build":
        raise ValueError("label protocol not frozen")
    if file_sha256(args.dataset) != protocol["frozen_dataset"]["sha256"]:
        raise ValueError("frozen source dataset mismatch")
    source = json.loads(args.dataset.read_text())
    rows = build_factorized_transition_rows(source["events"])
    audit = audit_factorized_transition_rows(rows, expected_tasks=protocol["frozen_dataset"]["tasks"])
    gate = protocol["stage_f1_label_gate"]
    checks = dict(audit["checks"])
    checks["minimum_classes"] = all(
        len(audit["class_counts"][factor]) >= gate["minimum_observed_classes_per_factor"]
        for factor in FACTOR_CLASSES
    )
    checks["maximum_dominance"] = all(
        value <= gate["maximum_single_class_fraction"]
        for value in audit["maximum_class_fraction"].values()
    )
    audit["checks"] = checks; audit["passed"] = all(checks.values())
    dataset = {
        "schema_version": "wmagentattack.factorized_transition_labels.v1",
        "source_dataset_sha256": protocol["frozen_dataset"]["sha256"],
        "factor_classes": FACTOR_CLASSES, "rows": rows,
    }
    write(args.output_dir / "dataset.json", dataset)
    audit["dataset_sha256"] = file_sha256(args.output_dir / "dataset.json")
    write(args.output_dir / "audit.json", audit)
    if not audit["passed"]: raise SystemExit(2)


if __name__ == "__main__":
    main()
