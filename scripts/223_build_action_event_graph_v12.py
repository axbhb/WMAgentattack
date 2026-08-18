"""Build and audit the deterministic v12 action-event graph."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wmagentattack.action_event_graph import (
    audit_action_event_graph_dataset, build_action_event_graph_dataset,
)
from wmagentattack.multisource_suitability import file_sha256


def _read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _write(path: Path, value: object):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--steps", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(); protocol = json.loads(args.protocol.read_text())
    if protocol["status"] != "preregistered_action_event_graph_data_gate":
        raise ValueError("v12 data protocol is not frozen")
    if file_sha256(args.steps) != protocol["sources"]["steps_sha256"]:
        raise ValueError("steps hash mismatch")
    if file_sha256(args.events) != protocol["sources"]["events_sha256"]:
        raise ValueError("events hash mismatch")
    steps = _read_jsonl(args.steps); event_dataset = json.loads(args.events.read_text())
    dataset = build_action_event_graph_dataset(steps, event_dataset["events"])
    dataset["source_steps_sha256"] = protocol["sources"]["steps_sha256"]
    dataset["source_events_sha256"] = protocol["sources"]["events_sha256"]
    audit = audit_action_event_graph_dataset(
        dataset, expected_rows=protocol["data_gate"]["expected_rows"],
        expected_tasks=protocol["data_gate"]["expected_tasks"],
    )
    _write(args.output_dir / "dataset.json", dataset)
    audit["dataset_sha256"] = file_sha256(args.output_dir / "dataset.json")
    _write(args.output_dir / "audit.json", audit)
    if not audit["passed"]: raise SystemExit(2)


if __name__ == "__main__": main()
