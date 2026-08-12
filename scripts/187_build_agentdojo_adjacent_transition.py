"""Build and audit the frozen AgentDojo adjacent-transition dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wmagentattack.adjacent_transition import build_adjacent_transition_dataset
from wmagentattack.io_utils import read_jsonl
from wmagentattack.multisource_suitability import file_sha256


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--unified-dataset", type=Path, required=True)
    parser.add_argument("--steps", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if protocol["status"] != "preregistered_before_preflight":
        raise ValueError("protocol is not preregistered before preflight")
    source = protocol["source"]
    if file_sha256(args.unified_dataset) != source["unified_dataset_sha256"]:
        raise ValueError("unified dataset hash mismatch")
    if file_sha256(args.steps) != source["agentdojo_steps_sha256"]:
        raise ValueError("AgentDojo steps hash mismatch")
    unified = json.loads(args.unified_dataset.read_text(encoding="utf-8"))
    dataset, audit = build_adjacent_transition_dataset(
        unified=unified,
        raw_steps=read_jsonl(args.steps),
        protocol=protocol,
    )
    audit["protocol_sha256"] = file_sha256(args.protocol)
    audit["unified_dataset_sha256"] = file_sha256(args.unified_dataset)
    audit["agentdojo_steps_sha256"] = file_sha256(args.steps)
    _write(args.output_dir / "dataset.json", dataset)
    audit["output_sha256"] = file_sha256(args.output_dir / "dataset.json")
    _write(args.output_dir / "audit.json", audit)
    print(json.dumps(audit, ensure_ascii=False, sort_keys=True))
    if not audit["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
