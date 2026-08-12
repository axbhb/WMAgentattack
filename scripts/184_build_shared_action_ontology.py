"""Annotate the frozen three-source dataset with a shared action ontology."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wmagentattack.multisource_suitability import file_sha256
from wmagentattack.shared_action_ontology import annotate_action_ontology


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if file_sha256(args.source) != protocol["source"]["dataset_sha256"]:
        raise ValueError("frozen three-source dataset hash mismatch")
    dataset = json.loads(args.source.read_text(encoding="utf-8"))
    output, audit = annotate_action_ontology(dataset)
    _write(args.output, output)
    audit.update(
        {
            "source_sha256": file_sha256(args.source),
            "protocol_sha256": file_sha256(args.protocol),
            "output_sha256": file_sha256(args.output),
        }
    )
    _write(args.audit, audit)
    print(json.dumps(audit, ensure_ascii=False, sort_keys=True))
    if not audit["passed"]:
        raise SystemExit("SHARED_ACTION_ONTOLOGY_PREFLIGHT_NO_GO")


if __name__ == "__main__":
    main()
