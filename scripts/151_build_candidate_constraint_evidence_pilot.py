"""Build the frozen development-only candidate-by-constraint schema pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from wmagentattack.candidate_constraint_dataset import (
    audit_candidate_constraint_pilot,
    build_candidate_constraint_pilot,
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--raw-dataset", type=Path, required=True)
    parser.add_argument("--semantic-dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if protocol.get("status") != "preregistered_before_pilot_build":
        raise ValueError("candidate-constraint pilot protocol is not frozen")
    raw_hash = _sha256(args.raw_dataset)
    semantic_hash = _sha256(args.semantic_dataset)
    if raw_hash != protocol["source"]["raw_dataset_sha256"]:
        raise ValueError("raw dataset hash differs from the frozen protocol")
    if semantic_hash != protocol["source"]["semantic_dataset_sha256"]:
        raise ValueError("semantic dataset hash differs from the frozen protocol")
    raw = json.loads(args.raw_dataset.read_text(encoding="utf-8"))
    semantic = json.loads(args.semantic_dataset.read_text(encoding="utf-8"))
    selection = protocol["selection"]
    dataset = build_candidate_constraint_pilot(
        raw,
        semantic,
        split=str(selection["split"]),
        track=str(selection["track"]),
        suites=tuple(selection["suites"]),
        difficulties=tuple(selection["difficulties"]),
    )
    dataset["protocol_id"] = protocol["protocol_id"]
    dataset["source"] = {
        "raw_dataset_sha256": raw_hash,
        "semantic_dataset_sha256": semantic_hash,
        "protocol_sha256": _sha256(args.protocol),
    }
    audit = audit_candidate_constraint_pilot(
        dataset,
        expected=protocol["expected_counts"],
        schema_gate=protocol["schema_gate"],
        readiness_gate=protocol["training_readiness_gate"],
    )
    if not audit["schema_pass"]:
        raise SystemExit("candidate-constraint schema gate failed")
    _write_json(args.output, dataset)
    audit["protocol_sha256"] = _sha256(args.protocol)
    audit["raw_dataset_sha256"] = raw_hash
    audit["semantic_dataset_sha256"] = semantic_hash
    audit["output_sha256"] = _sha256(args.output)
    _write_json(args.audit, audit)
    print(json.dumps(audit, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
