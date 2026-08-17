"""Build the deterministic four-cell auxiliary-label view for v5."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wmagentattack.joint_outcome_auxiliary import build_joint_outcome_dataset
from wmagentattack.multisource_suitability import file_sha256


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--adjacent", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if file_sha256(args.adjacent) != protocol["source"]["adjacent_sha256"]:
        raise ValueError("adjacent source hash mismatch")
    if file_sha256(args.metadata) != protocol["source"]["metadata_sha256"]:
        raise ValueError("metadata source hash mismatch")
    adjacent = json.loads(args.adjacent.read_text(encoding="utf-8"))
    dataset, audit = build_joint_outcome_dataset(
        adjacent=adjacent,
        metadata=_read_jsonl(args.metadata),
        dirichlet_prior=float(protocol["labels"]["dirichlet_prior_per_cell"]),
    )
    if not audit["passed"]:
        raise RuntimeError(f"joint outcome preflight failed: {audit['checks']}")
    _write(args.output_dir / "dataset.json", dataset)
    _write(args.output_dir / "audit.json", audit)
    print(json.dumps(audit, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
