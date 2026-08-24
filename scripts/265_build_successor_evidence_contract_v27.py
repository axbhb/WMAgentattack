"""Build one deterministic v27 typed successor-evidence dataset replica."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wmagentattack.successor_evidence_contract import build_successor_evidence_dataset


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--v17", type=Path, required=True)
    p.add_argument("--v18", type=Path, required=True)
    p.add_argument("--v19", type=Path, required=True)
    p.add_argument("--union", type=Path, required=True)
    p.add_argument("--hard", type=Path, required=True)
    p.add_argument("--support-execution", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--audit", type=Path, required=True)
    a = p.parse_args()
    protocol = json.loads(a.protocol.read_text(encoding="utf-8"))
    if protocol["status"] != "preregistered_before_results":
        raise ValueError("v27 protocol is not frozen")
    sources = protocol["sources"]
    for name, path in (
        ("v17", a.v17), ("v18", a.v18), ("v19", a.v19),
        ("v20_union", a.union), ("v21_hard", a.hard),
        ("v25_support_execution", a.support_execution),
    ):
        if sha256(path) != sources[name]["sha256"]:
            raise ValueError(f"v27 source hash mismatch: {name}")
    payloads = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in (a.v17, a.v18, a.v19, a.union, a.hard, a.support_execution)
    ]
    dataset, audit = build_successor_evidence_dataset(*payloads)
    write(a.output, dataset)
    audit["dataset_sha256"] = sha256(a.output)
    write(a.audit, audit)
    print(json.dumps({
        "confirmation_rows": audit["confirmation_rows"],
        "support_rows": audit["support_rows"],
        "full_render_matches": audit["full_render_matches"],
        "hard_render_matches": audit["hard_render_matches"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
