"""Build one deterministic v29 relational successor dataset replica."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wmagentattack.relational_successor_contract import build_relational_dataset


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--protocol", type=Path, required=True)
    for name in ("v17", "v18", "v19", "hard", "structured", "support_execution", "base_adapters", "extension_adapters", "output_schemas"):
        p.add_argument("--" + name.replace("_", "-"), dest=name, type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--audit", type=Path, required=True)
    a = p.parse_args()
    protocol = read(a.protocol)
    if protocol["status"] != "preregistered_before_results":
        raise ValueError("v29 protocol is not frozen")
    for name in ("v17", "v18", "v19", "hard", "structured", "support_execution", "base_adapters", "extension_adapters", "output_schemas"):
        path = getattr(a, name)
        if sha256(path) != protocol["sources"][name]["sha256"]:
            raise ValueError(f"v29 source hash mismatch: {name}")
    payloads = [read(getattr(a, name)) for name in (
        "v17", "v18", "v19", "hard", "structured", "support_execution",
        "base_adapters", "extension_adapters", "output_schemas",
    )]
    dataset, audit = build_relational_dataset(*payloads)
    write(a.output, dataset)
    audit["dataset_sha256"] = sha256(a.output)
    write(a.audit, audit)
    print(json.dumps({
        "confirmation_rows": audit["confirmation_rows"],
        "support_rows": audit["support_rows"],
        "records_with_goal_links": audit["records_with_goal_links"],
        "static_candidate_count": audit["static_candidate_count"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
