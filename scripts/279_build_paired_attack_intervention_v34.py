"""Build the frozen v34 task-disjoint paired attack manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wmagentattack.paired_attack_intervention import build_paired_manifest


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.source_manifest.read_text(encoding="utf-8"))
    manifest, audit = build_paired_manifest(source["rows"])
    if not audit["passed"]:
        raise RuntimeError(f"v34 manifest audit failed: {audit}")
    _write(args.output, manifest)
    _write(args.audit, audit)
    print(json.dumps(audit, sort_keys=True))


if __name__ == "__main__":
    main()
