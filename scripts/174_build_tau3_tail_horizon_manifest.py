"""Build the frozen label-blind tau3 bounded-tail panel."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wmagentattack.tau3_multistep import file_sha256
from wmagentattack.tau3_tail_horizon import select_tail_panel


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--parent-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if protocol["status"] != "preregistered_not_implemented_or_submitted":
        raise ValueError("tail protocol is not in preregistered state")
    if file_sha256(args.parent_manifest) != protocol["binding_parent_result"]["manifest_sha256"]:
        raise ValueError("tail parent manifest hash differs")
    parent = json.loads(args.parent_manifest.read_text(encoding="utf-8"))
    manifest, audit = select_tail_panel(parent, protocol)
    _write(args.output, manifest)
    audit.update({
        "protocol_file_sha256": file_sha256(args.protocol),
        "parent_manifest_file_sha256": file_sha256(args.parent_manifest),
        "output_file_sha256": file_sha256(args.output),
    })
    _write(args.audit, audit)
    print(json.dumps(audit, sort_keys=True))
    if not audit["passed"]:
        raise SystemExit("TAU3_TAIL_MANIFEST_NO_GO")


if __name__ == "__main__":
    main()
