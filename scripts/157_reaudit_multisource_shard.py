"""Re-audit an immutable failed shard without model calls or output rewrites."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wmagentattack.multisource_release_audit import audit_sharded_output, file_sha256


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--original-audit", type=Path, required=True)
    parser.add_argument("--recovery-audit", type=Path, required=True)
    parser.add_argument("--chunk-index", type=int, required=True)
    parser.add_argument("--num-chunks", type=int, required=True)
    args = parser.parse_args()

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    output_payload = json.loads(args.output.read_text(encoding="utf-8"))
    original_audit = json.loads(args.original_audit.read_text(encoding="utf-8"))
    audit = audit_sharded_output(
        manifest=manifest,
        protocol=protocol,
        output_payload=output_payload,
        original_audit=original_audit,
        chunk_index=args.chunk_index,
        num_chunks=args.num_chunks,
    )
    audit.update(
        {
            "protocol_file_sha256": file_sha256(args.protocol),
            "manifest_file_sha256": file_sha256(args.manifest),
            "immutable_output_file_sha256": file_sha256(args.output),
            "original_audit_file_sha256": file_sha256(args.original_audit),
        }
    )
    args.recovery_audit.parent.mkdir(parents=True, exist_ok=True)
    args.recovery_audit.write_text(
        json.dumps(audit, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(audit, ensure_ascii=False, sort_keys=True))
    if not audit["passed"]:
        raise SystemExit("SHARD_RELEASE_REAUDIT_FAILED")


if __name__ == "__main__":
    main()
