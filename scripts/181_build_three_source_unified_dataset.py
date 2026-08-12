"""Build the frozen AgentDojo + ToolSandbox + InjecAgent action dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wmagentattack.multisource_suitability import file_sha256
from wmagentattack.three_source_unified import build_three_source_dataset


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--agentdojo-root", type=Path, required=True)
    parser.add_argument("--base-multisource", type=Path, required=True)
    parser.add_argument("--replication", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    frozen = protocol["sources"]
    inputs = {
        "agentdojo_steps_sha256": file_sha256(args.agentdojo_root / "steps.jsonl"),
        "agentdojo_metadata_sha256": file_sha256(args.agentdojo_root / "metadata.jsonl"),
        "base_multisource_sha256": file_sha256(args.base_multisource),
        "replication_sha256": file_sha256(args.replication),
    }
    for key, digest in inputs.items():
        if digest != frozen[key]:
            raise ValueError(f"frozen source hash mismatch: {key}")
    dataset, audit = build_three_source_dataset(
        agentdojo_root=args.agentdojo_root,
        base_records=_read_jsonl(args.base_multisource),
        replication_records=_read_jsonl(args.replication),
        protocol=protocol,
    )
    _write(args.output, dataset)
    audit.update(
        {
            "input_sha256": inputs,
            "protocol_sha256": file_sha256(args.protocol),
            "output_sha256": file_sha256(args.output),
        }
    )
    _write(args.audit, audit)
    print(json.dumps(audit, ensure_ascii=False, sort_keys=True))
    if not audit["passed"]:
        raise SystemExit("THREE_SOURCE_UNIFIED_PREFLIGHT_NO_GO")


if __name__ == "__main__":
    main()
