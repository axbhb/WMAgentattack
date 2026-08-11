"""Build label-blind multi-seed replicas of the frozen ToolSandbox/InjecAgent panel."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wmagentattack.tau3_multistep import file_sha256, stable_hash


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def replicate(parent: dict[str, Any], protocol: dict[str, Any], source: str) -> tuple[dict[str, Any], dict[str, Any]]:
    seeds = list(map(int, protocol["sources"][source]["replication_seeds"]))
    rows = []
    for original in parent["rows"]:
        prefix = str(original["row_id"]).rsplit("::seed", 1)[0]
        for seed in seeds:
            row = copy.deepcopy(original)
            row["row_id"] = f"{prefix}::seed{seed}"
            row["run_seed"] = seed
            rows.append(row)
    chunks = int(protocol["sources"][source].get("generation_chunks", 1))
    if source == "injecagent" and chunks > 1:
        pairs: dict[tuple[str, int], list[dict[str, Any]]] = {}
        for row in rows:
            pairs.setdefault((str(row["group_id"]), int(row["run_seed"])), []).append(row)
        ordered_pairs = [pairs[key] for key in sorted(pairs)]
        if len(ordered_pairs) % chunks:
            raise ValueError("InjecAgent pair count must divide generation chunks")
        per_chunk: list[list[dict[str, Any]]] = [[] for _ in range(chunks)]
        for index, pair in enumerate(ordered_pairs):
            if len(pair) != 2:
                raise ValueError("InjecAgent replication pair is incomplete")
            per_chunk[index % chunks].extend(sorted(pair, key=lambda row: str(row["variant"])))
        if len({len(chunk) for chunk in per_chunk}) != 1:
            raise ValueError("InjecAgent pair-preserving chunks are unbalanced")
        rows = [per_chunk[chunk][offset] for offset in range(len(per_chunk[0])) for chunk in range(chunks)]
    manifest = {
        "schema_version": parent["schema_version"],
        "protocol_id": protocol["protocol_id"],
        "scale": "replication",
        "source": source,
        "source_commit": parent["source_commit"],
        "llm_contract_sha256": parent["llm_contract_sha256"],
        "real_external_endpoint_calls": 0,
        "rows": rows,
        "execution_preflight": copy.deepcopy(parent["execution_preflight"]),
    }
    pair_counts = Counter((row["group_id"], row["run_seed"]) for row in rows)
    expected_pair_size = 2 if source == "injecagent" else 1
    audit = {
        "source": source,
        "rows": len(rows),
        "seeds": seeds,
        "generation_chunks": chunks,
        "unique_row_ids": len({row["row_id"] for row in rows}) == len(rows),
        "all_groups_complete": all(value == expected_pair_size for value in pair_counts.values()),
        "every_injecagent_pair_within_one_modulo_chunk": (
            source != "injecagent"
            or all(
                len({index % chunks for index, row in enumerate(rows) if (str(row["group_id"]), int(row["run_seed"])) == key}) == 1
                for key in pair_counts
            )
        ),
        "real_external_endpoint_calls": 0,
        "manifest_content_sha256": stable_hash(manifest),
    }
    audit["passed"] = audit["unique_row_ids"] and audit["all_groups_complete"] and audit["every_injecagent_pair_within_one_modulo_chunk"] and len(rows) == int(protocol["sources"][source]["replication_expected_rows"])
    return manifest, audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--source", choices=("tool_sandbox", "injecagent"), required=True)
    parser.add_argument("--parent-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if protocol["status"] != "preregistered_before_replication_manifest_build":
        raise ValueError("replication protocol is not preregistered")
    expected = protocol["sources"][args.source]["parent_manifest_sha256"]
    if file_sha256(args.parent_manifest) != expected:
        raise ValueError("replication parent manifest hash differs")
    parent = json.loads(args.parent_manifest.read_text(encoding="utf-8"))
    manifest, audit = replicate(parent, protocol, args.source)
    _write(args.output, manifest)
    audit.update({"output_file_sha256": file_sha256(args.output), "protocol_file_sha256": file_sha256(args.protocol)})
    _write(args.audit, audit)
    print(json.dumps(audit, sort_keys=True))
    if not audit["passed"]:
        raise SystemExit("MULTISOURCE_REPLICATION_MANIFEST_NO_GO")


if __name__ == "__main__":
    main()
