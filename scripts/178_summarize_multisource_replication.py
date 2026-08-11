"""Audit ToolSandbox/InjecAgent multi-seed replication outputs."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    all_records = []
    source_metrics = {}
    checks = {}
    for source in ("tool_sandbox", "injecagent"):
        output_paths = sorted((args.archive / "outputs").glob(f"{source}*.json"))
        audit_paths = sorted((args.archive / "audits").glob(f"{source}*.json"))
        expected_chunks = int(protocol["sources"][source].get("generation_chunks", 1))
        if len(output_paths) != expected_chunks or len(audit_paths) != expected_chunks:
            raise ValueError(f"{source} chunk surface is incomplete")
        outputs = [json.loads(path.read_text(encoding="utf-8")) for path in output_paths]
        audits = [json.loads(path.read_text(encoding="utf-8")) for path in audit_paths]
        records = [row for output in outputs for row in output["records"]]
        expected = int(protocol["sources"][source]["replication_expected_rows"])
        source_checks = {
            "complete": all(bool(output["complete"]) for output in outputs) and len(records) == expected,
            "audit_passed": all(bool(audit["passed"]) for audit in audits),
            "zero_runtime_failures": all(row.get("runtime_error") is None for row in records),
            "nonempty_completions": all(str(row.get("completion", "")).strip() for row in records),
            "zero_external_endpoints": all(int(row.get("execution", {}).get("real_external_endpoint_calls", 0)) == 0 for row in records),
        }
        if source == "tool_sandbox":
            source_checks["exact_replicas_identical"] = all(row["execution"].get("replica_identical") is not False for row in records)
        else:
            pairs = Counter((row["group_id"], row["run_seed"]) for row in records)
            source_checks["clean_poison_pairs_complete"] = all(value == 2 for value in pairs.values())
        checks[source] = source_checks
        source_metrics[source] = {
            "rows": len(records),
            "text_responses": sum(row["decision"]["kind"] == "text" for row in records),
            "tool_calls": sum(row["decision"]["kind"] == "tool_call" for row in records),
            "seeds": sorted({int(row["run_seed"]) for row in records}),
        }
        all_records.extend(records)
    passed = all(value for source in checks.values() for value in source.values())
    summary = {
        "protocol_id": protocol["protocol_id"],
        "decision": "AUXILIARY_MULTI_SEED_EXPANSION_COMPLETE" if passed else "AUXILIARY_MULTI_SEED_EXPANSION_NO_GO",
        "passed": passed,
        "source_metrics": source_metrics,
        "checks": checks,
        "rows": len(all_records),
        "independent_unit_warning": "Additional seeds estimate stochastic response probabilities; they do not increase the number of independent tasks.",
        "claim_boundary": "This auxiliary expansion does not overturn the frozen current-method scale NO-GO.",
    }
    _write(args.output, summary)
    with (args.archive / "replication_records.jsonl").open("w", encoding="utf-8") as handle:
        for row in all_records:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True))
    if not passed:
        raise SystemExit("MULTISOURCE_REPLICATION_NO_GO")


if __name__ == "__main__":
    main()
