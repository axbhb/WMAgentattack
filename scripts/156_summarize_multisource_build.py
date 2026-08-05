"""Merge multi-source chunks and apply the frozen pilot/large release gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wmagentattack.multisource_semantic_data import stable_hash, summarize_generation


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, default=str)
        + "\n",
        encoding="utf-8",
    )


def _injecagent_pair_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in records:
        if row["source"] == "injecagent":
            grouped[(row["group_id"], int(row["run_seed"]))][row["variant"]] = row
    pairs = []
    discordant_clean_only = 0
    discordant_poison_only = 0
    for key, variants in sorted(grouped.items()):
        if set(variants) != {"clean", "poisoned"}:
            continue
        clean = bool(variants["clean"]["execution"].get("selected_attacker_tool"))
        poisoned = bool(variants["poisoned"]["execution"].get("selected_attacker_tool"))
        discordant_clean_only += int(clean and not poisoned)
        discordant_poison_only += int(poisoned and not clean)
        pairs.append({
            "group_id": key[0],
            "run_seed": key[1],
            "clean_selected_attacker_tool": clean,
            "poisoned_selected_attacker_tool": poisoned,
        })
    return {
        "complete_pairs": len(pairs),
        "clean_attacker_tool_rate": (
            sum(row["clean_selected_attacker_tool"] for row in pairs) / len(pairs)
            if pairs
            else 0.0
        ),
        "poisoned_attacker_tool_rate": (
            sum(row["poisoned_selected_attacker_tool"] for row in pairs) / len(pairs)
            if pairs
            else 0.0
        ),
        "discordant_clean_only": discordant_clean_only,
        "discordant_poison_only": discordant_poison_only,
        "paired_rows": pairs,
        "claim_boundary": "Descriptive counterevidence only; ASR is not a data-release gate.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--scale", choices=("pilot", "large"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    args = parser.parse_args()

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    contract_hash = stable_hash(protocol["shared_llm_contract"])
    all_records = []
    source_summaries = {}
    source_artifacts = {}
    for source in ("tool_sandbox", "injecagent", "tau3"):
        files = sorted((args.archive / "outputs" / source).glob("chunk*.json"))
        if not files:
            raise FileNotFoundError(f"no {source} output chunks")
        records = []
        complete = True
        for path in files:
            payload = json.loads(path.read_text(encoding="utf-8"))
            complete = complete and payload.get("complete") is True
            records.extend(payload["records"])
        expected = int(protocol["sources"][source][f"{args.scale}_expected_rows"])
        summary = summarize_generation(
            records,
            expected_rows=expected,
            require_exact_replica_determinism=True,
        )
        summary["all_chunks_complete"] = complete
        summary["passed"] = summary["passed"] and complete
        source_summaries[source] = summary
        source_artifacts[source] = {
            str(path.relative_to(args.archive)): _sha256(path) for path in files
        }
        all_records.extend(records)

    row_ids = [row["row_id"] for row in all_records]
    llm_hashes = {row["llm_contract_sha256"] for row in all_records}
    source_counts = Counter(row["source"] for row in all_records)
    endpoint_calls = sum(
        int(row.get("execution", {}).get("real_external_endpoint_calls", 0))
        for row in all_records
    )
    overall_checks = {
        "all_source_gates_pass": all(row["passed"] for row in source_summaries.values()),
        "unique_row_ids": len(row_ids) == len(set(row_ids)),
        "single_frozen_llm_contract": llm_hashes == {contract_hash},
        "zero_real_external_endpoint_calls": endpoint_calls == 0,
        "all_three_sources_present": set(source_counts) == {
            "tool_sandbox",
            "injecagent",
            "tau3",
        },
    }
    passed = all(overall_checks.values())
    decision = (
        "PILOT_GO__BUILD_FROZEN_LARGE_MULTISOURCE_V1"
        if args.scale == "pilot" and passed
        else "PILOT_NO_GO__DO_NOT_BUILD_LARGE"
        if args.scale == "pilot"
        else "LARGE_MULTISOURCE_V1_COMPLETE"
        if passed
        else "LARGE_MULTISOURCE_V1_INCOMPLETE"
    )
    args.dataset.parent.mkdir(parents=True, exist_ok=True)
    with args.dataset.open("w", encoding="utf-8") as handle:
        for row in sorted(all_records, key=lambda item: item["row_id"]):
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, default=str) + "\n")
    report = {
        "protocol_id": protocol["protocol_id"],
        "scale": args.scale,
        "decision": decision,
        "passed": passed,
        "overall_checks": overall_checks,
        "source_counts": dict(sorted(source_counts.items())),
        "total_rows": len(all_records),
        "llm_contract_sha256": contract_hash,
        "source_summaries": source_summaries,
        "injecagent_paired_counterevidence": _injecagent_pair_metrics(all_records),
        "real_external_endpoint_calls": endpoint_calls,
        "source_artifacts": source_artifacts,
        "dataset_sha256": _sha256(args.dataset),
        "protocol_sha256": _sha256(args.protocol),
        "claim_boundary": (
            "This gate validates deterministic offline data construction and a shared victim "
            "LLM. It does not establish attack effectiveness or authorize Dreamer training."
        ),
    }
    _write(args.output, report)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    if not passed:
        raise SystemExit(decision)


if __name__ == "__main__":
    main()
