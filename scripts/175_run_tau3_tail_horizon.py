"""Run one chunk of the frozen tau3 20/20/80 tail pilot."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from wmagentattack.tau3_multistep import file_sha256
from wmagentattack.tau3_tail_horizon import effective_tail_protocol
from importlib import import_module

interactive = import_module("168_run_tau3_interactive_llm")


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, default=str) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--chunk-index", type=int, required=True)
    parser.add_argument("--num-chunks", type=int, required=True)
    args = parser.parse_args()
    frozen = json.loads(args.protocol.read_text(encoding="utf-8"))
    if frozen["status"] != "manifest_frozen_before_interactive_outcomes":
        raise ValueError("tail manifest is not frozen")
    if file_sha256(args.manifest) != frozen["frozen_manifest"]["sha256"]:
        raise ValueError("tail manifest hash differs")
    for relative, expected in frozen["implementation_sha256"].items():
        if file_sha256(ROOT / relative) != expected:
            raise ValueError(f"tail implementation differs: {relative}")
    base_path = ROOT / frozen["execution_base"]["path"]
    if file_sha256(base_path) != frozen["execution_base"]["sha256"]:
        raise ValueError("tail execution-base protocol differs")
    base = json.loads(base_path.read_text(encoding="utf-8"))
    protocol = effective_tail_protocol(frozen, base)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=args.source_root, check=True, capture_output=True, text=True).stdout.strip()
    if commit != frozen["source"]["commit"]:
        raise ValueError("tau3 source commit differs")
    if not 0 <= args.chunk_index < args.num_chunks:
        raise ValueError("invalid tail chunk")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    rows = [row for index, row in enumerate(manifest["rows"]) if index % args.num_chunks == args.chunk_index]
    sys.path.insert(0, str(args.source_root / "src"))
    from loguru import logger
    from tau2.registry import registry
    logger.remove()
    shared = interactive.FrozenSharedConversationModel(protocol["shared_model_identity"])
    episodes, failures = [], []
    for row in rows:
        try:
            episodes.append(interactive.collect_episode(shared=shared, registry=registry, row=row, protocol=protocol))
        except Exception as exception:
            failures.append({"episode_id": row["episode_id"], "type": type(exception).__name__, "message": str(exception)})
        _write(args.output, {"protocol_id": frozen["protocol_id"], "chunk_index": args.chunk_index, "num_chunks": args.num_chunks, "episodes": episodes, "failures": failures, "real_external_endpoint_calls": 0})
    audit = {
        "chunk_index": args.chunk_index,
        "expected_rows": len(rows),
        "completed_episodes": len(episodes),
        "runtime_failures": failures,
        "physical_calls": sum(row["agent_physical_calls"] + row["user_physical_calls"] for row in episodes),
        "forced_budget_stops": sum(row["agent_forced_budget_stop"] or row["user_forced_budget_stop"] for row in episodes),
        "nondeterministic_sequences": sum(not event["replica_identical"] for row in episodes for event in row["combined_tool_events"]),
        "agent_private_scenario_exposures": sum(row["agent_private_scenario_exposures"] for row in episodes),
        "real_external_endpoint_calls": 0,
        "output_sha256": file_sha256(args.output),
        "passed": len(episodes) == len(rows) and not failures,
    }
    _write(args.audit, audit)
    print(json.dumps(audit, sort_keys=True))
    if not audit["passed"]:
        raise SystemExit("TAU3_TAIL_CHUNK_FAILED")


if __name__ == "__main__":
    main()
