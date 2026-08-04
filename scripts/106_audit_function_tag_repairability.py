"""Audit deterministic parser repairability of clean zero-tool failures."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from wmagentattack.qwen_agentdojo import TransformersQwenLLM


def _content(message: dict[str, Any]) -> str:
    return " ".join(
        block.get("content", "")
        for block in message.get("content", [])
        if isinstance(block, dict)
    ).strip()


def audit(
    archive: Path,
    *,
    suite: str,
    seeds: tuple[int, ...] = (101, 103, 107),
    chunks: int = 4,
) -> dict[str, Any]:
    rows = []
    for seed in seeds:
        for chunk in range(chunks):
            payload = json.loads(
                (archive / f"seed{seed}" / f"chunk{chunk}.json").read_text(
                    encoding="utf-8"
                )
            )
            for result in payload["results"]:
                if result["suite"] != suite or result.get("status") != "completed":
                    continue
                trace = json.loads(Path(result["raw_trace"]).read_text(encoding="utf-8"))
                calls = sum(
                    len(message.get("tool_calls") or [])
                    for message in trace.get("messages", [])
                    if message.get("role") == "assistant"
                )
                if result["utility"] or calls:
                    continue
                assistant_text = "\n".join(
                    _content(message)
                    for message in trace.get("messages", [])
                    if message.get("role") == "assistant"
                ).strip()
                repaired = TransformersQwenLLM._parse_repaired_completion(
                    assistant_text
                )
                repair_calls = repaired["tool_calls"]
                rows.append(
                    {
                        "seed": seed,
                        "suite": suite,
                        "user_task_id": result["user_task_id"],
                        "messages": len(trace.get("messages", [])),
                        "repairable": bool(repair_calls),
                        "repaired_function": (
                            repair_calls[0].function if repair_calls else None
                        ),
                        "text_prefix": assistant_text[:500],
                        "raw_trace": result["raw_trace"],
                    }
                )
    functions = Counter(
        row["repaired_function"] for row in rows if row["repairable"]
    )
    return {
        "scope": "clean-only deterministic function-tag repairability audit",
        "attack_outcomes_read": False,
        "runtime_validation_note": "The offline audit recognizes syntax only; the live candidate additionally rejects names outside the active FunctionsRuntime.",
        "counts": {
            "zero_tool_utility_failures": len(rows),
            "repairable_unambiguous_calls": sum(row["repairable"] for row in rows),
            "unrepairable": sum(not row["repairable"] for row in rows),
            "first_assistant_turn_only": sum(row["messages"] == 3 for row in rows),
        },
        "repaired_function_histogram": dict(sorted(functions.items())),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--suite", default="travel")
    parser.add_argument("--seeds", type=int, nargs="+", default=[101, 103, 107])
    parser.add_argument("--chunks", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(
        args.archive,
        suite=args.suite,
        seeds=tuple(args.seeds),
        chunks=args.chunks,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"counts": result["counts"], "repaired_function_histogram": result["repaired_function_histogram"]}, indent=2))


if __name__ == "__main__":
    main()
