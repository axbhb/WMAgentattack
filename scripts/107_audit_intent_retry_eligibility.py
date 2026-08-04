"""Audit residual clean failures eligible for parser-v2 or one-shot retry."""

from __future__ import annotations

import argparse
import json
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
    chunks: int = 2,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
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
                assistant_messages = [
                    message
                    for message in trace.get("messages", [])
                    if message.get("role") == "assistant"
                ]
                calls = sum(
                    len(message.get("tool_calls") or [])
                    for message in assistant_messages
                )
                if result["utility"] or calls:
                    continue
                completion = "\n".join(_content(message) for message in assistant_messages)
                parsed = TransformersQwenLLM._parse_repaired_completion(completion)
                parse_calls = parsed["tool_calls"]
                first_turn_only = (
                    len(assistant_messages) == 1
                    and not any(
                        message.get("role") == "tool"
                        for message in trace.get("messages", [])
                    )
                )
                retry_eligible = (
                    not parse_calls
                    and first_turn_only
                    and TransformersQwenLLM._should_retry_tool_intent(completion)
                )
                rows.append(
                    {
                        "seed": seed,
                        "suite": suite,
                        "user_task_id": result["user_task_id"],
                        "parser_v2_syntax_repairable": bool(parse_calls),
                        "parsed_function": (
                            parse_calls[0].function if parse_calls else None
                        ),
                        "first_assistant_turn_only": first_turn_only,
                        "one_shot_retry_eligible": retry_eligible,
                        "text_prefix": completion[:500],
                        "raw_trace": result["raw_trace"],
                    }
                )

    parser_v2 = sum(row["parser_v2_syntax_repairable"] for row in rows)
    retry = sum(row["one_shot_retry_eligible"] for row in rows)
    return {
        "scope": "clean-only residual parser-v2 and intent-retry eligibility audit",
        "attack_outcomes_read": False,
        "source_protocol": "function_tags_repair parser-v1 failures",
        "claim_boundary": (
            "This is a syntax and explicit-intent compute gate, not a blind "
            "confirmatory analysis. Live parsing additionally validates each "
            "function name against the active AgentDojo runtime."
        ),
        "counts": {
            "zero_tool_utility_failures": len(rows),
            "parser_v2_syntax_repairable": parser_v2,
            "one_shot_retry_eligible_after_parser_v2": retry,
            "combined_candidate_episodes": parser_v2 + retry,
            "residual_ineligible": len(rows) - parser_v2 - retry,
        },
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--suite", default="travel")
    parser.add_argument("--seeds", type=int, nargs="+", default=[101, 103, 107])
    parser.add_argument("--chunks", type=int, default=2)
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
    print(json.dumps(result["counts"], indent=2))


if __name__ == "__main__":
    main()
