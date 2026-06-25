"""Replay selected AgentDojo task/injection pairs with a local HF model."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentdojo.agent_pipeline import AgentPipeline
from agentdojo.agent_pipeline.agent_pipeline import PipelineConfig
from agentdojo.attacks.attack_registry import load_attack
from agentdojo.benchmark import benchmark_suite_with_injections
from agentdojo.logging import OutputLogger
from agentdojo.task_suite.load_suites import get_suite

from wmagentattack.qwen_agentdojo import TransformersQwenLLM


def _pair_result(results: dict, user_task_id: str, injection_task_id: str) -> dict:
    key = (user_task_id, injection_task_id)
    utility = bool(results["utility_results"].get(key, False))
    security = bool(results["security_results"].get(key, False))
    return {
        "user_task_id": user_task_id,
        "injection_task_id": injection_task_id,
        "utility": utility,
        "security": security,
    }


def _aggregate(rows: list[dict]) -> dict:
    count = len(rows)
    utility_success = sum(row["utility"] for row in rows)
    security_success = sum(row["security"] for row in rows)
    return {
        "count": count,
        "utility_success": utility_success,
        "utility_rate": utility_success / count if count else 0.0,
        "security_success": security_success,
        "targeted_asr": security_success / count if count else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--selection-name", action="append", dest="selection_names")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-label", default="hf-local")
    parser.add_argument("--benchmark-version", default="v1.2.2")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-tool-output-chars", type=int, default=12_000)
    parser.add_argument("--prompt-profile", choices=["base", "robust"], default="base")
    parser.add_argument("--max-input-tokens", type=int, default=8_192)
    parser.add_argument(
        "--protocol", choices=["function_tags", "native"], default="function_tags"
    )
    parser.add_argument("--quantization", choices=["bf16", "4bit"], default="4bit")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--agentdojo-local-alias", action="store_true")
    parser.add_argument("--logdir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force-rerun", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    selection_payload = json.loads(args.selection.read_text(encoding="utf-8"))
    selection_names = args.selection_names or list(
        selection_payload["selections"].keys()
    )

    llm = TransformersQwenLLM(
        args.model_path,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
        quantization=args.quantization,
        max_tool_output_chars=args.max_tool_output_chars,
        prompt_profile=args.prompt_profile,
        max_input_tokens=args.max_input_tokens,
        protocol=args.protocol,
        model_label=args.model_label,
        trust_remote_code=args.trust_remote_code,
    )
    pipeline = AgentPipeline.from_config(
        PipelineConfig(
            llm=llm,
            model_id=None,
            defense=None,
            system_message_name=None,
            system_message=None,
            tool_output_format=None,
        )
    )
    if args.agentdojo_local_alias:
        pipeline.name = f"{pipeline.name}-local"

    output = {
        "scope": "selected_real_agentdojo_replay",
        "selection": str(args.selection.resolve()),
        "model_path": str(args.model_path),
        "model_label": args.model_label,
        "pipeline_name": pipeline.name,
        "benchmark_version": args.benchmark_version,
        "results": {},
    }

    with OutputLogger(str(args.logdir)):
        for selection_name in selection_names:
            rows = selection_payload["selections"][selection_name]
            replayed = []
            for row in rows:
                suite = get_suite(args.benchmark_version, row["suite"])
                attack = load_attack(row["attack"], suite, pipeline)
                results = benchmark_suite_with_injections(
                    pipeline,
                    suite,
                    attack,
                    logdir=args.logdir / selection_name,
                    force_rerun=args.force_rerun,
                    user_tasks=[row["user_task_id"]],
                    injection_tasks=[row["injection_task_id"]],
                    verbose=False,
                    benchmark_version=args.benchmark_version,
                )
                replayed.append(
                    {
                        **row,
                        **_pair_result(
                            results,
                            row["user_task_id"],
                            row["injection_task_id"],
                        ),
                    }
                )
            output["results"][selection_name] = {
                "aggregate": _aggregate(replayed),
                "rows": replayed,
            }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
