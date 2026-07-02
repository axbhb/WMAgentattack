"""Run full AgentDojo clean/attack batches with a local HF chat model.

This is intended for server-side open-model experiments. It loads one local
Transformers model and then runs all selected suites/attacks through AgentDojo's
official environments and evaluators.
"""

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
from agentdojo.benchmark import (
    benchmark_suite_with_injections,
    benchmark_suite_without_injections,
)
from agentdojo.logging import OutputLogger
from agentdojo.task_suite.load_suites import get_suite, get_suites

from wmagentattack.qwen_agentdojo import TransformersQwenLLM


def _summarize_results(results: dict) -> dict:
    utility = list(results["utility_results"].values())
    payload = {
        "utility_count": len(utility),
        "utility_success": sum(utility),
        "utility_rate": sum(utility) / len(utility) if utility else 0.0,
    }
    if "security_results" in results:
        security = list(results["security_results"].values())
        payload.update(
            {
                "security_count": len(security),
                "security_success": sum(security),
                "targeted_asr": sum(security) / len(security)
                if security
                else 0.0,
                "injection_task_utility": results[
                    "injection_tasks_utility_results"
                ],
            }
        )
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-label", default="hf-local")
    parser.add_argument("--benchmark-version", default="v1.2.2")
    parser.add_argument("--suite", action="append", dest="suites")
    parser.add_argument(
        "--attack",
        action="append",
        dest="attacks",
        help="Repeat to run multiple attacks. Omit with --clean-only.",
    )
    parser.add_argument("--clean-only", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-tool-output-chars", type=int, default=12_000)
    parser.add_argument("--prompt-profile", choices=["base", "robust"], default="base")
    parser.add_argument("--max-input-tokens", type=int, default=8_192)
    parser.add_argument(
        "--protocol",
        choices=["function_tags", "native"],
        default="function_tags",
        help="Use function_tags for Llama-style AgentDojo local prompt parity.",
    )
    parser.add_argument("--quantization", choices=["bf16", "4bit"], default="4bit")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument(
        "--agentdojo-local-alias",
        action="store_true",
        help="Append '-local' so name-aware AgentDojo attacks use the local alias.",
    )
    parser.add_argument(
        "--logdir", type=Path, default=ROOT / "runs" / "agentdojo_full_hf"
    )
    parser.add_argument("--force-rerun", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    suites = args.suites or list(get_suites(args.benchmark_version).keys())
    attacks = [] if args.clean_only else (
        args.attacks or ["important_instructions_no_model_name"]
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
        seed=args.seed,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
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

    summary: dict = {
        "scope": "AgentDojo full-suite HF local model",
        "model_path": str(args.model_path),
        "model_label": args.model_label,
        "pipeline_name": pipeline.name,
        "benchmark_version": args.benchmark_version,
        "suites": suites,
        "attacks": attacks,
        "seed": args.seed,
        "do_sample": args.do_sample,
        "temperature": args.temperature if args.do_sample else None,
        "top_p": args.top_p if args.do_sample else None,
        "clean": {},
        "attack": {},
    }

    with OutputLogger(str(args.logdir)):
        for suite_name in suites:
            suite = get_suite(args.benchmark_version, suite_name)
            clean_results = benchmark_suite_without_injections(
                pipeline,
                suite,
                logdir=args.logdir,
                force_rerun=args.force_rerun,
                benchmark_version=args.benchmark_version,
            )
            summary["clean"][suite_name] = _summarize_results(clean_results)

            for attack_name in attacks:
                attack = load_attack(attack_name, suite, pipeline)
                attack_results = benchmark_suite_with_injections(
                    pipeline,
                    suite,
                    attack,
                    logdir=args.logdir,
                    force_rerun=args.force_rerun,
                    benchmark_version=args.benchmark_version,
                )
                summary["attack"].setdefault(attack_name, {})[suite_name] = (
                    _summarize_results(attack_results)
                )

    args.logdir.mkdir(parents=True, exist_ok=True)
    summary_path = args.logdir / f"summary_{args.model_label}.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
