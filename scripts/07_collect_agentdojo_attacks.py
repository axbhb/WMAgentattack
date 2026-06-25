"""Collect a small batch of real AgentDojo sandbox attack trajectories."""

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

DEFAULT_MODEL = Path(
    r"F:\hf_cache\models--Qwen--Qwen2.5-7B-Instruct"
    r"\snapshots\a09a35458c702b33eeacc393d103063234e8bc28"
)
DEFAULT_USER_TASKS = ["user_task_1", "user_task_24"]
DEFAULT_INJECTION_TASKS = ["injection_task_0", "injection_task_1"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--benchmark-version", default="v1.2.2")
    parser.add_argument("--suite", default="workspace")
    parser.add_argument(
        "--attack",
        action="append",
        dest="attacks",
        help="AgentDojo attack name. Repeat to collect multiple attacks.",
    )
    parser.add_argument("--user-task", action="append", dest="user_tasks")
    parser.add_argument(
        "--injection-task", action="append", dest="injection_tasks"
    )
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-tool-output-chars", type=int, default=12_000)
    parser.add_argument(
        "--prompt-profile", choices=["base", "robust"], default="robust"
    )
    parser.add_argument("--max-input-tokens", type=int, default=8_192)
    parser.add_argument(
        "--protocol",
        choices=["function_tags", "native"],
        default="native",
    )
    parser.add_argument(
        "--logdir", type=Path, default=ROOT / "runs" / "agentdojo_attacks"
    )
    parser.add_argument("--force-rerun", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    user_tasks = args.user_tasks or DEFAULT_USER_TASKS
    injection_tasks = args.injection_tasks or DEFAULT_INJECTION_TASKS

    llm = TransformersQwenLLM(
        args.model_path,
        max_new_tokens=args.max_new_tokens,
        quantization="4bit",
        max_tool_output_chars=args.max_tool_output_chars,
        prompt_profile=args.prompt_profile,
        max_input_tokens=args.max_input_tokens,
        protocol=args.protocol,
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
    # AgentDojo's name-aware attacks recognize the built-in "local" model
    # alias. Keep the concrete Qwen configuration in the prefix for provenance.
    pipeline.name = f"{pipeline.name}-local"
    suite = get_suite(args.benchmark_version, args.suite)
    attacks = args.attacks or ["direct"]
    attack_results = {}
    with OutputLogger(str(args.logdir)):
        for attack_name in attacks:
            attack = load_attack(attack_name, suite, pipeline)
            results = benchmark_suite_with_injections(
                pipeline,
                suite,
                attack,
                logdir=args.logdir,
                force_rerun=args.force_rerun,
                user_tasks=user_tasks,
                injection_tasks=injection_tasks,
                benchmark_version=args.benchmark_version,
            )
            attack_results[attack_name] = {
                "utility_results": {
                    "|".join(key): value
                    for key, value in results["utility_results"].items()
                },
                "security_results": {
                    "|".join(key): value
                    for key, value in results["security_results"].items()
                },
                "injection_task_utility": results[
                    "injection_tasks_utility_results"
                ],
            }

    payload = {
        "scope": "AgentDojo sandbox only",
        "prompt_profile": args.prompt_profile,
        "protocol": args.protocol,
        "attacks": attacks,
        "user_tasks": user_tasks,
        "injection_tasks": injection_tasks,
        "results": attack_results,
        "logdir": str(args.logdir.resolve()),
    }
    args.logdir.mkdir(parents=True, exist_ok=True)
    summary_name = "batch_summary_" + "_".join(attacks) + ".json"
    (args.logdir / summary_name).write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
