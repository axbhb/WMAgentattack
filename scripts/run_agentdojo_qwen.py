"""Run clean AgentDojo tasks with one local Qwen2.5-7B model load."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from agentdojo.agent_pipeline import AgentPipeline
from agentdojo.agent_pipeline.agent_pipeline import PipelineConfig
from agentdojo.benchmark import benchmark_suite_without_injections
from agentdojo.logging import OutputLogger
from agentdojo.task_suite.load_suites import get_suite

from wmagentattack.qwen_agentdojo import TransformersQwenLLM

DEFAULT_MODEL = Path(
    r"F:\hf_cache\models--Qwen--Qwen2.5-7B-Instruct"
    r"\snapshots\a09a35458c702b33eeacc393d103063234e8bc28"
)
DEFAULT_BASELINE_TASKS = [
    "user_task_0",
    "user_task_1",
    "user_task_2",
    "user_task_5",
    "user_task_10",
    "user_task_24",
    "user_task_26",
    "user_task_27",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--benchmark-version", default="v1.2.2")
    parser.add_argument("--suite", default="workspace")
    parser.add_argument(
        "--user-task",
        action="append",
        dest="user_tasks",
        help="Task ID to run. Repeat the option for multiple tasks.",
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Run the predefined small clean baseline task set.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument(
        "--quantization",
        choices=["bf16", "4bit"],
        default="4bit",
        help="Model weight precision. 4bit leaves more VRAM for long AgentDojo contexts.",
    )
    parser.add_argument(
        "--max-tool-output-chars",
        type=int,
        default=12_000,
        help="Compact oversized tool outputs only in model context; 0 keeps full output.",
    )
    parser.add_argument(
        "--prompt-profile",
        choices=["base", "robust"],
        default="robust",
        help="Use the unmodified AgentDojo local prompt or a stricter tool-use profile.",
    )
    parser.add_argument("--max-input-tokens", type=int, default=8_192)
    parser.add_argument(
        "--protocol",
        choices=["function_tags", "native"],
        default="native",
    )
    parser.add_argument("--logdir", type=Path, default=PROJECT_ROOT / "runs")
    parser.add_argument(
        "--load-only",
        action="store_true",
        help="Load the model and exit without running an AgentDojo task.",
    )
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help="Re-run tasks even when a trace already exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    llm = TransformersQwenLLM(
        args.model_path,
        max_new_tokens=args.max_new_tokens,
        quantization=args.quantization,
        max_tool_output_chars=args.max_tool_output_chars,
        prompt_profile=args.prompt_profile,
        max_input_tokens=args.max_input_tokens,
        protocol=args.protocol,
    )
    if args.load_only:
        print(
            json.dumps(
                {
                    "status": "model_loaded",
                    "model_path": str(args.model_path),
                    "pipeline_name": llm.name,
                },
                indent=2,
            )
        )
        return

    if args.baseline and args.user_tasks:
        raise ValueError("Use either --baseline or explicit --user-task options, not both")
    user_tasks = DEFAULT_BASELINE_TASKS if args.baseline else (args.user_tasks or ["user_task_0"])

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
    suite = get_suite(args.benchmark_version, args.suite)
    with OutputLogger(str(args.logdir)):
        results = benchmark_suite_without_injections(
            pipeline,
            suite,
            user_tasks=user_tasks,
            logdir=args.logdir,
            force_rerun=args.force_rerun,
            benchmark_version=args.benchmark_version,
        )
    print(
        json.dumps(
            {
                "suite": args.suite,
                "user_tasks": user_tasks,
                "utility_results": {
                    "|".join(key): value for key, value in results["utility_results"].items()
                },
                "logdir": str(args.logdir.resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
