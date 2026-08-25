"""Execute the frozen v34 clean/attack pairs in AgentDojo's local sandbox."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import random
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentdojo.agent_pipeline import AgentPipeline
from agentdojo.agent_pipeline.agent_pipeline import PipelineConfig
from agentdojo.benchmark import run_task_with_injection_tasks
from agentdojo.logging import OutputLogger
from agentdojo.task_suite.load_suites import get_suite

from wmagentattack.agentdojo_v2 import V2_SCOPE, build_manifest_attack, set_episode_seed
from wmagentattack.qwen_agentdojo import TransformersQwenLLM


def _atomic_write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _episode_seed(run_seed: int, row_id: str) -> int:
    value = int(run_seed)
    for character in row_id:
        value = (value * 131 + ord(character)) % (2**31 - 1)
    return value


def _summary(results: list[dict[str, Any]], expected: int) -> dict[str, Any]:
    completed = [row for row in results if row.get("status") == "completed"]
    failed = [row for row in results if row.get("status") == "failed"]
    return {
        "expected": expected,
        "recorded": len(results),
        "completed": len(completed),
        "failed": len(failed),
        "remaining": max(0, expected - len(completed)),
        "utility_success": sum(bool(row.get("utility")) for row in completed),
        "attack_success": sum(bool(row.get("security")) for row in completed),
        "completed_by_family": dict(Counter(row["attack_family"] for row in completed)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--run-seed", type=int, required=True)
    parser.add_argument("--benchmark-version", default="v1.2.2")
    parser.add_argument("--quantization", choices=["bf16", "4bit"], default="4bit")
    parser.add_argument("--max-input-tokens", type=int, default=8192)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-tool-output-chars", type=int, default=12000)
    parser.add_argument("--protocol", default="function_tags")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--logdir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("protocol_id") != "0825_paired_single_factor_attack_v34":
        raise ValueError("unexpected v34 manifest protocol")
    if manifest.get("scope") != V2_SCOPE:
        raise ValueError("v34 manifest is outside the AgentDojo sandbox scope")
    if manifest["safety_contract"]["allow_real_network_endpoints"] is not False:
        raise ValueError("real endpoints must remain disabled")
    rows = list(manifest["rows"])
    row_ids = [str(row["row_id"]) for row in rows]
    if len(rows) != 40 or len(set(row_ids)) != 40:
        raise ValueError("v34 requires exactly forty unique rows")

    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    args.logdir.mkdir(parents=True, exist_ok=True)
    llm = TransformersQwenLLM(
        args.model_path,
        max_new_tokens=args.max_new_tokens,
        device="cuda:0",
        quantization=args.quantization,
        max_tool_output_chars=args.max_tool_output_chars,
        prompt_profile="base",
        max_input_tokens=args.max_input_tokens,
        protocol=args.protocol,
        model_label=args.model_label,
        seed=args.run_seed,
        do_sample=True,
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
    pipeline.name = f"{pipeline.name}-local"

    existing: list[dict[str, Any]] = []
    if args.output.exists():
        prior = json.loads(args.output.read_text(encoding="utf-8"))
        if prior.get("protocol_id") != manifest["protocol_id"] or prior.get("row_ids") != row_ids:
            raise ValueError("existing output belongs to a different frozen row set")
        existing = list(prior.get("results", []))
    by_id = {str(row["row_id"]): row for row in existing}
    suites: dict[str, Any] = {}
    with OutputLogger(str(args.logdir)):
        for row in rows:
            row_id = str(row["row_id"])
            if by_id.get(row_id, {}).get("status") == "completed":
                continue
            episode_seed = _episode_seed(args.run_seed, row_id)
            set_episode_seed(episode_seed)
            suite_name = str(row["suite"])
            suite = suites.setdefault(
                suite_name, get_suite(args.benchmark_version, suite_name)
            )
            attack = build_manifest_attack(row, suite, pipeline)
            started = time.time()
            try:
                utility, security = run_task_with_injection_tasks(
                    suite,
                    pipeline,
                    suite.get_user_task_by_id(str(row["user_task_id"])),
                    attack,
                    args.logdir,
                    False,
                    [str(row["injection_task_id"])],
                    args.benchmark_version,
                )
                key = (str(row["user_task_id"]), str(row["injection_task_id"]))
                raw_path = (
                    args.logdir
                    / str(pipeline.name)
                    / suite_name
                    / str(row["user_task_id"])
                    / str(attack.name)
                    / f"{row['injection_task_id']}.json"
                )
                if not raw_path.exists():
                    raise FileNotFoundError(f"missing AgentDojo raw trace: {raw_path}")
                result = {
                    "row_id": row_id,
                    "run_seed": args.run_seed,
                    "episode_seed": episode_seed,
                    "status": "completed",
                    "suite": suite_name,
                    "user_task_id": row["user_task_id"],
                    "injection_task_id": row["injection_task_id"],
                    "attack_family": row["attack_family"],
                    "attack_variant": row["attack_variant"],
                    "utility": bool(utility[key]),
                    "security": bool(security[key]),
                    "raw_trace": str(raw_path.resolve()),
                    "raw_trace_exists": True,
                    "elapsed_seconds": time.time() - started,
                }
            except Exception as error:
                result = {
                    "row_id": row_id,
                    "run_seed": args.run_seed,
                    "episode_seed": episode_seed,
                    "status": "failed",
                    "suite": suite_name,
                    "user_task_id": row["user_task_id"],
                    "injection_task_id": row["injection_task_id"],
                    "attack_family": row["attack_family"],
                    "attack_variant": row["attack_variant"],
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "elapsed_seconds": time.time() - started,
                }
            by_id[row_id] = result
            ordered = [by_id[item] for item in row_ids if item in by_id]
            payload = {
                "protocol_id": manifest["protocol_id"],
                "manifest": str(args.manifest.resolve()),
                "row_ids": row_ids,
                "model_label": args.model_label,
                "run_seed": args.run_seed,
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
                "summary": _summary(ordered, len(rows)),
                "results": ordered,
            }
            _atomic_write(args.output, payload)
            print(json.dumps({"row_id": row_id, "status": result["status"], **payload["summary"]}), flush=True)
            if result["status"] == "failed":
                raise RuntimeError(f"v34 episode failed: {result}")

    final = json.loads(args.output.read_text(encoding="utf-8"))
    if final["summary"]["completed"] != 40 or final["summary"]["failed"]:
        raise RuntimeError(f"incomplete v34 seed: {final['summary']}")


if __name__ == "__main__":
    main()
