"""Run one chunk of a clean-only AgentDojo task manifest."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentdojo.agent_pipeline import AgentPipeline
from agentdojo.agent_pipeline.agent_pipeline import PipelineConfig
from agentdojo.benchmark import run_task_without_injection_tasks
from agentdojo.logging import OutputLogger
from agentdojo.task_suite.load_suites import get_suite

from wmagentattack.agentdojo_v2 import set_episode_seed, stable_episode_seed
from wmagentattack.qwen_agentdojo import TransformersQwenLLM


FRESH_CLEAN_SCOPE = "AgentDojo sandbox only; clean-task solvability screen"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def _summary(results: list[dict[str, Any]], expected: int) -> dict[str, Any]:
    completed = [row for row in results if row.get("status") == "completed"]
    return {
        "expected": expected,
        "recorded": len(results),
        "completed": len(completed),
        "failed": sum(row.get("status") == "failed" for row in results),
        "remaining": max(0, expected - len(completed)),
        "utility_success": sum(bool(row.get("utility")) for row in completed),
        "utility_rate": (
            sum(bool(row.get("utility")) for row in completed) / len(completed)
            if completed
            else 0.0
        ),
        "completed_by_suite": dict(Counter(row["suite"] for row in completed)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--benchmark-version", default="v1.2.2")
    parser.add_argument("--run-seed", type=int, required=True)
    parser.add_argument("--chunk-index", type=int, default=0)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-tool-output-chars", type=int, default=12_000)
    parser.add_argument("--max-input-tokens", type=int, default=8_192)
    parser.add_argument(
        "--prompt-profile",
        choices=["base", "format_only", "constraint_checklist", "robust"],
        default="base",
    )
    parser.add_argument(
        "--protocol",
        choices=[
            "function_tags",
            "function_tags_repair",
            "function_tags_repair_retry",
            "native",
        ],
        default="function_tags",
    )
    parser.add_argument("--quantization", choices=["bf16", "4bit"], default="4bit")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--agentdojo-local-alias", action="store_true")
    parser.add_argument(
        "--custom-task-module",
        help="Optional module that registers a frozen custom task panel before suite lookup.",
    )
    parser.add_argument("--logdir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force-rerun", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()
    if args.num_chunks < 1 or not 0 <= args.chunk_index < args.num_chunks:
        raise ValueError("Invalid chunk specification")
    if args.custom_task_module:
        importlib.import_module(args.custom_task_module)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("scope") != FRESH_CLEAN_SCOPE:
        raise ValueError("Unexpected clean-screen manifest scope")
    safety = manifest.get("safety_contract", {})
    if safety.get("allow_real_network_endpoints") is not False or safety.get(
        "clean_tasks_only"
    ) is not True:
        raise ValueError("Manifest does not enforce the clean sandbox boundary")
    frozen_custom_module = manifest.get("custom_task_module")
    if frozen_custom_module != args.custom_task_module:
        raise ValueError(
            "Runner custom-task module differs from the frozen manifest: "
            f"manifest={frozen_custom_module!r} runner={args.custom_task_module!r}"
        )
    rows = [
        row
        for index, row in enumerate(manifest["rows"])
        if index % args.num_chunks == args.chunk_index
    ]
    if not rows:
        raise RuntimeError("Chunk selected no tasks")

    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    args.logdir.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
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
        seed=args.run_seed,
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

    manifest_hash = _sha256(args.manifest)
    existing = []
    if args.output.exists() and not args.force_rerun:
        prior = json.loads(args.output.read_text(encoding="utf-8"))
        if prior.get("manifest_sha256") != manifest_hash:
            raise ValueError("Existing output used a different manifest")
        existing = list(prior.get("results", []))
    by_id = {str(row["row_id"]): row for row in existing}
    suites: dict[str, Any] = {}
    with OutputLogger(str(args.logdir)):
        for row in rows:
            row_id = str(row["row_id"])
            if not args.force_rerun and by_id.get(row_id, {}).get("status") == "completed":
                continue
            episode_seed = stable_episode_seed(args.run_seed, row_id)
            set_episode_seed(episode_seed)
            suite_name = str(row["suite"])
            suite = suites.setdefault(
                suite_name, get_suite(args.benchmark_version, suite_name)
            )
            task_id = str(row["user_task_id"])
            started = time.time()
            try:
                utility, security = run_task_without_injection_tasks(
                    suite,
                    pipeline,
                    suite.get_user_task_by_id(task_id),
                    args.logdir,
                    args.force_rerun,
                    args.benchmark_version,
                )
                raw_path = (
                    args.logdir
                    / str(pipeline.name)
                    / suite_name
                    / task_id
                    / "none"
                    / "none.json"
                )
                result = {
                    "row_id": row_id,
                    "status": "completed",
                    "suite": suite_name,
                    "user_task_id": task_id,
                    "episode_seed": episode_seed,
                    "utility": bool(utility),
                    "security": bool(security),
                    "raw_trace": str(raw_path.resolve()),
                    "raw_trace_exists": raw_path.exists(),
                    "elapsed_seconds": time.time() - started,
                }
                if not raw_path.exists():
                    raise FileNotFoundError(f"Missing expected trace: {raw_path}")
            except Exception as error:
                result = {
                    "row_id": row_id,
                    "status": "failed",
                    "suite": suite_name,
                    "user_task_id": task_id,
                    "episode_seed": episode_seed,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "elapsed_seconds": time.time() - started,
                }
            by_id[row_id] = result
            ordered = [
                by_id[str(item["row_id"])]
                for item in rows
                if str(item["row_id"]) in by_id
            ]
            payload = {
                "scope": FRESH_CLEAN_SCOPE,
                "manifest": str(args.manifest.resolve()),
                "manifest_sha256": manifest_hash,
                "pipeline_name": str(pipeline.name),
                "run_seed": args.run_seed,
                "chunk_index": args.chunk_index,
                "num_chunks": args.num_chunks,
                "custom_task_module": args.custom_task_module,
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
                "summary": _summary(ordered, len(rows)),
                "results": ordered,
            }
            _atomic_write(args.output, payload)
            if result["status"] == "failed" and not args.continue_on_error:
                raise RuntimeError(f"Clean task failed: {row_id}: {result['error']}")

    final = json.loads(args.output.read_text(encoding="utf-8"))
    if final["summary"]["remaining"] or final["summary"]["failed"]:
        raise RuntimeError(f"Incomplete clean-screen chunk: {final['summary']}")
    print(json.dumps(final["summary"], indent=2))


if __name__ == "__main__":
    main()
