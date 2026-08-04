"""Execute one chunk of the AgentDojo-v2 manifest with a local HF model.

Only AgentDojo's synthetic, in-memory environments are used.  Payload strings
are never interpreted as network commands by this runner.
"""

from __future__ import annotations

import argparse
import hashlib
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
from agentdojo.benchmark import run_task_with_injection_tasks
from agentdojo.logging import OutputLogger
from agentdojo.task_suite.load_suites import get_suite

from wmagentattack.agentdojo_v2 import (
    V2_SCOPE,
    build_manifest_attack,
    set_episode_seed,
    stable_episode_seed,
)
from wmagentattack.qwen_agentdojo import TransformersQwenLLM


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    temporary.replace(path)


def _summarize(results: list[dict[str, Any]], expected: int) -> dict[str, Any]:
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
        "utility_rate": (
            sum(bool(row.get("utility")) for row in completed) / len(completed)
            if completed
            else 0.0
        ),
        "targeted_asr": (
            sum(bool(row.get("security")) for row in completed) / len(completed)
            if completed
            else 0.0
        ),
        "completed_by_family": dict(
            Counter(str(row["attack_family"]) for row in completed)
        ),
    }


def _result_payload(
    *,
    args: argparse.Namespace,
    manifest_path: Path,
    manifest_sha256: str,
    pipeline_name: str,
    selected_rows: list[dict[str, Any]],
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "scope": V2_SCOPE,
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": manifest_sha256,
        "dataset_version": args.dataset_version,
        "model_path": str(args.model_path.resolve()),
        "model_label": args.model_label,
        "pipeline_name": pipeline_name,
        "benchmark_version": args.benchmark_version,
        "run_seed": args.run_seed,
        "do_sample": args.do_sample,
        "temperature": args.temperature if args.do_sample else None,
        "top_p": args.top_p if args.do_sample else None,
        "chunk_index": args.chunk_index,
        "num_chunks": args.num_chunks,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "summary": _summarize(results, len(selected_rows)),
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-label", default="hf-local-agentdojo-v2")
    parser.add_argument("--dataset-version", default="v2.0-screen")
    parser.add_argument("--benchmark-version", default="v1.2.2")
    parser.add_argument("--run-seed", type=int, default=7)
    parser.add_argument("--chunk-index", type=int, default=0)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--include-split", action="append", dest="include_splits")
    parser.add_argument("--max-episodes", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-tool-output-chars", type=int, default=12_000)
    parser.add_argument("--prompt-profile", choices=["base", "format_only", "robust"], default="base")
    parser.add_argument("--max-input-tokens", type=int, default=8_192)
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
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--agentdojo-local-alias", action="store_true")
    parser.add_argument("--logdir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force-rerun", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()

    if args.num_chunks < 1:
        raise ValueError("num_chunks must be positive")
    if not 0 <= args.chunk_index < args.num_chunks:
        raise ValueError("chunk_index must satisfy 0 <= index < num_chunks")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("scope") != V2_SCOPE:
        raise ValueError("Manifest is missing the AgentDojo-v2 sandbox scope")
    if manifest.get("safety_contract", {}).get("allow_real_network_endpoints") is not False:
        raise ValueError("Manifest safety contract does not forbid real network endpoints")
    rows = list(manifest["rows"])
    if args.include_splits:
        allowed = set(args.include_splits)
        rows = [row for row in rows if row["task_split"] in allowed]
    selected_rows = [
        row for index, row in enumerate(rows) if index % args.num_chunks == args.chunk_index
    ]
    if args.max_episodes > 0:
        selected_rows = selected_rows[: args.max_episodes]
    if not selected_rows:
        raise RuntimeError("No manifest rows selected for this chunk")

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
        trust_remote_code=args.trust_remote_code,
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
    existing: list[dict[str, Any]] = []
    if args.output.exists() and not args.force_rerun:
        prior = json.loads(args.output.read_text(encoding="utf-8"))
        if prior.get("manifest_sha256") != manifest_hash:
            raise ValueError("Existing output was produced from a different manifest")
        existing = list(prior.get("results", []))
    by_id = {str(row["row_id"]): row for row in existing}

    suites: dict[str, Any] = {}
    with OutputLogger(str(args.logdir)):
        for row in selected_rows:
            row_id = str(row["row_id"])
            if not args.force_rerun and by_id.get(row_id, {}).get("status") == "completed":
                continue
            episode_seed = stable_episode_seed(args.run_seed, row_id)
            set_episode_seed(episode_seed)
            suite_name = str(row["suite"])
            suite = suites.setdefault(
                suite_name, get_suite(args.benchmark_version, suite_name)
            )
            attack = build_manifest_attack(row, suite, pipeline)
            started = time.time()
            try:
                utility_results, security_results = run_task_with_injection_tasks(
                    suite,
                    pipeline,
                    suite.get_user_task_by_id(str(row["user_task_id"])),
                    attack,
                    args.logdir,
                    args.force_rerun,
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
                result = {
                    "row_id": row_id,
                    "status": "completed",
                    "suite": suite_name,
                    "task_split": row["task_split"],
                    "user_task_id": row["user_task_id"],
                    "injection_task_id": row["injection_task_id"],
                    "attack_name": attack.name,
                    "attack_family": row["attack_family"],
                    "attack_variant": row["attack_variant"],
                    "episode_seed": episode_seed,
                    "utility": bool(utility_results[key]),
                    "security": bool(security_results[key]),
                    "raw_trace": str(raw_path.resolve()),
                    "raw_trace_exists": raw_path.exists(),
                    "elapsed_seconds": time.time() - started,
                }
                if not raw_path.exists():
                    raise FileNotFoundError(f"AgentDojo did not write expected trace: {raw_path}")
            except Exception as error:
                result = {
                    "row_id": row_id,
                    "status": "failed",
                    "suite": suite_name,
                    "task_split": row["task_split"],
                    "user_task_id": row["user_task_id"],
                    "injection_task_id": row["injection_task_id"],
                    "attack_name": row["attack_name"],
                    "attack_family": row["attack_family"],
                    "attack_variant": row["attack_variant"],
                    "episode_seed": episode_seed,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "elapsed_seconds": time.time() - started,
                }
            by_id[row_id] = result
            ordered_results = [
                by_id[str(item["row_id"])]
                for item in selected_rows
                if str(item["row_id"]) in by_id
            ]
            payload = _result_payload(
                args=args,
                manifest_path=args.manifest,
                manifest_sha256=manifest_hash,
                pipeline_name=str(pipeline.name),
                selected_rows=selected_rows,
                results=ordered_results,
            )
            _atomic_write(args.output, payload)
            print(
                json.dumps(
                    {
                        "row_id": row_id,
                        "status": result["status"],
                        "completed": payload["summary"]["completed"],
                        "expected": payload["summary"]["expected"],
                    }
                ),
                flush=True,
            )
            if result["status"] == "failed" and not args.continue_on_error:
                raise RuntimeError(f"v2 episode failed: {result}")

    final_results = [
        by_id[str(item["row_id"])]
        for item in selected_rows
        if str(item["row_id"]) in by_id
    ]
    final = _result_payload(
        args=args,
        manifest_path=args.manifest,
        manifest_sha256=manifest_hash,
        pipeline_name=str(pipeline.name),
        selected_rows=selected_rows,
        results=final_results,
    )
    _atomic_write(args.output, final)
    print(json.dumps(final["summary"], indent=2))


if __name__ == "__main__":
    main()
