"""One GPU/model load, sixty frozen clean episodes; no attack execution."""
import argparse
import ctypes
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wmagentattack.clean_pairing import build_clean_plan, clean_gate, block_python_network


def write(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    args = parser.parse_args()
    p = json.loads(args.protocol.read_text())
    if p["status"] != "preregistered_before_v37_outcomes":
        raise ValueError("protocol not frozen")
    archive = Path(p["archive"])
    archive.mkdir(parents=True, exist_ok=True)
    # Exclusive attempt marker: neither a scheduler requeue nor an uncertain
    # connection may silently rerun completed/partial outcomes.
    with (archive / "execution.lock").open("x") as handle:
        handle.write(os.environ.get("SLURM_JOB_ID", "NO_SLURM"))
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("This experiment must run in remote Slurm")
    source = json.loads(Path(p["source_manifest"]).read_text())
    plan = build_clean_plan(p, source["rows"])
    write(archive / "run_plan.json", plan)
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    import numpy as np
    import torch
    assigned, visible = os.environ.get("SLURM_JOB_GPUS"), os.environ.get("CUDA_VISIBLE_DEVICES")
    cuinit = int(ctypes.CDLL("libcuda.so.1").cuInit(0))
    smi = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
    preflight = {"assigned_gpu": assigned, "visible_gpu": visible, "cuInit": cuinit,
                 "torch_cuda": torch.cuda.is_available(), "device_count": torch.cuda.device_count(),
                 "nvidia_smi_exit": smi.returncode, "nvidia_smi": smi.stdout}
    preflight["passed"] = bool(assigned and assigned == visible and cuinit == 0 and preflight["torch_cuda"] and preflight["device_count"] == 1 and smi.returncode == 0)
    write(archive / "cuda_preflight.json", preflight)
    if not preflight["passed"]:
        raise RuntimeError("CUDA preflight failed; no model or episode executed")
    from agentdojo.agent_pipeline import AgentPipeline
    from agentdojo.agent_pipeline.agent_pipeline import PipelineConfig
    from agentdojo.benchmark import run_task_without_injection_tasks
    from agentdojo.logging import OutputLogger
    from agentdojo.task_suite.load_suites import get_suite
    from wmagentattack.qwen_agentdojo import TransformersQwenLLM
    model = p["model"]
    results = []
    with block_python_network() as network:
        llm = TransformersQwenLLM(Path(model["path"]), device="cuda:0", quantization=model["quantization"],
                                  model_label="meta-llama-3.1-70b-v37", seed=p["run_seeds"][0],
                                  max_new_tokens=model["max_new_tokens"], max_input_tokens=model["max_input_tokens"],
                                  max_tool_output_chars=model["max_tool_output_chars"], prompt_profile=model["prompt_profile"],
                                  protocol=model["protocol"], do_sample=model["do_sample"],
                                  temperature=model["temperature"], top_p=model["top_p"])
        pipeline = AgentPipeline.from_config(PipelineConfig(llm=llm, model_id=None, defense=None,
                    system_message_name=None, system_message=None, tool_output_format=None))
        pipeline.name = f"{pipeline.name}-local"
        for episode in plan:
            task_name = episode["task"]
            suite_name, task_id = task_name.split("|", 1)
            seed = episode["episode_seed"]
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            llm.seed = seed
            logdir = archive / "raw" / f"seed{episode['run_seed']}"
            logdir.mkdir(parents=True, exist_ok=True)
            suite = get_suite("v1.2.2", suite_name)
            started = time.monotonic()
            try:
                with OutputLogger(str(logdir)):
                    utility, _ = run_task_without_injection_tasks(suite, pipeline, suite.get_user_task_by_id(task_id), logdir, False, "v1.2.2")
                raw = logdir / str(pipeline.name) / suite_name / task_id / "none" / "none.json"
                if not raw.is_file():
                    raise FileNotFoundError(f"Missing raw trace {raw}")
                json.loads(raw.read_text())
                if network["blocked_attempts"]:
                    raise RuntimeError("Unexpected attempted network access")
                result = {**episode, "status": "completed", "utility": bool(utility), "raw_trace": str(raw),
                          "blocked_network_attempts": network["blocked_attempts"], "elapsed_seconds": time.monotonic() - started}
            except Exception as error:
                result = {**episode, "status": "failed", "error_type": type(error).__name__, "error": str(error),
                          "blocked_network_attempts": network["blocked_attempts"]}
                results.append(result)
                write(archive / "results.json", results)
                write(archive / "gate.json", {"decision": "INVALID_CLEAN_PAIRING_V37", "scientific_result": False, "recorded": len(results), "failure": result})
                raise
            results.append(result)
            write(archive / "results.json", results)
            print(json.dumps({"recorded": len(results), "expected": 60, **result}), flush=True)
    def raw_valid(row):
        raw = Path(row.get("raw_trace", ""))
        return raw.is_relative_to(archive / "raw") and raw.is_file() and bool(json.loads(raw.read_text()))
    gate = clean_gate(results, p, raw_valid)
    gate["protocol_id"] = p["protocol_id"]
    gate["slurm_job_id"] = os.environ.get("SLURM_JOB_ID")
    gate["model_contract"] = model
    write(archive / "gate.json", gate)
    if not gate["scientific_result"]:
        raise RuntimeError("Final data integrity gate failed")
    print(json.dumps(gate, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
