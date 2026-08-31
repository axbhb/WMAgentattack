"""Frozen same-seed clean comparisons; one GPU load and no attack execution."""
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
from wmagentattack.clean_pairing import block_python_network
from wmagentattack.protocol_recovery_eval import make_plan, evaluate

PROTOCOLS = {"strict": "function_tags", "syntax": "function_tags_repair", "syntax_retry": "function_tags_repair_retry"}


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--stage", choices=["v39", "v40"], required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Must run in remote Slurm")
    protocol = json.loads(args.protocol.read_text())
    parent = json.loads(Path(protocol["parent_protocol"]).read_text())
    stage = protocol[args.stage]
    archive = Path(stage["archive"])
    archive.mkdir(parents=True, exist_ok=True)
    if json.loads((Path(protocol["v38"]["archive"]) / "gate.json").read_text())["decision"] != "GO_PARSER_VIABILITY_V38":
        raise RuntimeError("Parser viability gate has not passed")
    prior = None
    if args.stage == "v40":
        prior = json.loads((Path(protocol["v39"]["archive"]) / "gate.json").read_text())
        if prior["decision"] != "GO_PROTOCOL_RECOVERY" or not prior["selected_arm"]:
            raise RuntimeError("No passing v39 candidate")
        arms = ["strict", prior["selected_arm"]]
    else:
        arms = stage["arms"]
    with (archive / "execution.lock").open("x") as handle:
        handle.write(os.environ["SLURM_JOB_ID"])
    plan = make_plan(parent["tasks"], stage["seeds"], arms)
    assert len(plan) == stage["episodes"]
    write(archive / "run_plan.json", plan)
    write(archive / "effective_contract.json", {"model": parent["model"], "tasks": parent["tasks"], "arms": arms, "stage": args.stage})
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    import numpy as np
    import torch
    assigned, visible = os.environ.get("SLURM_JOB_GPUS"), os.environ.get("CUDA_VISIBLE_DEVICES")
    init = int(ctypes.CDLL("libcuda.so.1").cuInit(0))
    smi = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
    preflight = {"assigned_gpu": assigned, "visible_gpu": visible, "cuInit": init,
                 "torch_cuda": torch.cuda.is_available(), "device_count": torch.cuda.device_count(),
                 "nvidia_smi_exit": smi.returncode, "nvidia_smi": smi.stdout}
    preflight["passed"] = bool(assigned and assigned == visible and init == 0 and preflight["torch_cuda"] and preflight["device_count"] == 1 and smi.returncode == 0)
    write(archive / "cuda_preflight.json", preflight)
    if not preflight["passed"]:
        raise RuntimeError("CUDA preflight failed before model loading")
    from agentdojo.agent_pipeline import AgentPipeline
    from agentdojo.agent_pipeline.agent_pipeline import PipelineConfig
    from agentdojo.benchmark import run_task_without_injection_tasks
    from agentdojo.logging import OutputLogger
    from agentdojo.task_suite.load_suites import get_suite
    from wmagentattack.protocol_recovery_adapter import ConservativeProtocolLLM, GenerationRecorder
    m = parent["model"]
    results = []
    with block_python_network() as network:
        llm = ConservativeProtocolLLM(m["path"], device="cuda:0", quantization=m["quantization"], model_label=f"meta-llama-3.1-70b-{args.stage}",
            seed=stage["seeds"][0], max_new_tokens=m["max_new_tokens"], max_input_tokens=m["max_input_tokens"],
            max_tool_output_chars=m["max_tool_output_chars"], prompt_profile=m["prompt_profile"],
            protocol="function_tags", do_sample=m["do_sample"], temperature=m["temperature"], top_p=m["top_p"])
        llm.model = GenerationRecorder(llm.model, llm.tokenizer)
        pipeline = AgentPipeline.from_config(PipelineConfig(llm=llm, model_id=None, defense=None, system_message_name=None, system_message=None, tool_output_format=None))
        for index, item in enumerate(plan):
            suite_name, task_id = item["task"].split("|", 1)
            seed = item["episode_seed"]
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
            llm.seed, llm.protocol, llm.recovered_calls = seed, PROTOCOLS[item["arm"]], 0
            llm.model.reset()
            pipeline.name = f"meta-llama-3.1-70b-{args.stage}-{item['arm']}-local"
            logdir = archive / "raw" / item["arm"] / f"seed{item['run_seed']}"
            logdir.mkdir(parents=True, exist_ok=True)
            start = time.monotonic()
            try:
                suite = get_suite("v1.2.2", suite_name)
                with OutputLogger(str(logdir)):
                    utility, _ = run_task_without_injection_tasks(suite, pipeline, suite.get_user_task_by_id(task_id), logdir, False, "v1.2.2")
                rawpath = logdir / pipeline.name / suite_name / task_id / "none" / "none.json"
                raw = json.loads(rawpath.read_text())
                if network["blocked_attempts"]:
                    raise RuntimeError("Network attempt in clean sandbox")
                diagpath = archive / "diagnostics" / f"episode_{index:03d}.json"
                diag = {"first_prompt_ids": llm.model.first_prompt_ids, "events": llm.model.events,
                        "recovered_calls": llm.recovered_calls}
                write(diagpath, diag)
                assistant_count = sum(msg["role"] == "assistant" for msg in raw["messages"])
                extra = len(diag["events"]) - assistant_count
                if extra not in ([0, 1] if item["arm"] == "syntax_retry" else [0]):
                    raise RuntimeError("Generation/retry budget mismatch")
                result = {**item, "status": "completed", "utility": bool(utility), "raw_trace": str(rawpath),
                          "diagnostics": str(diagpath), "blocked_network_attempts": network["blocked_attempts"],
                          "parsed_tool_calls": sum(len(msg.get("tool_calls") or []) for msg in raw["messages"] if msg["role"] == "assistant"),
                          "generation_calls": len(diag["events"]), "extra_retry_generations": extra,
                          "elapsed_seconds": time.monotonic() - start}
            except Exception as error:
                result = {**item, "status": "failed", "error_type": type(error).__name__, "error": str(error)}
                results.append(result)
                write(archive / "results.json", results)
                write(archive / "gate.json", {"decision": "INVALID_PROTOCOL_RECOVERY", "scientific_result": False, "failure": result})
                raise
            results.append(result)
            write(archive / "results.json", results)
            print(json.dumps({"recorded": len(results), "expected": len(plan), **result}), flush=True)
    def read_artifacts(row):
        rawpath, diagpath = Path(row["raw_trace"]), Path(row["diagnostics"])
        if not rawpath.is_relative_to(archive / "raw") or not diagpath.is_relative_to(archive / "diagnostics"):
            raise ValueError("Artifact outside experiment archive")
        return json.loads(rawpath.read_text()), json.loads(diagpath.read_text())
    gate = evaluate(results, parent["tasks"], stage["seeds"], arms, stage["gate"], read_artifacts, prior=prior)
    gate.update(stage=args.stage, slurm_job_id=os.environ["SLURM_JOB_ID"])
    write(archive / "gate.json", gate)
    if not gate["scientific_result"]:
        raise RuntimeError("Final integrity gate failed")
    print(json.dumps(gate), flush=True)


if __name__ == "__main__":
    main()
