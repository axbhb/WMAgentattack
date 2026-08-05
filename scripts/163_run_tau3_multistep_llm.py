"""Collect one frozen chunk of exact multi-step tau3 victim trajectories."""

from __future__ import annotations

import argparse
import copy
import json
import random
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wmagentattack.multisource_semantic_data import (
    function_tag_prompt,
    parse_function_tag_completion,
    should_retry_tool_intent,
)
from wmagentattack.tau3_multistep import (
    append_ledger_event,
    file_sha256,
    source_prefix,
    stable_hash,
    visible_observation,
)


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, default=str)
        + "\n",
        encoding="utf-8",
    )


class FrozenSharedToolAgent:
    """The same local Transformers contract used by the prior three-source build."""

    def __init__(self, contract: Mapping[str, Any]) -> None:
        model_path = Path(contract["model_snapshot"])
        if not model_path.exists():
            raise FileNotFoundError(model_path)
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for the frozen 70B victim")
        self.contract = dict(contract)
        self.device = "cuda:0"
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=bool(contract.get("trust_remote_code", False)),
        )
        kwargs: dict[str, Any] = {
            "local_files_only": True,
            "trust_remote_code": bool(contract.get("trust_remote_code", False)),
            "device_map": {"": self.device},
            "low_cpu_mem_usage": True,
            "attn_implementation": "sdpa",
        }
        if contract["quantization"] == "4bit":
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        else:
            raise ValueError("only the frozen 4-bit victim is authorized")
        self.model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
        self.model.eval()

    def _inputs(self, messages: list[dict[str, str]]) -> dict[str, torch.Tensor]:
        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        length = int(inputs["input_ids"].shape[1])
        maximum = int(self.contract["max_input_tokens"])
        if maximum > 0 and length > maximum:
            prefix = min(1024, maximum // 4)
            suffix = maximum - prefix
            for key, value in list(inputs.items()):
                if value.ndim == 2 and value.shape[1] == length:
                    inputs[key] = torch.cat(
                        [value[:, :prefix], value[:, -suffix:]], dim=1
                    )
        return {key: value.to(self.device) for key, value in inputs.items()}

    def _generate(self, messages: list[dict[str, str]]) -> str:
        inputs = self._inputs(messages)
        kwargs: dict[str, Any] = {
            "max_new_tokens": int(self.contract["max_new_tokens"]),
            "do_sample": bool(self.contract["do_sample"]),
            "use_cache": True,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if kwargs["do_sample"]:
            kwargs.update(
                temperature=float(self.contract["temperature"]),
                top_p=float(self.contract["top_p"]),
            )
        outputs = self.model.generate(**inputs, **kwargs)
        generated = outputs[0, inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()

    @torch.inference_mode()
    def query(self, model_input: dict[str, Any], seed: int) -> dict[str, Any]:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        messages = function_tag_prompt(model_input)
        completion = self._generate(messages)
        allowed = {
            tool["function"]["name"] for tool in model_input["tool_schemas"]
        }
        decision = parse_function_tag_completion(completion, allowed)
        retry_completion = None
        if decision["kind"] != "tool_call" and should_retry_tool_intent(completion):
            retry_messages = [
                *messages,
                {"role": "assistant", "content": completion},
                {
                    "role": "user",
                    "content": (
                        "[Tool-call serialization correction]\nYour previous response "
                        "described a tool action but did not execute one. Emit exactly "
                        "one valid tool call now using "
                        "`<function=name>{\"arg\": \"value\"}</function>`. "
                        "Use `{}` when there are no arguments. Do not explain."
                    ),
                },
            ]
            retry_completion = self._generate(retry_messages)
            repaired = parse_function_tag_completion(retry_completion, allowed)
            if repaired["kind"] == "tool_call":
                decision = repaired
        return {
            "completion": completion,
            "retry_completion": retry_completion,
            "decision": decision,
            "prompt_sha256": stable_hash(messages),
        }


def _task(row: Mapping[str, Any]) -> Any:
    from tau2.runner.helpers import load_tasks

    matches = [
        task
        for task in load_tasks(row["domain"], row["source_split"])
        if str(task.id) == str(row["task_id"])
    ]
    if len(matches) != 1:
        raise ValueError("manifest task does not resolve uniquely")
    return matches[0]


def _reset(registry: Any, row: Mapping[str, Any], task: Any) -> Any:
    environment = registry.get_env_constructor(row["domain"])()
    initial = task.initial_state
    environment.set_state(
        initial.initialization_data if initial else None,
        initial.initialization_actions if initial else None,
        initial.message_history if initial and initial.message_history else [],
    )
    return environment


def _replay(
    registry: Any,
    row: Mapping[str, Any],
    task: Any,
    calls: Sequence[Mapping[str, Any]],
    seed: int,
) -> dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed)
    environment = _reset(registry, row, task)
    transitions = []
    for index, call in enumerate(calls):
        before = environment.get_db_hash()
        output: Any = None
        error: dict[str, str] | None = None
        try:
            output = environment.make_tool_call(
                call["name"], requestor="assistant", **call["arguments"]
            )
        except Exception as exception:
            error = {"type": type(exception).__name__, "message": str(exception)}
        if hasattr(output, "model_dump"):
            output = output.model_dump(mode="json")
        after = environment.get_db_hash()
        transitions.append(
            {
                "index": index,
                "action": {
                    "name": str(call["name"]),
                    "arguments": copy.deepcopy(call["arguments"]),
                },
                "status": "error" if error else "success",
                "error": error,
                "output": output,
                "state_before_sha256": before,
                "state_after_sha256": after,
                "state_changed": before != after,
            }
        )
    return {
        "transitions": transitions,
        "final_state_sha256": environment.get_db_hash(),
    }


def _transcript(events: Sequence[Mapping[str, Any]]) -> str:
    if not events:
        return ""
    lines = [
        "The following is the complete victim-visible transcript. Continue from "
        "the latest observation; do not repeat already completed actions."
    ]
    for event in events:
        lines.append(
            f"Step {event['index']} assistant tool call: "
            + json.dumps(event["action"], ensure_ascii=False, sort_keys=True)
        )
        lines.append(
            f"Step {event['index']} tool observation: {visible_observation(event)}"
        )
    return "\n".join(lines)


def collect_episode(
    *,
    agent: FrozenSharedToolAgent,
    registry: Any,
    row: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    task = _task(row)
    base_model_input = copy.deepcopy(row["model_input"])
    prior_events: list[dict[str, Any]] = []
    calls: list[dict[str, Any]] = []
    prefixes: list[dict[str, Any]] = []
    query_records: list[dict[str, Any]] = []
    ledger: dict[str, Any] = {
        "records": [],
        "conflicts": [],
        "execution_receipts": [],
    }
    maximum_transitions = int(
        protocol["trajectory_collection"][
            "maximum_executed_tool_transitions_per_episode"
        ]
    )
    maximum_decisions = int(
        protocol["trajectory_collection"]["maximum_decisions_per_episode"]
    )
    termination = "maximum_decisions_exhausted"
    for prefix_index in range(maximum_decisions):
        model_input = copy.deepcopy(base_model_input)
        transcript = _transcript(prior_events)
        if transcript:
            model_input["observation"] = transcript
        step_seed = int(row["llm_seed"]) * 1009 + prefix_index
        query = agent.query(model_input, step_seed)
        if not str(query["completion"]).strip():
            raise ValueError("victim completion is empty")
        decision = query["decision"]
        prefix = source_prefix(
            episode_id=row["episode_id"],
            domain=row["domain"],
            model_input=base_model_input,
            prefix_index=prefix_index,
            prior_events=prior_events,
            ledger=ledger,
            decision=decision,
        )
        prefixes.append(prefix)
        query_records.append(
            {
                "prefix_index": prefix_index,
                "step_seed": step_seed,
                "completion": query["completion"],
                "retry_completion": query["retry_completion"],
                "decision": decision,
                "prompt_sha256": query["prompt_sha256"],
            }
        )
        if decision["kind"] != "tool_call":
            termination = "text_response"
            break
        if len(prior_events) >= maximum_transitions:
            termination = "tool_action_censored_at_horizon"
            break
        call = {"name": decision["name"], "arguments": decision["arguments"]}
        proposed_calls = [*calls, call]
        first = _replay(registry, row, task, proposed_calls, step_seed)
        second = _replay(registry, row, task, proposed_calls, step_seed)
        if stable_hash(first) != stable_hash(second):
            raise ValueError("exact prefix execution replicas differ")
        event = copy.deepcopy(first["transitions"][-1])
        event["replica_identical"] = True
        event["prefix_replay_sha256"] = stable_hash(first)
        prior_events.append(event)
        calls.append(call)
        ledger = append_ledger_event(
            ledger,
            episode_id=row["episode_id"],
            domain=row["domain"],
            event=event,
        )
    return {
        "episode_id": row["episode_id"],
        "task_key": row["task_key"],
        "domain": row["domain"],
        "source_split": row["source_split"],
        "task_id": row["task_id"],
        "split": row["experimental_split"],
        "structural_stratum": row["structural_stratum"],
        "llm_seed": row["llm_seed"],
        "prefixes": prefixes,
        "transitions": prior_events,
        "queries": query_records,
        "termination": termination,
        "runtime_error": None,
        "real_external_endpoint_calls": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--chunk-index", type=int, required=True)
    parser.add_argument("--num-chunks", type=int, required=True)
    args = parser.parse_args()

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if protocol["status"] != "manifest_frozen_before_victim_outcomes":
        raise ValueError("tau3 multi-step manifest was not frozen before LLM outcomes")
    frozen = protocol["frozen_manifest"]
    if file_sha256(args.manifest) != frozen["sha256"]:
        raise ValueError("manifest file hash differs from frozen protocol")
    for relative_path, expected in protocol["implementation_sha256"].items():
        if file_sha256(ROOT / relative_path) != expected:
            raise ValueError(f"implementation hash differs: {relative_path}")
    if not 0 <= args.chunk_index < args.num_chunks:
        raise ValueError("invalid chunk index")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=args.source_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if commit != protocol["source"]["commit"]:
        raise ValueError("tau3 source commit differs from frozen protocol")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    rows = [
        row
        for index, row in enumerate(manifest["rows"])
        if index % args.num_chunks == args.chunk_index
    ]
    sys.path.insert(0, str(args.source_root / "src"))
    from loguru import logger
    from tau2.registry import registry

    logger.remove()
    agent = FrozenSharedToolAgent(protocol["shared_llm_contract"])
    episodes = []
    failures = []
    for row in rows:
        try:
            episodes.append(
                collect_episode(
                    agent=agent, registry=registry, row=row, protocol=protocol
                )
            )
        except Exception as exception:
            failures.append(
                {
                    "episode_id": row["episode_id"],
                    "type": type(exception).__name__,
                    "message": str(exception),
                }
            )
    payload = {
        "protocol_id": protocol["protocol_id"],
        "chunk_index": args.chunk_index,
        "num_chunks": args.num_chunks,
        "episodes": episodes,
        "failures": failures,
        "real_external_endpoint_calls": 0,
    }
    _write(args.output, payload)
    audit = {
        "chunk_index": args.chunk_index,
        "expected_rows": len(rows),
        "completed_episodes": len(episodes),
        "runtime_failures": failures,
        "decisions": sum(len(row["prefixes"]) for row in episodes),
        "adjacent_transitions": sum(len(row["transitions"]) for row in episodes),
        "nondeterministic_transitions": sum(
            not event["replica_identical"]
            for row in episodes
            for event in row["transitions"]
        ),
        "real_external_endpoint_calls": 0,
        "output_sha256": file_sha256(args.output),
        "passed": len(episodes) == len(rows) and not failures,
    }
    _write(args.audit, audit)
    print(json.dumps(audit, ensure_ascii=False, sort_keys=True))
    if not audit["passed"]:
        raise SystemExit("TAU3_MULTISTEP_CHUNK_FAILED")


if __name__ == "__main__":
    main()
