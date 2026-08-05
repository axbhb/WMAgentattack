"""Run one frozen local LLM over a frozen multi-source manifest chunk."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wmagentattack.multisource_semantic_data import (
    SCHEMA_VERSION,
    function_tag_prompt,
    parse_function_tag_completion,
    should_retry_tool_intent,
    stable_hash,
    summarize_generation,
)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, default=str)
        + "\n",
        encoding="utf-8",
    )


class FrozenSharedToolAgent:
    """The exact local Transformers scaffold shared with AgentDojo runs."""

    def __init__(self, contract: dict[str, Any]) -> None:
        model_path = Path(contract["model_snapshot"])
        if not model_path.exists():
            raise FileNotFoundError(model_path)
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for Llama-3.1-70B data construction")
        self.contract = contract
        self.device = "cuda:0"
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=bool(contract.get("trust_remote_code", False)),
        )
        quantization = contract["quantization"]
        kwargs: dict[str, Any] = {
            "local_files_only": True,
            "trust_remote_code": bool(contract.get("trust_remote_code", False)),
            "device_map": {"": self.device},
            "low_cpu_mem_usage": True,
            "attn_implementation": "sdpa",
        }
        if quantization == "4bit":
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        elif quantization == "bf16":
            kwargs["dtype"] = torch.bfloat16
        else:
            raise ValueError(f"unsupported quantization: {quantization}")
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
        length = inputs["input_ids"].shape[1]
        maximum = int(self.contract["max_input_tokens"])
        if maximum > 0 and length > maximum:
            prefix = min(1_024, maximum // 4)
            suffix = maximum - prefix
            for key, value in list(inputs.items()):
                if value.ndim == 2 and value.shape[1] == length:
                    inputs[key] = torch.cat([value[:, :prefix], value[:, -suffix:]], dim=1)
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
                        "[Tool-call serialization correction]\nYour previous response described "
                        "a tool action but did not execute one. Emit exactly one valid "
                        "tool call now using `<function=name>{\"arg\": \"value\"}</function>`. "
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


def _toolsandbox_state(context: Any) -> dict[str, Any]:
    from tool_sandbox.common.execution_context import DatabaseNamespace

    state = {}
    for namespace in DatabaseNamespace:
        if namespace == DatabaseNamespace.SANDBOX:
            continue
        try:
            state[str(namespace)] = context.get_database(namespace).to_dicts()
        except (IndexError, KeyError, TypeError, ValueError):
            continue
    return state


def _execute_toolsandbox_once(
    scenario: Any,
    decision: dict[str, Any],
    seed: int,
    logical_clock_iso: str,
) -> dict[str, Any]:
    from tool_sandbox.common.execution_context import get_current_context, set_current_context
    from wmagentattack.counterfactual_execution import frozen_sandbox_clock

    random.seed(seed)
    np.random.seed(seed)
    context = copy.deepcopy(scenario.starting_context)
    set_current_context(context)
    tools = context.get_available_tools(False)
    before = _toolsandbox_state(context)
    output: Any = None
    error: dict[str, str] | None = None
    try:
        with frozen_sandbox_clock(logical_clock_iso):
            output = tools[decision["name"]](**decision["arguments"])
    except Exception as exception:
        error = {"type": type(exception).__name__, "message": str(exception)}
    after = _toolsandbox_state(get_current_context())
    return {
        "status": "error" if error else "success",
        "error": error,
        "output": output,
        "state_before_sha256": stable_hash(before),
        "state_after_sha256": stable_hash(after),
        "state_changed": stable_hash(before) != stable_hash(after),
    }


def _tau_task(row: dict[str, Any]) -> tuple[Any, Any]:
    from tau2.registry import registry
    from tau2.runner.helpers import load_tasks

    domain = row["metadata"]["domain"]
    split = row["metadata"]["split"]
    task_id = row["metadata"]["task_id"]
    task = next(task for task in load_tasks(domain, split) if str(task.id) == task_id)
    environment = registry.get_env_constructor(domain)()
    initial = task.initial_state
    environment.set_state(
        initial.initialization_data if initial else None,
        initial.initialization_actions if initial else None,
        initial.message_history if initial and initial.message_history else [],
    )
    return environment, task


def _execute_tau_once(row: dict[str, Any], decision: dict[str, Any], seed: int) -> dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed)
    environment, _ = _tau_task(row)
    before = environment.get_db_hash()
    output: Any = None
    error: dict[str, str] | None = None
    try:
        output = environment.make_tool_call(
            decision["name"], requestor="assistant", **decision["arguments"]
        )
    except Exception as exception:
        error = {"type": type(exception).__name__, "message": str(exception)}
    if hasattr(output, "model_dump"):
        output = output.model_dump(mode="json")
    after = environment.get_db_hash()
    return {
        "status": "error" if error else "success",
        "error": error,
        "output": output,
        "state_before_sha256": before,
        "state_after_sha256": after,
        "state_changed": before != after,
    }


def _execution(
    source: str,
    row: dict[str, Any],
    decision: dict[str, Any],
    source_root: Path,
    seed: int,
    cache: dict[str, Any],
) -> dict[str, Any]:
    if decision["kind"] != "tool_call":
        return {"tier": "not_executed_text_response", "replica_identical": None}
    if source == "injecagent":
        attacker_tools = set(row["simulator_audit_only"]["attacker_tools"])
        return {
            "tier": "observation_only",
            "replica_identical": None,
            "selected_attacker_tool": decision["name"] in attacker_tools,
            "selected_user_tool": decision["name"] == row["metadata"]["user_tool"],
            "real_external_endpoint_calls": 0,
        }
    if source == "tool_sandbox":
        if "scenarios" not in cache:
            sys.path.insert(0, str(source_root))
            from tool_sandbox.common.tool_discovery import ToolBackend
            from tool_sandbox.scenarios import named_scenarios

            random.seed(int(cache["enumeration_seed"]))
            cache["scenarios"] = named_scenarios(ToolBackend.DEFAULT)
        scenario = cache["scenarios"][row["metadata"]["scenario_name"]]
        logical_clock_iso = str(cache["frozen_logical_clock_iso"])
        first = _execute_toolsandbox_once(
            scenario, decision, seed, logical_clock_iso
        )
        second = _execute_toolsandbox_once(
            scenario, decision, seed, logical_clock_iso
        )
    elif source == "tau3":
        sys.path.insert(0, str(source_root / "src"))
        first = _execute_tau_once(row, decision, seed)
        second = _execute_tau_once(row, decision, seed)
    else:
        raise ValueError(source)
    return {
        "tier": "exact",
        "replica_0": first,
        "replica_1": second,
        "replica_identical": stable_hash(first) == stable_hash(second),
        "real_external_endpoint_calls": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--chunk-index", type=int, default=0)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument(
        "--cached-input",
        type=Path,
        help="Reparse frozen completions without making new LLM calls.",
    )
    args = parser.parse_args()

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    implementation_hashes = protocol.get("execution_adapter", {}).get(
        "implementation_sha256", {}
    )
    for relative_path, expected_hash in implementation_hashes.items():
        implementation_path = ROOT / relative_path
        if _file_sha256(implementation_path) != expected_hash:
            raise ValueError(
                f"frozen execution implementation differs: {relative_path}"
            )
    contract = protocol["shared_llm_contract"]
    contract_hash = stable_hash(contract)
    if manifest["llm_contract_sha256"] != contract_hash:
        raise ValueError("manifest LLM contract differs from frozen protocol")
    if manifest["source_commit"] != protocol["sources"][manifest["source"]]["commit"]:
        raise ValueError("manifest source commit differs from frozen protocol")
    if _file_sha256(args.manifest) != protocol["frozen_manifests"][manifest["scale"]][manifest["source"]]:
        raise ValueError("manifest file hash differs from frozen protocol")
    if args.num_chunks <= 0 or not 0 <= args.chunk_index < args.num_chunks:
        raise ValueError("invalid chunk specification")
    rows = [
        row
        for index, row in enumerate(manifest["rows"])
        if index % args.num_chunks == args.chunk_index
    ]
    cached_by_row = None
    if args.cached_input is not None:
        cached_payload = json.loads(args.cached_input.read_text(encoding="utf-8"))
        cached_by_row = {row["row_id"]: row for row in cached_payload["records"]}
        if set(cached_by_row) != {row["row_id"] for row in rows}:
            raise ValueError("cached completions do not exactly match this manifest chunk")
        agent = None
    else:
        agent = FrozenSharedToolAgent(contract)
    cache: dict[str, Any] = {
        "enumeration_seed": protocol["sources"]["tool_sandbox"]["enumeration_seed"],
        "frozen_logical_clock_iso": protocol["sources"]["tool_sandbox"][
            "frozen_logical_clock_iso"
        ],
    }
    records = []
    for index, row in enumerate(rows):
        print(f"ROW_START {index + 1}/{len(rows)} {row['row_id']}", flush=True)
        record = {
            "schema_version": SCHEMA_VERSION,
            **row,
            "completion": "",
            "retry_completion": None,
            "decision": {"kind": "text", "text": "", "repair": None},
            "execution": {"tier": "not_started", "replica_identical": None},
            "runtime_error": None,
        }
        try:
            if cached_by_row is None:
                assert agent is not None
                generated = agent.query(row["model_input"], int(row["run_seed"]))
            else:
                cached = cached_by_row[row["row_id"]]
                completion = str(cached["completion"])
                retry_completion = cached.get("retry_completion")
                allowed = {
                    tool["function"]["name"]
                    for tool in row["model_input"]["tool_schemas"]
                }
                decision = parse_function_tag_completion(completion, allowed)
                if decision["kind"] != "tool_call" and retry_completion:
                    decision = parse_function_tag_completion(
                        str(retry_completion), allowed
                    )
                generated = {
                    "completion": completion,
                    "retry_completion": retry_completion,
                    "decision": decision,
                    "prompt_sha256": stable_hash(function_tag_prompt(row["model_input"])),
                }
            record.update(generated)
            record["execution"] = _execution(
                manifest["source"],
                row,
                generated["decision"],
                args.source_root,
                int(row["run_seed"]),
                cache,
            )
        except torch.cuda.OutOfMemoryError:
            record["runtime_error"] = {"type": "CUDAOutOfMemoryError", "message": "OOM"}
            records.append(record)
            _write(args.output, {"records": records, "complete": False})
            raise
        except Exception as exception:
            record["runtime_error"] = {
                "type": type(exception).__name__,
                "message": str(exception),
            }
        records.append(record)
        _write(args.output, {"records": records, "complete": len(records) == len(rows)})
        print(f"ROW_DONE {index + 1}/{len(rows)} {row['row_id']}", flush=True)

    audit = summarize_generation(
        records,
        expected_rows=len(rows),
        require_exact_replica_determinism=True,
    )
    audit.update(
        {
            "source": manifest["source"],
            "scale": manifest["scale"],
            "chunk_index": args.chunk_index,
            "num_chunks": args.num_chunks,
            "manifest_file_sha256": _file_sha256(args.manifest),
            "protocol_file_sha256": _file_sha256(args.protocol),
            "llm_contract_sha256": contract_hash,
            "output_file_sha256": _file_sha256(args.output),
            "cached_input_sha256": (
                _file_sha256(args.cached_input) if args.cached_input else None
            ),
        }
    )
    _write(args.audit, audit)
    print(json.dumps(audit, ensure_ascii=False, sort_keys=True), flush=True)
    if not audit["passed"]:
        raise SystemExit(f"{manifest['source']} generation gate failed")


if __name__ == "__main__":
    main()
