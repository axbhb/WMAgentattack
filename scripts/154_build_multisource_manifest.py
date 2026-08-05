"""Build frozen, label-blind ToolSandbox/InjecAgent/tau3 manifests."""

from __future__ import annotations

import argparse
import copy
import inspect
import json
import random
import subprocess
import sys
import types
from collections import Counter
from pathlib import Path
from typing import Any, Literal, Union, get_args, get_origin

from pydantic import BaseModel


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wmagentattack.multisource_semantic_data import (
    MANIFEST_SCHEMA_VERSION,
    audit_manifest,
    build_model_input,
    injecagent_tool_schema,
    stable_hash,
)


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, default=str)
        + "\n",
        encoding="utf-8",
    )


def _git_commit(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def _rank(value: Any, seed: str) -> str:
    return stable_hash({"seed": seed, "value": value})


def _source_row(
    *,
    source: str,
    source_commit: str,
    group_id: str,
    variant: str,
    run_seed: int,
    llm_contract_sha256: str,
    model_input: dict[str, Any],
    metadata: dict[str, Any],
    simulator_audit_only: dict[str, Any],
) -> dict[str, Any]:
    row_id = f"{source}::{group_id}::{variant}::seed{run_seed}"
    return {
        "schema_version": "wmagentattack.multisource_semantic_row.v1",
        "row_id": row_id,
        "source": source,
        "source_commit": source_commit,
        "group_id": f"{source}::{group_id}",
        "variant": variant,
        "run_seed": run_seed,
        "llm_contract_sha256": llm_contract_sha256,
        "model_input": model_input,
        "metadata": metadata,
        "simulator_audit_only": simulator_audit_only,
    }


def _toolsandbox_task(context: Any) -> str:
    from tool_sandbox.common.execution_context import DatabaseNamespace, RoleType

    rows = context.get_database(
        DatabaseNamespace.SANDBOX,
        drop_sandbox_message_index=False,
        get_all_history_snapshots=True,
    ).to_dicts()
    candidates = [
        row
        for row in rows
        if row.get("sender") == RoleType.USER
        and row.get("recipient") == RoleType.AGENT
        and row.get("content")
    ]
    if not candidates:
        raise ValueError("ToolSandbox scenario has no user-to-agent task")
    return str(candidates[-1]["content"])


def _toolsandbox_reference_calls(scenario: Any) -> list[dict[str, Any]]:
    from tool_sandbox.common.execution_context import DatabaseNamespace

    calls: list[dict[str, Any]] = []
    seen: set[str] = set()
    milestones = scenario.evaluation.milestone_matcher.milestones
    for milestone in milestones:
        for constraint in milestone.snapshot_constraints:
            if constraint.database_namespace != DatabaseNamespace.SANDBOX:
                continue
            frame = constraint.target_dataframe
            if frame is None or "tool_trace" not in frame.columns:
                continue
            for raw in frame["tool_trace"].drop_nulls().to_list():
                try:
                    trace = json.loads(raw)
                except (TypeError, json.JSONDecodeError):
                    continue
                traces = trace if isinstance(trace, list) else [trace]
                for item in traces:
                    if not isinstance(item, dict):
                        continue
                    name = item.get("tool_name")
                    arguments = item.get(
                        "arguments", item.get("tool_arguments", {})
                    )
                    if not isinstance(name, str) or not isinstance(arguments, dict):
                        continue
                    call = {"name": name, "arguments": arguments}
                    fingerprint = stable_hash(call)
                    if fingerprint not in seen:
                        calls.append(call)
                        seen.add(fingerprint)
    return calls


def _toolsandbox_tool_schemas(tools: dict[str, Any]) -> list[dict[str, Any]]:
    """Build schemas directly, avoiding ToolSandbox's legacy LangChain shim."""

    primitive_types = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
    }

    def annotation_schema(annotation: Any) -> dict[str, Any]:
        if annotation in primitive_types:
            return {"type": primitive_types[annotation]}
        if annotation in {Any, inspect.Parameter.empty}:
            return {}
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            return annotation.model_json_schema()
        origin = get_origin(annotation)
        arguments = get_args(annotation)
        if origin is Literal:
            values = list(arguments)
            schema: dict[str, Any] = {"enum": values}
            if values and type(values[0]) in primitive_types:
                schema["type"] = primitive_types[type(values[0])]
            return schema
        if origin in {list, tuple, set}:
            return {
                "type": "array",
                "items": annotation_schema(arguments[0]) if arguments else {},
            }
        if origin is dict:
            return {"type": "object", "additionalProperties": True}
        if origin in {Union, types.UnionType}:
            real = [
                item
                for item in arguments
                if item is not type(None) and getattr(item, "__name__", "") != "NotGiven"
            ]
            if len(real) == 1:
                return annotation_schema(real[0])
            if real and all(item in {int, float} for item in real):
                return {"type": "number"}
            variants = [annotation_schema(item) for item in real]
            return {"anyOf": variants} if variants else {}
        return {}

    def descriptions(tool: Any) -> tuple[str, dict[str, str]]:
        doc = inspect.getdoc(tool) or ""
        blocks = doc.split("\n\n")
        description_parts = []
        arguments: dict[str, str] = {}
        for block in blocks:
            if block.startswith("Args:"):
                current = None
                for line in block.splitlines()[1:]:
                    if ":" in line:
                        current, text = line.split(":", 1)
                        current = current.strip()
                        arguments[current] = text.strip()
                    elif current:
                        arguments[current] += " " + line.strip()
            elif not block.startswith(("Returns:", "Raises:", "Example:")):
                description_parts.append(block)
        return " ".join(description_parts), arguments

    schemas = []
    for name, tool in sorted(tools.items()):
        description, arg_descriptions = descriptions(tool)
        properties = {}
        required = []
        for parameter_name, parameter in inspect.signature(tool).parameters.items():
            properties[parameter_name] = annotation_schema(parameter.annotation)
            if parameter_name in arg_descriptions:
                properties[parameter_name]["description"] = arg_descriptions[
                    parameter_name
                ]
            if parameter.default is inspect.Parameter.empty:
                required.append(parameter_name)
        schemas.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": required,
                    },
                },
            }
        )
    return schemas


def _toolsandbox_state(context: Any) -> dict[str, Any]:
    from tool_sandbox.common.execution_context import DatabaseNamespace

    state: dict[str, Any] = {}
    for namespace in DatabaseNamespace:
        if namespace == DatabaseNamespace.SANDBOX:
            continue
        try:
            state[str(namespace)] = context.get_database(namespace).to_dicts()
        except (IndexError, KeyError, TypeError, ValueError):
            continue
    return state


def _run_toolsandbox_reference(
    scenario: Any, calls: list[dict[str, Any]], logical_clock_iso: str
) -> dict[str, Any]:
    from tool_sandbox.common.execution_context import get_current_context, set_current_context
    from wmagentattack.counterfactual_execution import frozen_sandbox_clock

    context = copy.deepcopy(scenario.starting_context)
    set_current_context(context)
    tools = context.get_available_tools(False)
    transitions = []
    for index, call in enumerate(calls):
        before = _toolsandbox_state(get_current_context())
        output: Any = None
        error: dict[str, str] | None = None
        try:
            with frozen_sandbox_clock(logical_clock_iso):
                output = tools[call["name"]](**call["arguments"])
        except Exception as exception:  # benchmark errors are data
            error = {"type": type(exception).__name__, "message": str(exception)}
        after = _toolsandbox_state(get_current_context())
        transitions.append(
            {
                "index": index,
                "action": call,
                "status": "error" if error else "success",
                "error": error,
                "output": output,
                "state_before_sha256": stable_hash(before),
                "state_after_sha256": stable_hash(after),
                "state_changed": stable_hash(before) != stable_hash(after),
            }
        )
    return {"transitions": transitions, "final_state_sha256": stable_hash(_toolsandbox_state(get_current_context()))}


def build_toolsandbox(
    source_root: Path, source_config: dict[str, Any], contract_hash: str, scale: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    sys.path.insert(0, str(source_root))
    from tool_sandbox.common.execution_context import ScenarioCategories
    from tool_sandbox.common.tool_discovery import ToolBackend
    from tool_sandbox.scenarios import named_scenarios

    random.seed(int(source_config["enumeration_seed"]))
    scenarios = named_scenarios(ToolBackend.DEFAULT)
    category_order = (
        ScenarioCategories.STATE_DEPENDENCY,
        ScenarioCategories.INSUFFICIENT_INFORMATION,
        ScenarioCategories.DISAMBIGUATION,
        ScenarioCategories.CANONICALIZATION,
    )
    excluded_suffixes = (
        "_3_distraction_tools",
        "_10_distraction_tools",
        "_all_tools",
        "_tool_description_scrambled",
        "_arg_type_scrambled",
        "_arg_description_scrambled",
        "_tool_name_scrambled",
    )
    eligible: list[tuple[str, Any, str]] = []
    unsafe: dict[str, list[str]] = {}
    for name, scenario in sorted(scenarios.items()):
        if any(suffix in name for suffix in excluded_suffixes):
            continue
        tools = scenario.starting_context.get_available_tools(False)
        external = sorted(
            tool_name
            for tool_name, tool in tools.items()
            if tool.__module__.split(".")[-1] == "rapid_api_search_tools"
        )
        if external:
            unsafe[name] = external
            continue
        primary = next(
            (str(category) for category in category_order if category in scenario.categories),
            "OTHER",
        )
        eligible.append((name, scenario, primary))

    if scale == "pilot":
        selected: list[tuple[str, Any, str]] = []
        strata = {
            "STATE_DEPENDENCY": int(source_config["pilot_per_stratum"]),
            "INSUFFICIENT_INFORMATION": int(source_config["pilot_per_stratum"]),
            "DISAMBIGUATION_OR_CANONICALIZATION": int(source_config["pilot_per_stratum"]),
        }
        grouped = {
            "STATE_DEPENDENCY": [row for row in eligible if row[2] == "STATE_DEPENDENCY"],
            "INSUFFICIENT_INFORMATION": [row for row in eligible if row[2] == "INSUFFICIENT_INFORMATION"],
            "DISAMBIGUATION_OR_CANONICALIZATION": [
                row for row in eligible if row[2] in {"DISAMBIGUATION", "CANONICALIZATION"}
            ],
        }
        for stratum, count in strata.items():
            ranked = sorted(grouped[stratum], key=lambda row: _rank(row[0], source_config["selection_seed"]))
            if len(ranked) < count:
                raise ValueError(f"not enough ToolSandbox rows in {stratum}")
            selected.extend(ranked[:count])
        seeds = list(source_config["pilot_seeds"])
    else:
        selected = eligible
        seeds = list(source_config["large_seeds"])

    source_commit = _git_commit(source_root)
    rows = []
    deterministic = 0
    reference_errors = 0
    for name, scenario, primary in sorted(selected, key=lambda row: row[0]):
        tools = scenario.starting_context.get_available_tools(False)
        schemas = _toolsandbox_tool_schemas(tools)
        reference_calls = _toolsandbox_reference_calls(scenario)
        first = _run_toolsandbox_reference(
            scenario, reference_calls, source_config["frozen_logical_clock_iso"]
        )
        second = _run_toolsandbox_reference(
            scenario, reference_calls, source_config["frozen_logical_clock_iso"]
        )
        replica_identical = stable_hash(first) == stable_hash(second)
        deterministic += int(replica_identical)
        reference_errors += sum(
            transition["status"] == "error" for transition in first["transitions"]
        )
        task = _toolsandbox_task(scenario.starting_context)
        for seed in seeds:
            rows.append(
                _source_row(
                    source="tool_sandbox",
                    source_commit=source_commit,
                    group_id=name,
                    variant="clean",
                    run_seed=int(seed),
                    llm_contract_sha256=contract_hash,
                    model_input=build_model_input(
                        trusted_goal=task,
                        tool_schemas=schemas,
                    ),
                    metadata={
                        "scenario_name": name,
                        "primary_category": primary,
                        "categories": sorted(str(category) for category in scenario.categories),
                        "exact_execution_available": True,
                    },
                    simulator_audit_only={
                        "reference_calls": reference_calls,
                        "reference_replay": first,
                        "reference_replica_identical": replica_identical,
                    },
                )
            )
    preflight = {
        "eligible_base_scenarios": len(eligible),
        "excluded_external_endpoint_scenarios": len(unsafe),
        "excluded_external_endpoint_tools": unsafe,
        "selected_groups": len(selected),
        "reference_replica_identical_groups": deterministic,
        "reference_execution_errors": reference_errors,
        "passed": deterministic == len(selected),
    }
    return rows, preflight


def _load_injecagent_tools(source_root: Path) -> dict[str, dict[str, Any]]:
    toolkits = json.loads((source_root / "data" / "tools.json").read_text(encoding="utf-8"))
    result = {}
    for toolkit in toolkits:
        for raw_tool in toolkit["tools"]:
            full = dict(raw_tool)
            full["name"] = str(toolkit["toolkit"]) + str(raw_tool["name"])
            result[full["name"]] = injecagent_tool_schema(full)
    return result


def build_injecagent(
    source_root: Path, source_config: dict[str, Any], contract_hash: str, scale: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    tools = _load_injecagent_tools(source_root)
    cases: list[tuple[str, int, dict[str, Any]]] = []
    for attack_family, filename in (
        ("direct_harm", "test_cases_dh_base.json"),
        ("data_stealing", "test_cases_ds_base.json"),
    ):
        payload = json.loads((source_root / "data" / filename).read_text(encoding="utf-8"))
        cases.extend((attack_family, index, row) for index, row in enumerate(payload))
    if scale == "pilot":
        selected = []
        per_family = int(source_config["pilot_per_family"])
        for family in ("direct_harm", "data_stealing"):
            pool = [row for row in cases if row[0] == family]
            pool.sort(
                key=lambda item: _rank(
                    {"family": family, "index": item[1], "user_tool": item[2]["User Tool"]},
                    source_config["selection_seed"],
                )
            )
            selected.extend(pool[:per_family])
        seeds = list(source_config["pilot_seeds"])
    else:
        selected = cases
        seeds = list(source_config["large_seeds"])

    source_commit = _git_commit(source_root)
    rows = []
    missing_tools: set[str] = set()
    clean_pair_differences = []
    for family, source_index, case in selected:
        names = [case["User Tool"], *case["Attacker Tools"]]
        schemas = []
        for name in names:
            if name not in tools:
                missing_tools.add(name)
            else:
                schemas.append(tools[name])
        poisoned = str(case["Tool Response"])
        attacker_instruction = str(case["Attacker Instruction"])
        clean = poisoned.replace(attacker_instruction, "[No additional instructions.]")
        if clean == poisoned:
            template = str(case["Tool Response Template"])
            clean = template.replace("<Attacker Instruction>", "[No additional instructions.]")
        observation_prefix = (
            f"Completed trusted tool call {case['User Tool']} with parameters "
            f"{case['Tool Parameters']}. Result:\n"
        )
        group = f"{family}::{source_index}"
        for seed in seeds:
            for variant, observation in (("clean", clean), ("poisoned", poisoned)):
                rows.append(
                    _source_row(
                        source="injecagent",
                        source_commit=source_commit,
                        group_id=group,
                        variant=variant,
                        run_seed=int(seed),
                        llm_contract_sha256=contract_hash,
                        model_input=build_model_input(
                            trusted_goal=str(case["User Instruction"]),
                            tool_schemas=schemas,
                            observation=observation_prefix + observation,
                        ),
                        metadata={
                            "attack_family": family,
                            "source_index": source_index,
                            "user_tool": case["User Tool"],
                            "available_tool_names": names,
                            "exact_execution_available": False,
                            "transition_tier": "observation_only",
                        },
                        simulator_audit_only={
                            "attacker_tools": case["Attacker Tools"],
                            "attacker_instruction": attacker_instruction,
                            "expected_achievements": case.get("Expected Achievements"),
                        },
                    )
                )
        clean_pair_differences.append(clean != poisoned)
    preflight = {
        "published_cases": len(cases),
        "selected_groups": len(selected),
        "missing_tool_schemas": sorted(missing_tools),
        "clean_poison_observation_diff_groups": sum(clean_pair_differences),
        "observation_only_groups": len(selected),
        "real_external_endpoint_calls": 0,
        "passed": not missing_tools and all(clean_pair_differences),
    }
    return rows, preflight


def _tau_goal(task: Any) -> str:
    instructions = task.user_scenario.model_dump(mode="json").get("instructions")
    if isinstance(instructions, dict):
        parts = [
            str(instructions[key]).strip()
            for key in ("reason_for_call", "known_info", "unknown_info", "task_instructions")
            if instructions.get(key)
        ]
        return "\n".join(parts)
    return str(instructions)


def _tau_reset(domain: str, task: Any) -> Any:
    from tau2.registry import registry

    environment = registry.get_env_constructor(domain)()
    initial = task.initial_state
    environment.set_state(
        initial.initialization_data if initial else None,
        initial.initialization_actions if initial else None,
        initial.message_history if initial and initial.message_history else [],
    )
    return environment


def _run_tau_reference(domain: str, task: Any, calls: list[dict[str, Any]]) -> dict[str, Any]:
    environment = _tau_reset(domain, task)
    transitions = []
    for index, call in enumerate(calls):
        before = environment.get_db_hash()
        output: Any = None
        error: dict[str, str] | None = None
        try:
            output = environment.make_tool_call(
                call["name"], requestor="assistant", **call["arguments"]
            )
        except Exception as exception:  # exact simulator errors are labels
            error = {"type": type(exception).__name__, "message": str(exception)}
        after = environment.get_db_hash()
        if hasattr(output, "model_dump"):
            output = output.model_dump(mode="json")
        transitions.append(
            {
                "index": index,
                "action": call,
                "status": "error" if error else "success",
                "error": error,
                "output": output,
                "state_before_sha256": before,
                "state_after_sha256": after,
                "state_changed": before != after,
            }
        )
    return {"transitions": transitions, "final_state_sha256": environment.get_db_hash()}


def _tau_invalid_probe(call: dict[str, Any]) -> dict[str, Any] | None:
    arguments = copy.deepcopy(call["arguments"])
    for key in sorted(arguments):
        if isinstance(arguments[key], str):
            arguments[key] = "__wmagentattack_missing_id__"
            return {"name": call["name"], "arguments": arguments}
    return None


def build_tau3(
    source_root: Path, source_config: dict[str, Any], contract_hash: str, scale: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    sys.path.insert(0, str(source_root / "src"))
    from tau2.registry import registry
    from tau2.runner.helpers import load_tasks

    task_rows: list[tuple[str, str, Any]] = []
    for domain in source_config["domains"]:
        for split in source_config["splits"]:
            task_rows.extend((domain, split, task) for task in load_tasks(domain, split))
    if scale == "pilot":
        selected = []
        per_domain = int(source_config["pilot_per_domain"])
        for domain in source_config["domains"]:
            pool = [
                row
                for row in task_rows
                if row[0] == domain
                and row[2].evaluation_criteria
                and any(
                    action.requestor == "assistant"
                    for action in (row[2].evaluation_criteria.actions or [])
                )
            ]
            pool.sort(key=lambda row: _rank({"domain": domain, "task": row[2].id}, source_config["selection_seed"]))
            if len(pool) < per_domain:
                raise ValueError(f"not enough tau3 tasks for {domain}")
            selected.extend(pool[:per_domain])
        seeds = list(source_config["pilot_seeds"])
    else:
        selected = task_rows
        seeds = list(source_config["large_seeds"])

    source_commit = _git_commit(source_root)
    rows = []
    deterministic = 0
    probe_errors = 0
    mutations = 0
    for domain, split, task in sorted(selected, key=lambda row: (row[0], row[1], str(row[2].id))):
        environment = registry.get_env_constructor(domain)()
        schemas = [tool.openai_schema for tool in environment.get_tools()]
        actions = task.evaluation_criteria.actions if task.evaluation_criteria else []
        calls = [
            {"name": action.name, "arguments": action.arguments}
            for action in (actions or [])
            if action.requestor == "assistant"
        ]
        first = _run_tau_reference(domain, task, calls)
        second = _run_tau_reference(domain, task, calls)
        replica_identical = stable_hash(first) == stable_hash(second)
        deterministic += int(replica_identical)
        mutations += sum(row["state_changed"] for row in first["transitions"])
        probe = _tau_invalid_probe(calls[0]) if calls else None
        probe_result = _run_tau_reference(domain, task, [probe]) if probe else None
        if probe_result:
            probe_errors += int(probe_result["transitions"][0]["status"] == "error")
        group = f"{domain}::{split}::{task.id}"
        for seed in seeds:
            rows.append(
                _source_row(
                    source="tau3",
                    source_commit=source_commit,
                    group_id=group,
                    variant="clean",
                    run_seed=int(seed),
                    llm_contract_sha256=contract_hash,
                    model_input=build_model_input(
                        trusted_goal=_tau_goal(task),
                        tool_schemas=schemas,
                        policy=environment.get_policy(),
                    ),
                    metadata={
                        "domain": domain,
                        "split": split,
                        "task_id": str(task.id),
                        "exact_execution_available": True,
                    },
                    simulator_audit_only={
                        "reference_calls": calls,
                        "reference_replay": first,
                        "reference_replica_identical": replica_identical,
                        "invalid_binding_probe": probe,
                        "invalid_binding_probe_result": probe_result,
                    },
                )
            )
    preflight = {
        "published_tasks": len(task_rows),
        "selected_groups": len(selected),
        "reference_replica_identical_groups": deterministic,
        "reference_state_mutations": mutations,
        "invalid_binding_probe_errors": probe_errors,
        "python_311_compatibility_shim": True,
        "passed": deterministic == len(selected),
    }
    return rows, preflight


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--source", choices=("tool_sandbox", "injecagent", "tau3"), required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--scale", choices=("pilot", "large"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if protocol["status"] not in {
        "preregistered_before_multisource_pilot",
        "pilot_manifests_frozen_before_any_multisource_llm_outcome",
        "pilot_complete_go_large_manifests_frozen_before_large_llm_outcome",
    }:
        raise ValueError("multi-source protocol is not frozen")
    reference_path = ROOT / protocol["agentdojo_llm_reference"]["config"]
    agentdojo_protocol = json.loads(reference_path.read_text(encoding="utf-8"))
    agentdojo_scaffold = agentdojo_protocol["frozen_execution_scaffold"]
    reference = protocol["agentdojo_llm_reference"]
    shared = protocol["shared_llm_contract"]
    same_llm_checks = {}
    for field in protocol["same_llm_gate"]["fields_required_equal_to_agentdojo"]:
        agentdojo_value = agentdojo_scaffold.get(field, reference.get(field))
        if field == "model":
            agentdojo_value = str(agentdojo_value).lower()
            shared_value = str(shared[field]).lower()
        else:
            shared_value = shared[field]
        same_llm_checks[field] = agentdojo_value == shared_value
    if not all(same_llm_checks.values()):
        raise ValueError(f"shared LLM differs from AgentDojo: {same_llm_checks}")
    contract_hash = stable_hash(protocol["shared_llm_contract"])
    source_config = protocol["sources"][args.source]
    expected_commit = source_config["commit"]
    actual_commit = _git_commit(args.source_root)
    if actual_commit != expected_commit:
        raise ValueError(
            f"{args.source} source commit mismatch: {actual_commit} != {expected_commit}"
        )
    builders = {
        "tool_sandbox": build_toolsandbox,
        "injecagent": build_injecagent,
        "tau3": build_tau3,
    }
    rows, execution_preflight = builders[args.source](
        args.source_root, source_config, contract_hash, args.scale
    )
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "protocol_id": protocol["protocol_id"],
        "scale": args.scale,
        "source": args.source,
        "source_commit": actual_commit,
        "llm_contract_sha256": contract_hash,
        "real_external_endpoint_calls": 0,
        "rows": rows,
        "execution_preflight": execution_preflight,
    }
    expected_rows = int(source_config[f"{args.scale}_expected_rows"])
    audit = audit_manifest(
        manifest,
        expected_rows=expected_rows,
        expected_llm_contract_sha256=contract_hash,
    )
    audit["source_execution_preflight"] = execution_preflight
    audit["same_llm_as_agentdojo_checks"] = same_llm_checks
    audit["passed"] = audit["passed"] and execution_preflight["passed"]
    _write(args.output, manifest)
    audit["output_file_sha256"] = __import__("hashlib").sha256(args.output.read_bytes()).hexdigest()
    _write(args.audit, audit)
    print(json.dumps(audit, ensure_ascii=False, sort_keys=True))
    if not audit["passed"]:
        raise SystemExit(f"{args.source} manifest gate failed")


if __name__ == "__main__":
    main()
