"""Build label-blind Travel runtime fixtures and audit structured ledger v2."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping
from xml.etree import ElementTree


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wmagentattack.clean_state_instrumentation import (
    canonical_call_signature,
    instrument_function_call,
)
from wmagentattack.decision_state import canonical_json_value
from wmagentattack.state_storage_v2 import (
    ContentAddressedStateStore,
    ModelTower,
    build_exact_state_transition,
)
from wmagentattack.structured_ledger_v2 import (
    AdapterMode,
    AdapterRegistry,
    ExecutionChannelStatus,
    StructuredEvidenceLedgerV2,
    load_adapter_registry,
    update_structured_ledger,
)


FORBIDDEN_OUTCOME_KEYS = {
    "utility",
    "security",
    "task_success",
    "attack_success",
    "expert_slot_coverage",
    "final_answer",
    "ground_truth_answer",
}


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def _junit_summary(path: Path) -> dict[str, int]:
    root = ElementTree.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    if not suites:
        raise ValueError("JUnit report contains no test suites")
    return {
        field: sum(int(suite.attrib.get(field, 0)) for suite in suites)
        for field in ("tests", "failures", "errors", "skipped")
    }


def _forbidden_paths(value: Any, path: tuple[str, ...] = ()) -> list[str]:
    rows = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            child = (*path, str(key))
            if str(key).lower() in FORBIDDEN_OUTCOME_KEYS:
                rows.append(".".join(child))
            rows.extend(_forbidden_paths(item, child))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            rows.extend(_forbidden_paths(item, (*path, str(index))))
    return rows


def _name_list_values(output: str) -> list[str]:
    if not output.strip():
        return []
    first, *remaining = output.splitlines()
    if ":" not in first:
        raise ValueError("fixture name list has no separator")
    first_name = first.split(":", 1)[1].strip()
    return [name for name in [first_name, *(row.strip() for row in remaining)] if name]


def _expected_entity_keys(spec, output: Any, arguments: Mapping[str, Any]) -> list[dict[str, Any]]:
    canonical = canonical_json_value(output)
    if spec.mode == AdapterMode.USER_FIELDS:
        return [canonical_json_value(spec.fixed_entity_key)]
    if spec.mode == AdapterMode.NAME_LIST_TEXT:
        return [{"name": name} for name in _name_list_values(canonical)]
    if spec.mode in {
        AdapterMode.ENTITY_MAP,
        AdapterMode.PRICE_RANGE_MAP,
        AdapterMode.RATING_REVIEWS_MAP,
    }:
        if not isinstance(canonical, Mapping):
            raise TypeError("expected mapping output")
        return [{"name": str(name)} for name in canonical]
    if spec.mode == AdapterMode.FLIGHT_LINES:
        rows = []
        for line in str(canonical).splitlines():
            fields = {}
            for part in line.split(", "):
                if ": " in part:
                    key, value = part.split(": ", 1)
                    fields[key] = value
            rows.append(
                {
                    "airline": fields["Airline"],
                    "flight_number": fields["Flight Number"],
                }
            )
        return rows
    if spec.mode == AdapterMode.OBJECT:
        if not isinstance(canonical, Mapping):
            raise TypeError("expected object output")
        return [
            {field: canonical[field] for field in spec.entity_key_fields if field in canonical}
        ]
    if spec.mode == AdapterMode.OBJECT_LIST:
        if not isinstance(canonical, list):
            raise TypeError("expected object-list output")
        return [
            {field: row[field] for field in spec.entity_key_fields if field in row}
            for row in canonical
        ]
    if spec.mode == AdapterMode.MUTATION_ACK:
        return [
            {
                field: canonical_json_value(arguments[field])
                for field in spec.entity_argument_fields
            }
        ]
    raise AssertionError(spec.mode)


def _synthetic_calls(environment: Any) -> dict[str, dict[str, Any]]:
    state = canonical_json_value(environment)
    company = state["car_rental"]["company_list"][0]["name"]
    restaurant = state["restaurants"]["restaurant_list"][0]["name"]
    event = state["calendar"]["initial_events"][0]
    day = str(event["start_time"]).split("T", 1)[0]
    return {
        "cancel_calendar_event": {"event_id": event["id_"]},
        "get_car_rental_address": {"company_name": [company]},
        "get_contact_information_for_restaurants": {
            "restaurant_names": [restaurant]
        },
        "get_day_calendar_events": {"day": day},
        "get_user_information": {},
        "reserve_car_rental": {
            "company": company,
            "start_time": "2024-05-16 09:00",
            "end_time": "2024-05-17 09:00",
        },
        "reserve_restaurant": {
            "restaurant": restaurant,
            "start_time": "2024-05-16 18:00",
        },
        "search_calendar_events": {"query": "Team", "date": day},
    }


def _mechanism_probes(registry: AdapterRegistry) -> dict[str, bool]:
    base = StructuredEvidenceLedgerV2()
    different = update_structured_ledger(
        base,
        registry,
        episode_id="mechanism",
        call_index=0,
        channel_status=ExecutionChannelStatus.EXECUTED_SUCCESS,
        tool_name="get_hotels_prices",
        arguments={"hotel_names": ["A", "B"]},
        runtime_output={
            "A": "Price range: 100 - 120",
            "B": "Price range: 200 - 220",
        },
    ).ledger
    first_price = update_structured_ledger(
        base,
        registry,
        episode_id="same-entity",
        call_index=0,
        channel_status=ExecutionChannelStatus.EXECUTED_SUCCESS,
        tool_name="get_price_for_restaurants",
        arguments={"restaurant_names": ["R"]},
        runtime_output={"R": 10.0},
    ).ledger
    conflicting = update_structured_ledger(
        first_price,
        registry,
        episode_id="same-entity",
        call_index=1,
        channel_status=ExecutionChannelStatus.EXECUTED_SUCCESS,
        tool_name="get_price_for_restaurants",
        arguments={"restaurant_names": ["R"]},
        runtime_output={"R": 20.0},
    ).ledger
    first_set = update_structured_ledger(
        base,
        registry,
        episode_id="set-valued",
        call_index=0,
        channel_status=ExecutionChannelStatus.EXECUTED_SUCCESS,
        tool_name="get_car_types_available",
        arguments={"company_name": ["C"]},
        runtime_output={"C": ["SUV"]},
    ).ledger
    changed_set = update_structured_ledger(
        first_set,
        registry,
        episode_id="set-valued",
        call_index=1,
        channel_status=ExecutionChannelStatus.EXECUTED_SUCCESS,
        tool_name="get_car_types_available",
        arguments={"company_name": ["C"]},
        runtime_output={"C": ["Sedan"]},
    ).ledger
    return {
        "different_entities_do_not_conflict": not different.conflicts,
        "same_entity_attribute_context_incompatible_values_conflict": (
            len(conflicting.conflicts) == 1
        ),
        "set_valued_members_do_not_conflict": not changed_set.conflicts,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--test-junit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if protocol.get("status") != "preregistered_before_execution":
        raise ValueError("ledger protocol is not preregistered")
    registry = load_adapter_registry(args.registry)
    test_summary = _junit_summary(args.test_junit)

    from agentdojo.functions_runtime import FunctionCall, FunctionsRuntime
    from agentdojo.task_suite.load_suites import get_suite

    suite = get_suite(protocol["benchmark_version"], protocol["suite"])
    runtime_tools = {tool.name for tool in suite.tools}
    if runtime_tools != set(registry.adapters):
        raise ValueError("adapter registry does not exactly cover suite tools")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    store = ContentAddressedStateStore(args.output_dir / "state_store")
    fixture_rows = []
    observed_tools = set()
    boundary_checks = []
    idempotence_checks = []
    state_replay_checks = []

    def process_call(
        *,
        source: str,
        episode_id: str,
        call_index: int,
        runtime: Any,
        environment: Any,
        initial_state: Any,
        function: str,
        arguments: Mapping[str, Any],
        ledger: StructuredEvidenceLedgerV2,
    ) -> StructuredEvidenceLedgerV2:
        transition, output = instrument_function_call(
            runtime,
            environment,
            event_index=call_index,
            function=function,
            arguments=arguments,
        )
        state_record = build_exact_state_transition(
            store,
            episode_id=episode_id,
            call_index=call_index,
            initial_state=initial_state,
            state_before=transition.canonical_state_before,
            state_after=transition.canonical_state_after,
            exact_delta=transition.canonical_state_delta,
            execution_status=transition.tool_execution_status,
            error_type=transition.tool_error_type,
        )
        state_replay_checks.append(
            store.get(
                state_record.state_before_ref, requesting_tower=ModelTower.SIMULATOR
            )
            == transition.canonical_state_before
            and store.get(
                state_record.state_after_ref, requesting_tower=ModelTower.SIMULATOR
            )
            == transition.canonical_state_after
        )
        status = (
            ExecutionChannelStatus.EXECUTED_ERROR
            if transition.tool_execution_status == "error"
            else ExecutionChannelStatus.EXECUTED_SUCCESS
        )
        result = update_structured_ledger(
            ledger,
            registry,
            episode_id=episode_id,
            call_index=call_index,
            channel_status=status,
            tool_name=function,
            arguments=arguments,
            runtime_output=output,
            error_type=transition.tool_error_type,
            state_changed=transition.state_changed,
            proposal_signature=canonical_call_signature(function, arguments),
        )
        replay = update_structured_ledger(
            result.ledger,
            registry,
            episode_id=episode_id,
            call_index=call_index,
            channel_status=status,
            tool_name=function,
            arguments=arguments,
            runtime_output=output,
            error_type=transition.tool_error_type,
            state_changed=transition.state_changed,
            proposal_signature=canonical_call_signature(function, arguments),
        )
        idempotence_checks.append(replay.ledger == result.ledger)
        additions = result.ledger.records[len(ledger.records) :]
        if status == ExecutionChannelStatus.EXECUTED_SUCCESS:
            expected = _expected_entity_keys(
                registry.adapters[function], output, arguments
            )
            actual = [record.entity_key for record in additions]
            boundary_checks.append(
                sorted(json.dumps(row, sort_keys=True) for row in expected)
                == sorted(json.dumps(row, sort_keys=True) for row in actual)
            )
        else:
            boundary_checks.append(
                len(additions) == 1 and additions[0].execution_status == "error"
            )
        fixture = {
            "source": source,
            "episode_id": episode_id,
            "call_index": call_index,
            "tool_name": function,
            "arguments": canonical_json_value(dict(arguments)),
            "execution_status": transition.tool_execution_status,
            "error_type": transition.tool_error_type,
            "runtime_output": canonical_json_value(output),
            "state_transition": state_record.model_dump(mode="json"),
            "structured_records": [record.model_dump(mode="json") for record in additions],
            "new_conflicts": [
                conflict.model_dump(mode="json")
                for conflict in result.ledger.conflicts[len(ledger.conflicts) :]
            ],
        }
        if _forbidden_paths(fixture):
            raise ValueError(f"outcome-like fields leaked into fixture: {_forbidden_paths(fixture)}")
        fixture_rows.append(fixture)
        observed_tools.add(function)
        return result.ledger

    expert_call_count = 0
    for task_id in sorted(suite.user_tasks):
        task = suite.get_user_task_by_id(task_id)
        environment = task.init_environment(suite.load_and_inject_default_environment({}))
        initial_state = canonical_json_value(environment)
        runtime = FunctionsRuntime(suite.tools)
        calls = list(task.ground_truth(environment.model_copy(deep=True)))
        ledger = StructuredEvidenceLedgerV2()
        for call_index, call in enumerate(calls):
            ledger = process_call(
                source="expert_structure_fixture",
                episode_id=f"expert::{task_id}",
                call_index=call_index,
                runtime=runtime,
                environment=environment,
                initial_state=initial_state,
                function=call.function,
                arguments=call.args,
                ledger=ledger,
            )
            expert_call_count += 1

    default_environment = suite.load_and_inject_default_environment({})
    missing_calls = _synthetic_calls(default_environment)
    expected_missing = runtime_tools - observed_tools
    if set(missing_calls) != expected_missing:
        raise ValueError(
            f"synthetic fixture calls differ from uncovered tools: {sorted(expected_missing)}"
        )
    for tool_name, arguments in sorted(missing_calls.items()):
        environment = suite.load_and_inject_default_environment({})
        process_call(
            source="synthetic_missing_tool_fixture",
            episode_id=f"synthetic::{tool_name}",
            call_index=0,
            runtime=FunctionsRuntime(suite.tools),
            environment=environment,
            initial_state=canonical_json_value(environment),
            function=tool_name,
            arguments=arguments,
            ledger=StructuredEvidenceLedgerV2(),
        )

    error_environment = suite.load_and_inject_default_environment({})
    process_call(
        source="synthetic_error_fixture",
        episode_id="synthetic::execution_error",
        call_index=0,
        runtime=FunctionsRuntime(suite.tools),
        environment=error_environment,
        initial_state=canonical_json_value(error_environment),
        function="get_hotels_prices",
        arguments={"hotel_names": 123},
        ledger=StructuredEvidenceLedgerV2(),
    )

    empty = StructuredEvidenceLedgerV2()
    no_update_counts = {}
    for status in (
        ExecutionChannelStatus.PROPOSED,
        ExecutionChannelStatus.TERMINAL_UNEXECUTED,
        ExecutionChannelStatus.CENSORED,
    ):
        result = update_structured_ledger(
            empty,
            registry,
            episode_id="channel-fixture",
            call_index=0,
            channel_status=status,
            tool_name="get_hotels_prices",
            arguments={"hotel_names": ["Hotel A"]},
            runtime_output={"Hotel A": "Price range: 1 - 2"},
        )
        no_update_counts[status.value] = len(result.ledger.records)

    mechanism = _mechanism_probes(registry)
    records = [record for fixture in fixture_rows for record in fixture["structured_records"]]
    conflicts = [row for fixture in fixture_rows for row in fixture["new_conflicts"]]
    unlinked_conflicts = sum(
        conflict["entity_id"].startswith("PROVISIONAL::") for conflict in conflicts
    )
    gates = {
        "registry_covers_all_suite_tools": observed_tools == runtime_tools,
        "record_boundaries_exact": all(boundary_checks),
        "deterministic_replay": all(idempotence_checks),
        "same_record_replay_idempotent": all(idempotence_checks),
        "no_future_or_outcome_fields": sum(
            len(_forbidden_paths(row)) for row in fixture_rows
        )
        == 0,
        "zero_terminal_unexecuted_updates": no_update_counts[
            ExecutionChannelStatus.TERMINAL_UNEXECUTED.value
        ]
        == 0,
        "zero_unlinked_derived_conflicts": unlinked_conflicts == 0,
        **mechanism,
        "all_records_victim_observed": all(
            record["observation_scope"] == "VICTIM_OBSERVED" for record in records
        ),
        "exact_state_fingerprint_replay": all(state_replay_checks),
        "tests_pass": (
            test_summary["tests"] > 0
            and test_summary["failures"] == 0
            and test_summary["errors"] == 0
        ),
    }
    frozen_gates = protocol["frozen_gates"]
    if frozen_gates != {name: True for name in gates}:
        raise ValueError("runtime gates differ from the frozen protocol")
    decision = (
        protocol["pass_decision"] if all(gates.values()) else protocol["failure_decision"]
    )
    fixture_path = args.output_dir / "fixtures.jsonl"
    with fixture_path.open("w", encoding="utf-8") as handle:
        for fixture in fixture_rows:
            handle.write(json.dumps(fixture, ensure_ascii=False) + "\n")
    audit = {
        "protocol_id": protocol["protocol_id"],
        "decision": decision,
        "gates": gates,
        "counts": {
            "suite_tools": len(runtime_tools),
            "tools_observed": len(observed_tools),
            "expert_tasks": len(suite.user_tasks),
            "expert_calls": expert_call_count,
            "synthetic_missing_tool_calls": len(missing_calls),
            "synthetic_error_calls": 1,
            "fixture_calls": len(fixture_rows),
            "structured_records": len(records),
            "conflicts": len(conflicts),
            "unlinked_derived_conflicts": unlinked_conflicts,
            "state_blobs": len(list((args.output_dir / "state_store" / "blobs").rglob("*.json"))),
        },
        "tool_fixture_counts": {
            tool: sum(row["tool_name"] == tool for row in fixture_rows)
            for tool in sorted(runtime_tools)
        },
        "nonexecuted_update_counts": no_update_counts,
        "test_summary": test_summary,
        "mechanism_probes": mechanism,
        "safety": {
            "outcome_labels_read": False,
            "utility_checker_called": False,
            "security_checker_called": False,
            "victim_model_calls": 0,
            "attacks": 0,
            "dreamer_training": False,
            "old_90_model_comparison": False,
        },
    }
    _write_json(args.output_dir / "audit.json", audit)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
