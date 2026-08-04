import importlib.util
import json
from pathlib import Path

from wmagentattack.structured_ledger_v2 import (
    AdapterMode,
    AdapterSpec,
    AttributeKind,
)


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "127_build_label_blind_ledger_v2_fixtures.py"
SPEC = importlib.util.spec_from_file_location("ledger_v2_fixture_builder", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
fixture_builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fixture_builder)


def _spec(mode: AdapterMode, **kwargs):
    return AdapterSpec(
        family="fixture",
        mode=mode,
        entity_type="fixture_entity",
        **kwargs,
    )


def test_forbidden_outcome_paths_are_recursive_and_label_blind_payload_passes():
    assert fixture_builder._forbidden_paths(
        {"runtime": [{"task_success": True}], "security": False}
    ) == ["runtime.0.task_success", "security"]
    assert fixture_builder._forbidden_paths(
        {
            "runtime_output": {"price": 10},
            "state_transition": {"outcome_labels_present": False},
        }
    ) == []


def test_expected_boundaries_preserve_entity_map_and_name_list_items():
    entity_map = _spec(
        AdapterMode.ENTITY_MAP,
        attribute_name="price",
        attribute_kind=AttributeKind.SINGLE_VALUED,
    )
    name_list = _spec(
        AdapterMode.NAME_LIST_TEXT,
        text_prefix="Hotel Names:",
        attribute_name="available",
        attribute_kind=AttributeKind.SINGLE_VALUED,
    )
    assert fixture_builder._expected_entity_keys(
        entity_map, {"A": 10, "B": 20}, {}
    ) == [{"name": "A"}, {"name": "B"}]
    assert fixture_builder._expected_entity_keys(
        name_list, "Hotel Names: A\nB", {"city": "X"}
    ) == [{"name": "A"}, {"name": "B"}]


def test_expected_boundaries_cover_flights_objects_and_mutations():
    flight = _spec(AdapterMode.FLIGHT_LINES)
    obj = _spec(AdapterMode.OBJECT, entity_key_fields=("id_",))
    mutation = _spec(
        AdapterMode.MUTATION_ACK,
        entity_argument_fields=("hotel",),
    )
    line = (
        "Airline: A, Flight Number: A1, Departure Time: T1, "
        "Arrival Time: T2, Price: 10, Contact Information: C"
    )
    assert fixture_builder._expected_entity_keys(flight, line, {}) == [
        {"airline": "A", "flight_number": "A1"}
    ]
    assert fixture_builder._expected_entity_keys(
        obj, {"id_": "event-1", "title": "Dinner"}, {}
    ) == [{"id_": "event-1"}]
    assert fixture_builder._expected_entity_keys(
        mutation, "Reservation successful", {"hotel": "H"}
    ) == [{"hotel": "H"}]


def test_frozen_gate_contract_uses_positive_boolean_semantics():
    protocol = json.loads(
        (ROOT / "configs" / "0726_structured_ledger_v2_protocol.json").read_text(
            encoding="utf-8"
        )
    )
    gates = protocol["frozen_gates"]
    assert gates
    assert set(gates.values()) == {True}
    assert "exact_state_fingerprint_replay" in gates
    assert not any("mismatch" in name for name in gates)


def test_junit_summary_is_computed_from_executed_test_report(tmp_path):
    report = tmp_path / "tests.xml"
    report.write_text(
        '<testsuites><testsuite tests="21" failures="0" errors="0" '
        'skipped="1"/></testsuites>',
        encoding="utf-8",
    )
    assert fixture_builder._junit_summary(report) == {
        "tests": 21,
        "failures": 0,
        "errors": 0,
        "skipped": 1,
    }


def test_builder_source_does_not_call_outcome_checkers():
    source = SCRIPT_PATH.read_text(encoding="utf-8")
    assert ".utility(" not in source
    assert ".security(" not in source
    assert "run_task(" not in source
