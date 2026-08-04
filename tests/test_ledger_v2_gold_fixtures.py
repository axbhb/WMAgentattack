import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

from wmagentattack.structured_ledger_v2 import (
    ExecutionChannelStatus,
    StructuredEvidenceLedgerV2,
    load_adapter_registry,
    update_structured_ledger,
)


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "128_audit_ledger_v2_gold_fixtures.py"
FIXTURES = ROOT / "configs" / "0726_travel_ledger_v2_gold_fixtures.json"
REGISTRY_PATH = ROOT / "configs" / "0726_travel_tool_output_adapter_registry_v2.json"
SPEC = importlib.util.spec_from_file_location("gold_audit", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
gold_audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gold_audit)


def test_gold_fixture_protocol_is_label_blind_and_fully_frozen():
    frozen = json.loads(FIXTURES.read_text(encoding="utf-8"))
    assert frozen["status"] == "preregistered_before_execution"
    assert set(frozen["frozen_gates"].values()) == {True}
    assert gold_audit._forbidden_paths(
        {"cases": frozen["cases"], "linkage_cases": frozen["linkage_cases"]}
    ) == []


def test_gold_tools_cover_every_registry_family_and_adapter_mode():
    frozen = json.loads(FIXTURES.read_text(encoding="utf-8"))
    registry = load_adapter_registry(REGISTRY_PATH)
    tools = {step["tool"] for case in frozen["cases"] for step in case["steps"]}
    assert {registry.adapters[tool].family for tool in tools} == set(
        frozen["expected_registry_families"]
    )
    assert {registry.adapters[tool].mode.value for tool in tools} == set(
        frozen["expected_adapter_modes"]
    )


def test_record_projection_matches_fixed_hotel_boundary_expectation():
    registry = load_adapter_registry(REGISTRY_PATH)
    result = update_structured_ledger(
        StructuredEvidenceLedgerV2(),
        registry,
        episode_id="projection",
        call_index=0,
        channel_status=ExecutionChannelStatus.EXECUTED_SUCCESS,
        tool_name="get_all_hotels_in_city",
        arguments={"city": "Paris"},
        runtime_output="Hotel Names: Hotel A\nHotel B",
    )
    projection = [gold_audit._record_projection(row) for row in result.ledger.records]
    frozen = json.loads(FIXTURES.read_text(encoding="utf-8"))
    expected = next(
        case for case in frozen["cases"] if case["case_id"] == "hotel_multi_record_modes"
    )["steps"][0]["expected_records"]
    assert projection == [gold_audit._canonical_record_projection(row) for row in expected]


def test_complete_gold_audit_passes_against_manual_expectations(tmp_path):
    junit = tmp_path / "tests.xml"
    junit.write_text(
        '<testsuites><testsuite tests="1" failures="0" errors="0" skipped="0"/>'
        "</testsuites>",
        encoding="utf-8",
    )
    output = tmp_path / "audit"
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--fixtures",
            str(FIXTURES),
            "--registry",
            str(REGISTRY_PATH),
            "--test-junit",
            str(junit),
            "--output-dir",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, result.stderr
    audit = json.loads((output / "audit.json").read_text(encoding="utf-8"))
    assert audit["decision"] == "LEDGER_V2_EXTRACTOR_GOLD_GATE_PASS"
    assert all(audit["gates"].values())
    assert audit["counts"]["mismatches"] == 0


def test_gold_audit_source_does_not_call_outcome_or_model_runners():
    source = SCRIPT.read_text(encoding="utf-8")
    assert ".utility(" not in source
    assert ".security(" not in source
    assert "run_task(" not in source
    assert "transformers" not in source
