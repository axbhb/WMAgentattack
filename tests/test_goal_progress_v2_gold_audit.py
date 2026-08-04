import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "129_audit_goal_progress_v2_gold.py"
PROTOCOL = ROOT / "configs" / "0726_goal_progress_v2_gold_protocol.json"
REGISTRY = ROOT / "configs" / "0726_travel_tool_output_adapter_registry_v2.json"
SPEC = importlib.util.spec_from_file_location("goal_gold_audit", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
goal_gold_audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(goal_gold_audit)


def test_goal_gold_protocol_is_preregistered_and_label_blind():
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    assert protocol["status"] == "preregistered_before_execution"
    assert set(protocol["frozen_gates"].values()) == {True}
    assert goal_gold_audit._forbidden_paths(protocol["cases"]) == []
    assert protocol["data_policy"]["expert_trajectory_read"] is False
    assert protocol["data_policy"]["future_calls_read"] is False
    assert protocol["data_policy"]["outcome_labels_read"] is False


def test_goal_gold_protocol_exercises_all_statuses_and_progress_types():
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    statuses = {
        value
        for case in protocol["cases"]
        for value in case["expected_statuses"].values()
    }
    progress_types = {
        atom["progress_type"]
        for case in protocol["cases"]
        for atom in case["atoms"]
    }
    assert statuses == {
        "UNSUPPORTED",
        "PARTIALLY_SUPPORTED",
        "SUPPORTED",
        "CONTRADICTED",
        "AMBIGUOUS",
    }
    assert progress_types == {"KNOWLEDGE", "ENVIRONMENT", "COMPLETION"}


def test_complete_goal_gold_audit_passes(tmp_path):
    junit = tmp_path / "tests.xml"
    junit.write_text(
        '<testsuites><testsuite tests="1" failures="0" errors="0" skipped="0"/>'
        "</testsuites>",
        encoding="utf-8",
    )
    output = tmp_path / "audit"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--protocol",
            str(PROTOCOL),
            "--registry",
            str(REGISTRY),
            "--test-junit",
            str(junit),
            "--output-dir",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
    )
    assert result.returncode == 0, result.stderr
    audit = json.loads((output / "audit.json").read_text(encoding="utf-8"))
    assert audit["decision"] == "GOAL_PROGRESS_V2_GOLD_GATE_PASS"
    assert all(audit["gates"].values())
    assert audit["counts"]["mismatches"] == 0


def test_goal_gold_audit_source_has_no_checker_or_model_calls():
    source = SCRIPT.read_text(encoding="utf-8")
    assert ".utility(" not in source
    assert ".security(" not in source
    assert "ground_truth(" not in source
    assert "run_task(" not in source
    assert "transformers" not in source
