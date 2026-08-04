from __future__ import annotations

import importlib.util
from pathlib import Path

from agentdojo.functions_runtime import FunctionCall
from agentdojo.task_suite.load_suites import get_suite


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "137_audit_custom_clean_panel_counterevidence.py"
SPEC = importlib.util.spec_from_file_location("custom_panel_counterevidence", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_runtime_schema_normalization_coerces_numeric_string() -> None:
    suite = get_suite("v1.2.2", "banking")
    tools = {tool.name: tool for tool in suite.tools}
    normalized, error = MODULE.normalize_call(
        FunctionCall(function="get_most_recent_transactions", args={"n": "100"}),
        tools,
    )
    assert error is None
    assert normalized.args == {"n": 100}


def test_counterevidence_categories_do_not_overwrite_frozen_success() -> None:
    assert (
        MODULE.classify_episode(
            strict_success=True,
            normalized_behavior_complete=False,
            output_fragments_complete=False,
        )
        == "frozen_success"
    )
    assert (
        MODULE.classify_episode(
            strict_success=False,
            normalized_behavior_complete=True,
            output_fragments_complete=True,
        )
        == "schema_coercion_false_negative_candidate"
    )
    assert (
        MODULE.classify_episode(
            strict_success=False,
            normalized_behavior_complete=True,
            output_fragments_complete=False,
        )
        == "lexical_reporting_false_negative_candidate"
    )
    assert (
        MODULE.classify_episode(
            strict_success=False,
            normalized_behavior_complete=False,
            output_fragments_complete=True,
        )
        == "model_or_evidence_failure"
    )
