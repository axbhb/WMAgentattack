import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "clean_tool_protocol_comparison_test",
    ROOT / "scripts" / "105_compare_clean_tool_protocols.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_compare_reports_paired_wins_and_retention():
    baseline = {
        (101, "travel", "a"): {"utility": False, "tool_calls": 0},
        (103, "travel", "a"): {"utility": False, "tool_calls": 0},
        (107, "travel", "a"): {"utility": True, "tool_calls": 2},
    }
    candidate = {
        (101, "travel", "a"): {"utility": True, "tool_calls": 2},
        (103, "travel", "a"): {"utility": True, "tool_calls": 3},
        (107, "travel", "a"): {"utility": False, "tool_calls": 1},
    }
    result = MODULE.compare(baseline, candidate, retention_successes=2)
    assert result["episodes"]["candidate_wins"] == 2
    assert result["episodes"]["candidate_losses"] == 1
    assert result["tasks"]["baseline_retained"] == 0
    assert result["tasks"]["candidate_retained"] == 1
    assert result["tool_execution"]["baseline_failures_without_tool_call"] == 2
    assert result["tool_execution"]["candidate_failures_without_tool_call"] == 0
