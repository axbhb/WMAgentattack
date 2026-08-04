import importlib.util
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "84_evaluate_v2_configuration_value.py"
)
SPEC = importlib.util.spec_from_file_location("configuration_value", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _row(group_id, task, value, observed):
    return {
        "group_id": group_id,
        "task_key": task,
        "configuration_value_score": value,
        "target_asr": observed / 2,
        "target_bup": observed / 2,
        "observed_asr": observed / 2,
        "observed_bup": observed / 2,
        "trials": 5,
    }


def test_direct_value_selection_is_task_balanced_and_budgeted():
    rows = [
        _row("a", "d|1", 0.2, 0.2),
        _row("b", "d|1", 0.9, 0.9),
        _row("c", "d|2", 0.8, 0.8),
        _row("d", "d|2", 0.1, 0.1),
    ]
    selected = MODULE._select_value(rows, budget_per_task=1)
    assert [row["group_id"] for row in selected] == ["b", "c"]


def test_value_quality_uses_normalized_zero_to_two_brier():
    rows = [
        _row("a", "d|1", 1.0, 0.0),
        _row("b", "d|1", 1.0, 2.0),
    ]
    result = MODULE._value_quality(rows)
    assert result["normalized_brier"] == pytest.approx(0.25)
    assert result["mae"] == pytest.approx(1.0)
