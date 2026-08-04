from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "141_summarize_custom_clean_panel_v2.py"
SPEC = importlib.util.spec_from_file_location("custom_panel_v2_summary", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
summary = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(summary)


def test_task_level_balance_gate_uses_each_split_independently():
    thresholds = {
        "training": {"PASS": 8, "FAIL": 8},
        "calibration": {"PASS": 4, "FAIL": 4},
        "confirmation": {"PASS": 4, "FAIL": 4},
    }
    counts = {
        "training": {"PASS": 8, "FAIL": 8},
        "calibration": {"PASS": 4, "FAIL": 4},
        "confirmation": {"PASS": 4, "FAIL": 4},
    }
    passed, conditions = summary._minimum_pass_fail_gate(counts, thresholds)
    assert passed is True
    assert all(conditions.values())
    counts["confirmation"]["FAIL"] = 3
    passed, conditions = summary._minimum_pass_fail_gate(counts, thresholds)
    assert passed is False
    assert conditions["confirmation_FAIL"] is False


def test_sampled_probabilities_are_computed_per_task_not_pooled():
    seeds = (307, 311, 313, 317, 331, 337)
    rows = []
    for index, seed in enumerate(seeds):
        value = "PASS" if index < 4 else "FAIL"
        rows.append(
            {
                "row_id": "clean::banking::user_task_2000",
                "suite": "banking",
                "split": "training",
                "task_difficulty": "L1",
                "run_seed": seed,
                "factorized": {
                    "labels": {
                        "state_action": "N/A",
                        "evidence": value,
                        "report": value,
                        "overall": value,
                    }
                },
            }
        )
    tasks = summary._sampled_task_rows(rows, seeds)
    assert len(tasks) == 1
    assert tasks[0]["complete_six_seeds"] is True
    assert tasks[0]["pass_counts"]["overall"] == 4
    assert tasks[0]["probabilities"]["overall"] == 4 / 6
    assert tasks[0]["overall_interior_probability"] is True


def test_repeated_greedy_labels_are_not_a_sampled_probability_track():
    seeds = (307, 311, 313, 317, 331, 337)
    rows = [
        {
            "row_id": "clean::banking::user_task_2000",
            "suite": "banking",
            "split": "training",
            "task_difficulty": "L1",
            "run_seed": seed,
            "factorized": {
                "labels": {
                    "state_action": "N/A",
                    "evidence": "PASS",
                    "report": "PASS",
                    "overall": "PASS",
                }
            },
        }
        for seed in seeds
    ]
    task = summary._sampled_task_rows(rows, seeds)[0]
    assert task["probabilities"]["overall"] == 1.0
    assert task["overall_interior_probability"] is False


def test_error_marker_scan_is_case_insensitive():
    assert summary._marker_hits("RuntimeError: CUDA allocation failed") == [
        "runtimeerror: cuda"
    ]
    assert summary._marker_hits("normal completion") == []
