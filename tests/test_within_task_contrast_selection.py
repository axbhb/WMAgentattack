import importlib.util
from pathlib import Path


def _module():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "36_select_within_task_contrast.py"
    )
    spec = importlib.util.spec_from_file_location("within_task_contrast", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _candidate(suite, task, injection, risk, view_shift=0.0):
    score = min(0.99, max(0.01, risk + view_shift))
    return {
        "suite": suite,
        "user_task_id": f"user_task_{task}",
        "injection_task_id": f"injection_task_{injection}",
        "attack": "important_instructions_no_model_name",
        "trajectory_id": f"{suite}-{task}-{injection}",
        "target_skill": f"skill_{injection % 3}",
        "source_trace": f"/tmp/{suite}-{task}-{injection}.json",
        "observed_security": injection % 2 == 0,
        "observed_utility": injection % 3 != 0,
        "risk_score": score,
        "rollout_mean_risk_score": score * 0.9,
        "utility_score": 1.0 - score * 0.4,
        "selection_utility_score": 1.0 - score * 0.4,
        "preservation_score": 1.0 - score * 0.5,
        "min_utility_score": 1.0 - score * 0.6,
        "final_utility_score": 1.0 - score * 0.45,
        "value_score": 0.3 + score * 0.2,
        "reward_score": score * 0.1,
        "target_skill_probability": 0.2 + score * 0.3,
        "rollout_mean_target_skill_probability": 0.1 + score * 0.2,
        "rollout_target_reached": float(injection % 2),
        "selection_score": score + 0.5,
    }


def test_frozen_design_selects_balanced_label_blind_contrasts():
    module = _module()
    primary = []
    secondary = []
    for suite_index, suite in enumerate(module.SUITES):
        for task in range(8):
            for injection in range(6):
                risk = 0.05 + 0.1 * task + 0.03 * injection + 0.01 * suite_index
                primary.append(_candidate(suite, task, injection, risk))
                secondary.append(
                    _candidate(suite, task, injection, risk, view_shift=0.02)
                )
    mappings = {
        "primary": {module._pair_key(row): row for row in primary},
        "secondary": {module._pair_key(row): row for row in secondary},
    }
    aggregated = module._aggregate_rows(primary, mappings, {})
    selected, task_metadata = module._select_contrast(
        aggregated,
        tasks_per_suite=6,
        pairs_per_task=4,
    )
    assert len(selected) == 96
    assert len(task_metadata) == 24
    assert {
        suite: sum(row["suite"] == suite for row in task_metadata)
        for suite in module.SUITES
    } == {suite: 6 for suite in module.SUITES}
    assert {
        stratum: sum(row["stratum"] == stratum for row in task_metadata)
        for stratum in (
            "high_score_span",
            "high_model_disagreement",
            "low_score_span_hard_control",
        )
    } == {
        "high_score_span": 8,
        "high_model_disagreement": 8,
        "low_score_span_hard_control": 8,
    }
    assert all(
        key not in row
        for row in selected
        for key in module.REMOVED_LABEL_KEYS
    )
    task_counts = {}
    for row in selected:
        task_counts[module._task_key(row)] = (
            task_counts.get(module._task_key(row), 0) + 1
        )
    assert set(task_counts.values()) == {4}
