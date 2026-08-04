import importlib.util
from pathlib import Path


def _module():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "31_select_validation_probability_pilot.py"
    )
    spec = importlib.util.spec_from_file_location("validation_probability_pilot", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_add_stratum_is_disjoint_and_enforces_task_cap():
    module = _module()
    rows = [
        {
            "suite": "workspace",
            "user_task_id": f"user_task_{index // 3}",
            "injection_task_id": f"injection_task_{index}",
        }
        for index in range(12)
    ]
    output = []
    module._add_stratum(
        output,
        rows,
        name="a",
        quota=6,
        max_per_user_task=2,
    )
    assert len(output) == 6
    task_counts = {}
    for row in output:
        key = (row["suite"], row["user_task_id"])
        task_counts[key] = task_counts.get(key, 0) + 1
    assert max(task_counts.values()) == 2
    assert all(row["pilot_stratum"] == "a" for row in output)


def test_uncertainty_score_prefers_mid_probability_and_disagreement():
    module = _module()
    certain = {
        "candidate_risk_score": 0.99,
        "candidate_expected_utility_score": 0.99,
        "risk_score": 0.99,
        "final_utility_score": 0.99,
    }
    uncertain = {
        "candidate_risk_score": 0.5,
        "candidate_expected_utility_score": 0.5,
        "risk_score": 0.8,
        "final_utility_score": 0.2,
    }
    assert module._uncertainty_score(uncertain) > module._uncertainty_score(certain)
