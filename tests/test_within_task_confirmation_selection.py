import importlib.util
from pathlib import Path


def _module():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "39_select_within_task_confirmation.py"
    )
    spec = importlib.util.spec_from_file_location("confirmation_selector", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_confirmation_tasks_are_balanced_and_disjoint():
    module = _module()
    candidates = []
    excluded = set()
    for suite_index, suite in enumerate(module.CONTRAST.SUITES):
        excluded.add((suite, "user_task_0"))
        for task in range(7):
            for injection in range(4):
                risk = 0.05 * (task + injection + suite_index)
                candidates.append(
                    {
                        "suite": suite,
                        "user_task_id": f"user_task_{task}",
                        "injection_task_id": f"injection_task_{injection}",
                        "target_skill": f"skill_{injection}",
                        "contrast_risk_score_mean": risk,
                        "contrast_final_utility_score_mean": 1 - risk,
                        "contrast_preservation_score_mean": 0.8 - risk / 2,
                        "contrast_target_skill_probability_mean": risk / 2,
                        "contrast_risk_score_std": task / 100,
                        "contrast_final_utility_score_std": task / 200,
                        "contrast_preservation_score_std": task / 300,
                    }
                )
    selected, metadata = module._select_confirmation(candidates, excluded)
    assert len(selected) == 32
    assert len(metadata) == 8
    assert all(module.CONTRAST._task_key(row) not in excluded for row in selected)
    assert {
        suite: sum(row["suite"] == suite for row in metadata)
        for suite in module.CONTRAST.SUITES
    } == {suite: 2 for suite in module.CONTRAST.SUITES}
