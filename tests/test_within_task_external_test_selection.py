import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "external_selector",
    ROOT / "scripts" / "42_select_within_task_external_test.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _candidate(suite, task, injection, value):
    return {
        "suite": suite,
        "user_task_id": task,
        "injection_task_id": injection,
        "target_skill": injection,
        "contrast_risk_score_mean": value,
        "contrast_final_utility_score_mean": 1.0 - value,
        "contrast_preservation_score_mean": 0.5 * value,
        "contrast_target_skill_probability_mean": value,
        "contrast_risk_score_std": value / 10.0,
        "contrast_final_utility_score_std": value / 20.0,
        "contrast_preservation_score_std": value / 30.0,
    }


def test_external_selection_is_suite_balanced_and_chunk_balanced():
    candidates = []
    for suite in MODULE.CONTRAST.SUITES:
        for task_index in range(5):
            for injection_index in range(2):
                value = 0.1 * task_index + 0.05 * injection_index
                candidates.append(
                    _candidate(
                        suite,
                        f"task_{task_index}",
                        f"injection_{injection_index}",
                        value,
                    )
                )
    excluded = {(suite, "task_0") for suite in MODULE.CONTRAST.SUITES}
    selected, metadata = MODULE._select_external(candidates, excluded)
    assert len(selected) == 16
    assert len(metadata) == 8
    for suite in MODULE.CONTRAST.SUITES:
        assert sum(row["suite"] == suite for row in metadata) == 2
    assert not ({(row["suite"], row["user_task_id"]) for row in selected} & excluded)
    assert {
        chunk: sum(row["contrast_replay_chunk"] == chunk for row in selected)
        for chunk in range(4)
    } == {0: 4, 1: 4, 2: 4, 3: 4}
