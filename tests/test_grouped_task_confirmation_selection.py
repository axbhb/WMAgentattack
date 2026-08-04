import importlib.util
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "grouped_confirmation_selector",
    ROOT / "scripts" / "46_select_grouped_task_confirmation.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _row(suite, task_index, injection_index, mode, seed):
    injection_effect = injection_index / 20.0 if mode == "injection" else 0.0
    return {
        "suite": suite,
        "user_task_id": f"task_{task_index}",
        "injection_task_id": f"injection_{injection_index}",
        "attack": "attack",
        "trajectory_id": f"trajectory_{suite}_{task_index}_{injection_index}",
        "target_skill": "target",
        "source_trace": f"/trace/{suite}/{task_index}/{injection_index}.json",
        "risk_score": 0.1 + 0.02 * task_index + injection_effect + seed / 10000.0,
        "utility_score": 0.8 - 0.01 * task_index - injection_effect / 2.0,
        "observed_security": injection_index % 2 == 0,
        "observed_utility": injection_index % 3 == 0,
    }


def test_grouped_confirmation_is_balanced_and_label_blind():
    mappings = {}
    for mode in MODULE.MODES:
        for seed in MODULE.SEEDS:
            rows = [
                _row(suite, task, injection, mode, seed)
                for suite in MODULE.SUITES
                for task in range(3)
                for injection in range(6)
            ]
            mappings[f"{mode}_seed{seed}"] = {
                MODULE._pair_key(row): row for row in rows
            }
    reference = list(mappings["clean_seed7"].values())
    candidates = MODULE._annotate_candidates(reference, mappings)
    selected, metadata = MODULE._select(candidates)

    assert len(selected) == 32
    assert len(metadata) == 8
    assert Counter(row["suite"] for row in metadata) == {
        suite: 2 for suite in MODULE.SUITES
    }
    assert set(Counter(MODULE._task_key(row) for row in selected).values()) == {4}
    assert not any(
        key in row for row in selected for key in MODULE.REMOVED_LABEL_KEYS
    )
    for row in selected:
        assert row["confirmation_predictions"]["dual_view"][
            "attack_probability"
        ] == row["confirmation_predictions"]["injection_view"][
            "attack_probability"
        ]
        assert row["confirmation_predictions"]["dual_view"][
            "utility_probability"
        ] == row["confirmation_predictions"]["clean_view"][
            "utility_probability"
        ]

