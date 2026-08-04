import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "remaining_selection",
    ROOT / "scripts" / "52_select_headwise_remaining_confirmation.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _row(injection, world, text, utility):
    method = lambda ar, ur, ap, up: {
        "attack_rank": ar,
        "utility_rank": ur,
        "attack_probability": ap,
        "utility_probability": up,
    }
    return {
        "suite": "banking",
        "user_task_id": "task",
        "injection_task_id": injection,
        "attack": "important_instructions_no_model_name",
        "trajectory_id": injection,
        "source_trace": f"/{injection}.json",
        "all_methods": {
            "world_pairwise_c0p03": method(world, utility, 0.4, 0.5),
            "text_pointwise": method(text, text, 0.3 + text / 10, 0.6),
            "clean_raw": method(0.2, utility, 0.2, 0.7),
        },
    }


def test_headwise_formula_uses_world_0p75_and_text_0p25():
    rows = [
        _row("a", 0.1, 0.4, 0.2),
        _row("b", 0.2, 0.3, 0.3),
        _row("c", 0.3, 0.2, 0.4),
        _row("d", 0.4, 0.1, 0.5),
    ]
    annotated = MODULE._annotate_task(rows)
    values = np.asarray(
        [
            row["confirmation_models"][MODULE.SELECTED_MODEL]["attack_rank"]
            for row in annotated
        ]
    )
    assert np.allclose(values, [0.25, 5 / 12, 7 / 12, 0.75])


def test_farthest_coverage_returns_four_unique_pairs():
    rows = MODULE._annotate_task(
        [_row(str(index), index, 6 - index, index / 10) for index in range(6)]
    )
    chosen = MODULE._farthest_coverage(rows)
    assert len(chosen) == 4
    assert len({row["injection_task_id"] for row in chosen}) == 4
