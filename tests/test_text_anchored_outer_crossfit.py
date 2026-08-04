import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "59_evaluate_text_anchored_outer_crossfit.py"
SPEC = importlib.util.spec_from_file_location("text_anchored_outer", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _candidate(index):
    def method(attack, utility):
        return {
            "attack_rank": attack,
            "utility_rank": utility,
            "attack_probability": attack,
            "utility_probability": utility,
        }

    return {
        "suite": "banking",
        "user_task_id": "task0",
        "injection_task_id": f"inj{index}",
        "all_methods": {
            "clean_raw": method(0.1 + index, 0.2 + index),
            "text_pointwise": method(0.3 + index, 0.4 + index),
            "text_borda_alpha_0p25": method(0.5 + index, 0.6 + index),
        },
    }


def test_safe_method_changes_only_attack_ordering():
    candidates = [_candidate(0), _candidate(1)]
    methods = MODULE._safe_methods(
        candidates, "text_borda_alpha_0p25", "text_pointwise"
    )
    safe = methods[MODULE.SAFE_NAME]
    text = methods["text_pointwise"]
    selected = methods["text_borda_alpha_0p25"]
    assert np.array_equal(safe["attack_rank"], selected["attack_rank"])
    assert np.array_equal(safe["utility_rank"], text["utility_rank"])
    assert np.array_equal(safe["attack_probability"], text["attack_probability"])
    assert np.array_equal(safe["utility_probability"], text["utility_probability"])
