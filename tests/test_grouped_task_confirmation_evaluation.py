import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "grouped_confirmation_evaluator",
    ROOT / "scripts" / "47_evaluate_grouped_task_confirmation.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _seed_result(value):
    return {
        "comparisons": {
            "dual_view__minus__clean_view": {
                "pairwise_accuracy_difference": value
            }
        }
    }


def test_confirmation_gate_separates_partial_and_strong_evidence():
    thresholds = {
        "pairwise_point_difference_min": 0.03,
        "pairwise_ci_lower_min": -0.02,
        "brier_difference_max": 0.01,
        "informative_task_count_min": 4,
        "strong_informative_task_count_min": 6,
        "positive_seed_count_min_secondary": 2,
    }
    per_seed = {
        "7": _seed_result(0.1),
        "13": _seed_result(-0.01),
        "21": _seed_result(0.05),
    }
    partial = MODULE._gate(
        {
            "pairwise_accuracy_difference": 0.08,
            "pairwise_accuracy_difference_95ci": [-0.01, 0.2],
            "brier_difference": 0.005,
            "informative_pairwise_task_count": 6,
        },
        thresholds,
        per_seed,
    )
    assert partial["decision"] == "PARTIAL_GO"
    assert partial["seed_robustness_secondary_pass"] is True

    strong = MODULE._gate(
        {
            "pairwise_accuracy_difference": 0.08,
            "pairwise_accuracy_difference_95ci": [0.01, 0.2],
            "brier_difference": 0.005,
            "informative_pairwise_task_count": 6,
        },
        thresholds,
        per_seed,
    )
    assert strong["decision"] == "CONFIRMED_STRONG"

