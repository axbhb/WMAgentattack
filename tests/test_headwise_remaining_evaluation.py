import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "remaining_evaluation",
    ROOT / "scripts" / "53_evaluate_headwise_remaining_confirmation.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_gate_requires_both_clean_and_text_comparisons():
    selected = "headwise"
    comparison = {
        "pairwise_accuracy_difference": 0.05,
        "pairwise_accuracy_difference_95ci": [-0.01, 0.1],
        "brier_difference": 0.0,
        "informative_pairwise_task_count": 6,
    }
    results = {
        selected: {
            "attack": {"within_task": {"pairwise_accuracy": 0.8}},
            "utility": {"within_task": {"pairwise_accuracy": 0.7}},
        },
        "text_pointwise": {
            "attack": {"within_task": {"pairwise_accuracy": 0.7}},
        },
        "clean_raw": {
            "utility": {"within_task": {"pairwise_accuracy": 0.7}},
        },
    }
    comparisons = {
        f"{selected}__minus__clean_raw": dict(comparison),
        f"{selected}__minus__text_pointwise": dict(comparison),
    }
    thresholds = {
        "pairwise_point_difference_min": 0.03,
        "pairwise_ci_lower_min": -0.02,
        "brier_difference_max": 0.01,
        "informative_task_count_min": 5,
    }
    assert MODULE._gate(comparisons, results, selected, thresholds)[
        "decision"
    ] == "GO"
    comparisons[f"{selected}__minus__text_pointwise"][
        "pairwise_accuracy_difference"
    ] = 0.0
    assert MODULE._gate(comparisons, results, selected, thresholds)[
        "decision"
    ] == "NO_GO"
