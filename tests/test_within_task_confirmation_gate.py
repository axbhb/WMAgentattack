import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "confirmation_gate",
    ROOT / "scripts" / "40_apply_within_task_confirmation_gate.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _protocol():
    return {
        "model_selection": {"comparator": "clean_world_ridge"},
        "go_criteria_all_required": {
            "minimum_pairwise_accuracy_difference": 0.03,
            "minimum_pairwise_accuracy_difference_95ci_lower": -0.02,
            "maximum_brier_difference": 0.01,
        },
        "decision": {"go": "confirm", "no_go": "stop"},
    }


def _summary(pairwise=0.04, lower=-0.01, brier=0.005):
    return {
        "selected_model": "hierarchical_beta_binomial",
        "comparisons": {
            "hierarchical_beta_binomial__minus__clean_world_ridge": {
                "pairwise_accuracy_difference": pairwise,
                "pairwise_accuracy_difference_95ci": [lower, 0.11],
                "brier_difference": brier,
            }
        },
    }


def test_gate_requires_every_predeclared_criterion():
    passed = MODULE.apply_gate(_summary(), _protocol())
    assert passed["go"] is True
    assert passed["decision"] == "GO"

    failed = MODULE.apply_gate(_summary(lower=-0.03), _protocol())
    assert failed["go"] is False
    assert failed["checks"]["pairwise_uncertainty_bound"]["passed"] is False
