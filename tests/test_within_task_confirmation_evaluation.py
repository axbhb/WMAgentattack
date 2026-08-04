import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "confirmation_evaluation",
    ROOT / "scripts" / "41_evaluate_within_task_confirmation.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_replication_status_is_predeclared_and_conservative():
    comparison = {
        "pairwise_accuracy_difference": 0.04,
        "pairwise_accuracy_difference_95ci": [-0.01, 0.09],
        "brier_difference": 0.005,
    }
    status = MODULE._replication_status(comparison)
    assert status["directional_replication"] is True
    assert status["strong_replication"] is False

    comparison["brier_difference"] = 0.02
    assert MODULE._replication_status(comparison)["claim_supported"] is False
