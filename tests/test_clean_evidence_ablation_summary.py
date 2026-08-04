import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "123_summarize_clean_evidence_ablation.py"
SPEC = importlib.util.spec_from_file_location("clean_evidence_summary", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_exact_sign_flip_is_two_sided_and_exhaustive():
    assert MODULE._exact_sign_flip([1.0, 1.0, 1.0]) == 0.25


def test_seed_average_preserves_targets_and_averages_predictions():
    base = {
        "variant": "semantic_markov_state_evidence",
        "fold": "fold0",
        "episode_id": "episode",
        "panel": "development",
        "data_seed": 101,
        "task_id": "user_task_1",
        "prefix_index": 1,
        "is_final_prefix": True,
        "progress_target": 0.5,
        "utility_target": 1.0,
    }
    rows = [
        {**base, "training_seed": 7, "progress_prediction": 0.2, "utility_probability": 0.4},
        {**base, "training_seed": 17, "progress_prediction": 0.6, "utility_probability": 0.8},
    ]
    averaged = MODULE._average_training_seeds(rows)
    assert len(averaged) == 1
    assert np.isclose(averaged[0]["progress_prediction"], 0.4)
    assert np.isclose(averaged[0]["utility_probability"], 0.6)


def test_leave_one_task_out_requires_every_omission_to_remain_positive():
    comparison = {
        "task_progress_gains": {"a": 0.2, "b": 0.1, "c": 0.3},
        "task_utility_gains": {"a": 0.1, "b": 0.2, "c": 0.05},
    }
    result = MODULE._leave_one_task_out(comparison)
    assert result["all_progress_positive"]
    assert result["all_utility_positive"]


def test_no_go_never_recommends_an_unaccepted_architecture():
    assert MODULE._preferred_architecture(
        decision="EVIDENCE_LEDGER_NO_INCREMENT_CLEAN_GATE_BLOCKED",
        accepted_decision="EVIDENCE_LEDGER_ARCHITECTURE_SIGNAL_CLEAN_GATE_BLOCKED",
        transformer_superiority=False,
    ) == "none_accepted"
