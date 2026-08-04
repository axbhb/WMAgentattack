import importlib.util
from pathlib import Path

import pytest


def _module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "27_seed_ensemble_candidates.py"
    spec = importlib.util.spec_from_file_location("seed_ensemble_candidates", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _row(score: float, utility: bool = True):
    return {
        "suite": "workspace",
        "user_task_id": "user_task_0",
        "injection_task_id": "injection_task_0",
        "observed_security": True,
        "observed_utility": utility,
        "candidate_risk_score": score,
        "selection_score": score + 0.25,
        "trajectory_id": f"seed-specific-{score}",
    }


def test_seed_ensemble_averages_only_scores_and_preserves_primary_metadata():
    module = _module()
    output = module._average_candidates([[_row(0.2)], [_row(0.8)]])
    assert output[0]["candidate_risk_score"] == pytest.approx(0.5)
    assert output[0]["selection_score"] == pytest.approx(0.75)
    assert output[0]["trajectory_id"] == "seed-specific-0.2"
    assert output[0]["candidate_ranker_fold"] == -2


def test_seed_ensemble_rejects_label_mismatch():
    module = _module()
    with pytest.raises(ValueError, match="Observed label mismatch"):
        module._average_candidates([[_row(0.2, True)], [_row(0.8, False)]])
