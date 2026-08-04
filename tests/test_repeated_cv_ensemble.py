import importlib.util
from pathlib import Path


def _module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "28_repeated_cv_ensemble.py"
    spec = importlib.util.spec_from_file_location("repeated_cv_ensemble", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _payload(score: float):
    return {
        "scope": "test",
        "candidates": [
            {
                "suite": "workspace",
                "user_task_id": "user_task_0",
                "injection_task_id": "injection_task_0",
                "observed_security": True,
                "observed_utility": True,
                "candidate_risk_score": score,
                "selection_score": score,
            }
        ],
    }


def test_repeated_cv_payload_averages_scores_and_records_inputs():
    module = _module()
    output = module._ensemble_payloads(
        [_payload(0.2), _payload(0.8)], [Path("fold_a"), Path("fold_b")]
    )
    assert output["candidates"][0]["candidate_risk_score"] == 0.5
    assert output["candidate_repeated_cv_ensemble"]["score_aggregation"] == (
        "arithmetic_mean"
    )
