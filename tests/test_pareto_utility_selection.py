import importlib.util
from pathlib import Path


def _module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "18_pareto_utility_selection.py"
    spec = importlib.util.spec_from_file_location("pareto_utility_selection", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_expected_attacked_utility_combines_clean_rate_and_preservation():
    module = _module()
    candidates = [
        {
            "suite": "workspace",
            "user_task_id": "user_task_1",
            "utility_score": 0.2,
            "preservation_score": 0.5,
        }
    ]
    annotated = module._annotate_clean_rates(
        candidates,
        {("workspace", "user_task_1"): 0.6},
        min_base_success_rate=0.5,
    )
    assert annotated[0]["expected_attacked_utility_score"] == 0.3
    assert annotated[0]["preservation_eval_eligible"]


def test_expected_attacked_utility_falls_back_to_utility_score():
    module = _module()
    candidates = [
        {
            "suite": "workspace",
            "user_task_id": "user_task_1",
            "utility_score": 0.4,
        }
    ]
    annotated = module._annotate_clean_rates(
        candidates,
        {},
        min_base_success_rate=0.5,
    )
    assert annotated[0]["expected_attacked_utility_score"] == 0.4
    assert annotated[0]["preservation_eval_eligible"]


def test_candidate_ranker_objective_overrides_step_level_heuristic():
    module = _module()
    row = {
        "candidate_objective_score": 0.25,
        "risk_score": 0.99,
        "rollout_mean_risk_score": 0.99,
        "target_skill_probability": 1.0,
        "rollout_target_reached": 1.0,
    }
    assert module._objective_score(row) == 0.25
