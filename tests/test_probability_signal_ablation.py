import importlib.util
from pathlib import Path


def _module():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "33_probability_signal_ablation.py"
    )
    spec = importlib.util.spec_from_file_location("probability_signal_ablation", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_variant_matrices_have_distinct_feature_sources():
    module = _module()
    row = {
        "suite": "slack",
        "user_task_id": "user_task_1",
        "injection_task_id": "injection_task_1",
        "candidate_risk_score": 0.8,
        "risk_score": 0.7,
        "target_skill_probability": 0.3,
        "rollout_target_reached": 1.0,
        "candidate_expected_utility_score": 0.6,
        "candidate_preservation_score": 0.7,
        "candidate_utility_score": 0.8,
        "final_utility_score": 0.75,
        "value_score": 0.65,
    }
    clean = {("slack", "user_task_1"): 0.8}
    _, context_names = module._variant_matrix(
        [row], clean, head="utility", variant="context_prior"
    )
    _, score_names = module._variant_matrix(
        [row], clean, head="utility", variant="world_scores"
    )
    assert "logit_clean_solvability" in context_names
    assert all(name.startswith("suite_") for name in context_names[1:])
    assert "logit_clean_solvability" not in score_names
    assert not any(name.startswith("suite_") for name in score_names)
