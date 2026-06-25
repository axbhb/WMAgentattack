from pathlib import Path

from wmagentattack.data_split import split_trajectories
from wmagentattack.metrics import evaluate_predictions
from wmagentattack.defense import evaluate_risk_detector
from wmagentattack.planner import evaluate_attack_strategies, rank_candidates
from wmagentattack.synthetic_generator import (
    flatten_steps,
    generate_synthetic_trajectories,
)
from wmagentattack.world_model import SklearnWorldModel


def test_synthetic_world_model_and_planner_round_trip(tmp_path: Path):
    trajectories = generate_synthetic_trajectories(120, seed=3)
    splits = split_trajectories(trajectories, seed=3)
    train_steps = flatten_steps(splits["train"])
    test_steps = flatten_steps(splits["test"])

    model = SklearnWorldModel(max_features=2_000).fit(train_steps)
    metrics = evaluate_predictions(test_steps, model.predict(test_steps))
    assert 0.0 <= metrics["next_skill_accuracy"] <= 1.0
    assert metrics["risk_auc"] is not None

    model_path = tmp_path / "model.joblib"
    model.save(model_path)
    loaded = SklearnWorldModel.load(model_path)
    attacked_step = next(step for step in test_steps if step.target_skill)
    ranking = rank_candidates(
        loaded, attacked_step, attacked_step.target_skill or "send_message"
    )
    assert len(ranking) == 4
    assert ranking[0]["score"] >= ranking[-1]["score"]
    states = {}
    for step in test_steps:
        if step.target_skill:
            states.setdefault(step.trajectory_id, step)
    attack_results = evaluate_attack_strategies(model, list(states.values()))
    assert set(attack_results) == {
        "random",
        "manual_template",
        "no_world_model",
        "world_model",
    }
    predictions = model.predict(test_steps)
    defense = evaluate_risk_detector(test_steps, predictions["risk_score"])
    assert 0.0 <= defense["unsafe_block_rate"] <= 1.0
