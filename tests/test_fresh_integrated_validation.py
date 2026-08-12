from __future__ import annotations

import torch

from wmagentattack.fresh_integrated_validation import (
    FreshIntegratedSemanticWorldModel,
    assert_no_unauthorized_heads,
    build_fresh_action_and_transition_rows,
)


def _step(trajectory: str, step: int, selected: str, previous: list[str]):
    return {
        "trajectory_id": trajectory,
        "domain": "banking",
        "task_id": "user_task_3000",
        "step_id": step,
        "user_goal": "Report balance",
        "candidate_skills": ["get", "finish"],
        "candidate_skill_descriptions": {"get": "Read data", "finish": "Stop"},
        "previous_skills": previous,
        "selected_skill": selected,
        "current_observation": "visible",
        "skill_output": "1810" if selected == "get" else "done",
        "tool_error": None,
        "attack_action": None,
    }


def test_fresh_builder_keeps_targets_outside_causal_input() -> None:
    steps = [_step("t1", 0, "get", []), _step("t1", 1, "finish", ["get"])]
    metadata = [
        {
            "trajectory_id": "t1",
            "row_id": "clean::banking::user_task_3000",
            "suite": "banking",
            "user_task_id": "user_task_3000",
            "attack_family": "clean",
            "security": False,
        }
    ]
    actions, transitions, catalog, audit = build_fresh_action_and_transition_rows(
        steps=steps, metadata=metadata, historical_catalog={}
    )
    assert len(actions) == 2
    assert len(transitions) == 2
    assert transitions[0]["next_target_candidate_id"] == actions[1]["target_candidate_id"]
    assert transitions[1]["next_target_candidate_id"] is None
    assert not audit["passed"]  # unit fixture is intentionally below formal count gates
    assert catalog
    for row in (*actions, *transitions):
        assert "target_candidate_id" not in row["causal_model_input"]
        assert "next_action" not in row["causal_model_input"]
        assert "utility" not in row["causal_model_input"]


def test_integrated_model_shares_encoders_and_masks_illegal_actions() -> None:
    model = FreshIntegratedSemanticWorldModel(
        state_size=11,
        candidate_size=7,
        hidden_size=13,
        source_count=3,
        source_specific_action_head=True,
        dropout=0.0,
    )
    states = torch.randn(4, 11)
    candidates = torch.randn(5, 7)
    sources = torch.tensor([0, 1, 2, 0])
    legal = torch.tensor(
        [[1, 1, 0, 0, 0], [0, 1, 1, 0, 0], [0, 0, 1, 1, 0], [1, 0, 0, 0, 1]],
        dtype=torch.bool,
    )
    current = model.current_action_logits(states, candidates, sources)
    current_p = model.probabilities(current, legal)
    selected = candidates[torch.tensor([0, 1, 2, 4])]
    tail, outcomes = model.transition_logits(states, selected, candidates)
    tail_p = model.probabilities(tail, legal)
    assert current.shape == tail.shape == (4, 5)
    assert outcomes.shape == (4, 3)
    assert torch.allclose(current_p[~legal], torch.zeros_like(current_p[~legal]))
    assert torch.allclose(tail_p[~legal], torch.zeros_like(tail_p[~legal]))
    assert len(model.current_action_heads) == 3
    assert_no_unauthorized_heads(model)
