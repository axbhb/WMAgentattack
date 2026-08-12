from __future__ import annotations

import torch

from wmagentattack.adjacent_transition import (
    OBSERVED_OUTCOME_TARGETS,
    ObservedAdjacentTransitionModel,
    build_adjacent_transition_dataset,
    evaluate_adjacent_transition_gate,
)


def _unified_row(trajectory: str, step: int, target: str) -> dict:
    legal = ["agentdojo::read", "agentdojo::finish"]
    causal = {
        "source": "agentdojo",
        "trusted_goal": "Find a record",
        "track": "agentdojo:test",
        "tool_schemas": [],
        "legal_tool_names": ["read"],
        "visible_observation": f"observation {step}",
        "visible_prior_tool": "<START>" if step == 0 else "read",
    }
    return {
        "row_id": f"agentdojo::{trajectory}::step{step}",
        "source": "agentdojo",
        "task_key": "task-key",
        "task_name": "test|task",
        "task_cohort": "original_test",
        "group_id": f"group::step{step}",
        "variant": "clean",
        "causal_model_input": causal,
        "causal_input_fingerprint": f"fp-{step}",
        "legal_candidate_ids": legal,
        "target_candidate_id": target,
    }


def test_adjacent_builder_preserves_causal_boundary() -> None:
    unified = {
        "rows": [
            _unified_row("trajectory", 0, "agentdojo::read"),
            _unified_row("trajectory", 1, "agentdojo::finish"),
        ],
        "candidate_catalog": {
            "agentdojo::read": {
                "source": "agentdojo",
                "kind": "tool",
                "function": {"name": "read"},
            },
            "agentdojo::finish": {
                "source": "agentdojo",
                "kind": "text_or_stop",
                "function": {"name": "TEXT"},
            },
        },
        "agentdojo_cohorts": {"original_test": ["test|task"]},
        "folds": [],
    }
    steps = [
        {
            "trajectory_id": "trajectory",
            "step_id": 0,
            "multiseed_group_id": "group",
            "skill_output": "record",
            "tool_error": None,
            "current_observation": "goal",
        },
        {
            "trajectory_id": "trajectory",
            "step_id": 1,
            "multiseed_group_id": "group",
            "skill_output": "done",
            "tool_error": None,
            "current_observation": "record",
        },
    ]
    protocol = {
        "protocol_id": "test",
        "source": {
            "expected_step_rows": 2,
            "expected_trajectories": 1,
            "expected_adjacent_transitions": 1,
            "expected_multistep_trajectories": 1,
            "expected_tasks_with_adjacent_transitions": 1,
        },
        "preflight_gate": {
            "minimum_execution_errors": 0,
            "minimum_tasks_with_execution_errors": 0,
        },
    }
    dataset, audit = build_adjacent_transition_dataset(
        unified=unified, raw_steps=steps, protocol=protocol
    )
    assert audit["passed"]
    assert len(dataset["events"]) == 2
    assert dataset["events"][0]["next_target_candidate_id"] == "agentdojo::finish"
    assert dataset["events"][1]["next_target_candidate_id"] is None
    assert "next_action" not in dataset["events"][0]["causal_model_input"]


def test_adjacent_model_masks_illegal_candidates() -> None:
    model = ObservedAdjacentTransitionModel(
        state_size=6, candidate_size=4, hidden_size=8, dropout=0.0
    )
    states = torch.randn(3, 6)
    selected = torch.randn(3, 4)
    candidates = torch.randn(5, 4)
    legal = torch.tensor(
        [[True, False, True, False, False]] * 3, dtype=torch.bool
    )
    probabilities = model.next_action_probabilities(
        states, selected, candidates, legal
    )
    assert probabilities.shape == (3, 5)
    assert torch.allclose(probabilities[:, ~legal[0]], torch.zeros(3, 3))
    _, outcomes = model(states, selected, candidates)
    assert outcomes.shape == (3, len(OBSERVED_OUTCOME_TARGETS))


def test_adjacent_gate_requires_tail_and_nontrivial_error_signal() -> None:
    gates = {
        "minimum_threshold_positive_seeds": 2,
        "minimum_tail_action_nll_gain": 0.01,
        "minimum_tail_action_accuracy_gain": 0.005,
        "minimum_positive_task_fraction": 0.55,
        "minimum_outcome_bce_gain_over_train_prior": 0.01,
        "minimum_execution_error_bce_gain": 0.001,
    }
    checks = evaluate_adjacent_transition_gate(
        action_nll_seed_gains=[0.02, 0.01, 0.00],
        action_accuracy_seed_gains=[0.01, 0.006, 0.0],
        action_task_gains=[1.0] * 12 + [-1.0] * 8,
        outcome_bce_seed_gains=[0.02, 0.01, 0.0],
        execution_error_bce_gain=0.002,
        all_predictions_legal=True,
        gates=gates,
    )
    assert all(checks.values())
    failed = evaluate_adjacent_transition_gate(
        action_nll_seed_gains=[0.02, 0.01, 0.00],
        action_accuracy_seed_gains=[0.01, 0.006, 0.0],
        action_task_gains=[1.0] * 12 + [-1.0] * 8,
        outcome_bce_seed_gains=[0.02, 0.01, 0.0],
        execution_error_bce_gain=-0.002,
        all_predictions_legal=True,
        gates=gates,
    )
    assert not failed["execution_error_bce_gain"]
