from copy import deepcopy

from wmagentattack.factorized_transition_labels import (
    FACTOR_CLASSES,
    build_factorized_transition_rows,
)


def event(step, observation, outcome, *, action="tool_read"):
    return {
        "event_id": f"e{step}", "trajectory_id": "t", "task_name": "travel|task",
        "step_id": step, "current_action_candidate_id": action,
        "observed_outcome": outcome,
        "causal_model_input": {
            "trusted_goal": "Find the hotel price and book it",
            "visible_observation": observation,
            "legal_tool_names": ["hotel_read", "hotel_book"],
            "tool_schemas": [{"function": {"name": "hotel_read", "description": "read hotel price"}}],
        },
    }


def test_labels_use_adjacent_visible_state_and_observed_execution_only():
    outcome = {"execution_error": False, "output_nonempty": True, "trajectory_continues": True}
    rows = build_factorized_transition_rows([
        event(0, "Find a hotel", outcome),
        event(1, "Find a hotel price", {**outcome, "trajectory_continues": False}),
    ])
    assert len(rows) == 1
    assert set(rows[0]["labels"]) == set(FACTOR_CLASSES)
    assert rows[0]["labels"]["execution_status"] == "productive_continue"


def test_final_outcomes_and_next_action_cannot_change_factor_labels():
    outcome = {"execution_error": False, "output_nonempty": True, "trajectory_continues": True}
    events = [event(0, "Find a hotel", outcome), event(1, "Hotel price found", {**outcome, "trajectory_continues": False})]
    left = build_factorized_transition_rows(events)
    changed = deepcopy(events)
    changed[0]["joint_outcome_target"] = {"attack1_utility1": 1.0}
    changed[1]["current_action_candidate_id"] = "completely_different_next_action"
    right = build_factorized_transition_rows(changed)
    assert left[0]["labels"] == right[0]["labels"]
    assert left[0]["label_fingerprint"] == right[0]["label_fingerprint"]


def test_error_status_is_separate_from_empty_success():
    error = {"execution_error": True, "output_nonempty": False, "trajectory_continues": True}
    empty = {"execution_error": False, "output_nonempty": False, "trajectory_continues": True}
    stop = {"execution_error": False, "output_nonempty": True, "trajectory_continues": False}
    error_row = build_factorized_transition_rows([event(0, "x", error), event(1, "x", stop)])[0]
    empty_row = build_factorized_transition_rows([event(0, "x", empty), event(1, "x", stop)])[0]
    assert error_row["labels"]["execution_status"] == "error_empty_continue"
    assert empty_row["labels"]["execution_status"] == "empty_continue"
