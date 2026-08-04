from copy import deepcopy

from wmagentattack.event_ontology import (
    EVENT_ONTOLOGY_VERSION,
    normalize_victim_event,
    ontology_fingerprint,
)


def _step():
    return {
        "trajectory_id": "trajectory-1",
        "dataset": "agentdojo",
        "domain": "travel",
        "task_id": "task-1",
        "victim_model": "victim",
        "agent_architecture": "tools",
        "defense": "none",
        "step_id": 0,
        "trusted_instruction": "trusted",
        "user_goal": "goal",
        "agent_history": "history",
        "current_observation": "observation",
        "untrusted_content": "untrusted",
        "candidate_tools": ["search"],
        "candidate_skills": ["finish", "restaurant_read"],
        "candidate_skill_descriptions": {
            "finish": "Finish",
            "restaurant_read": "Read restaurants",
        },
        "previous_skills": [],
        "selected_tool": "search",
        "selected_skill": "restaurant_read",
        "skill_arguments": {"city": "Paris", "date": "tomorrow"},
        "skill_output": "secret output",
        "tool_error": None,
        "attack_action": "attack",
        "attack_location": "result",
        "target_skill": "send",
        "task_success": True,
        "attack_success": False,
        "target_skill_success": False,
        "policy_violation": False,
        "risk_level": 0,
    }


def test_ontology_is_label_blind_and_candidate_order_canonical():
    first = _step()
    second = deepcopy(first)
    second.update(
        {
            "task_success": False,
            "attack_success": True,
            "policy_violation": True,
            "risk_level": 3,
            "skill_output": "different post-action output",
            "current_observation": "different raw text",
        }
    )
    second["candidate_skills"] = list(reversed(second["candidate_skills"]))
    event_a = normalize_victim_event(first, is_last_observed_event=False)
    event_b = normalize_victim_event(second, is_last_observed_event=False)
    assert event_a == event_b
    assert event_a.ontology_version == EVENT_ONTOLOGY_VERSION
    assert [item.name for item in event_a.candidates] == [
        "finish",
        "restaurant_read",
    ]
    assert event_a.normalized_argument_slots == ("city", "date")


def test_ontology_fingerprint_is_stable_and_missing_state_fields_are_explicit():
    event = normalize_victim_event(_step(), is_last_observed_event=True)
    assert len(ontology_fingerprint()) == 64
    assert event.canonical_state_delta is None
    assert event.task_progress_delta is None
    assert event.irreversible_effect is None
    assert "canonical_state_delta" in event.unavailable_fields
