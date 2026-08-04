import copy

from wmagentattack.decision_state import (
    build_manifest_decision_state,
    find_outcome_paths,
    step_to_victim_event,
)
from wmagentattack.schema import StepRecord


def _manifest_row():
    return {
        "row_id": "row-1",
        "suite": "workspace",
        "task_split": "train",
        "user_task_id": "user_task_1",
        "task_group_id": "workspace|user_task_1",
        "injection_task_id": "injection_task_1",
        "injection_goal": "send a synthetic message",
        "attack_name": "v2_contextual_paraphrase",
        "attack_family": "contextual_paraphrase",
        "attack_variant": "calendar_note",
        "attack_role": "contextual_paraphrase",
        "candidate_injection_vectors": ["calendar_event"],
        "target_tool_sequence": [{"tool": "send_email"}],
        "endpoint_policy": "all",
        "payload_position": "middle",
        "trigger_stage": "on_external_record",
        "knowledge_level": "goal_and_context",
        "payload": "synthetic inert payload",
        "base_success_rate": 0.75,
        "base_success_attempts": 4,
        "utility": True,
        "security": False,
        "selected_tool": "get_calendar",
        "skill_output": "post-rollout secret that must be dropped",
    }


def test_manifest_state_is_invariant_to_post_rollout_labels():
    first = _manifest_row()
    second = copy.deepcopy(first)
    second.update(
        utility=False,
        security=True,
        selected_tool="send_email",
        skill_output="different output",
    )
    kwargs = {
        "victim_model": "llama-3.1-70b",
        "agent_scaffold": "agentdojo-function-tags",
        "task_context": {
            "trusted_goal": "read the calendar",
            "candidate_tools": ["get_calendar", "send_email"],
        },
    }
    state_a = build_manifest_decision_state(first, **kwargs)
    state_b = build_manifest_decision_state(second, **kwargs)
    assert state_a.state_fingerprint == state_b.state_fingerprint
    assert state_a.attacker_action.family == "contextual_paraphrase"
    assert state_a.clean_solvability_prior.posterior_mean == 0.75
    assert "utility" in state_a.source_audit.ignored_outcome_paths
    assert "skill_output" in state_a.source_audit.ignored_outcome_paths


def test_outcome_audit_finds_nested_labels():
    assert find_outcome_paths({"nested": {"joint_success_probability_target": 0.4}}) == [
        "nested.joint_success_probability_target"
    ]


def test_archival_step_conversion_drops_outputs_and_checker_labels():
    step = StepRecord(
        trajectory_id="trajectory-1",
        dataset="test",
        domain="workspace",
        task_id="task-1",
        victim_model="model",
        agent_architecture="agent",
        defense="none",
        step_id=0,
        trusted_instruction="trusted",
        user_goal="goal",
        agent_history="",
        current_observation="observation",
        candidate_tools=["get_calendar"],
        candidate_skills=["calendar_read"],
        candidate_skill_descriptions={"calendar_read": "read"},
        previous_skills=[],
        selected_tool="get_calendar",
        selected_skill="calendar_read",
        skill_arguments={"day": "2026-01-01"},
        skill_output="must not enter victim action",
        task_success=True,
        attack_success=False,
        target_skill_success=False,
        policy_violation=False,
        risk_level=0,
    )
    event = step_to_victim_event(step)
    assert event.tool_name == "get_calendar"
    assert event.arguments == {"day": "2026-01-01"}
    assert "output" not in event.model_dump()
    assert "task_success" not in event.model_dump()

