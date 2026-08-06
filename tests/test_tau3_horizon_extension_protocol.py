import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _protocol():
    return json.loads(
        (ROOT / "configs" / "0806_tau3_horizon_extension_protocol.json").read_text(
            encoding="utf-8"
        )
    )


def _parent_protocol():
    return json.loads(
        (
            ROOT
            / "configs"
            / "0806_tau3_interaction_faithful_repair_protocol.json"
        ).read_text(encoding="utf-8")
    )


def test_horizon_pilot_completed_and_only_authorizes_confirmation():
    protocol = _protocol()
    assert protocol["status"] == "completed_go_authorize_full_96_confirmation_only"
    assert protocol["jobs"] == {"generation_array": 6534, "summary": 6535}
    assert protocol["result"]["passed"]
    assert (
        protocol["result"]["decision"]
        == "HORIZON_PILOT_GO__AUTHORIZE_FULL_96_CONFIRMATION"
    )
    assert protocol["result"]["failed_gate_clauses"] == []
    assert (
        protocol["parent_result"]["decision"]
        == "INTERACTION_DATA_NO_GO__DO_NOT_SCALE_OR_RUN_METHOD_TEST"
    )
    assert protocol["frozen_manifest"]["byte_identical_double_build"]
    assert protocol["frozen_manifest"]["label_blind_selection_audit_passed"]


def test_horizon_is_the_only_mutable_generation_mechanism():
    protocol = _protocol()
    mechanism = protocol["single_mutable_mechanism"]
    assert mechanism["name"] == "role_horizon_extension"
    assert mechanism["agent_generation_calls_per_episode_parent"] == 8
    assert mechanism["agent_generation_calls_per_episode_candidate"] == 16
    assert mechanism["user_generation_calls_per_episode_parent"] == 8
    assert mechanism["user_generation_calls_per_episode_candidate"] == 16
    assert mechanism["orchestrator_steps_parent"] == 32
    assert mechanism["orchestrator_steps_candidate"] == 64
    assert mechanism["all_other_generation_and_data_contracts_identical"]

    parent = _parent_protocol()
    for role in ("agent", "user"):
        candidate_role = dict(protocol["role_contracts"][role])
        parent_role = dict(parent["role_contracts"][role])
        assert candidate_role.pop("maximum_generation_calls_per_episode") == 16
        assert parent_role.pop("maximum_generation_calls_per_episode") == 8
        assert candidate_role == parent_role
    candidate_interaction = dict(protocol["interaction"])
    parent_interaction = dict(parent["interaction"])
    assert candidate_interaction.pop("maximum_orchestrator_steps") == 64
    assert parent_interaction.pop("maximum_orchestrator_steps") == 32
    assert candidate_interaction == parent_interaction
    assert protocol["shared_model_identity"] == parent["shared_model_identity"]
    assert protocol["exact_execution"] == parent["exact_execution"]


def test_panel_selection_and_targets_remain_label_blind():
    protocol = _protocol()
    forbidden = set(protocol["pilot_panel"]["selection_inputs_forbidden"])
    assert "state_changed labels" in forbidden
    assert "utility or final outcome labels" in forbidden
    assert protocol["held_fixed"]["assistant_only_transition_targets"]
    assert protocol["held_fixed"]["user_tool_events_are_exogenous_context"]
    assert protocol["pilot_panel"]["same_task_same_seed_parent_pairs_required"]


def test_gate_cannot_authorize_training_or_scale():
    protocol = _protocol()
    gate = protocol["pilot_gate"]
    assert gate["all_clauses_required"]
    assert gate["maximum_forced_budget_stop_episodes"] == 6
    assert gate["minimum_state_changed_assistant_transitions"] == 4
    assert gate["minimum_domains_with_state_changed_assistant_transition"] == 2
    assert gate["minimum_supported_transition_targets"] == 4
    boundary = protocol["authorization_boundary"]
    assert boundary["pilot_go_authorizes_only_full_96_episode_horizon_confirmation"]
    assert boundary["pilot_go_does_not_authorize_method_training"]
    assert boundary["pilot_go_does_not_authorize_large_scale_collection"]
    assert boundary["attack_generation_forbidden"]
    assert boundary["dreamer_or_planner_training_forbidden"]


def test_frozen_implementation_hashes_match_the_worktree():
    protocol = _protocol()
    for relative, expected in protocol["implementation_sha256"].items():
        observed = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        assert observed == expected
