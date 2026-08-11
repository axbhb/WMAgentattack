import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _protocol():
    return json.loads(
        (
            ROOT / "configs" / "0807_tau3_tail_horizon_protocol.json"
        ).read_text(encoding="utf-8")
    )


def test_tail_horizon_manifest_is_frozen_before_outcomes():
    protocol = _protocol()
    assert protocol["status"] == "manifest_frozen_before_interactive_outcomes"
    assert protocol["implementation_commit"]
    assert protocol["implementation_sha256"]
    assert protocol["frozen_manifest"]["episodes"] == 24
    assert protocol["frozen_manifest"]["tasks"] == 12
    assert protocol["frozen_manifest"]["byte_identical_double_build"]
    assert protocol["jobs"] == {"generation_array": 6657, "summary": 6658}
    assert protocol["result"] is None


def test_parent_no_go_is_binding_and_hashed():
    parent = _protocol()["binding_parent_result"]
    assert parent["decision"] == (
        "HORIZON_CONFIRMATION_NO_GO__DO_NOT_RUN_METHOD_TEST_OR_SCALE"
    )
    assert parent["gate_sha256"] == (
        "d4936d31edcee855a0be42ae75f9e673616c85ff4a344a87d94cbfc1452d8521"
    )
    assert parent["forced_budget_stop_episodes"] == 26
    assert parent["agent_tool_decision_rate"] < 0.35


def test_only_mutable_mechanism_is_the_fixed_four_call_tail():
    protocol = _protocol()
    mechanism = protocol["single_mutable_mechanism"]
    assert mechanism["name"] == "bounded_tail_horizon_extension"
    assert mechanism["agent_generation_calls_per_episode_parent"] == 16
    assert mechanism["agent_generation_calls_per_episode_candidate"] == 20
    assert mechanism["user_generation_calls_per_episode_parent"] == 16
    assert mechanism["user_generation_calls_per_episode_candidate"] == 20
    assert mechanism["orchestrator_steps_parent"] == 64
    assert mechanism["orchestrator_steps_candidate"] == 80
    assert mechanism["tail_generation_calls_added_per_role"] == 4
    assert mechanism["all_other_generation_and_data_contracts_identical"]
    held = protocol["held_fixed"]
    assert held["agent_and_user_tool_permissions_unchanged"]
    assert held["function_tag_parser_unchanged"]
    assert held["assistant_only_transition_targets"]
    assert held["user_tool_events_are_exogenous_context"]
    assert held["complete_live_tool_sequence_replay_replicas"] == 2


def test_panel_selection_is_label_blind_and_double_built():
    panel = _protocol()["pilot_panel"]
    assert panel["tasks"] == 12
    assert panel["episodes"] == 24
    assert panel["tasks_per_domain"] == 4
    assert panel["domains"] == ["airline", "retail", "telecom"]
    assert panel["seeds"] == [401, 409]
    assert "out-of-pilot" in panel["candidate_pool"]
    assert panel["manifest_must_be_built_twice_byte_identically_before_outcomes"]
    forbidden = " ".join(panel["selection_inputs_forbidden"])
    assert "completion" in forbidden
    assert "forced-stop" in forbidden
    assert "state_changed" in forbidden
    assert "final outcome" in forbidden


def test_gate_retains_strict_data_and_error_clauses():
    gate = _protocol()["pilot_gate"]
    assert gate["expected_complete_episodes"] == 24
    assert gate["maximum_forced_budget_stop_episodes"] == 6
    assert gate["minimum_relative_forced_stop_reduction_vs_paired_parent"] == 0.5
    assert gate["minimum_agent_tool_decision_rate"] == 0.35
    assert gate["minimum_state_changed_assistant_transitions"] == 4
    assert gate["minimum_tasks_with_state_changed_assistant_transition"] == 2
    assert gate["minimum_domains_with_state_changed_assistant_transition"] == 2
    assert gate["minimum_paired_state_changed_transition_gain"] == 3
    assert gate["minimum_supported_transition_targets"] == 4
    assert gate["maximum_assistant_tool_error_rate_increase_over_paired_parent"] == 0.05
    assert gate["require_parent_prefix_equivalence"]
    assert gate["require_label_blind_panel_selection"]
    assert gate["all_clauses_required"]


def test_timestamp_rule_is_prospective_and_raw_evidence_is_preserved():
    contract = _protocol()["prospective_reproducibility_contract"]
    assert contract["current_confirmation_decision_is_immutable"]
    assert contract["raw_episode_records_and_hashes_must_be_preserved"]
    assert contract["excluded_volatile_fields"] == ["trajectory[*].timestamp"]
    assert contract["all_agent_decisions_must_match"]
    assert contract["all_user_generations_must_match"]
    assert contract["all_tool_events_and_state_fingerprints_must_match"]
    assert contract["raw_mismatch_diagnostics_must_be_reported"]


def test_no_downstream_work_is_authorized():
    boundary = _protocol()["authorization_boundary"]
    assert boundary["pilot_go_authorizes_only_same_contract_96_episode_confirmation"]
    assert boundary["pilot_go_does_not_authorize_method_training"]
    assert boundary["pilot_go_does_not_authorize_large_scale_collection"]
    assert boundary["full_data_gate_and_later_method_gate_required_before_scale"]
    assert boundary["real_external_endpoints_forbidden"]
    assert boundary["attack_generation_forbidden"]
    assert boundary["dreamer_or_planner_training_forbidden"]
