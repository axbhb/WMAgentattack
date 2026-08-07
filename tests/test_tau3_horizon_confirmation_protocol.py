import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _protocol():
    return json.loads(
        (
            ROOT / "configs" / "0806_tau3_horizon_confirmation_protocol.json"
        ).read_text(encoding="utf-8")
    )


def test_confirmation_completed_with_binding_no_go():
    protocol = _protocol()
    assert protocol["status"] == "completed_no_go"
    assert protocol["jobs"]["generation_array"] == 6565
    assert protocol["jobs"]["summary"] == 6568
    result = protocol["result"]
    assert not result["passed"]
    assert result["decision"] == (
        "HORIZON_CONFIRMATION_NO_GO__DO_NOT_RUN_METHOD_TEST_OR_SCALE"
    )
    assert result["gate_sha256"] == (
        "d4936d31edcee855a0be42ae75f9e673616c85ff4a344a87d94cbfc1452d8521"
    )
    assert result["failed_gate_clauses"] == [
        "forced_budget_stop_episodes",
        "minimum_agent_tool_decision_rate",
        "pilot_overlap_reproducibility",
        "integrity::all_pilot_overlap_episodes_reproduced",
    ]
    assert (
        protocol["pilot_go"]["decision"]
        == "HORIZON_PILOT_GO__AUTHORIZE_FULL_96_CONFIRMATION"
    )


def test_confirmation_manifest_and_implementation_are_frozen():
    protocol = _protocol()
    manifest = protocol["frozen_manifest"]
    assert manifest["sha256"] == (
        "bea0961bcd1208af3df41057bf27826a94ceef0e77c28f6f3cc691472c034ea8"
    )
    assert manifest["byte_identical_double_build"]
    assert manifest["label_blind_selection_audit_passed"]
    assert manifest["forbidden_outcome_inputs_read"] == []
    assert manifest["pilot_overlap_episodes"] == 24
    assert manifest["out_of_pilot_episodes"] == 72
    assert len(protocol["implementation_sha256"]) == 11


def test_confirmation_reuses_the_exact_passed_horizon_contract():
    protocol = _protocol()
    mechanism = protocol["single_mutable_mechanism"]
    assert mechanism["identical_to_passed_pilot_mechanism"]
    assert protocol["role_contracts"]["agent"][
        "maximum_generation_calls_per_episode"
    ] == 16
    assert protocol["role_contracts"]["user"][
        "maximum_generation_calls_per_episode"
    ] == 16
    assert protocol["interaction"]["maximum_orchestrator_steps"] == 64
    assert protocol["exact_execution"]["fresh_replay_replicas"] == 2


def test_full_surface_and_out_of_pilot_gate_are_frozen():
    protocol = _protocol()
    surface = protocol["confirmation_surface"]
    assert surface["episodes"] == 96
    assert surface["tasks"] == 48
    assert surface["pilot_overlap_episodes"] == 24
    assert surface["out_of_pilot_episodes"] == 72
    assert surface["out_of_pilot_tasks"] == 36
    gate = protocol["confirmation_gate"]
    assert gate["maximum_forced_budget_stop_episodes"] == 24
    assert gate["minimum_state_changed_assistant_transitions"] == 15
    assert gate["minimum_paired_state_changed_transition_gain"] == 10
    assert gate["minimum_out_of_pilot_state_changed_assistant_transitions"] == 5
    assert gate[
        "minimum_out_of_pilot_tasks_with_state_changed_assistant_transition"
    ] == 4
    assert gate[
        "minimum_out_of_pilot_domains_with_state_changed_assistant_transition"
    ] == 2
    assert gate["require_pilot_overlap_reproducibility"]
    assert gate["all_clauses_required"]


def test_post_gate_timestamp_diagnostic_does_not_recompute_the_no_go():
    counterevidence = _protocol()["result"]["post_gate_counterevidence"]
    assert not counterevidence["current_decision_recomputed"]
    assert counterevidence["pilot_overlap_agent_decision_records_identical"] == 24
    assert counterevidence["pilot_overlap_user_generation_records_identical"] == 24
    assert counterevidence["pilot_overlap_full_raw_episode_records_identical"] == 0
    assert counterevidence["mismatched_top_level_fields"] == ["trajectory"]
    assert (
        counterevidence["observed_trajectory_difference"]
        == "trajectory[*].timestamp only"
    )
    assert counterevidence["user_tool_events_remain_exogenous"]


def test_confirmation_no_go_authorizes_nothing_downstream():
    result = _protocol()["result"]
    assert result["authorization"] == (
        "No predictive-method test, confirmation rerun, or scale-up is authorized."
    )
    boundary = _protocol()["authorization_boundary"]
    assert boundary["confirmation_go_authorizes_only_frozen_method_comparison"]
    assert boundary["confirmation_go_does_not_authorize_large_scale_collection"]
    assert boundary["large_scale_requires_later_method_gate"]
    assert boundary["real_external_endpoints_forbidden"]
    assert boundary["attack_generation_forbidden"]
    assert boundary["dreamer_or_planner_training_forbidden"]
