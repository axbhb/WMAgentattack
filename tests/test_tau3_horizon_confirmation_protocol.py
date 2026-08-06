import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _protocol():
    return json.loads(
        (
            ROOT / "configs" / "0806_tau3_horizon_confirmation_protocol.json"
        ).read_text(encoding="utf-8")
    )


def test_confirmation_manifest_is_frozen_and_not_run():
    protocol = _protocol()
    assert protocol["status"] == "manifest_frozen_before_interactive_outcomes"
    assert protocol["jobs"]["generation_array"] == 6565
    assert protocol["jobs"]["summary"] == 6568
    assert protocol["result"] is None
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


def test_confirmation_go_still_does_not_authorize_scale_or_attacks():
    boundary = _protocol()["authorization_boundary"]
    assert boundary["confirmation_go_authorizes_only_frozen_method_comparison"]
    assert boundary["confirmation_go_does_not_authorize_large_scale_collection"]
    assert boundary["large_scale_requires_later_method_gate"]
    assert boundary["real_external_endpoints_forbidden"]
    assert boundary["attack_generation_forbidden"]
    assert boundary["dreamer_or_planner_training_forbidden"]
