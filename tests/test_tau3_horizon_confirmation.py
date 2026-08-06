from wmagentattack.tau3_horizon_confirmation import (
    build_confirmation_manifest,
    episode_reproducibility,
    evaluate_confirmation_gate,
)


def _parent_and_pilot():
    rows = []
    for domain_index, domain in enumerate(("airline", "retail", "telecom")):
        for task_index in range(16):
            split = (
                "training"
                if task_index < 10
                else "calibration"
                if task_index < 13
                else "confirmation"
            )
            task_key = f"{domain}-task-{task_index}"
            for seed in (401, 409):
                episode_id = f"episode::{domain}::{task_index}::{seed}"
                rows.append(
                    {
                        "episode_id": episode_id,
                        "parent_episode_id": f"grandparent::{episode_id}",
                        "task_key": task_key,
                        "domain": domain,
                        "experimental_split": split,
                        "structural_stratum": "reference_mutating",
                        "llm_seed": seed,
                        "shared_model_identity_sha256": "model",
                        "role_contract_sha256": "parent-role",
                    }
                )
    parent = {
        "schema_version": "manifest-v1",
        "source_commit": "source",
        "shared_model_identity_sha256": "model",
        "role_contract_sha256": "parent-role",
        "rows": rows,
    }
    pilot_rows = [
        row
        for row in rows
        if int(row["task_key"].rsplit("-", 1)[1]) < 4
    ]
    pilot = {
        "role_contract_sha256": "candidate-role",
        "rows": pilot_rows,
    }
    protocol = {
        "protocol_id": "confirmation",
        "paired_parent": {"manifest_sha256": "parent-file"},
        "pilot_go": {"manifest_sha256": "pilot-file"},
        "confirmation_surface": {
            "tasks": 48,
            "episodes": 96,
            "domains": ["airline", "retail", "telecom"],
            "seeds": [401, 409],
            "pilot_overlap_episodes": 24,
            "pilot_overlap_tasks": 12,
            "out_of_pilot_episodes": 72,
            "out_of_pilot_tasks": 36,
            "selection_inputs_allowed": ["episode_id"],
        },
    }
    return parent, pilot, protocol


def test_confirmation_manifest_is_the_complete_label_blind_surface():
    parent, pilot, protocol = _parent_and_pilot()
    manifest, audit = build_confirmation_manifest(parent, pilot, protocol)
    assert audit["passed"]
    assert len(manifest["rows"]) == 96
    assert audit["tasks"] == 48
    assert audit["pilot_overlap_episodes"] == 24
    assert audit["out_of_pilot_episodes"] == 72
    assert audit["forbidden_outcome_inputs_read"] == []
    assert {row["role_contract_sha256"] for row in manifest["rows"]} == {
        "candidate-role"
    }


def test_episode_reproducibility_is_exact():
    episode = {"episode_id": "x", "agent_decisions": [{"completion": "a"}]}
    assert episode_reproducibility(episode, dict(episode))
    changed = {"episode_id": "x", "agent_decisions": [{"completion": "b"}]}
    assert not episode_reproducibility(changed, episode)


def test_confirmation_gate_requires_holdout_and_pilot_reproduction():
    gate = {
        "expected_complete_episodes": 96,
        "maximum_forced_budget_stop_episodes": 24,
        "minimum_relative_forced_stop_reduction_vs_paired_parent": 0.5,
        "minimum_adjacent_assistant_tool_transitions": 100,
        "minimum_state_changed_assistant_transitions": 15,
        "minimum_state_unchanged_assistant_transitions": 30,
        "minimum_tasks_with_state_changed_assistant_transition": 8,
        "minimum_domains_with_state_changed_assistant_transition": 2,
        "minimum_paired_state_changed_transition_gain": 10,
        "minimum_supported_transition_targets": 4,
        "maximum_assistant_tool_error_rate_increase_over_paired_parent": 0.05,
        "minimum_natural_user_messages": 96,
        "minimum_episodes_with_two_or_more_assistant_transitions": 36,
        "minimum_tasks_with_at_least_one_assistant_transition": 30,
        "minimum_unique_executed_assistant_tools": 10,
        "minimum_agent_tool_decision_rate": 0.35,
        "maximum_agent_tool_decision_rate": 0.9,
        "maximum_dominant_agent_action_fraction": 0.65,
        "expected_out_of_pilot_episodes": 72,
        "expected_out_of_pilot_tasks": 36,
        "minimum_out_of_pilot_state_changed_assistant_transitions": 5,
        "minimum_out_of_pilot_tasks_with_state_changed_assistant_transition": 4,
        "minimum_out_of_pilot_domains_with_state_changed_assistant_transition": 2,
    }
    metrics = {
        "episodes_complete": 96,
        "runtime_failures": 0,
        "agent_private_scenario_exposures": 0,
        "real_external_endpoint_calls": 0,
        "forced_budget_stop_episodes": 24,
        "relative_forced_stop_reduction_vs_parent": 0.5,
        "adjacent_assistant_tool_transitions": 100,
        "state_changed_assistant_transitions": 15,
        "state_unchanged_assistant_transitions": 30,
        "tasks_with_state_changed_assistant_transition": 8,
        "domains_with_state_changed_assistant_transition": 2,
        "paired_state_changed_transition_gain": 10,
        "supported_transition_targets": 4,
        "assistant_tool_error_rate_increase_over_parent": 0.05,
        "communication_error_terminations": 0,
        "natural_user_messages": 96,
        "episodes_with_two_or_more_assistant_transitions": 36,
        "tasks_with_at_least_one_assistant_transition": 30,
        "unique_executed_assistant_tools": 10,
        "agent_tool_decision_rate": 0.35,
        "dominant_agent_action_fraction": 0.65,
        "out_of_pilot_episodes": 72,
        "out_of_pilot_tasks": 36,
        "out_of_pilot_state_changed_assistant_transitions": 5,
        "out_of_pilot_tasks_with_state_changed_assistant_transition": 4,
        "out_of_pilot_domains_with_state_changed_assistant_transition": 2,
    }
    integrity = {
        "all_parent_prefixes_equivalent": True,
        "label_blind_panel_selection": True,
        "all_pilot_overlap_episodes_reproduced": True,
        "label_blind_full_surface": True,
    }
    assert all(evaluate_confirmation_gate(metrics, integrity, gate).values())
    integrity["all_pilot_overlap_episodes_reproduced"] = False
    checks = evaluate_confirmation_gate(metrics, integrity, gate)
    assert not checks["pilot_overlap_reproducibility"]
