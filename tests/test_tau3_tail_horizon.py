from wmagentattack.tau3_tail_horizon import (
    effective_tail_protocol,
    evaluate_tail_gate,
    select_tail_panel,
)


def _parent():
    rows = []
    out = []
    for domain in ("airline", "retail", "telecom"):
        for task_index in range(6):
            task_key = f"{domain}-{task_index}"
            for seed in (401, 409):
                episode_id = f"{task_key}-{seed}"
                out.append(episode_id)
                rows.append(
                    {
                        "episode_id": episode_id,
                        "task_key": task_key,
                        "domain": domain,
                        "llm_seed": seed,
                        "structural_stratum": "mutating" if task_index < 3 else "read_only",
                        "experimental_split": ("training", "calibration", "confirmation")[task_index % 3],
                    }
                )
    return {"rows": rows, "out_of_pilot_episode_ids": out, "source_commit": "abc"}


def _protocol():
    return {
        "protocol_id": "tail-test",
        "binding_parent_result": {"manifest_sha256": "parent"},
        "pilot_panel": {
            "domains": ["airline", "retail", "telecom"],
            "tasks_per_domain": 4,
            "tasks": 12,
            "episodes": 24,
            "seeds": [401, 409],
            "selection_inputs_allowed": ["domain", "task_key"],
        },
    }


def test_tail_panel_is_deterministic_balanced_and_task_disjoint():
    first, audit = select_tail_panel(_parent(), _protocol())
    second, second_audit = select_tail_panel(_parent(), _protocol())
    assert first == second
    assert audit["manifest_content_sha256"] == second_audit["manifest_content_sha256"]
    assert audit["passed"]
    assert audit["episodes"] == 24
    assert audit["tasks"] == 12
    assert audit["domain_episode_counts"] == {"airline": 8, "retail": 8, "telecom": 8}


def test_effective_protocol_changes_only_horizons_and_budget():
    base = {
        "protocol_id": "base",
        "status": "old",
        "role_contracts": {"agent": {"maximum_generation_calls_per_episode": 16}, "user": {"maximum_generation_calls_per_episode": 16}},
        "interaction": {"maximum_orchestrator_steps": 64},
        "fixed_budget": {"old": True},
    }
    protocol = {
        "protocol_id": "tail",
        "single_mutable_mechanism": {
            "agent_generation_calls_per_episode_candidate": 20,
            "user_generation_calls_per_episode_candidate": 20,
            "orchestrator_steps_candidate": 80,
        },
        "fixed_budget": {"episodes": 24},
    }
    effective = effective_tail_protocol(protocol, base)
    assert effective["role_contracts"]["agent"]["maximum_generation_calls_per_episode"] == 20
    assert effective["role_contracts"]["user"]["maximum_generation_calls_per_episode"] == 20
    assert effective["interaction"]["maximum_orchestrator_steps"] == 80
    assert effective["fixed_budget"] == {"episodes": 24}
    assert base["interaction"]["maximum_orchestrator_steps"] == 64


def test_tail_gate_requires_every_integrity_clause():
    gate = {
        "expected_complete_episodes": 1,
        "maximum_forced_budget_stop_episodes": 1,
        "minimum_relative_forced_stop_reduction_vs_paired_parent": 0.5,
        "minimum_natural_user_messages": 1,
        "minimum_adjacent_assistant_tool_transitions": 1,
        "minimum_episodes_with_two_or_more_assistant_transitions": 1,
        "minimum_tasks_with_at_least_one_assistant_transition": 1,
        "minimum_unique_executed_assistant_tools": 1,
        "minimum_agent_tool_decision_rate": 0.35,
        "maximum_agent_tool_decision_rate": 0.9,
        "maximum_dominant_agent_action_fraction": 0.65,
        "minimum_state_changed_assistant_transitions": 1,
        "minimum_state_unchanged_assistant_transitions": 1,
        "minimum_tasks_with_state_changed_assistant_transition": 1,
        "minimum_domains_with_state_changed_assistant_transition": 1,
        "minimum_paired_state_changed_transition_gain": 1,
        "minimum_supported_transition_targets": 1,
        "maximum_assistant_tool_error_rate_increase_over_paired_parent": 0.05,
    }
    metrics = {
        "episodes_complete": 1, "runtime_failures": 0, "agent_private_scenario_exposures": 0,
        "real_external_endpoint_calls": 0, "communication_error_terminations": 0,
        "forced_budget_stop_episodes": 0, "relative_forced_stop_reduction_vs_parent": 1.0,
        "natural_user_messages": 1, "adjacent_assistant_tool_transitions": 2,
        "episodes_with_two_or_more_assistant_transitions": 1, "tasks_with_at_least_one_assistant_transition": 1,
        "unique_executed_assistant_tools": 1, "agent_tool_decision_rate": 0.5,
        "dominant_agent_action_fraction": 0.5, "state_changed_assistant_transitions": 1,
        "state_unchanged_assistant_transitions": 1, "tasks_with_state_changed_assistant_transition": 1,
        "domains_with_state_changed_assistant_transition": 1, "paired_state_changed_transition_gain": 1,
        "supported_transition_targets": 1, "assistant_tool_error_rate_increase_over_parent": 0.0,
    }
    checks = evaluate_tail_gate(metrics, {"frozen": True}, gate)
    assert all(checks.values())
    checks = evaluate_tail_gate(metrics, {"frozen": False}, gate)
    assert not checks["integrity::frozen"]
