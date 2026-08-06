from wmagentattack.tau3_horizon import (
    compare_parent_prefixes,
    evaluate_horizon_gate,
    largest_remainder_quotas,
    select_horizon_panel,
)


def _row(domain, task, stratum, seed, split="training"):
    episode = f"{domain}::{task}::{seed}"
    return {
        "episode_id": episode,
        "parent_episode_id": f"old::{episode}",
        "task_key": task,
        "domain": domain,
        "source_split": "train",
        "task_id": task,
        "experimental_split": split,
        "structural_stratum": stratum,
        "llm_seed": seed,
        "shared_model_identity_sha256": "model",
        "role_contract_sha256": "parent-role",
        "agent_interface": {"tool_schemas": [], "policy": "synthetic"},
        "user_private_input": {"scenario": "private", "tool_schemas": []},
    }


def _manifest_and_protocol():
    rows = []
    for domain in ("airline", "retail"):
        for index in range(4):
            for seed in (401, 409):
                rows.append(_row(domain, f"{domain}-m{index}", "mutating", seed))
        for index in range(2):
            for seed in (401, 409):
                rows.append(_row(domain, f"{domain}-r{index}", "read", seed))
    for index in range(6):
        for seed in (401, 409):
            rows.append(_row("telecom", f"telecom-m{index}", "mutating", seed))
    manifest = {
        "schema_version": "test",
        "source_commit": "source",
        "role_contract_sha256": "parent-role",
        "shared_model_identity_sha256": "model",
        "rows": rows,
    }
    protocol = {
        "protocol_id": "horizon-test",
        "parent_result": {"manifest_sha256": "parent"},
        "single_mutable_mechanism": {"name": "role_horizon_extension"},
        "pilot_panel": {
            "domains": ["airline", "retail", "telecom"],
            "tasks_per_domain": 4,
            "tasks": 12,
            "episodes": 24,
            "seeds": [401, 409],
            "selection_inputs_allowed": [
                "domain",
                "task_key",
                "structural_stratum",
                "experimental_split",
                "llm_seed",
            ],
        },
    }
    return manifest, protocol


def test_largest_remainder_preserves_three_to_one_ratio():
    assert largest_remainder_quotas({"mutating": 12, "read": 4}, 4) == {
        "mutating": 3,
        "read": 1,
    }


def test_panel_is_deterministic_and_uses_both_seeds():
    parent, protocol = _manifest_and_protocol()
    first, audit_a = select_horizon_panel(parent, protocol)
    second, audit_b = select_horizon_panel(parent, protocol)
    assert first == second
    assert audit_a == audit_b
    assert audit_a["passed"]
    assert len(first["rows"]) == 24
    assert len({row["task_key"] for row in first["rows"]}) == 12
    for task in {row["task_key"] for row in first["rows"]}:
        assert sorted(
            row["llm_seed"] for row in first["rows"] if row["task_key"] == task
        ) == [401, 409]


def test_prefix_comparison_allows_only_candidate_suffix():
    parent = {
        "agent_decisions": [{"completion": "a"}],
        "user_generations": [{"completion": "u"}],
        "combined_tool_events": [{"action": "read"}],
    }
    candidate = {
        "agent_decisions": [{"completion": "a"}, {"completion": "later"}],
        "user_generations": [{"completion": "u"}],
        "combined_tool_events": [{"action": "read"}, {"action": "write"}],
    }
    assert compare_parent_prefixes(candidate, parent)["all_equal"]
    candidate["agent_decisions"][0]["completion"] = "changed"
    assert not compare_parent_prefixes(candidate, parent)["all_equal"]


def test_horizon_gate_requires_every_scientific_clause():
    gate = {
        "expected_complete_episodes": 24,
        "maximum_forced_budget_stop_episodes": 6,
        "minimum_relative_forced_stop_reduction_vs_paired_parent": 0.5,
        "minimum_adjacent_assistant_tool_transitions": 25,
        "minimum_state_changed_assistant_transitions": 4,
        "minimum_state_unchanged_assistant_transitions": 8,
        "minimum_tasks_with_state_changed_assistant_transition": 2,
        "minimum_domains_with_state_changed_assistant_transition": 2,
        "minimum_paired_state_changed_transition_gain": 3,
        "minimum_supported_transition_targets": 4,
        "maximum_assistant_tool_error_rate_increase_over_paired_parent": 0.05,
    }
    metrics = {
        "episodes_complete": 24,
        "runtime_failures": 0,
        "agent_private_scenario_exposures": 0,
        "real_external_endpoint_calls": 0,
        "forced_budget_stop_episodes": 6,
        "relative_forced_stop_reduction_vs_parent": 0.5,
        "adjacent_assistant_tool_transitions": 25,
        "state_changed_assistant_transitions": 4,
        "state_unchanged_assistant_transitions": 8,
        "tasks_with_state_changed_assistant_transition": 2,
        "domains_with_state_changed_assistant_transition": 2,
        "paired_state_changed_transition_gain": 3,
        "supported_transition_targets": 4,
        "assistant_tool_error_rate_increase_over_parent": 0.05,
    }
    integrity = {
        "all_parent_prefixes_equivalent": True,
        "label_blind_panel_selection": True,
        "exact": True,
    }
    assert all(evaluate_horizon_gate(metrics, integrity, gate).values())
    metrics["state_changed_assistant_transitions"] = 3
    checks = evaluate_horizon_gate(metrics, integrity, gate)
    assert not checks["state_changed_assistant_transitions"]
