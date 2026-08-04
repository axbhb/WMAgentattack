from __future__ import annotations

import importlib.util
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "139_build_custom_clean_panel_v2.py"
SPEC = importlib.util.spec_from_file_location("custom_panel_v2_manifest", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


def test_all_generated_objects_are_deterministic():
    for function in (
        builder.build_greedy_manifest,
        builder.build_stochastic_manifest,
        builder.build_contract_registry,
        builder.build_run_plan,
    ):
        first = function()
        second = function()
        assert first == second
        assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


def test_greedy_manifest_is_balanced_and_label_blind():
    manifest = builder.build_greedy_manifest()
    assert manifest["summary"]["tasks"] == 48
    assert manifest["summary"]["by_split"] == {
        "calibration": 12,
        "confirmation": 12,
        "training": 24,
    }
    assert manifest["summary"]["by_suite"] == {
        "banking": 12,
        "slack": 12,
        "travel": 12,
        "workspace": 12,
    }
    assert set(manifest["summary"]["by_suite_difficulty_split"].values()) == {1, 2}
    assert manifest["independence_contract"][
        "new_panel_victim_outcomes_read_during_task_authoring"
    ] is False
    forbidden = {"utility", "success", "outcome", "prediction"}
    assert not any(forbidden & set(row) for row in manifest["rows"])


def test_stochastic_subset_is_preselected_and_stratified():
    manifest = builder.build_stochastic_manifest()
    assert manifest["summary"]["tasks"] == 16
    assert manifest["summary"]["by_suite"] == {
        "banking": 4,
        "slack": 4,
        "travel": 4,
        "workspace": 4,
    }
    assert manifest["summary"]["by_split"] == {
        "calibration": 4,
        "confirmation": 4,
        "training": 8,
    }
    assert manifest["summary"]["by_difficulty"] == {"L1": 6, "L2": 5, "L3": 5}
    assert {row["row_id"] for row in manifest["rows"]} == set(
        builder.STOCHASTIC_TASK_IDS
    )


def test_run_plan_uses_the_fixed_48_plus_96_budget():
    plan = builder.build_run_plan()
    assert plan["budget"] == {
        "total_clean_episodes": 144,
        "greedy_independent_tasks": 48,
        "stochastic_tasks": 16,
        "stochastic_samples_per_task": 6,
        "stochastic_episodes": 96,
        "attack_episodes": 0,
        "model_training_runs": 0,
    }
    assert len(plan["episodes"]) == 144
    assert len({row["episode_id"] for row in plan["episodes"]}) == 144
    assert Counter(row["track"] for row in plan["episodes"]) == {
        "deterministic_greedy": 48,
        "stochastic_policy": 96,
    }
    sampled = [row for row in plan["episodes"] if row["track"] == "stochastic_policy"]
    assert all(row["do_sample"] is True for row in sampled)
    assert all(row["temperature"] == 0.7 and row["top_p"] == 0.95 for row in sampled)
    assert all(
        len({row["episode_seed"] for row in sampled if row["row_id"] == task_id}) == 6
        for task_id in builder.STOCHASTIC_TASK_IDS
    )


def test_contract_registry_is_fresh_and_outcome_blind():
    registry = builder.build_contract_registry()
    assert registry["development_only"] is False
    assert registry["barred_from_fresh_confirmation"] is False
    assert registry["frozen_before_first_victim_outcome"] is True
    assert len(registry["contracts"]) == 48
    assert all(contract["outcome_labels_present"] is False for contract in registry["contracts"])
