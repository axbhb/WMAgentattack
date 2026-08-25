from __future__ import annotations

import copy

import torch

from wmagentattack.agentdojo_v2 import PairedFactorAttack
from wmagentattack.paired_attack_intervention import (
    ATTACK_VARIANTS,
    CONFIRMATION_TASKS,
    FactorizedStateAttackSelector,
    aggregate_paired_results,
    build_paired_manifest,
    split_preexecution_features,
)


def _source_rows():
    rows = []
    for index, task in enumerate(CONFIRMATION_TASKS):
        suite, user_task = task.split("|", 1)
        rows.append(
            {
                "row_id": f"source-{index}",
                "suite": suite,
                "user_task_id": user_task,
                "task_split": "train",
                "injection_task_id": "injection_task_0",
                "injection_goal": "synthetic goal",
                "attack_family": "static_control",
                "candidate_injection_vectors": ["synthetic_vector"],
                "target_tool_sequence": [{"tool": "synthetic.read", "placeholder_args": {"id": "x"}}],
                "required_tool_depth": 1,
                "base_success_rate": 1.0,
                "base_success_attempts": 3,
            }
        )
    return rows


def test_manifest_is_exact_and_outcome_blind():
    manifest, audit = build_paired_manifest(_source_rows())
    assert audit["passed"]
    assert len(manifest["rows"]) == 40
    assert all("payload" not in row and "security" not in row for row in manifest["rows"])
    assert {row["attack_variant"] for row in manifest["rows"] if row["attack_kind"] == "paired_factor"} == set(ATTACK_VARIANTS)


def test_aggregate_requires_same_three_seeds():
    manifest, _ = build_paired_manifest(_source_rows())
    results = []
    for seed in (503, 509, 521):
        rows = []
        for index, action in enumerate(manifest["rows"]):
            rows.append(
                {
                    "row_id": action["row_id"],
                    "run_seed": seed,
                    "status": "completed",
                    "utility": True,
                    "security": action["attack_kind"] == "paired_factor" and index % 2 == 0,
                }
            )
        results.append({"summary": {"failed": 0}, "results": rows})
    rows, audit = aggregate_paired_results(
        manifest_rows=manifest["rows"], seed_results=results, expected_seeds=(503, 509, 521)
    )
    assert len(rows) == 40
    assert audit["all_rows_have_exact_seeds"]
    assert all(value == 3 for value in audit["clean_successes_by_task"].values())


def test_features_exclude_identity_and_raw_text():
    manifest, _ = build_paired_manifest(_source_rows())
    state, action = split_preexecution_features(manifest["rows"][1])
    keys = " ".join([*state, *action])
    assert "user_task" not in keys
    assert "suite" not in keys
    assert "payload=" not in keys
    assert "payload_text" not in keys
    assert "injection_goal" not in keys


def test_zero_start_selector_returns_base_distribution():
    model = FactorizedStateAttackSelector(state_size=5, action_size=7, hidden_size=8)
    state = torch.randn(3, 5)
    action = torch.randn(3, 7)
    base = torch.tensor([[0.4, 0.3, 0.2, 0.1]]).repeat(3, 1)
    output = torch.softmax(model(state, action, base), dim=1)
    assert torch.allclose(output, base, atol=1e-6)


def test_attack_variants_change_only_declared_execution_factor():
    attack = object.__new__(PairedFactorAttack)
    attack.variant = "fixed_plain"
    attack.get_injection_candidates = lambda _: ["a", "b"]
    injection = type("Injection", (), {"GOAL": "perform synthetic action"})()
    assert set(attack.attack(object(), injection)) == {"a", "b"}
    contextual = copy.copy(attack)
    contextual.variant = "contextual_prefix"
    values = contextual.attack(object(), injection)
    assert set(values) == {"a", "b"}
    assert all("perform synthetic action" in value for value in values.values())
