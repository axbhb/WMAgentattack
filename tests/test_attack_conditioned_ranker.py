from __future__ import annotations

import torch

from wmagentattack.attack_conditioned_ranker import (
    AttackConditionedResidualRanker,
    align_attack_candidates,
    base_distribution_from_p11,
    forbidden_feature_keys,
    four_cell_target_from_label_group,
    ranking_metrics,
    structured_attack_features,
    within_task_pairs,
)
from wmagentattack.joint_outcome_auxiliary import JOINT_OUTCOME_CLASSES


def _manifest_row(task: int, candidate: int) -> dict:
    return {
        "row_id": f"row-{task}-{candidate}",
        "suite": ["banking", "slack", "travel", "workspace"][task % 4],
        "user_task_id": f"user_task_{task}",
        "injection_task_id": f"injection_task_{candidate % 4}",
        "attack_kind": "manifest_payload",
        "attack_role": f"role_{candidate % 5}",
        "attack_family": f"family_{candidate % 5}",
        "attack_name": "name",
        "attack_variant": "variant",
        "knowledge_level": "goal_and_target_tools",
        "endpoint_policy": "all",
        "payload_position": "middle",
        "trigger_stage": "on_external_record",
        "target_tool_sequence": [
            {"step": 1, "tool": "send_message", "placeholder_args": {"recipient": "x"}}
        ],
        "candidate_injection_vectors": ["message_body"],
        "required_tool_depth": 1,
        "payload": "forbidden raw text",
        "injection_goal": "forbidden raw goal",
    }


def test_structured_features_exclude_ids_payload_goal_and_outcomes() -> None:
    row = _manifest_row(0, 0)
    row.update({"utility": True, "security": True, "joint_outcome_counts": {"x": 1}})
    features = structured_attack_features(row)
    assert not forbidden_feature_keys(features)
    assert not any("family_" in key or "attack_name" in key for key in features)
    assert features["target_tool=send_message"] == 1.0
    assert features["target_arg=recipient"] == 1.0


def test_zero_start_ranker_preserves_base_distribution() -> None:
    prior = torch.tensor([0.4, 0.3, 0.2, 0.1])
    base = base_distribution_from_p11(torch.tensor([0.25, 0.75]), prior)
    model = AttackConditionedResidualRanker(input_size=6, hidden_size=8, dropout=0.0)
    probability = torch.softmax(model(torch.randn(2, 6), base), dim=1)
    assert torch.allclose(probability, base, atol=1e-6)
    assert torch.allclose(base.sum(dim=1), torch.ones(2))


def test_legacy_label_group_reconstructs_exact_four_cells() -> None:
    target = four_cell_target_from_label_group(
        {
            "joint_success_probability_trials": 5,
            "attack_probability_successes": 3,
            "utility_probability_successes": 4,
            "joint_success_probability_successes": 2,
        }
    )
    # Empirical counts are n00=0, n01=2, n10=1, n11=2; alpha=0.5 each.
    assert target == [0.5 / 7.0, 2.5 / 7.0, 1.5 / 7.0, 2.5 / 7.0]


def test_alignment_audit_passes_for_rectangular_fixture() -> None:
    manifest = []
    labels = []
    predictions = []
    for task in range(20):
        for candidate in range(20):
            row = _manifest_row(task, candidate)
            manifest.append(row)
            p11 = (candidate % 5 + 0.5) / 7.0
            other = (1.0 - p11) / 3.0
            labels.append(
                {
                    "source_kind": "attack",
                    "row_id": row["row_id"],
                    "joint_outcome_probability_target": {
                        JOINT_OUTCOME_CLASSES[0]: other,
                        JOINT_OUTCOME_CLASSES[1]: other,
                        JOINT_OUTCOME_CLASSES[2]: other,
                        JOINT_OUTCOME_CLASSES[3]: p11,
                    },
                }
            )
            for seed in (7, 17, 29):
                predictions.append(
                    {
                        "arm": "structured_joint_aux",
                        "joint_trainable": 1.0,
                        "joint_group_id": f"attack::{row['row_id']}",
                        "task_name": f"{row['suite']}|{row['user_task_id']}",
                        "fold": task % 5,
                        "joint_p11": p11 + seed * 0.0,
                    }
                )
    aligned, audit = align_attack_candidates(
        manifest_rows=manifest, label_groups=labels, v5_predictions=predictions
    )
    assert audit["passed"] is True
    assert len(aligned) == 400
    assert {row["v5_prediction_count"] for row in aligned} == {3}


def test_pair_builder_and_task_level_ranking() -> None:
    rows = [
        {"row_id": "a", "task_name": "t", "target_p11": 0.2, "score": 0.1},
        {"row_id": "b", "task_name": "t", "target_p11": 0.8, "score": 0.9},
        {"row_id": "c", "task_name": "u", "target_p11": 0.6, "score": 0.7},
        {"row_id": "d", "task_name": "u", "target_p11": 0.1, "score": 0.2},
    ]
    pairs = within_task_pairs(rows, minimum_target_gap=0.1)
    assert pairs.shape == (2, 4)
    metrics = ranking_metrics(rows, "score")
    assert metrics["top1_target_p11"] == 0.7
    assert metrics["top1_regret"] == 0.0
    assert metrics["pairwise_accuracy"] == 1.0
