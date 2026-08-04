from copy import deepcopy

import numpy as np
import pytest

from wmagentattack.markov_sufficiency import (
    FROZEN_SUFFICIENCY_VARIANTS,
    evaluate_sufficiency_gate,
    full_history_diagnostic_feature_vector,
    representation_feature_size,
    representation_feature_vector,
    semantic_markov_feature_vector,
    validate_dataset_alignment,
)
from wmagentattack.semantic_state_v3 import semantic_state_v3_payload


def _source_prefix(index: int, observation: str = ""):
    return {
        "prefix_index": index,
        "features": {
            "trusted_goal": "Find the London hotel price.",
            "track": "deterministic_greedy",
            "prefix_index": index,
            "legal_tools": ["STOP", "travel::lookup"],
            "last_action": (
                {"function": "<START>", "arguments": {}}
                if index == 0
                else {"function": "lookup", "arguments": {"city": "London"}}
            ),
            "last_observation": observation,
            "execution_receipt": {
                "status": "start" if index == 0 else "success",
                "error_type": None,
            },
            "causal_state_summary": {"hidden_oracle": index},
            "ledger_v2": {
                "records": [],
                "conflicts": [],
                "execution_receipts": [],
            },
        },
        "targets": {
            "next_action": "travel::lookup" if index == 0 else "STOP",
            "argument_keys": [] if index == 0 else ["city"],
            "stop": index > 0,
            "evidence_obligations": [],
        },
    }


def _semantic_prefix(source):
    return {
        "prefix_index": source["prefix_index"],
        "features": {
            "semantic_state_v3": semantic_state_v3_payload(source["features"])
        },
        "targets": deepcopy(source["targets"]),
    }


def test_all_representations_have_identical_frozen_dimension_and_are_stable():
    source = [_source_prefix(0), _source_prefix(1, "Hotel A costs 250")]
    semantic = [_semantic_prefix(row) for row in source]
    rows = {}
    for variant in FROZEN_SUFFICIENCY_VARIANTS:
        first = representation_feature_vector(
            variant=variant,
            source_prefixes=source,
            semantic_prefixes=semantic,
            prefix_index=1,
            hash_dimension=16,
        )
        second = representation_feature_vector(
            variant=variant,
            source_prefixes=deepcopy(source),
            semantic_prefixes=deepcopy(semantic),
            prefix_index=1,
            hash_dimension=16,
        )
        assert first.shape == (representation_feature_size(16),)
        assert np.array_equal(first, second)
        rows[variant] = first
    assert not np.array_equal(
        rows["semantic_markov"], rows["structured_markov_v3"]
    )


def test_semantic_markov_ignores_hidden_oracle_and_raw_observation():
    baseline = _source_prefix(1, "first observation")
    changed = deepcopy(baseline)
    changed["features"]["causal_state_summary"] = {"secret": 999}
    changed["features"]["last_observation"] = "different observation"
    assert np.array_equal(
        semantic_markov_feature_vector(baseline, hash_dimension=16),
        semantic_markov_feature_vector(changed, hash_dimension=16),
    )


def test_full_history_uses_earlier_visible_observations_but_not_future_rows():
    first = [_source_prefix(0), _source_prefix(1, "visible now")]
    changed = deepcopy(first)
    changed[0]["features"]["last_observation"] = "changed earlier evidence"
    assert not np.array_equal(
        full_history_diagnostic_feature_vector(
            first, prefix_index=1, hash_dimension=16
        ),
        full_history_diagnostic_feature_vector(
            changed, prefix_index=1, hash_dimension=16
        ),
    )
    future = [*deepcopy(first), _source_prefix(2, "future secret")]
    assert np.array_equal(
        full_history_diagnostic_feature_vector(
            first, prefix_index=1, hash_dimension=16
        ),
        full_history_diagnostic_feature_vector(
            future, prefix_index=1, hash_dimension=16
        ),
    )


def test_outcome_fields_fail_closed():
    prefix = _source_prefix(0)
    prefix["features"]["utility"] = True
    with pytest.raises(ValueError, match="leakage"):
        semantic_markov_feature_vector(prefix, hash_dimension=8)


def _dataset():
    source = [_source_prefix(0), _source_prefix(1, "Hotel A")]
    semantic = [_semantic_prefix(row) for row in source]
    metadata = {
        "episode_id": "episode-1",
        "task_id": "task-1",
        "suite": "travel",
        "split": "training",
        "track": "deterministic_greedy",
        "run_seed": 7,
    }
    return (
        {"episodes": [{**metadata, "prefixes": source}]},
        {"episodes": [{**metadata, "prefixes": semantic}]},
    )


def test_alignment_accepts_identical_identity_and_targets():
    source, semantic = _dataset()
    validate_dataset_alignment(source, semantic)


def test_alignment_rejects_target_or_task_changes():
    source, semantic = _dataset()
    semantic["episodes"][0]["task_id"] = "different"
    with pytest.raises(ValueError, match="alignment"):
        validate_dataset_alignment(source, semantic)


def _gates():
    return {
        "minimum_action_nll_gain": 0.02,
        "minimum_evidence_bce_gain": 0.01,
        "minimum_threshold_positive_seeds": 2,
        "minimum_confirmation_positive_tasks": 6,
        "maximum_action_nll_gap_to_full_history": 0.05,
        "maximum_evidence_bce_gap_to_full_history": 0.02,
    }


def test_frozen_sufficiency_gate_requires_both_heads_and_full_history_control():
    checks = evaluate_sufficiency_gate(
        action_seed_gains=[0.03, 0.02, 0.01],
        evidence_seed_gains=[0.02, 0.01, 0.0],
        action_task_gains=[1.0] * 6 + [-1.0] * 6,
        evidence_task_gains=[1.0] * 7 + [-1.0] * 5,
        structured_minus_full_action_nll=0.04,
        structured_minus_full_evidence_bce=0.01,
        gates=_gates(),
    )
    assert all(checks.values())


def test_frozen_sufficiency_gate_fails_when_evidence_does_not_replicate():
    checks = evaluate_sufficiency_gate(
        action_seed_gains=[0.03, 0.03, 0.03],
        evidence_seed_gains=[0.02, 0.0, -0.01],
        action_task_gains=[1.0] * 8 + [-1.0] * 4,
        evidence_task_gains=[1.0] * 5 + [-1.0] * 7,
        structured_minus_full_action_nll=0.0,
        structured_minus_full_evidence_bce=0.0,
        gates=_gates(),
    )
    assert not checks["structured_evidence_mean_gain"]
    assert not checks["structured_evidence_seed_replication"]
    assert not checks["structured_evidence_paired_tasks"]
    evaluate_sufficiency_gate,
