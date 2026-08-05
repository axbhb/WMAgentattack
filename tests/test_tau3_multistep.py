import copy

import numpy as np

from wmagentattack.markov_sufficiency import (
    representation_feature_size,
    semantic_markov_feature_vector,
)
from wmagentattack.tau3_multistep import (
    allocate_stratum,
    append_ledger_event,
    build_dataset,
    candidate_id,
    observed_semantic_markov_v4_feature_vector,
    source_prefix,
    stable_hash,
    transition_target,
)


def _schema(name="lookup"):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": "Look up a customer record.",
            "parameters": {
                "type": "object",
                "properties": {"customer_id": {"type": "string"}},
                "required": ["customer_id"],
            },
        },
    }


def _model_input():
    return {
        "trusted_goal": "Find customer Alice and report the status.",
        "tool_schemas": [_schema()],
        "policy": "Use tools when needed.",
    }


def _event(output=None, *, changed=False, status="success"):
    return {
        "index": 0,
        "action": {"name": "lookup", "arguments": {"customer_id": "alice"}},
        "status": status,
        "error": None if status == "success" else {"type": "LookupError"},
        "output": {"name": "Alice", "status": "active"}
        if output is None
        else output,
        "state_before_sha256": "before",
        "state_after_sha256": "after" if changed else "before",
        "state_changed": changed,
        "replica_identical": True,
    }


def test_stratified_task_allocation_is_deterministic_and_disjoint():
    keys = [stable_hash(["task", index]) for index in range(12)]
    counts = {"training": 8, "calibration": 2, "confirmation": 2}
    first = allocate_stratum(
        keys, seed="frozen", domain="retail", stratum="mutating", counts=counts
    )
    second = allocate_stratum(
        list(reversed(keys)),
        seed="frozen",
        domain="retail",
        stratum="mutating",
        counts=counts,
    )
    assert first == second
    assert {split: list(first.values()).count(split) for split in counts} == counts
    assert set(first) == set(keys)


def test_prefix_is_causal_label_blind_and_maps_only_presented_tools():
    prefix = source_prefix(
        episode_id="episode",
        domain="retail",
        model_input=_model_input(),
        prefix_index=0,
        prior_events=[],
        ledger={"records": [], "conflicts": [], "execution_receipts": []},
        decision={
            "kind": "tool_call",
            "name": "lookup",
            "arguments": {"customer_id": "alice"},
        },
    )
    assert prefix["targets"]["next_action"] == candidate_id("retail", _schema())
    assert prefix["targets"]["next_action"] in prefix["features"]["legal_tools"]
    assert "targets" not in prefix["features"]
    assert "state_changed" not in prefix["features"]


def test_observation_aware_v4_uses_visible_receipt_without_width_increase():
    event = _event()
    ledger = append_ledger_event(
        {"records": [], "conflicts": [], "execution_receipts": []},
        episode_id="episode",
        domain="retail",
        event=event,
    )
    prefix = source_prefix(
        episode_id="episode",
        domain="retail",
        model_input=_model_input(),
        prefix_index=1,
        prior_events=[event],
        ledger=ledger,
        decision={"kind": "text"},
    )
    changed = copy.deepcopy(prefix)
    changed["features"]["last_observation"] = "TOOL_OUTPUT different visible fact"
    dimension = 16
    semantic_a = semantic_markov_feature_vector(prefix, hash_dimension=dimension)
    semantic_b = semantic_markov_feature_vector(changed, hash_dimension=dimension)
    v4_a = observed_semantic_markov_v4_feature_vector(
        prefix, hash_dimension=dimension
    )
    v4_b = observed_semantic_markov_v4_feature_vector(
        changed, hash_dimension=dimension
    )
    assert np.array_equal(semantic_a, semantic_b)
    assert not np.array_equal(v4_a, v4_b)
    assert v4_a.shape == (representation_feature_size(dimension),)
    assert np.isfinite(v4_a).all()


def test_transition_targets_use_only_current_and_prior_visible_events():
    first = _event(changed=True)
    target = transition_target(
        trusted_goal="Find Alice and report active status.",
        prior_events=[],
        event=first,
    )
    assert target == {
        "state_changed": 1.0,
        "execution_error": 0.0,
        "output_nonempty": 1.0,
        "goal_overlap_gained": 1.0,
        "novel_observation": 1.0,
    }
    repeated = _event(changed=False)
    repeated_target = transition_target(
        trusted_goal="Find Alice and report active status.",
        prior_events=[first],
        event=repeated,
    )
    assert repeated_target["goal_overlap_gained"] == 0.0
    assert repeated_target["novel_observation"] == 0.0


def test_dataset_builder_preserves_task_split_and_adjacent_transition():
    model_input = _model_input()
    event = _event(changed=True)
    empty = {"records": [], "conflicts": [], "execution_receipts": []}
    prefix0 = source_prefix(
        episode_id="episode",
        domain="retail",
        model_input=model_input,
        prefix_index=0,
        prior_events=[],
        ledger=empty,
        decision={
            "kind": "tool_call",
            "name": "lookup",
            "arguments": {"customer_id": "alice"},
        },
    )
    ledger = append_ledger_event(
        empty, episode_id="episode", domain="retail", event=event
    )
    prefix1 = source_prefix(
        episode_id="episode",
        domain="retail",
        model_input=model_input,
        prefix_index=1,
        prior_events=[event],
        ledger=ledger,
        decision={"kind": "text"},
    )
    manifest = {
        "protocol_id": "test",
        "rows": [
            {
                "episode_id": "episode",
                "model_input": model_input,
            }
        ],
    }
    episodes = [
        {
            "episode_id": "episode",
            "task_key": "task",
            "domain": "retail",
            "split": "training",
            "llm_seed": 401,
            "prefixes": [prefix0, prefix1],
            "transitions": [event],
            "termination": "text_response",
        }
    ]
    dataset, audit = build_dataset(manifest, episodes)
    assert audit["episodes"] == 1
    assert audit["prefixes"] == 2
    assert audit["adjacent_transitions"] == 1
    assert audit["task_disjoint"]
    assert audit["causal_label_blind_states"]
    assert dataset["episodes"][0]["transitions"][0]["prefix_index"] == 0
