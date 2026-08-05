import numpy as np

from wmagentattack.tau3_multistep import (
    append_ledger_event,
    build_dataset,
    source_prefix,
)
from wmagentattack.tau3_multistep_experiment import (
    NEURAL_VARIANTS,
    build_arrays,
    evaluate_method_gate,
    flatten_dataset,
    frequency_action_probabilities,
    frequency_transition_probabilities,
    task_balanced_weights,
    two_step_task_map,
)


TARGET_NAMES = (
    "state_changed",
    "execution_error",
    "output_nonempty",
    "goal_overlap_gained",
    "novel_observation",
)


def _model_input():
    return {
        "trusted_goal": "Find Alice and report active status.",
        "policy": "Use only the provided tool.",
        "tool_schemas": [
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "description": "Lookup a customer.",
                    "parameters": {
                        "type": "object",
                        "properties": {"customer_id": {"type": "string"}},
                        "required": ["customer_id"],
                    },
                },
            }
        ],
    }


def _episode(episode_id, task, split, changed):
    model_input = _model_input()
    empty = {"records": [], "conflicts": [], "execution_receipts": []}
    decision = {
        "kind": "tool_call",
        "name": "lookup",
        "arguments": {"customer_id": "alice"},
    }
    prefix0 = source_prefix(
        episode_id=episode_id,
        domain="retail",
        model_input=model_input,
        prefix_index=0,
        prior_events=[],
        ledger=empty,
        decision=decision,
    )
    event = {
        "index": 0,
        "action": {"name": "lookup", "arguments": {"customer_id": "alice"}},
        "status": "success",
        "error": None,
        "output": {"name": "Alice", "status": "active"},
        "state_before_sha256": "before",
        "state_after_sha256": "after" if changed else "before",
        "state_changed": changed,
        "replica_identical": True,
    }
    ledger = append_ledger_event(
        empty, episode_id=episode_id, domain="retail", event=event
    )
    prefix1 = source_prefix(
        episode_id=episode_id,
        domain="retail",
        model_input=model_input,
        prefix_index=1,
        prior_events=[event],
        ledger=ledger,
        decision={"kind": "text"},
    )
    manifest_row = {"episode_id": episode_id, "model_input": model_input}
    episode = {
        "episode_id": episode_id,
        "task_key": task,
        "domain": "retail",
        "split": split,
        "llm_seed": 401,
        "prefixes": [prefix0, prefix1],
        "transitions": [event],
        "termination": "text_response",
    }
    return manifest_row, episode


def _dataset():
    train_manifest, train = _episode("train", "task-train", "training", True)
    confirm_manifest, confirm = _episode(
        "confirm", "task-confirm", "confirmation", False
    )
    return build_dataset(
        {"protocol_id": "test", "rows": [train_manifest, confirm_manifest]},
        [train, confirm],
    )[0]


def test_flatten_and_all_neural_representations_keep_the_same_surface():
    dataset = _dataset()
    prefixes, transitions = flatten_dataset(dataset, TARGET_NAMES)
    assert len(prefixes) == 4
    assert len(transitions) == 2
    shapes = set()
    candidates = None
    for variant in NEURAL_VARIANTS:
        arrays = build_arrays(
            prefixes,
            dataset["candidate_catalog"],
            variant=variant,
            hash_dimension=16,
        )
        shapes.add(arrays["states"].shape)
        candidates = candidates or arrays["candidates"]
        assert arrays["candidates"] == candidates
        assert np.all(arrays["legal"].sum(axis=1) >= 1)
    assert shapes == {(4, 130)}


def test_frequency_controls_are_legal_and_use_training_only():
    dataset = _dataset()
    prefixes, transitions = flatten_dataset(dataset, TARGET_NAMES)
    arrays = build_arrays(
        prefixes,
        dataset["candidate_catalog"],
        variant="semantic_markov",
        hash_dimension=8,
    )
    action = frequency_action_probabilities(prefixes, arrays)
    assert np.allclose(action.sum(axis=1), 1.0)
    assert np.all(action[~arrays["legal"]] == 0.0)
    transition = frequency_transition_probabilities(
        transitions, target_count=len(TARGET_NAMES)
    )
    assert transition.shape == (2, 5)
    assert np.all((transition > 0.0) & (transition < 1.0))


def test_task_balancing_and_two_step_metric_use_tasks_as_units():
    weights = task_balanced_weights(["a", "a", "b"])
    assert np.isclose(weights[:2].sum(), weights[2])
    rows = [
        {
            "task_id": "a",
            "episode_id": "e",
            "split": "confirmation",
            "prefix_index": 0,
            "action_correct": 1.0,
        },
        {
            "task_id": "a",
            "episode_id": "e",
            "split": "confirmation",
            "prefix_index": 1,
            "action_correct": 0.0,
        },
    ]
    assert two_step_task_map(rows, split="confirmation") == {"a": 0.0}


def test_method_gate_requires_every_preregistered_clause():
    gate = {
        "minimum_candidate_nll_gain_over_frequency": 0.05,
        "minimum_candidate_accuracy_gain_over_frequency": 0.03,
        "minimum_threshold_positive_seeds": 2,
        "minimum_positive_task_fraction": 0.6,
        "maximum_candidate_nll_gap_to_full_history": 0.02,
        "minimum_two_step_sequence_accuracy_gain_over_frequency": 0.03,
        "minimum_transition_brier_gain_over_frequency": 0.01,
        "require_legal_prediction_rate": 1.0,
    }
    passed = evaluate_method_gate(
        nll_seed_gains=[0.06, 0.07, 0.04],
        accuracy_seed_gains=[0.04, 0.05, 0.02],
        paired_task_nll_gains=[0.1, 0.1, 0.1, -0.1, 0.1],
        candidate_minus_full_history_nll=0.01,
        two_step_seed_gains=[0.04, 0.05, 0.01],
        transition_brier_seed_gains=[0.02, 0.03, 0.0],
        legal_prediction_rate=1.0,
        data_gate_passed=True,
        two_step_surface_available=True,
        gate=gate,
    )
    assert all(passed.values())
    failed = evaluate_method_gate(
        nll_seed_gains=[0.06, 0.07, 0.04],
        accuracy_seed_gains=[0.04, 0.05, 0.02],
        paired_task_nll_gains=[0.1, 0.1, 0.1, -0.1, 0.1],
        candidate_minus_full_history_nll=0.01,
        two_step_seed_gains=[],
        transition_brier_seed_gains=[0.02, 0.03, 0.0],
        legal_prediction_rate=1.0,
        data_gate_passed=True,
        two_step_surface_available=False,
        gate=gate,
    )
    assert not failed["two_step_surface_available"]
    assert not failed["two_step_mean_gain"]
