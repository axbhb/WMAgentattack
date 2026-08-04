import numpy as np

from wmagentattack.clean_evidence_probe import (
    CleanEvidenceProbe,
    VectorEncoder,
    build_within_task_cyclic_donors,
    fit_progress_then_utility,
    predict_probe,
    task_balanced_weights,
    task_macro_errors,
    vector_features,
)


def _prefix(evidence_text="Hotel A costs 180", state_value="open"):
    return {
        "features": {
            "trusted_goal": "Find Hotel A and report its price",
            "last_event": {
                "tool_name": "get_hotels_prices",
                "arguments": {"hotel_names": ["Hotel A"]},
                "execution_status": "success",
                "error_type": None,
            },
            "state_summary": {
                "last_state_changed": False,
                "cumulative_state_changes": 0,
                "cumulative_errors": 0,
                "delta_roots": {},
            },
            "canonical_state": {"reservation": {"status": state_value}},
            "evidence_text": evidence_text,
            "new_evidence_text": evidence_text,
            "evidence_length": {
                "item_count": 1,
                "character_count": 17,
                "token_count": 4,
                "mean_goal_overlap": 0.5,
                "conflict_item_count": 0,
            },
            "prefix_length": 1,
        },
        "targets": {"expert_slot_coverage": 0.5, "is_final_prefix": True},
    }


def test_output_length_control_cannot_see_evidence_semantics():
    first = _prefix("Hotel A costs 180")
    second = _prefix("Flight ZZ999 departs tomorrow")
    length_first = vector_features(
        first, variant="semantic_markov_state_output_length", hash_dimension=16
    )
    length_second = vector_features(
        second, variant="semantic_markov_state_output_length", hash_dimension=16
    )
    assert np.array_equal(length_first, length_second)
    semantic_first = vector_features(
        first, variant="semantic_markov_state_evidence", hash_dimension=16
    )
    semantic_second = vector_features(
        second, variant="semantic_markov_state_evidence", hash_dimension=16
    )
    assert not np.array_equal(semantic_first, semantic_second)


def test_state_branch_uses_exact_canonical_state():
    first = vector_features(_prefix(state_value="open"), variant="state_only", hash_dimension=16)
    second = vector_features(_prefix(state_value="closed"), variant="state_only", hash_dimension=16)
    assert not np.array_equal(first, second)


def test_within_task_shuffle_is_cyclic_and_never_crosses_tasks():
    episodes = [
        {"episode_id": "a1", "task_id": "a"},
        {"episode_id": "a2", "task_id": "a"},
        {"episode_id": "b1", "task_id": "b"},
        {"episode_id": "b2", "task_id": "b"},
    ]
    donors = build_within_task_cyclic_donors(episodes)
    assert donors == {"a1": "a2", "a2": "a1", "b1": "b2", "b2": "b1"}


def test_task_balanced_weights_give_each_task_equal_mass():
    weights = task_balanced_weights(["a", "a", "a", "b"])
    assert np.isclose(weights[:3].sum(), weights[3:].sum())


def test_progress_encoder_is_frozen_before_utility_head_training():
    inputs = np.asarray(
        [[0.0, 0.0], [0.2, 0.3], [0.8, 0.7], [1.0, 1.0]], dtype=np.float32
    )
    targets = np.asarray([0.0, 0.25, 0.75, 1.0], dtype=np.float32)
    model = CleanEvidenceProbe(VectorEncoder(2, 8, 0.0), 8)
    result = fit_progress_then_utility(
        model,
        inputs=inputs,
        masks=None,
        progress_targets=targets,
        task_ids=["a", "a", "b", "b"],
        final_indices=np.asarray([0, 1, 2, 3]),
        utility_targets=np.asarray([0.0, 0.0, 1.0, 1.0], dtype=np.float32),
        progress_epochs=2,
        utility_epochs=2,
        batch_size=2,
        learning_rate=1e-3,
        weight_decay=0.0,
        seed=7,
        device="cpu",
    )
    assert np.isfinite(result["final_progress_training_loss"])
    assert all(not parameter.requires_grad for parameter in model.encoder.parameters())
    progress, utility = predict_probe(
        model, inputs=inputs, masks=None, batch_size=2, device="cpu"
    )
    assert progress.shape == utility.shape == (4,)
    assert np.all((progress >= 0) & (progress <= 1))
    assert np.all((utility >= 0) & (utility <= 1))


def test_task_macro_errors_do_not_pool_tasks():
    rows = [
        {
            "task_id": "a",
            "progress_prediction": 0.0,
            "progress_target": 0.0,
            "is_final_prefix": True,
            "utility_probability": 0.1,
            "utility_target": 0.0,
        },
        {
            "task_id": "b",
            "progress_prediction": 0.0,
            "progress_target": 1.0,
            "is_final_prefix": True,
            "utility_probability": 0.1,
            "utility_target": 1.0,
        },
    ]
    metrics = task_macro_errors(rows)
    assert metrics["task_macro_progress_mae"] == 0.5
    assert np.isclose(metrics["task_macro_utility_brier"], 0.41)
