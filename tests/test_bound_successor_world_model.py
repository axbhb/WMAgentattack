import numpy as np
import torch

from wmagentattack.bound_successor_world_model import (
    BoundSuccessorRecordTransition,
    clipped_poisson_binomial,
    record_signature,
    render_effect_probabilities,
)


def test_record_signature_is_order_invariant_but_binding_preserving():
    left = {
        "entity_type": "file",
        "link_status": "LINKED",
        "attributes": [
            {"name": "owner", "kind": "SINGLE_VALUED"},
            {"name": "size", "kind": "SINGLE_VALUED"},
        ],
    }
    right = {**left, "attributes": list(reversed(left["attributes"]))}
    other = {**left, "entity_type": "email"}
    assert record_signature(left) == record_signature(right)
    assert record_signature(left) != record_signature(other)


def test_clipped_poisson_binomial_is_normalized_and_has_tail_bin():
    probability = clipped_poisson_binomial(torch.tensor([1.0, 1.0, 1.0, 1.0]))
    assert torch.allclose(probability, torch.tensor([0.0, 0.0, 0.0, 1.0]))
    mixed = clipped_poisson_binomial(torch.tensor([0.2, 0.3]))
    assert torch.isclose(mixed.sum(), torch.tensor(1.0))


def test_renderer_keeps_entity_attribute_binding_and_derives_count():
    records = [
        record_signature({
            "entity_type": "file",
            "link_status": "LINKED",
            "attributes": [{"name": "owner", "kind": "SINGLE_VALUED"}],
        }),
        record_signature({
            "entity_type": "email",
            "link_status": "UNLINKED",
            "attributes": [{"name": "subject", "kind": "SINGLE_VALUED"}],
        }),
    ]
    vocabulary = [
        "entity=file",
        "attribute=file::owner::SINGLE_VALUED",
        "attribute=file::subject::SINGLE_VALUED",
        "matched_count=3",
        "execution=success",
        "delta_bit_0=1",
    ]
    rendered = render_effect_probabilities(
        np.asarray([[0.9, 0.1]]),
        [np.asarray([0.99, 0.99, 0.99, 0.99])],
        np.asarray([[0.8, 0.1, 0.1, 0.1, 0.1]]),
        np.asarray([0.2]),
        np.zeros((1, 0)),
        records,
        [],
        vocabulary,
    )[0]
    assert rendered[0] > 0.85
    assert rendered[1] > 0.85
    assert rendered[2] < 1e-5
    assert rendered[3] > 0.95
    assert rendered[4] == 0.8
    assert rendered[5] == 0.8


def test_model_scores_variable_candidate_sets_and_zero_starts_residual():
    model = BoundSuccessorRecordTransition(8, 6, 12, 10, 7, 5)
    states = torch.randn(3, 8)
    actions = torch.randn(3, 6)
    records = torch.randn(4, 10)
    conflicts = torch.randn(2, 5)
    initial = model.initial_hidden(states)
    following, _ = model.advance_with_execution(initial, actions)
    assert torch.allclose(following, model.next_norm(initial))
    record_logits, delta_logits, conflict_logits, execution = model(
        states, actions, records, conflicts
    )
    assert record_logits.shape == (3, 4)
    assert delta_logits.shape == (3, 5)
    assert conflict_logits.shape == (3, 2)
    assert execution.shape == (3,)
    assert model.pointer_logits(following[:1], torch.randn(9, 7)).shape == (9,)
