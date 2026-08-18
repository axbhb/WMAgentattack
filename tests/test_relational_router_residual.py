import numpy as np
import torch

from wmagentattack.relational_router_residual import (
    DenseRelationalSignatureResidual,
    SparseRelationalSignatureResidual,
    parameter_gap_fraction,
    stack_relation_signature_features,
)


def slots() -> dict[str, object]:
    features = np.zeros((3, 5, 40), dtype=np.float32)
    features[:, :, :24] = np.arange(24, dtype=np.float32)
    features[:, :3, 24:] = np.arange(16, dtype=np.float32)
    node_types = np.zeros((3, 5), dtype=np.int64)
    node_types[:, 1] = 4; node_types[:, 2] = 7
    relations = np.zeros((3, 5, 5), dtype=np.int64)
    relations[:, np.arange(3), np.arange(3)] = 1
    relations[:, 1, 2] = 8
    mask = np.zeros((3, 5), dtype=bool); mask[:, :3] = True
    return {"features": features, "node_types": node_types, "relations": relations, "mask": mask}


def models():
    common = dict(candidate_size=128, route_feature_size=50, hidden_size=96, dropout=0.0)
    dense = DenseRelationalSignatureResidual(**common, dense_bottleneck_size=96)
    sparse = SparseRelationalSignatureResidual(
        **common, experts=4, active_experts=2, expert_bottleneck_size=24,
        router_hidden_size=32,
    )
    return dense, sparse


def test_signature_discards_every_lexical_hash_coordinate():
    source = slots()
    left = stack_relation_signature_features(source, hash_dimension=24)
    source["features"][:, :, :24] = np.random.default_rng(7).normal(size=(3, 5, 24))
    right = stack_relation_signature_features(source, hash_dimension=24)
    np.testing.assert_array_equal(left, right)
    assert left.shape == (3, 50)


def test_signature_changes_with_visible_relation_state():
    source = slots()
    left = stack_relation_signature_features(source, hash_dimension=24)
    source["relations"][0, 1, 2] = 9
    source["features"][1, 0, 24] = 99
    right = stack_relation_signature_features(source, hash_dimension=24)
    assert not np.array_equal(left[0], right[0])
    assert not np.array_equal(left[1], right[1])


def test_zero_gates_are_an_exact_v6_noop():
    dense, sparse = models()
    context = torch.randn(6, 96); signature = torch.randn(6, 50)
    dense_hidden, _ = dense.initial_hidden(context, signature)
    sparse_hidden, _ = sparse.initial_hidden(context, signature)
    torch.testing.assert_close(dense_hidden, context, rtol=0, atol=0)
    torch.testing.assert_close(sparse_hidden, context, rtol=0, atol=0)


def test_sparse_router_activates_exactly_two_experts_per_state():
    _, sparse = models()
    weights = sparse.routing_weights(torch.randn(31, 50))
    assert torch.all((weights > 0).sum(1) == 2)
    torch.testing.assert_close(weights.sum(1), torch.ones(31))


def test_sparse_and_dense_capacity_are_preregistered_close():
    dense, sparse = models()
    assert parameter_gap_fraction(dense, sparse) <= 0.02
