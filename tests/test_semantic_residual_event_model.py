import pytest


torch = pytest.importorskip("torch")

from wmagentattack.semantic_residual_event_model import (
    SemanticResidualEventConfig,
    SemanticResidualEventWorldModel,
    build_skill_token_incidence,
)


def _model():
    names = ["<PAD>", "<UNK>", "<BOS>", "restaurant_read", "restaurant_generate", "finish"]
    token_vocab, incidence = build_skill_token_incidence(names)
    config = SemanticResidualEventConfig(
        num_skills=len(names),
        num_skill_tokens=len(token_vocab),
        semantic_cardinalities=(3, 4),
        num_domains=2,
        num_argument_signatures=4,
        hidden_size=24,
        num_layers=1,
        num_heads=4,
        feedforward_size=48,
        max_sequence_length=5,
    )
    return SemanticResidualEventWorldModel(config, incidence), names, token_vocab, incidence


def test_skill_names_are_compositional_and_cover_unselected_catalog_entries():
    _, names, token_vocab, incidence = _model()
    generate_index = names.index("restaurant_generate")
    assert incidence[generate_index, token_vocab["restaurant"]] > 0
    assert incidence[generate_index, token_vocab["generate"]] > 0
    assert incidence[generate_index].sum().item() == pytest.approx(1.0)


def test_candidate_mask_static_anchor_count_likelihood_and_gradients():
    model, names, _, _ = _model()
    model.eval()
    skill_ids = torch.tensor([[2, 3, 5], [2, 4, 0]])
    attention = torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.bool)
    event_mask = torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.bool)
    candidate_mask = torch.zeros(2, 3, len(names), dtype=torch.bool)
    candidate_mask[:, :, 3:] = True
    outputs = model(
        skill_ids,
        torch.tensor([[1, 2], [2, 3]]),
        torch.tensor([0, 1]),
        torch.tensor([0.8, 0.4]),
        attention,
        candidate_mask,
    )
    assert outputs["next_skill_logits"].shape == (2, 3, len(names))
    assert torch.all(outputs["next_skill_logits"][..., :3] < -1e20)
    assert outputs["static_joint_concentration"].shape == (2, 4)
    assert outputs["dynamic_joint_concentration"].shape == (2, 4)
    assert torch.all(outputs["static_joint_concentration"] > 0)
    losses = model.loss(
        outputs,
        event_loss_mask=event_mask,
        next_skill_targets=torch.tensor([[3, 5, 0], [4, 0, 0]]),
        candidate_mask=candidate_mask,
        argument_signature_targets=torch.tensor([[1, 2, 0], [3, 0, 0]]),
        stop_targets=torch.tensor([[0, 1, 0], [0, 0, 0]]),
        joint_outcome_counts=torch.tensor([[1, 2, 0, 2], [3, 0, 1, 1]]),
        joint_sample_weight=torch.tensor([0.2, 0.2]),
    )
    assert torch.isfinite(losses["total"])
    losses["total"].backward()
    assert model.skill_token_embedding.weight.grad is not None
    assert model.dynamic_joint_residual_head[-1].weight.grad is not None


def test_static_value_is_prefix_invariant_and_disallowed_target_is_rejected():
    model, names, _, _ = _model()
    model.eval()
    candidates = torch.zeros(1, 3, len(names), dtype=torch.bool)
    candidates[:, :, 3:] = True
    common = (
        torch.tensor([[1, 2]]),
        torch.tensor([0]),
        torch.tensor([0.6]),
        torch.ones(1, 3, dtype=torch.bool),
        candidates,
    )
    left = model(torch.tensor([[2, 3, 5]]), *common)
    right = model(torch.tensor([[2, 4, 5]]), *common)
    assert torch.allclose(
        left["static_joint_concentration"], right["static_joint_concentration"]
    )
    bad_targets = torch.tensor([[1, 5, 0]])
    with pytest.raises(ValueError, match="absent from its candidate set"):
        model.loss(
            left,
            event_loss_mask=torch.tensor([[1, 1, 0]], dtype=torch.bool),
            next_skill_targets=bad_targets,
            candidate_mask=candidates,
        )
