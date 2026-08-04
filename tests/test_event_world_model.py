import pytest

torch = pytest.importorskip("torch")

from wmagentattack.event_world_model import (
    EventWorldModelConfig,
    FactorizedEventWorldModel,
)


def test_event_world_model_shapes_count_likelihood_and_gradients():
    config = EventWorldModelConfig(
        num_tools=8,
        num_attack_contexts=3,
        num_domains=2,
        num_argument_signatures=5,
        hidden_size=32,
        num_layers=1,
        num_heads=4,
        feedforward_size=64,
        max_sequence_length=6,
    )
    model = FactorizedEventWorldModel(config)
    tool_ids = torch.tensor([[2, 3, 4], [2, 5, 0]])
    mask = torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.bool)
    outputs = model(
        tool_ids,
        torch.tensor([1, 2]),
        torch.tensor([0, 1]),
        torch.tensor([0.8, 0.4]),
        mask,
    )
    assert outputs["next_tool_logits"].shape == (2, 3, 8)
    assert outputs["joint_concentration"].shape == (2, 4)
    assert torch.all(outputs["joint_concentration"] > 0)
    losses = model.loss(
        outputs,
        attention_mask=mask,
        next_tool_targets=torch.tensor([[3, 4, 1], [5, 1, 0]]),
        argument_signature_targets=torch.tensor([[1, 2, 0], [3, 0, 0]]),
        stop_targets=torch.tensor([[0, 0, 1], [0, 1, 0]]),
        joint_outcome_counts=torch.tensor([[1, 2, 0, 2], [3, 0, 1, 1]]),
        utility_residual_targets=torch.tensor([0.2, float("nan")]),
    )
    assert torch.isfinite(losses["total"])
    losses["total"].backward()
    assert model.next_tool_head.weight.grad is not None


def test_joint_probabilities_use_attack_then_utility_order():
    concentration = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    probabilities = FactorizedEventWorldModel.outcome_probabilities(concentration)
    assert probabilities["utility"].item() == pytest.approx(0.6)
    assert probabilities["attack"].item() == pytest.approx(0.7)
    assert probabilities["attack_and_utility"].item() == pytest.approx(0.4)

