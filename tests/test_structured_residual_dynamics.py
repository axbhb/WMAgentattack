import torch

from wmagentattack.structured_residual_dynamics import StructuredResidualDynamics


def test_residual_starts_with_exactly_zero_one_step_delta() -> None:
    model=StructuredResidualDynamics(candidate_size=8,hidden_size=16,dropout=0.0)
    delta=model.one_step_delta_logits(torch.randn(4,16),torch.randn(6,8))
    assert torch.equal(delta,torch.zeros_like(delta))


def test_residual_rollout_shapes() -> None:
    model=StructuredResidualDynamics(candidate_size=8,hidden_size=16,dropout=0.0)
    hidden=model.advance(torch.randn(4,16),torch.randn(4,8))
    assert model.rollout_logits(hidden,torch.randn(6,8)).shape==(4,6)
    assert model.projected_context(hidden).shape==(4,16)
    assert model.joint_logits(hidden).shape==(4,4)
