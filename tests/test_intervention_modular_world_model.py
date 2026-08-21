import torch

from wmagentattack.intervention_modular_world_model import (
    DirectStructuredTransition,
    InterventionModularTransition,
    RecurrentResidualTransition,
    assert_transition_only,
    trainable_parameter_count,
)


def test_all_v20_arms_emit_effect_and_execution_logits() -> None:
    state = torch.randn(4, 18)
    action = torch.randn(4, 9)
    for model in (
        DirectStructuredTransition(18, 9, 16, 7),
        RecurrentResidualTransition(18, 9, 16, 7),
        InterventionModularTransition(18, 9, 16, 7),
    ):
        effects, execution = model(state, action)
        assert effects.shape == (4, 7)
        assert execution.shape == (4,)
        assert trainable_parameter_count(model) > 0
        assert_transition_only(model)


def test_modular_residual_is_exactly_zero_initialized() -> None:
    model = InterventionModularTransition(18, 9, 16, 7)
    hidden = model.initial_hidden(torch.randn(4, 18))
    following, _ = model.advance_with_execution(hidden, torch.randn(4, 9))
    expected = model.next_norm(hidden)
    assert torch.equal(following, expected)


def test_v6_style_residual_gate_starts_at_identity() -> None:
    model = RecurrentResidualTransition(18, 9, 16, 7)
    hidden = model.initial_hidden(torch.randn(4, 18))
    following = model.advance(hidden, torch.randn(4, 9))
    assert torch.equal(following, hidden)
