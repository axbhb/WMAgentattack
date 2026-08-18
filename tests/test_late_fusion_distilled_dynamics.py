import torch

from wmagentattack.late_fusion_distilled_dynamics import LateFusionDistilledDynamics


def _model():
    return LateFusionDistilledDynamics(graph_size=263, candidate_size=128, hidden_size=96, latent_size=32, dropout=0)


def test_zero_gates_are_exact_noop():
    model = _model(); hidden = torch.randn(3, 96); exact = torch.randn(3, 263); evidence = torch.randn(3, 263)
    torch.testing.assert_close(model.condition(hidden, exact, evidence), hidden, rtol=0, atol=0)


def test_latent_advance_shapes():
    model = _model(); hidden = torch.randn(3, 96); action = torch.randn(3, 128)
    advanced, latent = model.advance_latent(hidden, action)
    assert advanced.shape == (3, 96); assert latent.shape == (3, 32)
