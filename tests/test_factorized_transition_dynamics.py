import torch

from wmagentattack.factorized_transition_dynamics import (
    FactorizedSemanticTransitionDynamics,
)
from wmagentattack.factorized_transition_labels import FACTOR_CLASSES


def model():
    return FactorizedSemanticTransitionDynamics(candidate_size=128, hidden_size=96, dropout=0.0)


def test_zero_factor_gate_is_exact_v6_context_noop():
    instance = model(); context = torch.randn(11, 96)
    hidden, logits = instance.initial_hidden(context)
    torch.testing.assert_close(hidden, context, rtol=0, atol=0)
    assert set(logits) == set(FACTOR_CLASSES)


def test_predicted_and_oracle_conditioning_have_same_shape():
    instance = model(); instance.factor_gate.data.fill_(0.2)
    context = torch.randn(7, 96)
    predicted, _ = instance.condition_predicted(context)
    indices = torch.stack([
        torch.randint(len(classes), (7,)) for classes in FACTOR_CLASSES.values()
    ], dim=1)
    oracle = instance.condition_oracle(context, indices)
    assert predicted.shape == oracle.shape == context.shape


def test_oracle_rejects_missing_factor_columns():
    instance = model(); context = torch.randn(3, 96)
    try:
        instance.condition_oracle(context, torch.zeros(3, 2, dtype=torch.long))
    except ValueError as error:
        assert "one oracle index" in str(error)
    else:
        raise AssertionError("invalid oracle factors accepted")
