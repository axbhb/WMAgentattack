import numpy as np

from wmagentattack.uncertainty_ensemble import uniform_categorical_ensemble


def test_identical_members_have_zero_epistemic_uncertainty():
    members = np.asarray([[[0.8, 0.2]], [[0.8, 0.2]], [[0.8, 0.2]]])
    mixture, predictive, expected, epistemic = uniform_categorical_ensemble(members)
    np.testing.assert_allclose(mixture, [[0.8, 0.2]])
    np.testing.assert_allclose(predictive, expected)
    np.testing.assert_allclose(epistemic, 0, atol=1e-12)


def test_disagreement_is_positive_and_uniform_mixture_is_exact():
    members = np.asarray([[[1.0, 0.0]], [[0.0, 1.0]], [[0.5, 0.5]]])
    mixture, predictive, expected, epistemic = uniform_categorical_ensemble(members)
    np.testing.assert_allclose(mixture, [[0.5, 0.5]])
    assert predictive[0] > expected[0]
    assert epistemic[0] > 0


def test_no_member_weight_or_label_input_exists():
    members = np.asarray([[[0.7, 0.3]], [[0.3, 0.7]]])
    mixture, *_ = uniform_categorical_ensemble(members)
    np.testing.assert_allclose(mixture, members.mean(axis=0))
