import numpy as np

from wmagentattack.multisource_suitability_experiment import (
    error_probe_supported,
    evaluate_action_gate,
    evaluate_error_gate,
    exact_sign_test,
    task_balanced_weights,
)


def test_task_balanced_weights_give_each_task_equal_mass():
    weights = task_balanced_weights(["a", "a", "a", "b"])
    assert np.isclose(weights[:3].sum(), weights[3:].sum())
    assert np.isclose(weights.mean(), 1.0)


def test_error_probe_support_requires_both_classes_in_train_and_confirmation():
    rows = []
    for split, errors, successes in (
        ("training", 3, 20),
        ("calibration", 1, 10),
        ("confirmation", 2, 14),
    ):
        for index in range(errors + successes):
            rows.append(
                {
                    "split": split,
                    "exact_outcome": {
                        "available": True,
                        "execution_error": index < errors,
                    },
                }
            )
    gate = {
        "minimum_exact_rows_for_error_probe": 50,
        "minimum_exact_errors_for_error_probe": 6,
        "minimum_exact_successes_for_error_probe": 10,
        "minimum_each_error_class_per_training_and_confirmation": 2,
    }
    supported, counts = error_probe_supported(rows, gate)
    assert supported
    assert counts["errors"] == 6
    rows[-16]["exact_outcome"]["execution_error"] = False
    supported, _ = error_probe_supported(rows, gate)
    assert not supported


def test_action_gate_enforces_seed_task_lexical_and_legal_checks():
    gate = {
        "minimum_structured_nll_gain_over_frequency": 0.05,
        "minimum_structured_accuracy_gain_over_frequency": 0.02,
        "minimum_threshold_positive_seeds": 2,
        "minimum_positive_task_fraction": 0.5,
        "maximum_structured_nll_gap_to_tfidf": 0.02,
    }
    checks = evaluate_action_gate(
        nll_seed_gains=[0.06, 0.07, 0.04],
        accuracy_seed_gains=[0.03, 0.04, 0.0],
        paired_nll_task_gains=[0.1, 0.2, -0.1, 0.0],
        structured_nll_gap_to_tfidf=0.01,
        legal_prediction_rate=1.0,
        gate=gate,
    )
    assert all(checks.values())
    failed = evaluate_action_gate(
        nll_seed_gains=[0.06, 0.07, 0.04],
        accuracy_seed_gains=[0.03, 0.04, 0.0],
        paired_nll_task_gains=[0.1, 0.2, -0.1, 0.0],
        structured_nll_gap_to_tfidf=0.03,
        legal_prediction_rate=1.0,
        gate=gate,
    )
    assert not failed["structured_within_tfidf_nll_gap"]


def test_error_gate_and_sign_test_keep_counterevidence():
    checks = evaluate_error_gate(
        bce_seed_gains=[0.02, 0.03, 0.0],
        paired_bce_task_gains=[0.1, 0.1, -0.1, -0.1],
        gate={
            "minimum_bce_gain_over_frequency": 0.01,
            "minimum_threshold_positive_seeds": 2,
            "minimum_positive_task_fraction": 0.5,
        },
    )
    assert all(checks.values())
    sign = exact_sign_test([1.0, 1.0, -1.0, 0.0])
    assert sign == {"wins": 2, "losses": 1, "ties": 1, "p_value": 1.0}
