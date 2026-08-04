import importlib.util
from pathlib import Path

import numpy as np


def _module():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "38_evaluate_hierarchical_contrast_models.py"
    )
    spec = importlib.util.spec_from_file_location("hierarchical_contrast", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _rows():
    module = _module()
    rows = []
    for suite in module.SUITES:
        for task in range(3):
            for injection in range(4):
                rows.append(
                    {
                        "suite": suite,
                        "user_task_id": f"user_task_{task}",
                        "injection_task_id": f"injection_task_{injection}",
                    }
                )
    return rows


def test_suite_balanced_folds_hold_out_whole_tasks():
    module = _module()
    rows = _rows()
    folds = module._suite_balanced_folds(rows, random_state=17)
    assert len(folds) == 3
    for train, valid in folds:
        train_tasks = {module._task_key(rows[index]) for index in train}
        valid_tasks = {module._task_key(rows[index]) for index in valid}
        assert not (train_tasks & valid_tasks)
        assert {task[0] for task in valid_tasks} == set(module.SUITES)


def test_hierarchical_beta_binomial_predicts_finite_probabilities():
    module = _module()
    rows = _rows()
    matrix = np.asarray(
        [
            [index / len(rows), float(index % 4), float(index % 3)]
            for index in range(len(rows))
        ]
    )
    trials = np.full(len(rows), 5)
    successes = np.asarray([index % 6 for index in range(len(rows))])
    train, valid = module._suite_balanced_folds(rows, random_state=19)[0]
    prediction, diagnostics = module._hierarchical_predict_compatible(
        matrix, successes, trials, rows, train, valid
    )
    assert prediction.shape == (len(valid),)
    assert np.isfinite(prediction).all()
    assert ((prediction > 0) & (prediction < 1)).all()
    assert diagnostics["concentration"] > 2.0


def test_beta_binomial_analytic_gradient_matches_finite_difference():
    module = _module()
    matrix = np.asarray(
        [
            [1.0, -0.5],
            [1.0, 0.2],
            [1.0, 0.8],
            [1.0, -0.1],
        ]
    )
    successes = np.asarray([0.0, 2.0, 4.0, 1.0])
    trials = np.full(4, 5.0)
    task_index = np.asarray([0, 0, 1, 1])
    parameters = np.asarray([0.1, -0.2, 0.05, -0.03, np.log(8.0)])

    def value(candidate):
        return module._beta_binomial_value_gradient(
            candidate,
            matrix,
            successes,
            trials,
            task_index,
            2,
            fixed_l2=1.0,
            task_l2=4.0,
        )[0]

    _, analytic = module._beta_binomial_value_gradient(
        parameters,
        matrix,
        successes,
        trials,
        task_index,
        2,
        fixed_l2=1.0,
        task_l2=4.0,
    )
    epsilon = 1e-6
    numerical = np.asarray(
        [
            (
                value(parameters + np.eye(len(parameters))[index] * epsilon)
                - value(parameters - np.eye(len(parameters))[index] * epsilon)
            )
            / (2 * epsilon)
            for index in range(len(parameters))
        ]
    )
    assert np.max(np.abs(analytic - numerical)) < 1e-4


def test_within_task_pairwise_metric_rewards_correct_ordering():
    module = _module()
    rows = [
        {
            "suite": "banking",
            "user_task_id": "user_task_1",
            "injection_task_id": f"injection_task_{index}",
        }
        for index in range(4)
    ]
    rates = np.asarray([0.0, 0.2, 0.6, 1.0])
    good = module._within_task_metrics(rows, rates, rates)
    bad = module._within_task_metrics(rows, rates, rates[::-1])
    assert good["pairwise_accuracy"] == 1.0
    assert bad["pairwise_accuracy"] == 0.0


def test_text_context_baseline_contains_no_world_model_features():
    module = _module()
    rows = [
        {
            "suite": suite,
            "contrast_clean_solvability": 0.5,
            "injection_text": f"untrusted instruction for {suite}",
        }
        for suite in module.SUITES
    ]
    matrix, names = module._feature_matrix(rows, "context_text")
    assert matrix.shape == (4, 37)
    assert any(name.startswith("injection_text_hash_") for name in names)
    assert not any(name.startswith("contrast_risk") for name in names)
    assert not any(name.startswith("injection_contrast") for name in names)


def test_task_bootstrap_ignores_tasks_with_only_tied_outcomes():
    module = _module()
    rows = []
    rates = []
    left = []
    right = []
    for suite in module.SUITES:
        for task, task_rates in (("varying", [0.0, 0.2, 0.8, 1.0]), ("tied", [0.4] * 4)):
            for injection, rate in enumerate(task_rates):
                rows.append(
                    {
                        "suite": suite,
                        "user_task_id": task,
                        "injection_task_id": f"injection_{injection}",
                    }
                )
                rates.append(rate)
                left.append(rate)
                right.append(1.0 - rate)
    rates = np.asarray(rates)
    left = np.asarray(left)
    right = np.asarray(right)
    result = module._task_bootstrap_difference(
        rows,
        left,
        left,
        right,
        right,
        rates,
        rates,
        samples=200,
        seed=11,
    )
    assert np.isfinite(result["pairwise_accuracy_difference"])
    assert np.isfinite(result["pairwise_accuracy_difference_95ci"]).all()
    assert result["informative_pairwise_task_count"] == 4
    assert result["total_task_count"] == 8
