from wmagentattack.medium_scale_diagnostic import evaluate_medium_scale_gate


def _action(nll, accuracy, *, seed=7, horizon=1, task="task"):
    return {
        "control": "free_latent_residual",
        "training_seed": seed,
        "horizon": horizon,
        "task_name": task,
        "action_nll": nll,
        "action_correct": accuracy,
        "legal_prediction": 1.0,
    }


def _effect(arm, nll, recall, rollout, *, fold=0, seed=7):
    return {
        "arm": arm,
        "split_suite": "task_disjoint",
        "fold_marker": fold,
        "seed": seed,
        "hard_task_macro_bce": 0.04,
        "hard_positive_task_macro_nll": nll,
        "hard_positive_task_macro_recall": recall,
        "execution_brier": 0.02,
        "pair_assignment_accuracy": 0.7,
        "v19_rollout_hard_bce": rollout,
        "unseen_positive_recall": 0.0,
    }


def test_medium_scale_gate_passes_improving_paired_candidate():
    baseline_action = []
    candidate_action = []
    for seed in (7, 17, 29):
        for horizon in (1, 2, 3, 4, 5):
            baseline_action.append(_action(1.0, 0.5, seed=seed, horizon=horizon))
            candidate_action.append(_action(0.9, 0.6, seed=seed, horizon=horizon))
    baseline_effect = [_effect("small", 0.2, 0.95, 0.03, fold=i, seed=s) for i, s in enumerate((7, 17, 29))]
    candidate_effect = [_effect("medium", 0.18, 0.96, 0.025, fold=i, seed=s) for i, s in enumerate((7, 17, 29))]
    thresholds = {
        "effect_baseline_arm": "small",
        "effect_candidate_arm": "medium",
        "required_model_fits": 39,
        "minimum_combined_parameters": 3_000_000,
        "maximum_combined_parameters": 10_000_000,
        "minimum_paired_key_coverage": 1.0,
        "maximum_h1_nll_degradation": 0.01,
        "maximum_h1_accuracy_degradation": 0.005,
        "minimum_h2_h5_nll_gain": 0.01,
        "minimum_h2_h5_positive_task_fraction": 0.55,
        "minimum_h2_h5_positive_seeds": 2,
        "maximum_effect_positive_nll_degradation": 0.03,
        "maximum_effect_recall_degradation": 0.01,
        "maximum_effect_rollout_bce_degradation": 0.01,
        "minimum_effect_noninferior_folds": 2,
        "minimum_effect_noninferior_seeds": 2,
    }
    result = evaluate_medium_scale_gate(
        action_baseline_rows=baseline_action,
        action_candidate_rows=candidate_action,
        effect_baseline_rows=baseline_effect,
        effect_candidate_rows=candidate_effect,
        training_metrics={
            "completed_model_fits": 39,
            "runtime_failures": 0,
            "device": "cuda",
            "parameter_counts": {
                "action_teacher": 1_000_000,
                "action_residual": 3_000_000,
                "effect_transition": 2_000_000,
            },
        },
        thresholds=thresholds,
    )
    assert result["decision"] == "GO_MEDIUM_SCALE_CAPACITY_V32"
    assert all(result["clauses"].values())


def test_medium_scale_gate_rejects_capacity_overfit():
    baseline_action = [_action(1.0, 0.5, horizon=h) for h in (1, 2, 3, 4, 5)]
    candidate_action = [_action(1.3, 0.3, horizon=h) for h in (1, 2, 3, 4, 5)]
    baseline_effect = [_effect("small", 0.2, 0.95, 0.03)]
    candidate_effect = [_effect("medium", 0.8, 0.5, 0.2)]
    thresholds = {
        "effect_baseline_arm": "small",
        "effect_candidate_arm": "medium",
        "required_model_fits": 39,
        "minimum_combined_parameters": 3_000_000,
        "maximum_combined_parameters": 10_000_000,
        "minimum_paired_key_coverage": 1.0,
        "maximum_h1_nll_degradation": 0.01,
        "maximum_h1_accuracy_degradation": 0.005,
        "minimum_h2_h5_nll_gain": 0.01,
        "minimum_h2_h5_positive_task_fraction": 0.55,
        "minimum_h2_h5_positive_seeds": 2,
        "maximum_effect_positive_nll_degradation": 0.03,
        "maximum_effect_recall_degradation": 0.01,
        "maximum_effect_rollout_bce_degradation": 0.01,
        "minimum_effect_noninferior_folds": 2,
        "minimum_effect_noninferior_seeds": 2,
    }
    result = evaluate_medium_scale_gate(
        action_baseline_rows=baseline_action,
        action_candidate_rows=candidate_action,
        effect_baseline_rows=baseline_effect,
        effect_candidate_rows=candidate_effect,
        training_metrics={
            "completed_model_fits": 39,
            "runtime_failures": 0,
            "device": "cuda",
            "parameter_counts": {
                "action_teacher": 1_000_000,
                "action_residual": 3_000_000,
                "effect_transition": 2_000_000,
            },
        },
        thresholds=thresholds,
    )
    assert result["decision"] == "NO_GO_MEDIUM_SCALE_CAPACITY_V32"
    assert not result["clauses"]["h1_nll_noninferiority"]
