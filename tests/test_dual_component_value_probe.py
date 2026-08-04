import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "87_probe_v2_dual_component_value.py"
)
SPEC = importlib.util.spec_from_file_location("dual_component_probe", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_guarded_scores_excludes_high_attack_candidate_below_utility_guard():
    rows = [
        {"task_key": "d|1"},
        {"task_key": "d|1"},
        {"task_key": "d|1"},
    ]
    scores = MODULE._guarded_scores(
        rows,
        base_scores=np.asarray([10.0, 1.0, 0.0]),
        utility_predictions=np.asarray([0.5, 0.8, 0.75]),
        tolerance=0.05,
    )
    assert int(np.argmax(scores)) == 1


def test_validation_selection_rejects_recipe_that_reduces_target_bup():
    validation = {
        recipe: {
            "top1_target_ASR_plus_BUP": 1.0,
            "top1_target_BUP": 0.6,
            "mean_task_spearman": 0.0,
        }
        for recipe in MODULE.RECIPES
    }
    validation["joint_control"]["top1_target_ASR_plus_BUP"] = 1.1
    validation["dual_sum"] = {
        "top1_target_ASR_plus_BUP": 1.4,
        "top1_target_BUP": 0.5,
        "mean_task_spearman": 0.5,
    }
    validation["attack_plus_2utility"] = {
        "top1_target_ASR_plus_BUP": 1.2,
        "top1_target_BUP": 0.6,
        "mean_task_spearman": 0.2,
    }
    selected, eligible = MODULE._select_recipe(validation)
    assert not eligible["dual_sum"]
    assert selected == "attack_plus_2utility"


def test_domain_interactions_distinguish_same_family_across_domains():
    rows = [
        {
            "task_key": "banking|user_task_1",
            "group_id": "attack::banking__user_task_1__injection_task_0__tool_knowledge__a",
        },
        {
            "task_key": "workspace|user_task_1",
            "group_id": "attack::workspace__user_task_1__injection_task_0__tool_knowledge__b",
        },
    ]
    vocab = MODULE._domain_interaction_vocab(rows)
    matrix = MODULE._domain_interaction_matrix(rows, vocab)
    assert matrix.shape[0] == 2
    assert not np.allclose(matrix[0], matrix[1])


def test_hierarchical_utility_preserves_domain_family_order_with_shrinkage():
    rows = [
        {
            "task_key": "workspace|user_task_1",
            "group_id": "attack::workspace__user_task_1__injection_task_0__tool_knowledge__a",
            "target_bup": 0.2,
        },
        {
            "task_key": "workspace|user_task_2",
            "group_id": "attack::workspace__user_task_2__injection_task_0__dynamic_multistage__b",
            "target_bup": 0.8,
        },
    ]
    model = MODULE._fit_hierarchical_utility(rows, shrinkage=1.0)
    prediction = MODULE._predict_hierarchical_utility(model, rows)
    assert prediction[1] > prediction[0]
    assert 0.2 < prediction[0] < 0.5
    assert 0.5 < prediction[1] < 0.8
