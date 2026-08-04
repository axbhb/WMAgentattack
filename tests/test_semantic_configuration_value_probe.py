import importlib.util
from pathlib import Path

import numpy as np
import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "85_probe_v2_semantic_configuration_value.py"
)
SPEC = importlib.util.spec_from_file_location("semantic_probe", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _rows():
    return [
        {"group_id": "a", "task_key": "x|1", "target_asr": 0.0, "target_bup": 0.2, "target": 0.2, "observed": 0.2, "observed_asr": 0.0, "observed_bup": 0.2},
        {"group_id": "b", "task_key": "x|1", "target_asr": 0.4, "target_bup": 0.8, "target": 1.2, "observed": 1.2, "observed_asr": 0.4, "observed_bup": 0.8},
        {"group_id": "c", "task_key": "x|2", "target_asr": 0.2, "target_bup": 0.2, "target": 0.4, "observed": 0.4, "observed_asr": 0.2, "observed_bup": 0.2},
        {"group_id": "d", "task_key": "x|2", "target_asr": 0.6, "target_bup": 0.8, "target": 1.4, "observed": 1.4, "observed_asr": 0.6, "observed_bup": 0.8},
    ]


def test_pairwise_ridge_uses_only_within_task_pairs_and_orders_targets():
    rows = _rows()
    matrix = np.asarray([[0.0], [1.0], [0.2], [1.2]])
    pairs = MODULE._largest_pairs(rows)
    assert len(pairs) == 2
    model = MODULE._ridge_fit(
        matrix, rows, estimator="pairwise_ridge", alpha=0.1
    )
    rank, prediction = MODULE._ridge_predict(model, matrix)
    assert rank[1] > rank[0]
    assert rank[3] > rank[2]
    assert np.all((prediction >= 0.0) & (prediction <= 2.0))


def test_evaluation_is_task_balanced_at_top1():
    rows = _rows()
    result = MODULE._evaluate(
        rows,
        rank_scores=np.asarray([0.0, 1.0, 0.0, 1.0]),
        predictions=np.asarray([0.2, 1.2, 0.4, 1.4]),
    )
    assert result["mean_task_spearman"] == pytest.approx(1.0)
    assert result["mean_top1_target_regret"] == 0.0
    assert result["top1_ASR_plus_BUP"] == pytest.approx(1.3)
    assert result["top1_target_ASR"] == pytest.approx(0.5)
    assert result["top1_target_BUP"] == pytest.approx(0.8)
    assert result["top1_ASR"] == pytest.approx(0.5)
    assert result["top1_BUP"] == pytest.approx(0.8)
    assert result["unique_top1_rate"] == 1.0


def test_structured_features_distinguish_attack_family_and_injection():
    rows = [
        {
            "group_id": "attack::d__user_task_1__injection_task_0__static_control__a"
        },
        {
            "group_id": "attack::d__user_task_1__injection_task_1__static_control__b"
        },
        {
            "group_id": "attack::d__user_task_1__injection_task_0__tool_knowledge__c"
        },
    ]
    vocab = MODULE._structured_vocab(rows)
    matrix = MODULE._structured_matrix(rows, vocab)
    assert matrix.shape[0] == 3
    assert len({tuple(row) for row in matrix}) == 3
