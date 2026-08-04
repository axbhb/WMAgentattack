import importlib.util
from pathlib import Path

import numpy as np
import pytest


def _module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "26_candidate_level_ranker.py"
    spec = importlib.util.spec_from_file_location("candidate_level_ranker", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _candidate(task: int, injection: int, security: bool, utility: bool):
    return {
        "suite": "workspace",
        "user_task_id": f"user_task_{task}",
        "injection_task_id": f"injection_task_{injection}",
        "observed_security": security,
        "observed_utility": utility,
        "risk_score": float(security) * 0.8 + 0.1,
        "utility_score": float(utility) * 0.8 + 0.1,
        "target_skill": "record_read",
        "rollout_imagined_skills": ["record_read", "finish"],
        "rollout_top_branch_summaries": [],
    }


def test_alignment_reorders_sources_without_changing_primary_order():
    module = _module()
    rows = [_candidate(0, 0, True, False), _candidate(1, 1, False, True)]
    primary, aligned = module._align_source_rows(
        {"a": rows, "b": list(reversed(rows))}, "a"
    )
    assert primary == rows
    assert [module._pair_key(row) for row in aligned["b"]] == [
        module._pair_key(row) for row in rows
    ]


def test_alignment_rejects_label_mismatch():
    module = _module()
    primary = [_candidate(0, 0, True, False)]
    mismatch = [_candidate(0, 0, False, False)]
    try:
        module._align_source_rows({"a": primary, "b": mismatch}, "a")
    except ValueError as error:
        assert "Label mismatch" in str(error)
    else:
        raise AssertionError("Expected label mismatch to fail")


def test_group_folds_are_disjoint_and_cover_every_candidate():
    module = _module()
    groups = np.asarray([f"g{i // 2}" for i in range(24)])
    labels = np.asarray([i % 4 for i in range(24)])
    folds = module._make_group_folds(
        labels, groups, n_splits=4, random_state=7
    )
    seen = np.zeros(len(groups), dtype=int)
    for train, valid in folds:
        assert not (set(groups[train]) & set(groups[valid]))
        seen[valid] += 1
    assert np.all(seen == 1)


def test_features_exclude_identifiers_and_labels():
    module = _module()
    rows = [_candidate(0, 0, True, False), _candidate(1, 1, False, True)]
    matrix, names = module._feature_dicts(
        rows,
        {"dreamer": rows},
        {("workspace", "user_task_0"): 1.0, ("workspace", "user_task_1"): 0.5},
    )
    assert matrix.shape == (2, len(names))
    assert not any("observed" in name for name in names)
    assert not any("user_task_id" in name for name in names)
    assert not any("injection_task_id" in name for name in names)
    assert not any("trajectory_id" in name for name in names)


def test_pointwise_and_pairwise_predictions_are_finite_probabilities():
    module = _module()
    rng = np.random.default_rng(3)
    x = rng.normal(size=(30, 5))
    y = np.asarray([0, 1] * 15)
    test_x = rng.normal(size=(7, 5))
    for estimator in ("pointwise", "pairwise"):
        scores = module._fit_predict(
            x,
            y,
            test_x,
            estimator=estimator,
            c_value=0.1,
            random_state=5,
            max_pairs=1000,
        )
        assert scores.shape == (7,)
        assert np.all(np.isfinite(scores))
        assert np.all((scores >= 0.0) & (scores <= 1.0))

    ordinal_y = np.asarray([0, 1, 2] * 10)
    for estimator in ("ordinal_pairwise", "ridge"):
        scores = module._fit_predict(
            x,
            ordinal_y,
            test_x,
            estimator=estimator,
            c_value=0.1,
            random_state=5,
            max_pairs=1000,
        )
        assert scores.shape == (7,)
        assert np.all(np.isfinite(scores))
        assert np.all((scores >= 0.0) & (scores <= 2.0))


def test_candidate_annotations_preserve_base_score_and_expose_new_objective():
    module = _module()
    rows = [{**_candidate(0, 0, True, True), "selection_score": 0.25}]
    annotated = module._annotate_candidates(
        rows,
        risk_scores=np.asarray([0.7]),
        utility_scores=np.asarray([0.6]),
        preservation_scores=np.asarray([0.5]),
        joint_scores=np.asarray([1.1]),
        clean_rates={("workspace", "user_task_0"): 0.8},
        fold_ids=np.asarray([2]),
    )
    assert annotated[0]["base_selection_score"] == 0.25
    assert annotated[0]["candidate_objective_score"] == 0.7
    assert annotated[0]["candidate_expected_utility_score"] == 0.4
    assert annotated[0]["candidate_marginal_sum_score"] == pytest.approx(1.3)
    assert annotated[0]["selection_score"] == pytest.approx(1.1)
    assert annotated[0]["candidate_ranker_fold"] == 2
