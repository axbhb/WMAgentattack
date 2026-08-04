import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "88_summarize_v2_dual_component_folds.py"
)
SPEC = importlib.util.spec_from_file_location("dual_component_summary", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_dual_component_summary_aggregates_frozen_folds(tmp_path):
    paths = []
    baseline_tasks = {}
    for fold in range(2):
        task = f"d|task{fold}"
        rows = [
            {
                "group_id": f"f{fold}-low",
                "task_key": task,
                "target_asr": 0.0,
                "target_bup": 0.2,
                "target": 0.2,
                "observed_asr": 0.0,
                "observed_bup": 0.2,
                "observed": 0.2,
                "rank_score": 0.1,
                "joint_prediction": 0.2,
            },
            {
                "group_id": f"f{fold}-high",
                "task_key": task,
                "target_asr": 0.4,
                "target_bup": 0.8,
                "target": 1.2,
                "observed_asr": 0.4,
                "observed_bup": 0.8,
                "observed": 1.2,
                "rank_score": 0.9,
                "joint_prediction": 1.2,
            },
        ]
        payload = {
            "protocol": {
                "representation": "e5_structured/full",
                "estimator": "pairwise_ridge",
                "ridge_alpha": 10.0,
                "utility_estimator": "pointwise_ridge",
                "utility_alpha": 10.0,
            },
            "applied_test_recipe": "attack_plus_2utility",
            "test": {"task_count": 1},
            "test_candidate_scores": rows,
        }
        path = tmp_path / f"fold{fold}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        paths.append(path)
        baseline_tasks[task] = {"ASR": 0.2, "BUP": 0.7, "ASR_plus_BUP": 0.9}
    baseline = tmp_path / "baseline.json"
    baseline.write_text(
        json.dumps(
            {"prospective_oof": {"pct100": {"1": {"by_task": baseline_tasks}}}}
        ),
        encoding="utf-8",
    )
    result = MODULE.summarize(
        paths,
        baseline_summary=baseline,
        bootstrap_samples=1_000,
        bootstrap_seed=7,
    )
    assert result["oof_aggregate"]["top1_ASR_plus_BUP"] == 1.2
    assert result["paired_baseline_comparison"]["metrics"]["ASR_plus_BUP"][
        "mean_delta"
    ] == pytest.approx(0.3)
