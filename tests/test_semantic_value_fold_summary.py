import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "86_summarize_v2_semantic_value_folds.py"
)
SPEC = importlib.util.spec_from_file_location("fold_summary", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_fold_summary_requires_disjoint_tasks_and_aggregates(tmp_path):
    paths = []
    for fold in range(2):
        rows = [
            {
                "group_id": f"f{fold}-low",
                "task_key": f"d|task{fold}",
                "target": 0.2,
                "target_asr": 0.0,
                "target_bup": 0.2,
                "observed": 0.2,
                "observed_asr": 0.0,
                "observed_bup": 0.2,
                "rank_score": 0.1,
                "prediction": 0.2,
            },
            {
                "group_id": f"f{fold}-high",
                "task_key": f"d|task{fold}",
                "target": 1.2,
                "target_asr": 0.4,
                "target_bup": 0.8,
                "observed": 1.2,
                "observed_asr": 0.4,
                "observed_bup": 0.8,
                "rank_score": 0.9,
                "prediction": 1.2,
            },
        ]
        path = tmp_path / f"fold{fold}.json"
        path.write_text(
            json.dumps(
                {
                    "frozen_method": {
                        "frozen_candidate": {
                            "representation": "e5_structured",
                            "view": "full",
                            "estimator": "pairwise_ridge",
                            "alpha": 10.0,
                        },
                        "test": {"task_count": 1},
                        "test_candidate_scores": rows,
                    }
                }
            ),
            encoding="utf-8",
        )
        paths.append(path)
    result = MODULE.summarize(paths)
    assert result["counts"] == {"tasks": 2, "configurations": 4}
    assert result["oof_aggregate"]["top1_ASR_plus_BUP"] == 1.2
    assert result["oof_aggregate"]["top1_ASR"] == 0.4
    assert result["oof_aggregate"]["top1_BUP"] == 0.8

    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(
        json.dumps(
            {
                "prospective_oof": {
                    "pct100": {
                        "1": {
                            "by_task": {
                                f"d|task{fold}": {
                                    "ASR": 0.2,
                                    "BUP": 0.7,
                                    "ASR_plus_BUP": 0.9,
                                }
                                for fold in range(2)
                            }
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    compared = MODULE.summarize(
        paths,
        baseline_summary=baseline_path,
        bootstrap_samples=1_000,
        bootstrap_seed=7,
    )["paired_baseline_comparison"]
    assert compared["metrics"]["ASR_plus_BUP"]["mean_delta"] == pytest.approx(0.3)
    assert compared["metrics"]["BUP"]["mean_delta"] == pytest.approx(0.1)
    assert compared["metrics"]["ASR_plus_BUP"]["exact_sign_flip"][
        "one_sided_p_delta_at_least_observed"
    ] == pytest.approx(0.25)
    assert compared["integration_gate"]["decision"] == "PILOT_ONLY_UNCONFIRMED"
