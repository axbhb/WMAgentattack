import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "92_summarize_v2_residual_preservation_folds.py"
)
SPEC = importlib.util.spec_from_file_location("residual_fold_summary", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _candidate(task, suffix, score, observed_asr, observed_bup):
    return {
        "group_id": f"{task}-{suffix}",
        "task_key": task,
        "target": observed_asr + observed_bup,
        "target_asr": observed_asr,
        "target_bup": observed_bup,
        "observed": observed_asr + observed_bup,
        "observed_asr": observed_asr,
        "observed_bup": observed_bup,
        "rank_score": score,
        "prediction": observed_asr + observed_bup,
    }


def test_summary_aggregates_frozen_folds_and_compares_candidate_baseline(tmp_path):
    fold_paths = []
    baseline_tasks = {}
    e5_tasks = {}
    for fold in range(2):
        task = f"d|task{fold}"
        rows = [
            _candidate(task, "low", 0.1, 0.0, 0.2),
            _candidate(task, "high", 0.9, 0.4, 0.8),
        ]
        path = tmp_path / f"fold{fold}.json"
        path.write_text(
            json.dumps(
                {
                    "protocol": {
                        "frozen_candidate": {
                            "target_kind": "logit_residual",
                            "alpha": 1.0,
                            "utility_weight": 2.0,
                            "uncertainty_penalty": 0.5,
                        }
                    },
                    "test": {"task_count": 1},
                    "test_candidate_scores": rows,
                }
            ),
            encoding="utf-8",
        )
        fold_paths.append(path)
        baseline_tasks[task] = {"ASR": 0.2, "BUP": 0.6, "ASR_plus_BUP": 0.8}
        e5_tasks[task] = {
            "selected_ASR": 0.3,
            "selected_BUP": 0.7,
            "selected_observed": 1.0,
        }

    dreamer = tmp_path / "dreamer.json"
    dreamer.write_text(
        json.dumps(
            {
                "prospective_oof": {
                    "pct100": {"1": {"by_task": baseline_tasks}}
                }
            }
        ),
        encoding="utf-8",
    )
    e5 = tmp_path / "e5.json"
    e5.write_text(
        json.dumps({"oof_aggregate": {"per_task": e5_tasks}}),
        encoding="utf-8",
    )

    result = MODULE.summarize(
        fold_paths,
        dreamer_summary=dreamer,
        e5_summary=e5,
        bootstrap_samples=1_000,
        bootstrap_seed=7,
    )
    assert result["counts"] == {"tasks": 2, "configurations": 4}
    assert result["oof_aggregate"]["top1_ASR_plus_BUP"] == pytest.approx(1.2)
    assert result["paired_comparisons"]["dreamer"]["metrics"][
        "ASR_plus_BUP"
    ]["mean_delta"] == pytest.approx(0.4)
    assert result["paired_comparisons"]["e5_joint_probe"]["metrics"][
        "BUP"
    ]["mean_delta"] == pytest.approx(0.1)
