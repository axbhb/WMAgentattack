import importlib.util
import json
from pathlib import Path

import numpy as np


def _module():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "32_fit_replay_probability_calibrators.py"
    )
    spec = importlib.util.spec_from_file_location("replay_probability", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _rows(count=24):
    module = _module()
    rows = []
    for index in range(count):
        probability = 0.08 + 0.84 * index / max(1, count - 1)
        rows.append(
            {
                "suite": module.SUITES[index % len(module.SUITES)],
                "user_task_id": f"user_task_{index // 2}",
                "injection_task_id": f"injection_task_{index % 2}",
                "candidate_risk_score": probability,
                "risk_score": 0.1 + 0.8 * probability,
                "target_skill_probability": 0.2 + 0.5 * probability,
                "rollout_target_reached": float(index % 3 == 0),
                "candidate_expected_utility_score": 0.9 - 0.7 * probability,
                "candidate_preservation_score": 0.85 - 0.5 * probability,
                "candidate_utility_score": 0.8 - 0.4 * probability,
                "final_utility_score": 0.75 - 0.3 * probability,
                "value_score": 0.7 - 0.2 * probability,
            }
        )
    return rows


def _attempts(count=24):
    output = []
    for index in range(count):
        attack_count = index % 4
        utility_count = 3 - ((index // 2) % 4)
        output.append(
            [
                (int(seed < attack_count), int(seed < utility_count))
                for seed in range(3)
            ]
        )
    return output


def test_grouped_folds_do_not_split_user_tasks():
    module = _module()
    rows = _rows()
    attempts = _attempts()
    security = np.asarray(
        [np.mean([outcome[0] for outcome in values]) for values in attempts]
    )
    utility = np.asarray(
        [np.mean([outcome[1] for outcome in values]) for values in attempts]
    )
    groups = np.asarray(
        [f"{row['suite']}::{row['user_task_id']}" for row in rows]
    )
    folds = module._make_folds(
        security, utility, groups, n_splits=3, random_state=17
    )
    assert len(folds) == 3
    for train, valid in folds:
        assert not (set(groups[train]) & set(groups[valid]))


def test_two_calibrators_emit_finite_crossfit_probabilities():
    module = _module()
    rows = _rows()
    attempts = _attempts()
    security = np.asarray(
        [np.mean([outcome[0] for outcome in values]) for values in attempts]
    )
    utility = np.asarray(
        [np.mean([outcome[1] for outcome in values]) for values in attempts]
    )
    clean = {
        (str(row["suite"]), str(row["user_task_id"])): 0.75 for row in rows
    }
    for model_name in module.MODEL_NAMES:
        result = module._crossfit_model(
            model_name,
            rows,
            rows[:7],
            clean,
            attempts,
            security,
            utility,
            cv_seeds=[11, 23],
            n_splits=3,
        )
        for head in ("attack", "utility"):
            assert result[head]["oof_mean"].shape == (len(rows),)
            assert result[head]["test_mean"].shape == (7,)
            assert np.isfinite(result[head]["oof_mean"]).all()
            assert ((result[head]["oof_mean"] > 0) & (result[head]["oof_mean"] < 1)).all()


def test_replay_loader_and_probability_selector(tmp_path):
    module = _module()
    rows = _rows(8)
    expected = {module._pair_key(row) for row in rows}
    replay_paths = []
    for seed in (7, 13, 21):
        path = tmp_path / f"seed{seed}.json"
        path.write_text(
            json.dumps(
                {
                    "seed": seed,
                    "do_sample": True,
                    "results": {
                        "validation_probability_pilot": {
                            "rows": [
                                {
                                    **row,
                                    "security": (index + seed) % 2 == 0,
                                    "utility": (index + seed) % 3 != 0,
                                }
                                for index, row in enumerate(rows)
                            ]
                        }
                    },
                }
            ),
            encoding="utf-8",
        )
        replay_paths.append(path)
    attempts, metadata = module._load_replay_attempts(
        replay_paths, "validation_probability_pilot", expected
    )
    assert len(metadata) == 3
    assert {len(values) for values in attempts.values()} == {3}

    annotated = [
        {
            **row,
            "replay_attack_probability": 0.9 - index * 0.05,
            "replay_utility_lcb": 0.25 + index * 0.08,
        }
        for index, row in enumerate(rows)
    ]
    selected, info = module._select_probability_rows(
        annotated,
        top_k=4,
        utility_floor=0.5,
        objective="joint_lcb",
        max_per_user_task=1,
        oof=False,
    )
    assert len(selected) == 4
    assert max(
        sum(module._task_key(left) == module._task_key(right) for left in selected)
        for right in selected
    ) == 1
    assert info["feasible_selected_count"] + info["fallback_selected_count"] == 4
