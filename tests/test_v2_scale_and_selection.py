from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_balanced_task_order_is_complete_and_family_interleaved():
    module = _load_script("71_build_v2_size_ablation.py")
    groups = [
        {
            "group_id": f"a{index}",
            "task_key": "suite|task",
            "attack_family": "a",
        }
        for index in range(3)
    ] + [
        {
            "group_id": f"b{index}",
            "task_key": "suite|task",
            "attack_family": "b",
        }
        for index in range(3)
    ]
    ordered = module._balanced_task_order(groups, seed=7)
    family_by_id = {row["group_id"]: row["attack_family"] for row in groups}
    assert len(ordered) == 6
    assert len(set(ordered)) == 6
    assert {family_by_id[group_id] for group_id in ordered[:2]} == {"a", "b"}


def test_downstream_selection_respects_per_task_budget():
    module = _load_script("70_evaluate_v2_downstream_selection.py")
    rows = []
    for task in ("a", "b"):
        for index, risk in enumerate((0.1, 0.9, 0.4)):
            rows.append(
                {
                    "group_id": f"{task}{index}",
                    "task_key": task,
                    "raw_risk_score": risk,
                    "calibrated_risk_score": risk,
                    "utility_score": 0.0,
                    "preservation_score": 0.0,
                    "observed_asr": float(index == 1),
                    "observed_bup": 0.5,
                    "target_asr": 0.5,
                    "target_bup": 0.5,
                    "trials": 5,
                }
            )
    selected = module._select(
        rows,
        risk_key="raw_risk_score",
        recipe="risk_only",
        budget_per_task=1,
    )
    metrics = module._metrics(selected)
    assert {row["group_id"] for row in selected} == {"a1", "b1"}
    assert metrics["selected_configuration_count"] == 2
    assert metrics["selected_episode_count"] == 10
    assert metrics["ASR"] == 1.0
    assert metrics["BUP"] == 0.5


def test_randomization_observed_metrics_use_selected_ids():
    module = _load_script("72_randomization_test_v2_selection.py")
    outcomes = {
        "a": {"ASR": 1.0, "BUP": 0.5, "ASR_plus_BUP": 1.5},
        "b": {"ASR": 0.0, "BUP": 1.0, "ASR_plus_BUP": 1.0},
    }
    metrics = module._selection_metrics(["a", "b"], outcomes)
    assert metrics == {"ASR": 0.5, "BUP": 0.75, "ASR_plus_BUP": 1.25}


def test_grouped_oof_cohorts_partition_one_task_per_domain():
    module = _load_script("74_build_v2_grouped_oof_folds.py")
    metadata = []
    for domain in ("a", "b", "c", "d"):
        for split, count in (("train", 3), ("val", 1), ("test", 1)):
            for index in range(count):
                metadata.append(
                    {
                        "suite": domain,
                        "user_task_id": f"{split}_{index}",
                        "task_split": split,
                    }
                )
    cohorts = module._cohorts(metadata)
    assert set(cohorts) == set(module.COHORT_ORDER)
    assert len(set().union(*(set(rows) for rows in cohorts.values()))) == 20
    assert all(len(rows) == 4 for rows in cohorts.values())
    assert all(
        len({task.split("|", 1)[0] for task in rows}) == 4
        for rows in cohorts.values()
    )


def test_oof_paired_contrast_reports_task_direction():
    module = _load_script("75_summarize_v2_grouped_oof.py")
    left = {
        f"task{index}": {"ASR": 0.0, "BUP": 0.5, "ASR_plus_BUP": 0.5}
        for index in range(20)
    }
    right = {
        key: {"ASR": 0.2, "BUP": 0.5, "ASR_plus_BUP": 0.7}
        for key in left
    }
    contrast = module._paired_contrast(left, right, draws=10000, seed=7)
    joint = contrast["metrics"]["ASR_plus_BUP"]
    assert abs(joint["observed_delta_pct100_minus_pct25"] - 0.2) < 1e-12
    assert abs(joint["observed_mean_delta"] - 0.2) < 1e-12
    assert joint["positive_task_count"] == 20
    assert joint["tie_task_count"] == 0
    assert joint["negative_task_count"] == 0
    assert joint["exact_sign_flip_one_sided_p"] == 1 / (1 << 20)


def test_oof_paired_contrast_treats_zero_mean_ties_as_extreme():
    module = _load_script("75_summarize_v2_grouped_oof.py")
    left = {
        f"task{index}": {"ASR": 0.0, "BUP": 0.5, "ASR_plus_BUP": 0.5}
        for index in range(20)
    }
    right = {key: dict(value) for key, value in left.items()}
    right["task0"]["BUP"] = 0.7
    right["task0"]["ASR_plus_BUP"] = 0.7
    right["task1"]["BUP"] = 0.3
    right["task1"]["ASR_plus_BUP"] = 0.3
    contrast = module._paired_contrast(left, right, draws=20000, seed=7)
    joint = contrast["metrics"]["ASR_plus_BUP"]
    assert abs(joint["observed_delta_pct100_minus_pct25"]) < 1e-12
    assert joint["exact_sign_flip_one_sided_p"] == 0.75
    assert joint["exact_sign_flip_effective_assignment_count"] == 4
    assert 0.72 < joint["sign_flip_one_sided_p"] < 0.78


def test_oof_diagnostic_jaccard_handles_overlap_and_empty_sets():
    module = _load_script("76_diagnose_v2_grouped_oof.py")
    assert module._jaccard(set(), set()) == 1.0
    assert module._jaccard({"a", "b"}, {"b", "c"}) == 1 / 3


def test_stability_aggregators_penalize_one_seed_outlier():
    module = _load_script("77_evaluate_v2_stability_ensemble.py")
    scores = {
        7: {"a": 10.0, "b": 1.0, "c": 0.0},
        13: {"a": 0.0, "b": 1.0, "c": 0.5},
        21: {"a": 0.0, "b": 1.0, "c": 0.5},
    }
    rows_by_seed = {
        seed: [
            {
                "group_id": group_id,
                "task_key": "suite|task",
                "raw_risk_score": score,
                "calibrated_risk_score": score,
                "utility_score": 0.0,
                "preservation_score": 0.0,
            }
            for group_id, score in values.items()
        ]
        for seed, values in scores.items()
    }
    aggregated = module._aggregate_rows(
        rows_by_seed,
        risk_key="calibrated_risk_score",
        recipe="risk_only",
        budget=1,
    )
    selected = {
        name: module._select_aggregated(rows, 1)[0]["group_id"]
        for name, rows in aggregated.items()
    }
    assert selected["mean_score"] == "a"
    assert selected["mean_borda"] == "b"
    assert selected["rank_lcb_0p5"] == "b"
    assert selected["consensus_borda_0p25"] == "b"


def test_headwise_risk_ucb_preserves_rare_high_risk_seed():
    module = _load_script("79_evaluate_v2_headwise_uncertainty.py")
    risk = {
        7: {"a": 0.9, "b": 0.4},
        13: {"a": 0.1, "b": 0.4},
        21: {"a": 0.1, "b": 0.4},
    }
    rows_by_seed = {
        seed: [
            {
                "group_id": group_id,
                "task_key": "suite|task",
                "raw_risk_score": score,
                "calibrated_risk_score": score,
                "utility_score": 0.5,
                "preservation_score": 0.5,
            }
            for group_id, score in values.items()
        ]
        for seed, values in risk.items()
    }
    aggregated = module._aggregate_rows(
        rows_by_seed,
        risk_key="calibrated_risk_score",
        recipe="risk_only",
        budget=1,
    )
    selected = {
        name: module.STABILITY._select_aggregated(rows, 1)[0]["group_id"]
        for name, rows in aggregated.items()
    }
    assert selected["mean_score"] == "b"
    assert selected["risk_ucb_0p5"] == "a"
    assert selected["utility_lcb_0p5"] == "b"
    assert selected["asymmetric_ucb_lcb_0p5"] == "a"
