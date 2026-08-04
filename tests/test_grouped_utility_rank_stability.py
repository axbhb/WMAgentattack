import importlib.util
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "83_diagnose_v2_group_utility_rank_stability.py"
)
SPEC = importlib.util.spec_from_file_location("rank_stability", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _row(group_id, utility, target, observed, task="d|1", risk=0.2):
    return {
        "group_id": group_id,
        "task_key": task,
        "target_asr": risk,
        "target_bup": target,
        "observed_asr": risk,
        "observed_bup": observed,
        "raw_risk_score": risk,
        "calibrated_risk_score": risk,
        "utility_score": utility,
        "preservation_score": utility,
        "critic_value_score": risk + utility,
        "configuration_value_score": risk + utility,
    }


def test_rank_summary_reports_perfect_and_reversed_utility_order():
    perfect = [
        _row("a", 0.1, 0.1, 0.0),
        _row("b", 0.5, 0.5, 0.4),
        _row("c", 0.9, 0.9, 0.8),
    ]
    reversed_rows = [
        _row("a", 0.9, 0.1, 0.0),
        _row("b", 0.5, 0.5, 0.4),
        _row("c", 0.1, 0.9, 0.8),
    ]

    good = MODULE._rank_summary(perfect, recipe="risk_plus_utility")
    bad = MODULE._rank_summary(reversed_rows, recipe="risk_plus_utility")

    assert good["utility"]["aggregate"]["mean_task_spearman_target"] == pytest.approx(1.0)
    assert good["utility"]["aggregate"]["mean_top1_target_regret"] == pytest.approx(0.0)
    assert bad["utility"]["aggregate"]["mean_task_spearman_target"] == pytest.approx(-1.0)
    assert bad["utility"]["aggregate"]["mean_top1_target_regret"] == pytest.approx(0.8)


def test_cross_seed_stability_detects_top1_disagreement():
    rows = {
        7: [_row("a", 0.9, 0.8, 0.8), _row("b", 0.2, 0.2, 0.2)],
        13: [_row("a", 0.4, 0.8, 0.8), _row("b", 0.8, 0.2, 0.2)],
        21: [_row("a", 0.7, 0.8, 0.8), _row("b", 0.6, 0.2, 0.2)],
    }

    result = MODULE._cross_seed_stability(rows, recipe="risk_plus_utility")
    utility = result["utility"]

    assert utility["aggregate"]["complete_top1_agreement_rate"] == pytest.approx(0.0)
    assert utility["aggregate"]["mean_top1_consensus_fraction"] == pytest.approx(2 / 3)
    assert utility["per_task"]["d|1"]["unique_seed_top1_count"] == 2


def test_selection_comparison_counts_changed_outcome_tie():
    baseline = [
        _row("a", 0.9, 0.8, 0.6),
        _row("b", 0.2, 0.7, 0.6),
    ]
    variant = [
        _row("a", 0.2, 0.8, 0.6),
        _row("b", 0.9, 0.7, 0.6),
    ]

    result = MODULE._selection_comparison(
        baseline,
        variant,
        baseline_recipe="risk_plus_utility",
        variant_recipe="risk_plus_utility",
    )

    assert result["aggregate"]["changed_selection_count"] == 1
    assert result["aggregate"]["changed_but_observed_outcome_tied_count"] == 1
    assert result["aggregate"]["mean_selected_observed_joint_delta"] == pytest.approx(0.0)
