"""Summarize the frozen semantic-residual event-model research round."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from itertools import combinations
from pathlib import Path


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _mean_std(values: list[float]) -> dict:
    return {
        "mean": statistics.fmean(values),
        "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "values": values,
    }


def _get(report: dict, split: str, mode: str, metric: str) -> float:
    return float(report["metrics"][split][mode][metric])


def _markov(report: dict, split: str, mode: str, metric: str) -> float:
    return float(
        report["baselines"]["candidate_hierarchical_markov"]["metrics"][split][
            mode
        ][metric]
    )


def _prediction_rows(path: Path) -> dict[str, dict]:
    rows = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            trajectory_id = row["trajectory_id"]
            if trajectory_id in rows:
                raise ValueError(f"duplicate prediction for {trajectory_id} in {path}")
            rows[trajectory_id] = row
    return rows


def _ensemble_diagnostics(seed_dirs: list[Path], split: str) -> dict:
    by_seed = [
        _prediction_rows(path / f"{split}_free_predictions.jsonl")
        for path in seed_dirs
    ]
    id_sets = [set(rows) for rows in by_seed]
    if any(ids != id_sets[0] for ids in id_sets[1:]):
        raise ValueError(f"{split} prediction trajectory IDs differ across seeds")
    first_agreement = []
    sequence_agreement = []
    pairwise_total_variation = []
    for trajectory_id in sorted(id_sets[0]):
        records = [rows[trajectory_id] for rows in by_seed]
        sequences = [tuple(row["generated_skill_path"]) for row in records]
        first = [sequence[0] if sequence else "<EMPTY>" for sequence in sequences]
        first_agreement.append(float(len(set(first)) == 1))
        sequence_agreement.append(float(len(set(sequences)) == 1))
        distributions = [
            [row["free_joint_probability"][key] for key in sorted(row["free_joint_probability"])]
            for row in records
        ]
        for left, right in combinations(distributions, 2):
            pairwise_total_variation.append(
                0.5 * sum(abs(a - b) for a, b in zip(left, right))
            )
    return {
        "trajectory_count": len(id_sets[0]),
        "all_seed_first_event_agreement": statistics.fmean(first_agreement),
        "all_seed_full_sequence_agreement": statistics.fmean(sequence_agreement),
        "mean_pairwise_joint_total_variation": statistics.fmean(
            pairwise_total_variation
        ),
        "max_pairwise_joint_total_variation": max(pairwise_total_variation),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--previous-summary", type=Path)
    args = parser.parse_args()

    metric_paths = sorted(args.root.glob("model/seed*/metrics.json"))
    if len(metric_paths) != 3:
        raise ValueError(f"Expected three semantic model seeds, found {len(metric_paths)}")
    reports = [_read(path) for path in metric_paths]
    seed_dirs = [path.parent for path in metric_paths]
    seeds = [int(path.parent.name.removeprefix("seed")) for path in metric_paths]
    if sorted(seeds) != [7, 17, 29]:
        raise ValueError(f"Unexpected fixed seed set: {seeds}")

    for report in reports:
        if report["data"]["vocabulary_source"] != "training candidate_skills only":
            raise ValueError("vocabulary provenance is not the frozen training catalog")
        if any(report["data"]["task_group_overlap"].values()):
            raise ValueError("task group overlap invalidates the round")

    metric_specs = {
        "validation_teacher_skill_nll": ("validation", "teacher", "next_skill_nll"),
        "validation_teacher_skill_accuracy": (
            "validation",
            "teacher",
            "next_skill_accuracy",
        ),
        "test_teacher_skill_nll": ("test", "teacher", "next_skill_nll"),
        "test_teacher_skill_accuracy": ("test", "teacher", "next_skill_accuracy"),
        "validation_free_normalized_edit": (
            "validation",
            "free",
            "normalized_edit_distance",
        ),
        "validation_free_exact_accuracy": (
            "validation",
            "free",
            "exact_sequence_accuracy",
        ),
        "test_free_normalized_edit": ("test", "free", "normalized_edit_distance"),
        "test_free_exact_accuracy": ("test", "free", "exact_sequence_accuracy"),
        "validation_teacher_dynamic_minus_static_joint_nll": (
            "validation",
            "teacher",
            "dynamic_minus_static_joint_nll",
        ),
        "test_teacher_dynamic_minus_static_joint_nll": (
            "test",
            "teacher",
            "dynamic_minus_static_joint_nll",
        ),
        "validation_free_dynamic_joint_nll": (
            "validation",
            "free",
            "free_dynamic_joint_count_nll",
        ),
        "test_free_dynamic_joint_nll": (
            "test",
            "free",
            "free_dynamic_joint_count_nll",
        ),
        "validation_conservative_truncation_fraction": (
            "validation",
            "free",
            "conservative_truncation_fraction",
        ),
        "test_conservative_truncation_fraction": (
            "test",
            "free",
            "conservative_truncation_fraction",
        ),
    }
    aggregate = {
        name: _mean_std([_get(report, *location) for report in reports])
        for name, location in metric_specs.items()
    }
    for name, values in aggregate.items():
        if any(not math.isfinite(value) for value in values["values"]):
            raise ValueError(f"non-finite aggregate metric: {name}")

    markov = {
        "validation_teacher_skill_nll": _markov(
            reports[0], "validation", "teacher", "next_skill_nll"
        ),
        "validation_teacher_skill_accuracy": _markov(
            reports[0], "validation", "teacher", "next_skill_accuracy"
        ),
        "test_teacher_skill_nll": _markov(
            reports[0], "test", "teacher", "next_skill_nll"
        ),
        "test_teacher_skill_accuracy": _markov(
            reports[0], "test", "teacher", "next_skill_accuracy"
        ),
        "validation_free_normalized_edit": _markov(
            reports[0], "validation", "free", "normalized_edit_distance"
        ),
        "validation_free_exact_accuracy": _markov(
            reports[0], "validation", "free", "exact_sequence_accuracy"
        ),
        "test_free_normalized_edit": _markov(
            reports[0], "test", "free", "normalized_edit_distance"
        ),
        "test_free_exact_accuracy": _markov(
            reports[0], "test", "free", "exact_sequence_accuracy"
        ),
    }
    for report in reports[1:]:
        for metric, expected in markov.items():
            split, mode, raw_metric = {
                "validation_teacher_skill_nll": ("validation", "teacher", "next_skill_nll"),
                "validation_teacher_skill_accuracy": (
                    "validation",
                    "teacher",
                    "next_skill_accuracy",
                ),
                "test_teacher_skill_nll": ("test", "teacher", "next_skill_nll"),
                "test_teacher_skill_accuracy": ("test", "teacher", "next_skill_accuracy"),
                "validation_free_normalized_edit": (
                    "validation",
                    "free",
                    "normalized_edit_distance",
                ),
                "validation_free_exact_accuracy": (
                    "validation",
                    "free",
                    "exact_sequence_accuracy",
                ),
                "test_free_normalized_edit": (
                    "test",
                    "free",
                    "normalized_edit_distance",
                ),
                "test_free_exact_accuracy": (
                    "test",
                    "free",
                    "exact_sequence_accuracy",
                ),
            }[metric]
            if abs(_markov(report, split, mode, raw_metric) - expected) > 1e-12:
                raise ValueError("deterministic Markov baseline changed across seeds")

    selected_oov = {
        split: [
            int(report["data"]["dataset_audit"][split]["selected_skill_oov_events"])
            for report in reports
        ]
        for split in ("train", "validation", "test")
    }
    static_joint_nll = {
        split: _mean_std(
            [
                _get(report, split, "teacher", "static_joint_count_nll")
                for report in reports
            ]
        )
        for split in ("validation", "test")
    }
    free_minus_static = {
        split: _mean_std(
            [
                _get(report, split, "free", "free_dynamic_joint_count_nll")
                - _get(report, split, "teacher", "static_joint_count_nll")
                for report in reports
            ]
        )
        for split in ("validation", "test")
    }

    thresholds = {
        "teacher_validation_min_nll_gain_vs_candidate_markov": 0.02,
        "teacher_test_max_nll_regression_vs_candidate_markov": 0.02,
        "free_validation_min_edit_gain_vs_candidate_markov": 0.02,
        "free_test_max_edit_regression_vs_candidate_markov": 0.02,
        "teacher_validation_min_dynamic_joint_nll_gain_vs_static": 0.02,
        "teacher_test_max_dynamic_joint_nll_regression_vs_static": 0.02,
        "free_validation_min_dynamic_joint_nll_gain_vs_static": 0.02,
        "free_test_max_dynamic_joint_nll_regression_vs_static": 0.02,
        "validation_teacher_accuracy_max_seed_std": 0.05,
        "selected_skill_oov_events": 0,
    }
    gates = {
        "zero_selected_skill_oov": all(
            value == 0 for values in selected_oov.values() for value in values
        ),
        "teacher_validation_nll_beats_candidate_markov": (
            aggregate["validation_teacher_skill_nll"]["mean"]
            <= markov["validation_teacher_skill_nll"] - 0.02
        ),
        "teacher_test_nll_not_worse_than_candidate_markov": (
            aggregate["test_teacher_skill_nll"]["mean"]
            <= markov["test_teacher_skill_nll"] + 0.02
        ),
        "free_validation_edit_beats_candidate_markov": (
            aggregate["validation_free_normalized_edit"]["mean"]
            <= markov["validation_free_normalized_edit"] - 0.02
        ),
        "free_test_edit_not_worse_than_candidate_markov": (
            aggregate["test_free_normalized_edit"]["mean"]
            <= markov["test_free_normalized_edit"] + 0.02
        ),
        "teacher_validation_dynamic_value_beats_static": (
            aggregate["validation_teacher_dynamic_minus_static_joint_nll"]["mean"]
            <= -0.02
        ),
        "teacher_test_dynamic_value_not_worse_than_static": (
            aggregate["test_teacher_dynamic_minus_static_joint_nll"]["mean"] <= 0.02
        ),
        "free_validation_dynamic_value_beats_static": (
            free_minus_static["validation"]["mean"] <= -0.02
        ),
        "free_test_dynamic_value_not_worse_than_static": (
            free_minus_static["test"]["mean"] <= 0.02
        ),
        "validation_teacher_accuracy_seed_stable": (
            aggregate["validation_teacher_skill_accuracy"]["std"] <= 0.05
        ),
        "clean_eligibility_gate": False,
    }
    architecture_gates = {
        key: value for key, value in gates.items() if key != "clean_eligibility_gate"
    }
    architecture_signal = all(architecture_gates.values())

    previous_comparison = None
    if args.previous_summary:
        previous = _read(args.previous_summary)
        previous_comparison = {
            "source": str(args.previous_summary),
            "previous_validation_teacher_accuracy": previous["event_summary"][
                "validation_next_tool_accuracy"
            ]["mean"],
            "current_validation_teacher_accuracy": aggregate[
                "validation_teacher_skill_accuracy"
            ]["mean"],
            "previous_test_teacher_accuracy": previous["event_summary"][
                "test_next_tool_accuracy"
            ]["mean"],
            "current_test_teacher_accuracy": aggregate[
                "test_teacher_skill_accuracy"
            ]["mean"],
            "important_caveat": (
                "The current model uses a candidate-constrained training-catalog "
                "vocabulary and is compared confirmatorily only with its stronger "
                "candidate-aware Markov baseline."
            ),
        }

    summary = {
        "scope": "fixed-budget second diagnostic on existing AgentDojo-v2 sandbox data",
        "confirmatory": False,
        "seed_set": seeds,
        "aggregate": aggregate,
        "candidate_markov": markov,
        "static_joint_nll": static_joint_nll,
        "free_dynamic_minus_static_joint_nll": free_minus_static,
        "selected_skill_oov_events": selected_oov,
        "ensemble_diagnostics": {
            split: _ensemble_diagnostics(seed_dirs, split)
            for split in ("validation", "test")
        },
        "thresholds_frozen_before_run": thresholds,
        "gates": gates,
        "architecture_signal": architecture_signal,
        "previous_round_context_only": previous_comparison,
        "decision": (
            "ARCHITECTURE_SIGNAL_ONLY_CLEAN_GATE_BLOCKED"
            if architecture_signal
            else "NO_GO_REVISE_SEMANTIC_RESIDUAL_MODEL"
        ),
        "research_boundary": (
            "No new attack data, exact-simulator H2 planning, or Dreamer training is "
            "authorized while the independently frozen clean gate is false."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
