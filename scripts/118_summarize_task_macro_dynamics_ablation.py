"""Summarize the frozen 0723 task-macro mechanism ablation."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


VARIANTS = ("length_semantic", "event_no_attack_semantics", "semantic_event")
SPLITS = ("validation", "test")


def _summary(values: list[float]) -> dict[str, Any]:
    if not values:
        raise ValueError("cannot summarize an empty metric")
    return {
        "values": values,
        "mean": statistics.fmean(values),
        "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "minimum": min(values),
        "maximum": max(values),
    }


def _nested(payload: dict, *keys: str):
    value: Any = payload
    for key in keys:
        value = value[key]
    return value


def _load_runs(root: Path, variants: tuple[str, ...], seeds: tuple[int, ...]):
    runs: dict[str, dict[int, dict]] = {}
    for variant in variants:
        runs[variant] = {}
        for seed in seeds:
            path = root / "model" / variant / f"seed{seed}" / "metrics.json"
            if not path.exists():
                raise FileNotFoundError(path)
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload["variant"] != variant:
                raise ValueError(f"variant mismatch in {path}")
            if int(payload["training"]["seed"]) != seed:
                raise ValueError(f"seed mismatch in {path}")
            runs[variant][seed] = payload
    return runs


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + end - 1) / 2.0
        for position in range(start, end):
            ranks[order[position]] = rank
        start = end
    return ranks


def _pearson(left: list[float], right: list[float]) -> float:
    left_mean = statistics.fmean(left)
    right_mean = statistics.fmean(right)
    numerator = sum(
        (lvalue - left_mean) * (rvalue - right_mean)
        for lvalue, rvalue in zip(left, right)
    )
    left_scale = sum((value - left_mean) ** 2 for value in left) ** 0.5
    right_scale = sum((value - right_mean) ** 2 for value in right) ** 0.5
    if left_scale == 0 or right_scale == 0:
        return 1.0 if left == right else 0.0
    return numerator / (left_scale * right_scale)


def _seed_prediction_stability(root: Path, seeds: tuple[int, ...]) -> dict:
    report = {}
    for split in SPLITS:
        by_seed = {}
        for seed in seeds:
            path = (
                root
                / "model"
                / "semantic_event"
                / f"seed{seed}"
                / f"{split}_free_predictions.jsonl"
            )
            rows = _read_jsonl(path)
            by_seed[seed] = {row["trajectory_id"]: row for row in rows}
        identifiers = sorted(by_seed[seeds[0]])
        if any(sorted(rows) != identifiers for rows in by_seed.values()):
            raise ValueError(f"free prediction IDs differ across seeds for {split}")
        sequence_agreement = statistics.fmean(
            len(
                {
                    tuple(by_seed[seed][identifier]["generated_skill_path"])
                    for seed in seeds
                }
            )
            == 1
            for identifier in identifiers
        )
        pair_rows = []
        for left_index, left_seed in enumerate(seeds):
            for right_seed in seeds[left_index + 1 :]:
                total_variations = []
                left_scores = []
                right_scores = []
                for identifier in identifiers:
                    left_probability = by_seed[left_seed][identifier][
                        "free_joint_probability"
                    ]
                    right_probability = by_seed[right_seed][identifier][
                        "free_joint_probability"
                    ]
                    names = sorted(left_probability)
                    if sorted(right_probability) != names:
                        raise ValueError("joint probability outcome names differ")
                    total_variations.append(
                        0.5
                        * sum(
                            abs(
                                float(left_probability[name])
                                - float(right_probability[name])
                            )
                            for name in names
                        )
                    )
                    left_scores.append(float(left_probability["attack1_utility1"]))
                    right_scores.append(float(right_probability["attack1_utility1"]))
                pair_rows.append(
                    {
                        "left_seed": left_seed,
                        "right_seed": right_seed,
                        "mean_joint_total_variation": statistics.fmean(
                            total_variations
                        ),
                        "maximum_joint_total_variation": max(total_variations),
                        "joint_success_ranking_spearman": _pearson(
                            _average_ranks(left_scores),
                            _average_ranks(right_scores),
                        ),
                    }
                )
        report[split] = {
            "trajectory_count": len(identifiers),
            "all_seed_full_sequence_agreement": sequence_agreement,
            "mean_pairwise_joint_total_variation": statistics.fmean(
                row["mean_joint_total_variation"] for row in pair_rows
            ),
            "maximum_pairwise_joint_total_variation": max(
                row["maximum_joint_total_variation"] for row in pair_rows
            ),
            "minimum_pairwise_joint_success_ranking_spearman": min(
                row["joint_success_ranking_spearman"] for row in pair_rows
            ),
            "pairwise": pair_rows,
            "independence_warning": (
                "trajectory-level ranking is a seed-stability diagnostic; repeated "
                "configurations are not independent tasks"
            ),
        }
    return report


def _metric_across_seeds(runs, variant, path):
    return _summary([float(_nested(payload, *path)) for payload in runs[variant].values()])


def _aggregate_variants(runs):
    report = {}
    for variant in VARIANTS:
        report[variant] = {}
        for split in SPLITS:
            report[variant][split] = {
                "teacher_task_macro": {
                    metric: _metric_across_seeds(
                        runs,
                        variant,
                        ("metrics", split, "teacher", "task_macro", metric),
                    )
                    for metric in (
                        "next_skill_nll",
                        "next_skill_accuracy",
                        "static_joint_count_nll",
                        "dynamic_joint_count_nll",
                        "dynamic_minus_static_joint_nll",
                    )
                },
                "teacher_micro": {
                    metric: _metric_across_seeds(
                        runs,
                        variant,
                        ("metrics", split, "teacher", "micro", metric),
                    )
                    for metric in ("next_skill_nll", "next_skill_accuracy")
                },
                "free_task_macro": {
                    metric: _metric_across_seeds(
                        runs,
                        variant,
                        ("metrics", split, "free", "task_macro", metric),
                    )
                    for metric in (
                        "exact_sequence_accuracy",
                        "normalized_edit_distance",
                    )
                },
            }
    return report


def _baseline(runs):
    reference = runs["semantic_event"][next(iter(runs["semantic_event"]))]
    report = {}
    for split in SPLITS:
        report[split] = {}
        for name in (
            "candidate_uniform",
            "candidate_hierarchical_markov",
            "candidate_factorized_semantic_markov",
        ):
            baseline = reference["baselines"][split][name]
            report[split][name] = baseline
            canonical = json.dumps(baseline, sort_keys=True)
            for variant_runs in runs.values():
                for payload in variant_runs.values():
                    candidate = payload["baselines"][split][name]
                    if json.dumps(candidate, sort_keys=True) != canonical:
                        raise ValueError(f"baseline changed across runs: {split}/{name}")
    return report


def _prefix_controls(runs):
    report = {}
    controls = (
        "static_semantic",
        "observed",
        "shuffled",
        "length_only",
        "random_length_matched",
        "markov_length_matched",
        "semantic_markov_length_matched",
    )
    for split in SPLITS:
        report[split] = {}
        for control in controls:
            report[split][control] = {
                metric: _metric_across_seeds(
                    runs,
                    "semantic_event",
                    ("prefix_value_controls", split, control, metric),
                )
                for metric in (
                    "task_macro_joint_count_nll",
                    "task_macro_joint_probability_log_score",
                )
            }
    return report


def _per_task_directions(runs, baseline, margin):
    report = {}
    for split in SPLITS:
        full_runs = list(runs["semantic_event"].values())
        tasks = sorted(
            full_runs[0]["metrics"][split]["teacher"]["per_task"]
        )
        plain_markov = baseline[split]["candidate_hierarchical_markov"]["teacher"][
            "per_task"
        ]
        semantic_markov = baseline[split]["candidate_factorized_semantic_markov"][
            "teacher"
        ]["per_task"]
        rows = {}
        for task in tasks:
            full_values = [
                float(
                    payload["metrics"][split]["teacher"]["per_task"][task][
                        "next_skill_nll"
                    ]
                )
                for payload in full_runs
            ]
            full_mean = statistics.fmean(full_values)
            plain_nll = float(plain_markov[task]["next_skill_nll"])
            semantic_nll = float(semantic_markov[task]["next_skill_nll"])
            markov_nll = min(plain_nll, semantic_nll)
            delta = full_mean - markov_nll
            rows[task] = {
                "full_seed_mean_nll": full_mean,
                "full_seed_std_nll": statistics.pstdev(full_values),
                "plain_markov_nll": plain_nll,
                "semantic_markov_nll": semantic_nll,
                "strongest_markov_nll": markov_nll,
                "strongest_markov": (
                    "candidate_hierarchical_markov"
                    if plain_nll <= semantic_nll
                    else "candidate_factorized_semantic_markov"
                ),
                "full_minus_strongest_markov_nll": delta,
                "materially_improved": delta <= -margin,
                "not_materially_worse": delta <= margin,
            }
        report[split] = {
            "tasks": rows,
            "materially_improved_count": sum(
                row["materially_improved"] for row in rows.values()
            ),
            "not_materially_worse_count": sum(
                row["not_materially_worse"] for row in rows.values()
            ),
            "task_count": len(rows),
        }
    return report


def _integrity(runs, ontology_audit, protocol):
    overlaps_empty = True
    task_counts_match = True
    clean_false = True
    for variant_runs in runs.values():
        for payload in variant_runs.values():
            overlaps = payload["data"]["task_group_overlap"]
            overlaps_empty &= not any(overlaps.values())
            audits = payload["data"]["dataset_audit"]
            task_counts_match &= (
                int(audits["train"]["task_count"]) == protocol["data"]["train_tasks"]
                and int(audits["validation"]["task_count"])
                == protocol["data"]["validation_tasks"]
                and int(audits["test"]["task_count"])
                == protocol["data"]["test_tasks"]
            )
            clean_false &= payload["clean_eligibility_gate"] is False
    ontology_match = (
        ontology_audit["ontology_fingerprint"]
        == protocol["frozen_representation"]["event_ontology_fingerprint"]
    )
    forbidden_empty = all(
        not row["forbidden_outcome_fields_seen"]
        for row in ontology_audit["splits"].values()
    )
    return {
        "all_nine_runs_present": all(len(rows) == 3 for rows in runs.values()),
        "task_group_overlaps_empty": overlaps_empty,
        "split_task_counts_match_protocol": task_counts_match,
        "all_run_clean_gates_false": clean_false,
        "ontology_fingerprint_matches_protocol": ontology_match,
        "no_forbidden_outcome_fields_in_ontology": forbidden_empty,
    }


def _mean(aggregate, variant, split, section, metric):
    return aggregate[variant][split][section][metric]["mean"]


def _evaluate_gates(
    protocol,
    aggregate,
    baseline,
    controls,
    directions,
    prediction_stability,
    integrity,
):
    threshold = protocol["thresholds_frozen_before_execution"]
    comparisons = {
        "full_vs_markov": {},
        "full_vs_semantic_markov": {},
        "full_vs_components": {},
        "prefix": {},
    }
    for split in SPLITS:
        full_nll = _mean(
            aggregate, "semantic_event", split, "teacher_task_macro", "next_skill_nll"
        )
        markov_nll = float(
            baseline[split]["candidate_hierarchical_markov"]["teacher"][
                "task_macro"
            ]["next_skill_nll"]
        )
        full_edit = _mean(
            aggregate,
            "semantic_event",
            split,
            "free_task_macro",
            "normalized_edit_distance",
        )
        markov_edit = float(
            baseline[split]["candidate_hierarchical_markov"]["free"]["task_macro"][
                "normalized_edit_distance"
            ]
        )
        comparisons["full_vs_markov"][split] = {
            "next_skill_nll_gain": markov_nll - full_nll,
            "free_edit_gain": markov_edit - full_edit,
            "task_macro_model_nll": full_nll,
            "task_macro_markov_nll": markov_nll,
            "task_macro_model_free_edit": full_edit,
            "task_macro_markov_free_edit": markov_edit,
        }
        semantic_markov = baseline[split]["candidate_factorized_semantic_markov"]
        semantic_markov_nll = float(
            semantic_markov["teacher"]["task_macro"]["next_skill_nll"]
        )
        semantic_markov_edit = float(
            semantic_markov["free"]["task_macro"]["normalized_edit_distance"]
        )
        comparisons["full_vs_semantic_markov"][split] = {
            "next_skill_nll_gain": semantic_markov_nll - full_nll,
            "free_edit_gain": semantic_markov_edit - full_edit,
            "task_macro_model_nll": full_nll,
            "task_macro_semantic_markov_nll": semantic_markov_nll,
            "task_macro_model_free_edit": full_edit,
            "task_macro_semantic_markov_free_edit": semantic_markov_edit,
        }
        comparisons["full_vs_components"][split] = {}
        for variant in ("length_semantic", "event_no_attack_semantics"):
            component_nll = _mean(
                aggregate, variant, split, "teacher_task_macro", "next_skill_nll"
            )
            comparisons["full_vs_components"][split][variant] = {
                "next_skill_nll_gain": component_nll - full_nll,
                "component_nll": component_nll,
                "full_nll": full_nll,
            }
        observed = controls[split]["observed"]["task_macro_joint_count_nll"][
            "mean"
        ]
        comparisons["prefix"][split] = {
            name: {
                "observed_nll": observed,
                "control_nll": controls[split][name][
                    "task_macro_joint_count_nll"
                ]["mean"],
                "observed_nll_gain": controls[split][name][
                    "task_macro_joint_count_nll"
                ]["mean"]
                - observed,
            }
            for name in (
                "static_semantic",
                "shuffled",
                "length_only",
                "random_length_matched",
                "markov_length_matched",
                "semantic_markov_length_matched",
            )
        }

    all_integrity = all(integrity.values())
    markov_gate = (
        comparisons["full_vs_markov"]["validation"]["next_skill_nll_gain"]
        >= threshold["full_validation_min_next_skill_nll_gain_vs_markov"]
        and -comparisons["full_vs_markov"]["test"]["next_skill_nll_gain"]
        <= threshold["full_test_max_next_skill_nll_regression_vs_markov"]
        and comparisons["full_vs_markov"]["validation"]["free_edit_gain"]
        >= threshold["full_validation_min_free_edit_gain_vs_markov"]
        and -comparisons["full_vs_markov"]["test"]["free_edit_gain"]
        <= threshold["full_test_max_free_edit_regression_vs_markov"]
        and comparisons["full_vs_semantic_markov"]["validation"][
            "next_skill_nll_gain"
        ]
        >= threshold["full_validation_min_next_skill_nll_gain_vs_semantic_markov"]
        and -comparisons["full_vs_semantic_markov"]["test"][
            "next_skill_nll_gain"
        ]
        <= threshold["full_test_max_next_skill_nll_regression_vs_semantic_markov"]
        and comparisons["full_vs_semantic_markov"]["validation"][
            "free_edit_gain"
        ]
        >= threshold["full_validation_min_free_edit_gain_vs_semantic_markov"]
        and -comparisons["full_vs_semantic_markov"]["test"]["free_edit_gain"]
        <= threshold["full_test_max_free_edit_regression_vs_semantic_markov"]
    )
    identity_gate = (
        comparisons["full_vs_components"]["validation"]["length_semantic"][
            "next_skill_nll_gain"
        ]
        >= threshold["full_validation_min_next_skill_nll_gain_vs_length_semantic"]
        and -comparisons["full_vs_components"]["test"]["length_semantic"][
            "next_skill_nll_gain"
        ]
        <= threshold["full_test_max_next_skill_nll_regression_vs_length_semantic"]
    )
    semantics_gate = (
        comparisons["full_vs_components"]["validation"][
            "event_no_attack_semantics"
        ]["next_skill_nll_gain"]
        >= threshold[
            "full_validation_min_next_skill_nll_gain_vs_no_attack_semantics"
        ]
        and -comparisons["full_vs_components"]["test"][
            "event_no_attack_semantics"
        ]["next_skill_nll_gain"]
        <= threshold[
            "full_test_max_next_skill_nll_regression_vs_no_attack_semantics"
        ]
    )
    val_min = threshold[
        "validation_min_observed_prefix_joint_nll_gain_vs_each_static_length_random_markov_control"
    ]
    test_max = threshold["test_max_observed_prefix_joint_nll_regression_vs_any_control"]
    content_controls = (
        "static_semantic",
        "length_only",
        "random_length_matched",
        "markov_length_matched",
        "semantic_markov_length_matched",
    )
    prefix_content = all(
        comparisons["prefix"]["validation"][name]["observed_nll_gain"] >= val_min
        for name in content_controls
    ) and all(
        -comparisons["prefix"]["test"][name]["observed_nll_gain"] <= test_max
        for name in content_controls
    )
    prefix_order = (
        comparisons["prefix"]["validation"]["shuffled"]["observed_nll_gain"]
        >= threshold["validation_min_observed_prefix_joint_nll_gain_vs_shuffled"]
        and -comparisons["prefix"]["test"]["shuffled"]["observed_nll_gain"]
        <= test_max
    )
    stability = (
        aggregate["semantic_event"]["validation"]["teacher_task_macro"][
            "next_skill_nll"
        ]["std"]
        <= threshold["validation_full_next_skill_nll_max_seed_std"]
        and aggregate["semantic_event"]["validation"]["free_task_macro"][
            "normalized_edit_distance"
        ]["std"]
        <= threshold["validation_full_free_edit_max_seed_std"]
    )
    prediction_stability_gate = (
        prediction_stability["validation"]["all_seed_full_sequence_agreement"]
        >= threshold["validation_min_all_seed_sequence_agreement"]
        and prediction_stability["test"]["all_seed_full_sequence_agreement"]
        >= threshold["test_min_all_seed_sequence_agreement"]
        and prediction_stability["validation"][
            "mean_pairwise_joint_total_variation"
        ]
        <= threshold["validation_max_mean_pairwise_joint_total_variation"]
        and prediction_stability["test"]["mean_pairwise_joint_total_variation"]
        <= threshold["test_max_mean_pairwise_joint_total_variation"]
        and prediction_stability["validation"][
            "minimum_pairwise_joint_success_ranking_spearman"
        ]
        >= threshold["validation_min_pairwise_joint_success_ranking_spearman"]
        and prediction_stability["test"][
            "minimum_pairwise_joint_success_ranking_spearman"
        ]
        >= threshold["test_min_pairwise_joint_success_ranking_spearman"]
    )
    task_direction = (
        directions["validation"]["materially_improved_count"]
        >= threshold["validation_min_tasks_improved_vs_strongest_markov"]
        and directions["test"]["not_materially_worse_count"]
        >= threshold["test_min_tasks_not_materially_worse_than_strongest_markov"]
    )
    gates = {
        "representation_integrity": all_integrity,
        "task_macro_predictive_and_free_vs_markov": markov_gate,
        "event_identity_beyond_length": identity_gate,
        "attack_semantics_increment": semantics_gate,
        "prefix_content_beyond_static_length_random_markov": prefix_content,
        "prefix_order_beyond_shuffled_multiset": prefix_order,
        "seed_stability": stability,
        "outcome_and_ranking_seed_stability": prediction_stability_gate,
        "per_task_direction": task_direction,
        "clean_eligibility_gate": False,
    }
    if not all_integrity:
        decision = "INVALID_INTEGRITY_FAILURE_CLEAN_GATE_BLOCKED"
    elif not markov_gate:
        decision = "SHORTCUT_NOT_RULED_OUT_CLEAN_GATE_BLOCKED"
    elif all(
        gates[name]
        for name in (
            "event_identity_beyond_length",
            "attack_semantics_increment",
            "prefix_content_beyond_static_length_random_markov",
            "prefix_order_beyond_shuffled_multiset",
            "seed_stability",
            "outcome_and_ranking_seed_stability",
            "per_task_direction",
        )
    ):
        decision = "TASK_MACRO_MECHANISM_SIGNAL_CLEAN_GATE_BLOCKED"
    elif prefix_content and not prefix_order:
        decision = "EVENT_CONTENT_WITHOUT_ORDER_SIGNAL_CLEAN_GATE_BLOCKED"
    else:
        decision = "ARCHITECTURE_SIGNAL_NOT_MECHANISTIC_CLEAN_GATE_BLOCKED"
    return comparisons, gates, decision


def _micro_macro_disagreement(aggregate, baseline):
    report = {}
    for split in SPLITS:
        full_micro = _mean(
            aggregate, "semantic_event", split, "teacher_micro", "next_skill_nll"
        )
        full_macro = _mean(
            aggregate, "semantic_event", split, "teacher_task_macro", "next_skill_nll"
        )
        by_baseline = {}
        for name in (
            "candidate_hierarchical_markov",
            "candidate_factorized_semantic_markov",
        ):
            teacher = baseline[split][name]["teacher"]
            by_baseline[name] = {
                "micro_nll_gain": float(teacher["micro"]["next_skill_nll"])
                - full_micro,
                "task_macro_nll_gain": float(
                    teacher["task_macro"]["next_skill_nll"]
                )
                - full_macro,
            }
        strongest_name = min(
            by_baseline,
            key=lambda name: baseline[split][name]["teacher"]["task_macro"][
                "next_skill_nll"
            ],
        )
        micro_gain = by_baseline[strongest_name]["micro_nll_gain"]
        macro_gain = by_baseline[strongest_name]["task_macro_nll_gain"]
        report[split] = {
            "by_baseline": by_baseline,
            "strongest_task_macro_markov": strongest_name,
            "micro_nll_gain_vs_strongest_markov": micro_gain,
            "task_macro_nll_gain_vs_strongest_markov": macro_gain,
            "direction_disagrees": (micro_gain >= 0) != (macro_gain >= 0),
        }
    return report


def _markdown(report: dict) -> str:
    decision = report["decision"]
    lines = [
        "# 0723 task-macro dynamics ablation",
        "",
        "## Decision",
        "",
        f"`{decision}`",
        "",
        "This is a development-only mechanism audit on four validation and four test tasks. "
        "It cannot establish formal task generalization and cannot override the failed clean gate.",
        "",
        "## Primary task-macro comparisons",
        "",
        "| Split | Full NLL | Best Markov NLL | NLL gain | Full free edit | Best Markov free edit | Edit gain |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for split in SPLITS:
        plain = report["comparisons"]["full_vs_markov"][split]
        semantic = report["comparisons"]["full_vs_semantic_markov"][split]
        best_nll = min(
            plain["task_macro_markov_nll"],
            semantic["task_macro_semantic_markov_nll"],
        )
        best_edit = min(
            plain["task_macro_markov_free_edit"],
            semantic["task_macro_semantic_markov_free_edit"],
        )
        lines.append(
            f"| {split} | {plain['task_macro_model_nll']:.4f} | "
            f"{best_nll:.4f} | {best_nll - plain['task_macro_model_nll']:+.4f} | "
            f"{plain['task_macro_model_free_edit']:.4f} | "
            f"{best_edit:.4f} | {best_edit - plain['task_macro_model_free_edit']:+.4f} |"
        )
    lines.extend(["", "## Frozen gates", ""])
    for name, value in report["gates"].items():
        lines.append(f"- {name}: `{str(value).upper()}`")
    lines.extend(["", "## Prefix-value counterfactual gains", ""])
    for split in SPLITS:
        lines.append(f"### {split}")
        lines.append("")
        for name, row in report["comparisons"]["prefix"][split].items():
            lines.append(f"- observed vs {name}: NLL gain {row['observed_nll_gain']:+.4f}")
        lines.append("")
    lines.extend(
        [
            "## Boundary and next action",
            "",
            "No attack data, H2 attack planning, selective-deployment claim, or Dreamer training is authorized. "
            "The next admissible stage is clean-only expansion of independent tasks/victims, while retaining these frozen task-level metrics and negative controls.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--ontology-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path)
    args = parser.parse_args()

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    seeds = tuple(int(seed) for seed in protocol["fixed_budget"]["seeds"])
    variants = tuple(protocol["fixed_budget"]["variants"])
    if variants != VARIANTS:
        raise ValueError("protocol variants differ from the frozen implementation")
    runs = _load_runs(args.root, variants, seeds)
    aggregate = _aggregate_variants(runs)
    baseline = _baseline(runs)
    controls = _prefix_controls(runs)
    prediction_stability = _seed_prediction_stability(args.root, seeds)
    ontology_audit = json.loads(args.ontology_audit.read_text(encoding="utf-8"))
    integrity = _integrity(runs, ontology_audit, protocol)
    margin = protocol["thresholds_frozen_before_execution"][
        "per_task_material_nll_margin"
    ]
    directions = _per_task_directions(runs, baseline, margin)
    comparisons, gates, decision = _evaluate_gates(
        protocol,
        aggregate,
        baseline,
        controls,
        directions,
        prediction_stability,
        integrity,
    )
    report = {
        "protocol_id": protocol["protocol_id"],
        "scope": protocol["research_scope"],
        "confirmatory": False,
        "independent_task_counts": {"validation": 4, "test": 4},
        "formal_significance_warning": protocol["claim_boundary"]["reason"],
        "integrity": integrity,
        "aggregate": aggregate,
        "baselines": baseline,
        "prefix_controls": controls,
        "comparisons": comparisons,
        "per_task_directions": directions,
        "seed_prediction_stability": prediction_stability,
        "micro_task_macro_disagreement": _micro_macro_disagreement(
            aggregate, baseline
        ),
        "gates": gates,
        "decision": decision,
        "clean_gate_override": protocol["decision_rule"]["clean_gate_override"],
        "next_admissible_stage": (
            "clean-only independent task/victim expansion; retain task-grouped "
            "evaluation and do not collect attack trajectories until clean GO"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if args.markdown_output is not None:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(_markdown(report), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
