"""Summarize and gate the frozen clean evidence-ledger ablation."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wmagentattack.clean_evidence_probe import FROZEN_VARIANTS, task_macro_errors


EVIDENCE = "semantic_markov_state_evidence"
EVENT_STATE = "semantic_markov_state"
SHUFFLE = "semantic_markov_state_shuffled_evidence"
OUTPUT_LENGTH = "semantic_markov_state_output_length"
TRANSFORMER = "event_transformer_state_evidence"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def _average_training_seeds(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                row["variant"],
                row["episode_id"],
                row["prefix_index"],
            )
        ].append(row)
    averaged = []
    for key, members in sorted(grouped.items()):
        first = members[0]
        utility_values = [
            float(member["utility_probability"])
            for member in members
            if member["utility_probability"] is not None
        ]
        averaged.append(
            {
                **{
                    field: first[field]
                    for field in (
                        "variant",
                        "fold",
                        "episode_id",
                        "panel",
                        "data_seed",
                        "task_id",
                        "prefix_index",
                        "is_final_prefix",
                        "progress_target",
                        "utility_target",
                    )
                },
                "progress_prediction": float(
                    np.mean([float(member["progress_prediction"]) for member in members])
                ),
                "utility_probability": (
                    float(np.mean(utility_values)) if utility_values else None
                ),
                "training_seed_count": len(members),
            }
        )
    return averaged


def _metrics_by_variant(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["variant"])].append(row)
    return {variant: task_macro_errors(values) for variant, values in sorted(grouped.items())}


def _metrics_by_variant_seed(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["variant"]), int(row["training_seed"]))].append(row)
    output: dict[str, dict[str, Any]] = defaultdict(dict)
    for (variant, seed), values in sorted(grouped.items()):
        output[variant][str(seed)] = task_macro_errors(values)
    return dict(output)


def _exact_sign_flip(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    observed = abs(float(array.mean()))
    extreme = 0
    total = 0
    for signs in itertools.product((-1.0, 1.0), repeat=len(array)):
        statistic = abs(float(np.mean(array * np.asarray(signs))))
        extreme += int(statistic >= observed - 1e-15)
        total += 1
    return extreme / total


def _comparison(
    metrics: Mapping[str, Mapping[str, Any]], candidate: str, control: str
) -> dict[str, Any]:
    candidate_tasks = metrics[candidate]["task_metrics"]
    control_tasks = metrics[control]["task_metrics"]
    tasks = sorted(set(candidate_tasks) & set(control_tasks))
    progress = {
        task: float(control_tasks[task]["progress_mae"] - candidate_tasks[task]["progress_mae"])
        for task in tasks
    }
    utility = {
        task: float(control_tasks[task]["utility_brier"] - candidate_tasks[task]["utility_brier"])
        for task in tasks
    }
    return {
        "candidate": candidate,
        "control": control,
        "positive_means_candidate_is_better": True,
        "task_macro_progress_mae_gain": float(np.mean(list(progress.values()))),
        "task_macro_utility_brier_gain": float(np.mean(list(utility.values()))),
        "progress_exact_sign_flip_p": _exact_sign_flip(list(progress.values())),
        "utility_exact_sign_flip_p": _exact_sign_flip(list(utility.values())),
        "progress_positive_tasks": sum(value > 0 for value in progress.values()),
        "utility_positive_tasks": sum(value > 0 for value in utility.values()),
        "task_count": len(tasks),
        "task_progress_gains": progress,
        "task_utility_gains": utility,
    }


def _leave_one_task_out(comparison: Mapping[str, Any]) -> dict[str, Any]:
    progress = comparison["task_progress_gains"]
    utility = comparison["task_utility_gains"]
    tasks = sorted(progress)
    progress_rows = {
        omitted: float(np.mean([progress[task] for task in tasks if task != omitted]))
        for omitted in tasks
    }
    utility_rows = {
        omitted: float(np.mean([utility[task] for task in tasks if task != omitted]))
        for omitted in tasks
    }
    return {
        "progress_gains": progress_rows,
        "utility_gains": utility_rows,
        "all_progress_positive": all(value > 0 for value in progress_rows.values()),
        "all_utility_positive": all(value > 0 for value in utility_rows.values()),
        "minimum_progress_gain": min(progress_rows.values()),
        "minimum_utility_gain": min(utility_rows.values()),
    }


def _mixed_outcome_direction(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_task: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["variant"] == EVIDENCE and row["is_final_prefix"]:
            by_task[str(row["task_id"])].append(row)
    directions = {}
    for task, values in sorted(by_task.items()):
        positives = [
            float(row["utility_probability"])
            for row in values
            if float(row["utility_target"]) == 1.0
        ]
        negatives = [
            float(row["utility_probability"])
            for row in values
            if float(row["utility_target"]) == 0.0
        ]
        if positives and negatives:
            directions[task] = float(np.mean(positives) - np.mean(negatives))
    return {
        "mixed_task_count": len(directions),
        "positive_direction_count": sum(value > 0 for value in directions.values()),
        "task_probability_gaps": directions,
    }


def _preferred_architecture(
    *, decision: str, accepted_decision: str, transformer_superiority: bool
) -> str:
    """Recommend a backbone only after the full evidence gate is accepted."""

    if decision != accepted_decision:
        return "none_accepted"
    return TRANSFORMER if transformer_superiority else EVIDENCE


def _integrity(audit: Mapping[str, Any], manifest: Mapping[str, Any], protocol: Mapping[str, Any]) -> dict[str, Any]:
    expected_runs = (
        len(protocol["frozen_variants"])
        * len(protocol["training"]["training_seeds"])
        * len(protocol["task_folds"])
    )
    checks = {
        "episodes_90": audit["counts"]["episodes"] == 90 == manifest["counts"]["episodes"],
        "tasks_15": audit["counts"]["tasks"] == 15 == manifest["counts"]["tasks"],
        "executed_calls_455": audit["counts"]["executed_calls"] == 455,
        "proposals_456": audit["counts"]["proposals"] == 456,
        "terminal_unexecuted_1": audit["counts"]["terminal_unexecuted"] == 1,
        "utility_successes_14": audit["counts"]["utility_successes"] == 14,
        "input_chunks_12": len(audit["input_chunks"]) == 12,
        "runs_complete": manifest["counts"]["runs"] == expected_runs,
        "variants_exact": tuple(manifest["variants"]) == FROZEN_VARIANTS,
        "outcome_gradient_blocked": manifest["outcome_gradient_into_progress_encoder"] is False,
        "no_test_selection": manifest["model_selection_on_test"] is False,
        "no_hyperparameter_grid": manifest["hyperparameter_grid"] is False,
    }
    return {"checks": checks, "passed": all(checks.values())}


def _markdown(summary: Mapping[str, Any]) -> str:
    metrics = summary["seed_averaged_metrics"]
    lines = [
        "# 0725 clean evidence-ledger architecture probe",
        "",
        "## Decision",
        "",
        f"`{summary['decision']}`",
        "",
        "This is a clean-only architecture result. It does not open attack-data construction or Dreamer training.",
        "",
        "## Frozen-grid results",
        "",
        "| Variant | Task-macro progress MAE | Task-macro utility Brier | Utility log loss |",
        "|---|---:|---:|---:|",
    ]
    for variant in FROZEN_VARIANTS:
        row = metrics[variant]
        lines.append(
            f"| `{variant}` | {row['task_macro_progress_mae']:.6f} | "
            f"{row['task_macro_utility_brier']:.6f} | {row['task_macro_utility_log_loss']:.6f} |"
        )
    lines.extend(["", "## Prespecified gates", ""])
    for gate, passed in summary["gates"].items():
        lines.append(f"- {'PASS' if passed else 'FAIL'} — `{gate}`")
    lines.extend(["", "## Evidence-ledger comparisons", ""])
    for name, comparison in summary["comparisons"].items():
        lines.append(
            f"- `{name}`: progress gain {comparison['task_macro_progress_mae_gain']:.6f}; "
            f"utility-Brier gain {comparison['task_macro_utility_brier_gain']:.6f}; "
            f"positive tasks {comparison['progress_positive_tasks']}/{comparison['task_count']} "
            f"and {comparison['utility_positive_tasks']}/{comparison['task_count']}."
        )
    mixed = summary["mixed_outcome_direction"]
    lines.extend(
        [
            "",
            "## Mixed-outcome and robustness checks",
            "",
            f"- Evidence utility direction was positive in {mixed['positive_direction_count']}/{mixed['mixed_task_count']} mixed-outcome tasks.",
            f"- Leave-one-task-out minimum progress gain: {summary['leave_one_task_out']['minimum_progress_gain']:.6f}.",
            f"- Leave-one-task-out minimum utility-Brier gain: {summary['leave_one_task_out']['minimum_utility_gain']:.6f}.",
            "",
            "## Claim boundary",
            "",
            "The panel has only 15 independent tasks and 14/90 clean successes. Even a positive architecture signal remains clean-gate blocked until a separately frozen stronger victim/task panel recovers the durable development-confirmation solvability gate.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--training-manifest", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    args = parser.parse_args()

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    manifest = json.loads(args.training_manifest.read_text(encoding="utf-8"))
    rows = _read_jsonl(args.predictions)
    averaged_rows = _average_training_seeds(rows)
    averaged_metrics = _metrics_by_variant(averaged_rows)
    per_seed_metrics = _metrics_by_variant_seed(rows)
    if set(averaged_metrics) != set(FROZEN_VARIANTS):
        raise ValueError("missing variant predictions")

    comparisons = {
        "evidence_vs_event_state": _comparison(averaged_metrics, EVIDENCE, EVENT_STATE),
        "evidence_vs_shuffled_evidence": _comparison(averaged_metrics, EVIDENCE, SHUFFLE),
        "evidence_vs_output_length": _comparison(averaged_metrics, EVIDENCE, OUTPUT_LENGTH),
        "evidence_vs_transformer": _comparison(averaged_metrics, EVIDENCE, TRANSFORMER),
    }
    baseline = comparisons["evidence_vs_event_state"]
    shuffle = comparisons["evidence_vs_shuffled_evidence"]
    output_length = comparisons["evidence_vs_output_length"]
    leave_one_out = _leave_one_task_out(baseline)
    mixed = _mixed_outcome_direction(averaged_rows)
    integrity = _integrity(audit, manifest, protocol)
    seeds = [str(seed) for seed in protocol["training"]["training_seeds"]]
    all_seed_progress = all(
        per_seed_metrics[EVIDENCE][seed]["task_macro_progress_mae"]
        < per_seed_metrics[EVENT_STATE][seed]["task_macro_progress_mae"]
        for seed in seeds
    )
    all_seed_utility = all(
        per_seed_metrics[EVIDENCE][seed]["task_macro_utility_brier"]
        < per_seed_metrics[EVENT_STATE][seed]["task_macro_utility_brier"]
        for seed in seeds
    )
    gates = {
        "complete_source_and_pairing_integrity": integrity["passed"],
        "evidence_progress_improves_over_event_state": baseline["task_macro_progress_mae_gain"] > 0,
        "episode_utility_improves_over_event_state": baseline["task_macro_utility_brier_gain"] > 0,
        "evidence_beats_within_task_shuffle_on_both_primary_targets": (
            shuffle["task_macro_progress_mae_gain"] > 0
            and shuffle["task_macro_utility_brier_gain"] > 0
        ),
        "evidence_beats_output_length_on_both_primary_targets": (
            output_length["task_macro_progress_mae_gain"] > 0
            and output_length["task_macro_utility_brier_gain"] > 0
        ),
        "positive_progress_and_utility_gain_in_all_three_training_seeds": (
            all_seed_progress and all_seed_utility
        ),
        "mixed_outcome_utility_direction_at_least_6_of_8_tasks": (
            mixed["mixed_task_count"] == 8 and mixed["positive_direction_count"] >= 6
        ),
        "leave_one_task_out_progress_and_utility_gains_remain_positive": (
            leave_one_out["all_progress_positive"]
            and leave_one_out["all_utility_positive"]
        ),
    }
    if all(gates.values()):
        decision = protocol["interpretation"]["both_targets_and_controls_pass"]
    elif (
        gates["evidence_progress_improves_over_event_state"]
        and not gates["episode_utility_improves_over_event_state"]
    ):
        decision = protocol["interpretation"]["progress_only"]
    elif (
        gates["evidence_progress_improves_over_event_state"]
        and gates["episode_utility_improves_over_event_state"]
        and (
            not gates["evidence_beats_within_task_shuffle_on_both_primary_targets"]
            or not gates["evidence_beats_output_length_on_both_primary_targets"]
        )
    ):
        decision = protocol["interpretation"]["controls_explain_gain"]
    else:
        decision = protocol["interpretation"]["no_increment"]

    transformer_comparison = comparisons["evidence_vs_transformer"]
    transformer_superiority_established = (
        transformer_comparison["task_macro_progress_mae_gain"] < 0
        and transformer_comparison["task_macro_utility_brier_gain"] < 0
        and transformer_comparison["progress_exact_sign_flip_p"] <= 0.05
        and transformer_comparison["utility_exact_sign_flip_p"] <= 0.05
    )
    summary = {
        "protocol_id": protocol["protocol_id"],
        "decision": decision,
        "clean_gate_blocked": True,
        "attack_data_authorized": False,
        "dreamer_training_authorized": False,
        "integrity": integrity,
        "gates": gates,
        "seed_averaged_metrics": averaged_metrics,
        "per_training_seed_metrics": per_seed_metrics,
        "comparisons": comparisons,
        "leave_one_task_out": leave_one_out,
        "mixed_outcome_direction": mixed,
        "transformer_superiority_established": transformer_superiority_established,
        "preferred_clean_architecture": _preferred_architecture(
            decision=decision,
            accepted_decision=protocol["interpretation"]["both_targets_and_controls_pass"],
            transformer_superiority=transformer_superiority_established,
        ),
        "claim_boundary": protocol["claim_boundary"],
    }
    _write_json(args.output_json, summary)
    args.output_markdown.write_text(_markdown(summary), encoding="utf-8")
    print(json.dumps({"decision": decision, "gates": gates}, indent=2))


if __name__ == "__main__":
    main()
