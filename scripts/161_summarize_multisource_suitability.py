"""Apply the frozen gates to the multi-source suitability experiment."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from wmagentattack.multisource_suitability import file_sha256
from wmagentattack.multisource_suitability_experiment import (
    BASELINE_VARIANTS,
    NEURAL_VARIANTS,
    SOURCE_SCOPES,
    TRAINING_SCOPES,
    evaluate_action_gate,
    evaluate_error_gate,
    exact_sign_test,
    paired_bootstrap,
    task_metric_map,
    task_macro,
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _select(
    rows: Sequence[Mapping[str, Any]],
    *,
    scope: str,
    variant: str,
    seed: int,
    split: str,
    source: str | None = None,
    exact_only: bool = False,
) -> list[Mapping[str, Any]]:
    selected = [
        row
        for row in rows
        if row["scope"] == scope
        and row["variant"] == variant
        and int(row["training_seed"]) == seed
        and row["split"] == split
        and (source is None or row["source"] == source)
        and (not exact_only or row["exact_outcome_available"])
    ]
    if not selected:
        raise ValueError(
            f"empty prediction surface: {scope}/{variant}/seed{seed}/{split}/{source}"
        )
    return selected


def _average_task_maps(maps: Sequence[Mapping[str, float]]) -> dict[str, float]:
    tasks = set(maps[0])
    if any(set(mapping) != tasks for mapping in maps):
        raise ValueError("seed task surfaces differ")
    return {
        task: float(np.mean([mapping[task] for mapping in maps]))
        for task in sorted(tasks)
    }


def _arm_summary(
    predictions: Sequence[Mapping[str, Any]],
    *,
    scope: str,
    variant: str,
    seeds: Sequence[int],
    split: str,
    source: str | None = None,
) -> dict[str, Any]:
    active_seeds = [0] if variant in BASELINE_VARIANTS else list(seeds)
    metrics: dict[str, Any] = {}
    for metric in ("action_nll", "action_correct", "tool_brier", "legal_prediction"):
        values = [
            task_macro(
                _select(
                    predictions,
                    scope=scope,
                    variant=variant,
                    seed=seed,
                    split=split,
                    source=source,
                ),
                metric,
            )
            for seed in active_seeds
        ]
        metrics[metric] = float(np.mean(values))
        metrics[f"{metric}_by_seed"] = dict(zip(map(str, active_seeds), values))
    return metrics


def _error_summary(
    predictions: Sequence[Mapping[str, Any]],
    *,
    scope: str,
    seeds: Sequence[int],
    split: str,
    support: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    if not support["authorized"]:
        return {
            "authorized": False,
            "support": support,
            "gate_passed": False,
            "status": "NOT_AUTHORIZED_BY_FROZEN_SUPPORT_GATE",
        }
    frequency_rows = _select(
        predictions,
        scope=scope,
        variant="frequency_prior",
        seed=0,
        split=split,
        exact_only=True,
    )
    frequency_bce = task_macro(frequency_rows, "execution_error_bce")
    frequency_brier = task_macro(frequency_rows, "execution_error_brier")
    structured_rows_by_seed = [
        _select(
            predictions,
            scope=scope,
            variant="structured_markov_v3",
            seed=seed,
            split=split,
            exact_only=True,
        )
        for seed in seeds
    ]
    bce_by_seed = [
        task_macro(rows, "execution_error_bce") for rows in structured_rows_by_seed
    ]
    brier_by_seed = [
        task_macro(rows, "execution_error_brier") for rows in structured_rows_by_seed
    ]
    seed_gains = [frequency_bce - value for value in bce_by_seed]
    frequency_tasks = task_metric_map(frequency_rows, "execution_error_bce")
    structured_tasks = _average_task_maps(
        [task_metric_map(rows, "execution_error_bce") for rows in structured_rows_by_seed]
    )
    if set(frequency_tasks) != set(structured_tasks):
        raise ValueError("error-probe paired task surfaces differ")
    task_gains = {
        task: float(frequency_tasks[task] - structured_tasks[task])
        for task in sorted(frequency_tasks)
    }
    checks = evaluate_error_gate(
        bce_seed_gains=seed_gains,
        paired_bce_task_gains=list(task_gains.values()),
        gate=protocol["execution_error_acceptance_gate"],
    )
    draws = int(protocol["uncertainty"]["paired_task_bootstrap_draws"])
    bootstrap_seed = int(protocol["uncertainty"]["bootstrap_seed"])
    return {
        "authorized": True,
        "support": support,
        "frequency_bce": frequency_bce,
        "frequency_brier": frequency_brier,
        "structured_bce": float(np.mean(bce_by_seed)),
        "structured_bce_by_seed": dict(zip(map(str, seeds), bce_by_seed)),
        "structured_brier": float(np.mean(brier_by_seed)),
        "bce_seed_gains": dict(zip(map(str, seeds), seed_gains)),
        "paired_task_bce_gains": task_gains,
        "paired_bootstrap": paired_bootstrap(
            list(task_gains.values()), draws=draws, seed=bootstrap_seed + 51
        ),
        "paired_sign_test": exact_sign_test(list(task_gains.values())),
        "gate_checks": checks,
        "gate_passed": all(checks.values()),
        "status": "PASS" if all(checks.values()) else "FAIL",
    }


def _scope_summary(
    predictions: Sequence[Mapping[str, Any]],
    *,
    scope: str,
    seeds: Sequence[int],
    split: str,
    support: Mapping[str, Any],
    protocol: Mapping[str, Any],
    audit: Mapping[str, Any],
) -> dict[str, Any]:
    variants = (*BASELINE_VARIANTS, *NEURAL_VARIANTS)
    arms = {
        variant: _arm_summary(
            predictions,
            scope=scope,
            variant=variant,
            seeds=seeds,
            split=split,
        )
        for variant in variants
    }
    frequency = arms["frequency_prior"]
    structured = arms["structured_markov_v3"]
    tfidf = arms["tfidf_candidate_logistic"]
    nll_seed_gains = [
        frequency["action_nll"]
        - structured["action_nll_by_seed"][str(seed)]
        for seed in seeds
    ]
    accuracy_seed_gains = [
        structured["action_correct_by_seed"][str(seed)]
        - frequency["action_correct"]
        for seed in seeds
    ]
    frequency_tasks = task_metric_map(
        _select(
            predictions,
            scope=scope,
            variant="frequency_prior",
            seed=0,
            split=split,
        ),
        "action_nll",
    )
    structured_seed_tasks = [
        task_metric_map(
            _select(
                predictions,
                scope=scope,
                variant="structured_markov_v3",
                seed=seed,
                split=split,
            ),
            "action_nll",
        )
        for seed in seeds
    ]
    structured_tasks = _average_task_maps(structured_seed_tasks)
    if set(frequency_tasks) != set(structured_tasks):
        raise ValueError("action paired task surfaces differ")
    paired_gains = {
        task: float(frequency_tasks[task] - structured_tasks[task])
        for task in sorted(frequency_tasks)
    }
    checks = evaluate_action_gate(
        nll_seed_gains=nll_seed_gains,
        accuracy_seed_gains=accuracy_seed_gains,
        paired_nll_task_gains=list(paired_gains.values()),
        structured_nll_gap_to_tfidf=(
            structured["action_nll"] - tfidf["action_nll"]
        ),
        legal_prediction_rate=structured["legal_prediction"],
        gate=protocol["action_acceptance_gate"],
    )
    action_passed = all(checks.values())
    error = _error_summary(
        predictions,
        scope=scope,
        seeds=seeds,
        split=split,
        support=support,
        protocol=protocol,
    )
    source_names = list(SOURCE_SCOPES) if scope == "combined" else [scope]
    transitions = sum(
        int(audit["source_audits"][source]["adjacent_semantic_transitions"])
        for source in source_names
    )
    full_checks = {
        "action_acceptance_gate": action_passed,
        "minimum_adjacent_semantic_transitions": transitions
        >= int(
            protocol["full_world_model_readiness_gate"][
                "minimum_adjacent_semantic_transitions"
            ]
        ),
        "minimum_positive_rows_for_all_five_evidence_deltas": False,
        "execution_error_probe_gate": error["gate_passed"],
    }
    full_passed = all(full_checks.values())
    if full_passed:
        decision = protocol["decision_policy"]["action_and_full_world_model_pass"]
    elif action_passed:
        decision = protocol["decision_policy"]["action_pass_full_world_model_fail"]
    elif scope == "injecagent" and audit["checks"]["injecagent_pairs_within_split"]:
        decision = protocol["decision_policy"][
            "injecagent_action_fail_but_paired_surface_valid"
        ]
    else:
        decision = protocol["decision_policy"]["action_fail"]

    draws = int(protocol["uncertainty"]["paired_task_bootstrap_draws"])
    bootstrap_seed = int(protocol["uncertainty"]["bootstrap_seed"])
    return {
        "scope": scope,
        "confirmation_tasks": len(frequency_tasks),
        "arms": arms,
        "action_comparison": {
            "structured_nll_seed_gains_over_frequency": dict(
                zip(map(str, seeds), nll_seed_gains)
            ),
            "structured_accuracy_seed_gains_over_frequency": dict(
                zip(map(str, seeds), accuracy_seed_gains)
            ),
            "structured_nll_gap_to_tfidf": structured["action_nll"]
            - tfidf["action_nll"],
            "structured_nll_gap_to_full_history": structured["action_nll"]
            - arms["full_history_diagnostic"]["action_nll"],
            "paired_task_nll_gains": paired_gains,
            "positive_task_fraction": sum(
                gain > 0.0 for gain in paired_gains.values()
            )
            / len(paired_gains),
            "paired_bootstrap": paired_bootstrap(
                list(paired_gains.values()), draws=draws, seed=bootstrap_seed
            ),
            "paired_sign_test": exact_sign_test(list(paired_gains.values())),
        },
        "action_gate_checks": checks,
        "action_gate_passed": action_passed,
        "execution_error_probe": error,
        "full_world_model_checks": full_checks,
        "full_world_model_gate_passed": full_passed,
        "structural_transition_count": transitions,
        "decision": decision,
        "large_scale_interpretation": (
            "IMMEDIATE_FULL_WORLD_MODEL_SCALE_AUTHORIZED"
            if full_passed
            else "TARGETED_MULTISTEP_COLLECTION_PILOT_ONLY__NOT_LARGE_SCALE"
            if action_passed
            else "AUXILIARY_ROBUSTNESS_DATA_ONLY__NOT_WORLD_MODEL_CORE"
            if decision.startswith("AUXILIARY")
            else "DO_NOT_SCALE_THIS_SOURCE_WITH_CURRENT_METHOD"
        ),
    }


def _combined_adapter_comparison(
    predictions: Sequence[Mapping[str, Any]],
    *,
    source_summaries: Mapping[str, Mapping[str, Any]],
    seeds: Sequence[int],
    split: str,
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    threshold = float(
        protocol["decision_policy"]["combined_model_degradation_threshold_nll"]
    )
    sources = {}
    for source in SOURCE_SCOPES:
        combined_values = [
            task_macro(
                _select(
                    predictions,
                    scope="combined",
                    variant="structured_markov_v3",
                    seed=seed,
                    split=split,
                    source=source,
                ),
                "action_nll",
            )
            for seed in seeds
        ]
        source_specific = float(
            source_summaries[source]["arms"]["structured_markov_v3"]["action_nll"]
        )
        combined = float(np.mean(combined_values))
        gap = combined - source_specific
        sources[source] = {
            "source_specific_structured_nll": source_specific,
            "combined_structured_nll": combined,
            "combined_minus_source_specific_nll": gap,
            "materially_degraded": gap > threshold,
        }
    retain = any(row["materially_degraded"] for row in sources.values())
    return {
        "threshold_nll": threshold,
        "sources": sources,
        "decision": (
            protocol["decision_policy"]["combined_model_degrades_source_specific"]
            if retain
            else "COMBINED_ADAPTER_NOT_MATERIALLY_WORSE"
        ),
    }


def _injecagent_pair_diagnostic(
    predictions: Sequence[Mapping[str, Any]],
    *,
    seeds: Sequence[int],
    split: str,
) -> dict[str, Any]:
    actual_rows = _select(
        predictions,
        scope="injecagent",
        variant="frequency_prior",
        seed=0,
        split=split,
    )
    actual: dict[str, dict[str, str]] = defaultdict(dict)
    for row in actual_rows:
        actual[str(row["group_id"])][str(row["record_variant"])] = str(
            row["target_candidate_id"]
        )
    complete = {
        group: variants
        for group, variants in actual.items()
        if set(variants) == {"clean", "poisoned"}
    }
    target_divergence = float(
        np.mean(
            [
                variants["clean"] != variants["poisoned"]
                for variants in complete.values()
            ]
        )
    )
    predicted_divergence = {}
    clean_accuracy = {}
    poisoned_accuracy = {}
    for seed in seeds:
        rows = _select(
            predictions,
            scope="injecagent",
            variant="structured_markov_v3",
            seed=seed,
            split=split,
        )
        grouped: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
        for row in rows:
            grouped[str(row["group_id"])][str(row["record_variant"])] = row
        pairs = [
            variants
            for variants in grouped.values()
            if set(variants) == {"clean", "poisoned"}
        ]
        predicted_divergence[str(seed)] = float(
            np.mean(
                [
                    pair["clean"]["predicted_candidate_id"]
                    != pair["poisoned"]["predicted_candidate_id"]
                    for pair in pairs
                ]
            )
        )
        clean_accuracy[str(seed)] = float(
            np.mean([pair["clean"]["action_correct"] for pair in pairs])
        )
        poisoned_accuracy[str(seed)] = float(
            np.mean([pair["poisoned"]["action_correct"] for pair in pairs])
        )
    return {
        "confirmation_pairs": len(complete),
        "target_action_divergence_rate": target_divergence,
        "structured_predicted_action_divergence_by_seed": predicted_divergence,
        "structured_clean_accuracy_by_seed": clean_accuracy,
        "structured_poisoned_accuracy_by_seed": poisoned_accuracy,
        "gate_role": "counterevidence_only__not_an_acceptance_gate",
    }


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def _markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Multi-source current-method suitability results",
        "",
        f"Overall decision: `{summary['overall_decision']}`",
        "",
        "## Frozen confirmation results",
        "",
        "| Source | Freq NLL / Acc | TF-IDF NLL / Acc | Structured-v3 NLL / Acc | Action gate | Full-WM gate | Decision |",
        "|---|---:|---:|---:|---|---|---|",
    ]
    for scope in TRAINING_SCOPES:
        row = summary["scopes"][scope]
        frequency = row["arms"]["frequency_prior"]
        tfidf = row["arms"]["tfidf_candidate_logistic"]
        structured = row["arms"]["structured_markov_v3"]
        lines.append(
            f"| {scope} | {_fmt(frequency['action_nll'])} / {_fmt(frequency['action_correct'])} | "
            f"{_fmt(tfidf['action_nll'])} / {_fmt(tfidf['action_correct'])} | "
            f"{_fmt(structured['action_nll'])} / {_fmt(structured['action_correct'])} | "
            f"{'PASS' if row['action_gate_passed'] else 'FAIL'} | "
            f"{'PASS' if row['full_world_model_gate_passed'] else 'FAIL'} | `{row['decision']}` |"
        )
    lines.extend(["", "## Source decisions", ""])
    for source in SOURCE_SCOPES:
        row = summary["scopes"][source]
        comparison = row["action_comparison"]
        error = row["execution_error_probe"]
        lines.extend(
            [
                f"### {source}",
                "",
                f"- Structured-v3 action NLL gain over frequency: {_fmt(float(np.mean(list(comparison['structured_nll_seed_gains_over_frequency'].values()))))}",
                f"- Structured-v3 accuracy gain over frequency: {_fmt(float(np.mean(list(comparison['structured_accuracy_seed_gains_over_frequency'].values()))))}",
                f"- Positive paired-task fraction: {_fmt(comparison['positive_task_fraction'])}",
                f"- Structured-v3 minus TF-IDF NLL: {_fmt(comparison['structured_nll_gap_to_tfidf'])}",
                f"- Execution-error probe: `{error['status']}`",
                f"- Scale interpretation: `{row['large_scale_interpretation']}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Combined versus source-specific adapter",
            "",
            "| Source | Source-specific NLL | Combined NLL | Gap | Material degradation |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for source, row in summary["combined_adapter_comparison"]["sources"].items():
        lines.append(
            f"| {source} | {_fmt(row['source_specific_structured_nll'])} | "
            f"{_fmt(row['combined_structured_nll'])} | "
            f"{_fmt(row['combined_minus_source_specific_nll'])} | "
            f"{'YES' if row['materially_degraded'] else 'NO'} |"
        )
    pair = summary["injecagent_pair_diagnostic"]
    lines.extend(
        [
            "",
            "## Required counterevidence",
            "",
            f"- All sources contain `0` adjacent semantic transitions; therefore none can pass the frozen full-world-model gate from these one-step records alone.",
            f"- InjecAgent confirmation clean/poison pairs: {pair['confirmation_pairs']}; target-action divergence rate: {_fmt(pair['target_action_divergence_rate'])}.",
            f"- Combined adapter decision: `{summary['combined_adapter_comparison']['decision']}`.",
            "- TF-IDF, full-history, seed replication, paired-task bootstrap intervals, and exact sign tests are retained in `summary.json`; they are counterevidence and cannot be removed post hoc.",
            "",
            "## Claim boundary",
            "",
            "This fixed-budget study tests task-disjoint one-step predictive signal in existing synthetic records. An action-only GO authorizes a targeted multi-step collection pilot for that source. It does not authorize immediate unrestricted data scaling, attack generation, Dreamer/value training, or a claim that free-running world-model rollouts are valid.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--run-metrics", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    run_metrics = json.loads(args.run_metrics.read_text(encoding="utf-8"))
    predictions = _read_jsonl(args.predictions)
    if run_metrics["protocol_sha256"] != file_sha256(args.protocol):
        raise ValueError("training used a different protocol hash")
    if run_metrics["audit_sha256"] != file_sha256(args.audit):
        raise ValueError("training used a different audit hash")
    if run_metrics["neural_runs"] != int(
        protocol["fixed_budget"]["neural_training_runs"]
    ):
        raise ValueError("neural run budget is incomplete")
    if run_metrics["frequency_fits"] != int(
        protocol["fixed_budget"]["frequency_fits"]
    ):
        raise ValueError("frequency fit budget is incomplete")
    if run_metrics["tfidf_fits"] != int(protocol["fixed_budget"]["tfidf_fits"]):
        raise ValueError("TF-IDF fit budget is incomplete")
    predictions_per_row = 2 + len(NEURAL_VARIANTS) * len(
        protocol["training"]["training_seeds"]
    )
    expected_predictions = predictions_per_row * sum(
        audit["source_audits"][scope]["rows"]
        if scope != "combined"
        else int(protocol["source"]["expected_rows"])
        for scope in TRAINING_SCOPES
    )
    if len(predictions) != expected_predictions:
        raise ValueError(
            f"prediction budget mismatch: {len(predictions)} != {expected_predictions}"
        )

    seeds = [int(seed) for seed in protocol["training"]["training_seeds"]]
    split = str(protocol["action_acceptance_gate"]["split"])
    scopes = {
        scope: _scope_summary(
            predictions,
            scope=scope,
            seeds=seeds,
            split=split,
            support=run_metrics["error_probe_support"][scope],
            protocol=protocol,
            audit=audit,
        )
        for scope in TRAINING_SCOPES
    }
    adapter = _combined_adapter_comparison(
        predictions,
        source_summaries=scopes,
        seeds=seeds,
        split=split,
        protocol=protocol,
    )
    pair = _injecagent_pair_diagnostic(predictions, seeds=seeds, split=split)
    source_action_passes = [scopes[source]["action_gate_passed"] for source in SOURCE_SCOPES]
    full_passes = [
        scopes[source]["full_world_model_gate_passed"] for source in SOURCE_SCOPES
    ]
    if any(full_passes):
        overall = "AT_LEAST_ONE_SOURCE_READY_FOR_FULL_WORLD_MODEL_SCALE"
    elif any(source_action_passes):
        overall = "ACTION_SIGNAL_ONLY__TARGETED_MULTISTEP_COLLECTION_REQUIRED"
    else:
        overall = "NO_SOURCE_PASSES_CURRENT_METHOD_ACTION_GATE__DO_NOT_SCALE"
    summary = {
        "protocol_sha256": file_sha256(args.protocol),
        "audit_sha256": file_sha256(args.audit),
        "run_metrics_sha256": file_sha256(args.run_metrics),
        "predictions_sha256": file_sha256(args.predictions),
        "prediction_rows": len(predictions),
        "fixed_budget_complete": True,
        "training_runs": run_metrics["neural_runs"],
        "tfidf_fits": run_metrics["tfidf_fits"],
        "frequency_fits": run_metrics["frequency_fits"],
        "new_llm_calls": run_metrics["new_llm_calls"],
        "new_tool_executions": run_metrics["new_tool_executions"],
        "scopes": scopes,
        "combined_adapter_comparison": adapter,
        "injecagent_pair_diagnostic": pair,
        "structural_counterevidence": audit["structural_counterevidence"],
        "overall_decision": overall,
    }
    _write_json(args.output, summary)
    args.report.write_text(_markdown(summary), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
