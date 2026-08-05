"""Apply the frozen tau3 multi-step predictive-method acceptance gate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wmagentattack.tau3_multistep import file_sha256
from wmagentattack.tau3_multistep_experiment import (
    BASELINE_VARIANTS,
    NEURAL_VARIANTS,
    average_task_maps,
    evaluate_method_gate,
    task_macro,
    task_metric_map,
    two_step_task_map,
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
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, default=str)
        + "\n",
        encoding="utf-8",
    )


def _select_action(
    rows: Sequence[Mapping[str, Any]],
    *,
    variant: str,
    seed: int,
    split: str,
    domain: str | None = None,
) -> list[Mapping[str, Any]]:
    selected = [
        row
        for row in rows
        if row["variant"] == variant
        and int(row["training_seed"]) == seed
        and row["split"] == split
        and (domain is None or row["domain"] == domain)
    ]
    if not selected:
        raise ValueError(f"empty action surface: {variant}/seed{seed}/{split}/{domain}")
    return selected


def _select_transition(
    rows: Sequence[Mapping[str, Any]],
    *,
    variant: str,
    seed: int,
    split: str,
    domain: str | None = None,
    target_name: str | None = None,
) -> list[Mapping[str, Any]]:
    selected = [
        row
        for row in rows
        if row["variant"] == variant
        and int(row["training_seed"]) == seed
        and row["split"] == split
        and bool(row["supported"])
        and (domain is None or row["domain"] == domain)
        and (target_name is None or row["target_name"] == target_name)
    ]
    if not selected:
        raise ValueError(
            f"empty transition surface: {variant}/seed{seed}/{split}/{domain}/{target_name}"
        )
    return selected


def _arm(
    action_rows: Sequence[Mapping[str, Any]],
    transition_rows: Sequence[Mapping[str, Any]],
    *,
    variant: str,
    seeds: Sequence[int],
    split: str,
) -> dict[str, Any]:
    active = [0] if variant in BASELINE_VARIANTS else list(seeds)
    action_nll = {
        str(seed): task_macro(
            _select_action(action_rows, variant=variant, seed=seed, split=split),
            "action_nll",
        )
        for seed in active
    }
    action_accuracy = {
        str(seed): task_macro(
            _select_action(action_rows, variant=variant, seed=seed, split=split),
            "action_correct",
        )
        for seed in active
    }
    legal = {
        str(seed): task_macro(
            _select_action(action_rows, variant=variant, seed=seed, split=split),
            "legal_prediction",
        )
        for seed in active
    }
    two_step = {}
    for seed in active:
        try:
            values = two_step_task_map(
                _select_action(
                    action_rows, variant=variant, seed=seed, split=split
                ),
                split=split,
            )
        except ValueError:
            values = {}
        two_step[str(seed)] = (
            float(np.mean(list(values.values()))) if values else None
        )
    output: dict[str, Any] = {
        "action_nll": float(np.mean(list(action_nll.values()))),
        "action_nll_by_seed": action_nll,
        "action_accuracy": float(np.mean(list(action_accuracy.values()))),
        "action_accuracy_by_seed": action_accuracy,
        "legal_prediction_rate": float(np.mean(list(legal.values()))),
        "legal_prediction_rate_by_seed": legal,
        "two_step_sequence_accuracy": (
            float(np.mean([value for value in two_step.values() if value is not None]))
            if any(value is not None for value in two_step.values())
            else None
        ),
        "two_step_sequence_accuracy_by_seed": two_step,
    }
    if variant != "tfidf_candidate_logistic":
        transition_brier = {
            str(seed): task_macro(
                _select_transition(
                    transition_rows, variant=variant, seed=seed, split=split
                ),
                "brier",
            )
            for seed in active
        }
        transition_bce = {
            str(seed): task_macro(
                _select_transition(
                    transition_rows, variant=variant, seed=seed, split=split
                ),
                "bce",
            )
            for seed in active
        }
        output.update(
            {
                "transition_brier": float(np.mean(list(transition_brier.values()))),
                "transition_brier_by_seed": transition_brier,
                "transition_bce": float(np.mean(list(transition_bce.values()))),
                "transition_bce_by_seed": transition_bce,
            }
        )
    return output


def _domain_diagnostics(
    action_rows: Sequence[Mapping[str, Any]],
    transition_rows: Sequence[Mapping[str, Any]],
    *,
    seeds: Sequence[int],
    split: str,
) -> dict[str, Any]:
    output = {}
    for domain in ("airline", "retail", "telecom"):
        domain_row = {}
        for variant in ("frequency_prior", *NEURAL_VARIANTS):
            active = [0] if variant == "frequency_prior" else list(seeds)
            action_nll = [
                task_macro(
                    _select_action(
                        action_rows,
                        variant=variant,
                        seed=seed,
                        split=split,
                        domain=domain,
                    ),
                    "action_nll",
                )
                for seed in active
            ]
            transition_brier = []
            for seed in active:
                try:
                    selected_transition = _select_transition(
                        transition_rows,
                        variant=variant,
                        seed=seed,
                        split=split,
                        domain=domain,
                    )
                except ValueError:
                    continue
                transition_brier.append(task_macro(selected_transition, "brier"))
            domain_row[variant] = {
                "action_nll": float(np.mean(action_nll)),
                "transition_brier": (
                    float(np.mean(transition_brier)) if transition_brier else None
                ),
            }
        output[domain] = domain_row
    return output


def _target_diagnostics(
    rows: Sequence[Mapping[str, Any]],
    *,
    supported_names: Sequence[str],
    seeds: Sequence[int],
    split: str,
) -> dict[str, Any]:
    output = {}
    for name in supported_names:
        frequency = task_macro(
            _select_transition(
                rows,
                variant="frequency_prior",
                seed=0,
                split=split,
                target_name=name,
            ),
            "brier",
        )
        candidate = [
            task_macro(
                _select_transition(
                    rows,
                    variant="observed_semantic_markov_v4",
                    seed=seed,
                    split=split,
                    target_name=name,
                ),
                "brier",
            )
            for seed in seeds
        ]
        output[name] = {
            "frequency_brier": frequency,
            "candidate_brier": float(np.mean(candidate)),
            "candidate_gain": frequency - float(np.mean(candidate)),
        }
    return output


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def _markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# tau3 multi-step scale-readiness method results",
        "",
        f"Decision: `{summary['decision']}`",
        "",
        "## Frozen confirmation comparison",
        "",
        "| Model | Action NLL | Action accuracy | Two-step accuracy | Transition Brier |",
        "|---|---:|---:|---:|---:|",
    ]
    for variant in (*BASELINE_VARIANTS, *NEURAL_VARIANTS):
        row = summary["arms"][variant]
        lines.append(
            f"| {variant} | {_fmt(row['action_nll'])} | {_fmt(row['action_accuracy'])} | "
            f"{_fmt(row['two_step_sequence_accuracy'])} | "
            f"{_fmt(row['transition_brier']) if row.get('transition_brier') is not None else 'n/a'} |"
        )
    lines.extend(["", "## Acceptance gate", ""])
    for name, passed in summary["gate_checks"].items():
        lines.append(f"- {name}: `{'PASS' if passed else 'FAIL'}`")
    comparison = summary["candidate_comparison"]
    lines.extend(
        [
            "",
            "## Required counterevidence",
            "",
            f"- Mean action-NLL gain over frequency: {_fmt(comparison['mean_nll_gain_over_frequency'])}.",
            f"- Mean action-accuracy gain over frequency: {_fmt(comparison['mean_accuracy_gain_over_frequency'])}.",
            f"- Positive paired-task fraction: {_fmt(comparison['positive_task_fraction'])}.",
            f"- Candidate minus full-history NLL: {_fmt(comparison['candidate_minus_full_history_nll'])}.",
            f"- Mean two-step gain over frequency: {_fmt(comparison['mean_two_step_gain_over_frequency'])}.",
            f"- Mean transition-Brier gain over frequency: {_fmt(comparison['mean_transition_brier_gain_over_frequency'])}.",
            "- TF-IDF, Semantic Markov, Structured v3, full history, all seeds, all domains, and every supported transition target remain in the archived metrics even when they contradict the candidate.",
            "",
            "## Authorization boundary",
            "",
            "A GO authorizes a frozen large tau3 multi-step data build under the same Llama-3.1-70B contract and exact sandbox execution. It does not authorize attacks, real endpoints, Dreamer, or a planner. A NO-GO forbids scale-up and requires a separately preregistered mechanism rather than threshold relaxation.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--pilot-gate", type=Path, required=True)
    parser.add_argument("--run-metrics", type=Path, required=True)
    parser.add_argument("--action-predictions", type=Path, required=True)
    parser.add_argument("--transition-predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    pilot_gate = json.loads(args.pilot_gate.read_text(encoding="utf-8"))
    run_metrics = json.loads(args.run_metrics.read_text(encoding="utf-8"))
    if protocol["status"] != "data_gate_passed_and_method_frozen_before_training":
        raise ValueError("method protocol status differs")
    if run_metrics["protocol_sha256"] != file_sha256(args.protocol):
        raise ValueError("training protocol hash differs")
    if run_metrics["pilot_gate_sha256"] != file_sha256(args.pilot_gate):
        raise ValueError("training pilot-gate hash differs")
    if run_metrics["action_predictions_sha256"] != file_sha256(
        args.action_predictions
    ):
        raise ValueError("action prediction hash differs")
    if run_metrics["transition_predictions_sha256"] != file_sha256(
        args.transition_predictions
    ):
        raise ValueError("transition prediction hash differs")
    budget = protocol["method_training_budget_if_data_gate_passes"]
    if run_metrics["neural_runs"] != int(budget["neural_training_runs"]):
        raise ValueError("neural run budget is incomplete")
    if run_metrics["frequency_fits"] != int(budget["frequency_fits"]):
        raise ValueError("frequency fit budget is incomplete")
    if run_metrics["tfidf_fits"] != int(budget["tfidf_fits"]):
        raise ValueError("TF-IDF fit budget is incomplete")
    action_rows = _read_jsonl(args.action_predictions)
    transition_rows = _read_jsonl(args.transition_predictions)
    expected_action = run_metrics["prefixes"] * (
        len(BASELINE_VARIANTS) + int(budget["neural_training_runs"])
    )
    expected_transition = (
        run_metrics["transitions"]
        * len(protocol["transition_targets"]["names"])
        * (1 + int(budget["neural_training_runs"]))
    )
    if len(action_rows) != expected_action:
        raise ValueError("action prediction budget differs")
    if len(transition_rows) != expected_transition:
        raise ValueError("transition prediction budget differs")
    seeds = [int(seed) for seed in budget["training_seeds"]]
    split = str(protocol["method_acceptance_gate"]["split"])
    arms = {
        variant: _arm(
            action_rows,
            transition_rows,
            variant=variant,
            seeds=seeds,
            split=split,
        )
        for variant in (*BASELINE_VARIANTS, *NEURAL_VARIANTS)
    }
    frequency = arms["frequency_prior"]
    candidate = arms["observed_semantic_markov_v4"]
    full_history = arms["full_history_diagnostic"]
    nll_seed_gains = [
        frequency["action_nll"] - candidate["action_nll_by_seed"][str(seed)]
        for seed in seeds
    ]
    accuracy_seed_gains = [
        candidate["action_accuracy_by_seed"][str(seed)]
        - frequency["action_accuracy"]
        for seed in seeds
    ]
    two_step_surface_available = (
        frequency["two_step_sequence_accuracy"] is not None
        and all(
            candidate["two_step_sequence_accuracy_by_seed"][str(seed)] is not None
            for seed in seeds
        )
    )
    two_step_seed_gains = (
        [
            candidate["two_step_sequence_accuracy_by_seed"][str(seed)]
            - frequency["two_step_sequence_accuracy"]
            for seed in seeds
        ]
        if two_step_surface_available
        else []
    )
    transition_brier_seed_gains = [
        frequency["transition_brier"]
        - candidate["transition_brier_by_seed"][str(seed)]
        for seed in seeds
    ]
    frequency_tasks = task_metric_map(
        _select_action(
            action_rows,
            variant="frequency_prior",
            seed=0,
            split=split,
        ),
        "action_nll",
    )
    candidate_tasks = average_task_maps(
        [
            task_metric_map(
                _select_action(
                    action_rows,
                    variant="observed_semantic_markov_v4",
                    seed=seed,
                    split=split,
                ),
                "action_nll",
            )
            for seed in seeds
        ]
    )
    if set(frequency_tasks) != set(candidate_tasks):
        raise ValueError("paired action task surface differs")
    paired_task_gains = {
        task: float(frequency_tasks[task] - candidate_tasks[task])
        for task in sorted(frequency_tasks)
    }
    candidate_minus_full = candidate["action_nll"] - full_history["action_nll"]
    legal_rate = candidate["legal_prediction_rate"]
    gate_checks = evaluate_method_gate(
        nll_seed_gains=nll_seed_gains,
        accuracy_seed_gains=accuracy_seed_gains,
        paired_task_nll_gains=list(paired_task_gains.values()),
        candidate_minus_full_history_nll=candidate_minus_full,
        two_step_seed_gains=two_step_seed_gains,
        transition_brier_seed_gains=transition_brier_seed_gains,
        legal_prediction_rate=legal_rate,
        data_gate_passed=bool(pilot_gate["passed"]),
        two_step_surface_available=two_step_surface_available,
        gate=protocol["method_acceptance_gate"],
    )
    passed = all(gate_checks.values())
    decision = (
        "METHOD_GO__AUTHORIZE_LARGE_SCALE_TAU3_MULTISTEP_V1"
        if passed
        else "METHOD_NO_GO__DO_NOT_SCALE__PREREGISTER_NEXT_MECHANISM"
    )
    comparison = {
        "nll_seed_gains_over_frequency": dict(zip(map(str, seeds), nll_seed_gains)),
        "mean_nll_gain_over_frequency": float(np.mean(nll_seed_gains)),
        "accuracy_seed_gains_over_frequency": dict(
            zip(map(str, seeds), accuracy_seed_gains)
        ),
        "mean_accuracy_gain_over_frequency": float(np.mean(accuracy_seed_gains)),
        "paired_task_nll_gains": paired_task_gains,
        "positive_task_fraction": sum(value > 0.0 for value in paired_task_gains.values())
        / len(paired_task_gains),
        "candidate_minus_full_history_nll": candidate_minus_full,
        "two_step_seed_gains_over_frequency": dict(
            zip(map(str, seeds), two_step_seed_gains)
        ),
        "mean_two_step_gain_over_frequency": (
            float(np.mean(two_step_seed_gains)) if two_step_seed_gains else None
        ),
        "transition_brier_seed_gains_over_frequency": dict(
            zip(map(str, seeds), transition_brier_seed_gains)
        ),
        "mean_transition_brier_gain_over_frequency": float(
            np.mean(transition_brier_seed_gains)
        ),
    }
    summary = {
        "decision": decision,
        "passed": passed,
        "gate_checks": gate_checks,
        "arms": arms,
        "candidate_comparison": comparison,
        "domain_diagnostics": _domain_diagnostics(
            action_rows,
            transition_rows,
            seeds=seeds,
            split=split,
        ),
        "transition_target_diagnostics": _target_diagnostics(
            transition_rows,
            supported_names=run_metrics["supported_transition_target_names"],
            seeds=seeds,
            split=split,
        ),
        "prefixes": run_metrics["prefixes"],
        "transitions": run_metrics["transitions"],
        "supported_transition_target_names": run_metrics[
            "supported_transition_target_names"
        ],
        "fixed_budget_complete": True,
        "protocol_sha256": file_sha256(args.protocol),
        "pilot_gate_sha256": file_sha256(args.pilot_gate),
        "run_metrics_sha256": file_sha256(args.run_metrics),
        "action_predictions_sha256": file_sha256(args.action_predictions),
        "transition_predictions_sha256": file_sha256(
            args.transition_predictions
        ),
        "claim_boundary": (
            "Only a complete GO authorizes a frozen larger tau3 multi-step build; "
            "attacks, Dreamer, planning, and real endpoints remain disabled."
        ),
    }
    _write_json(args.output, summary)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(_markdown(summary), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
