"""Summarize the frozen 0721 factorized world-model diagnostic budget."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _mean_std(values):
    return {
        "mean": statistics.fmean(values),
        "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "values": values,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    io_reports = sorted(args.root.glob("io_hmm/state*/metrics.json"))
    event_reports = sorted(args.root.glob("event/seed*/metrics.json"))
    if len(io_reports) != 3:
        raise ValueError(f"Expected three IO-HMM state runs, found {len(io_reports)}")
    if len(event_reports) != 3:
        raise ValueError(f"Expected three Event Transformer seeds, found {len(event_reports)}")

    io_rows = []
    for path in io_reports:
        report = _read(path)
        validation = report["metrics"]["validation"]
        test = report["metrics"]["test"]
        markov_val = report["counterbaseline"]["metrics"]["validation"]
        markov_test = report["counterbaseline"]["metrics"]["test"]
        io_rows.append(
            {
                "states": int(report["config"]["num_states"]),
                "validation_mean_nll": validation["mean_event_nll"],
                "validation_accuracy": validation["next_event_accuracy"],
                "validation_markov_mean_nll": markov_val["mean_event_nll"],
                "validation_nll_delta_vs_markov": (
                    validation["mean_event_nll"] - markov_val["mean_event_nll"]
                ),
                "test_mean_nll": test["mean_event_nll"],
                "test_accuracy": test["next_event_accuracy"],
                "test_markov_mean_nll": markov_test["mean_event_nll"],
                "test_nll_delta_vs_markov": (
                    test["mean_event_nll"] - markov_test["mean_event_nll"]
                ),
                "path": str(path),
            }
        )
    best_io = min(io_rows, key=lambda row: row["validation_mean_nll"])

    event_rows = []
    for path in event_reports:
        report = _read(path)
        metrics = report["metrics"]
        baseline = report["joint_constant_baseline"]["count_nll"]
        seed = int(path.parent.name.removeprefix("seed"))
        event_rows.append(
            {
                "seed": seed,
                "validation_next_tool_accuracy": metrics["validation"][
                    "next_tool_accuracy"
                ],
                "validation_joint_count_nll": metrics["validation"][
                    "joint_count_nll"
                ],
                "validation_constant_joint_nll": baseline["validation"],
                "validation_joint_nll_delta_vs_constant": (
                    metrics["validation"]["joint_count_nll"]
                    - baseline["validation"]
                ),
                "test_next_tool_accuracy": metrics["test"]["next_tool_accuracy"],
                "test_joint_count_nll": metrics["test"]["joint_count_nll"],
                "test_constant_joint_nll": baseline["test"],
                "test_joint_nll_delta_vs_constant": (
                    metrics["test"]["joint_count_nll"] - baseline["test"]
                ),
                "path": str(path),
            }
        )
    for row in event_rows:
        for key, value in row.items():
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"Non-finite Event Transformer result: {key}")

    event_summary = {
        "validation_next_tool_accuracy": _mean_std(
            [row["validation_next_tool_accuracy"] for row in event_rows]
        ),
        "validation_joint_nll_delta_vs_constant": _mean_std(
            [row["validation_joint_nll_delta_vs_constant"] for row in event_rows]
        ),
        "test_next_tool_accuracy": _mean_std(
            [row["test_next_tool_accuracy"] for row in event_rows]
        ),
        "test_joint_nll_delta_vs_constant": _mean_std(
            [row["test_joint_nll_delta_vs_constant"] for row in event_rows]
        ),
    }
    thresholds = {
        "io_hmm_min_validation_nll_gain": 0.02,
        "io_hmm_max_test_nll_regression": 0.02,
        "event_min_validation_accuracy_vs_best_io": -0.02,
        "event_max_validation_accuracy_std": 0.05,
        "event_joint_nll_must_beat_constant": True,
    }
    gates = {
        "io_hmm_validation_beats_markov": (
            best_io["validation_nll_delta_vs_markov"] <= -0.02
        ),
        "io_hmm_test_not_worse_than_markov": (
            best_io["test_nll_delta_vs_markov"] <= 0.02
        ),
        "event_validation_accuracy_not_materially_below_io_hmm": (
            event_summary["validation_next_tool_accuracy"]["mean"]
            >= best_io["validation_accuracy"] - 0.02
        ),
        "event_seed_stability": (
            event_summary["validation_next_tool_accuracy"]["std"] <= 0.05
        ),
        "event_joint_count_nll_beats_constant": (
            event_summary["validation_joint_nll_delta_vs_constant"]["mean"] < 0.0
        ),
        "clean_eligibility_gate": False,
    }
    architecture_signal = all(
        value for key, value in gates.items() if key != "clean_eligibility_gate"
    )
    summary = {
        "scope": "fixed-budget diagnostic on existing AgentDojo-v2 sandbox data",
        "confirmatory": False,
        "reason_non_confirmatory": (
            "Frozen unseen-seed clean gate has zero durable tasks; this run cannot "
            "authorize new attack data or a paper-level world-model claim."
        ),
        "io_hmm_runs": io_rows,
        "best_io_hmm_selected_by_validation_only": best_io,
        "event_runs": event_rows,
        "event_summary": event_summary,
        "thresholds_frozen_before_run": thresholds,
        "gates": gates,
        "architecture_signal": architecture_signal,
        "decision": (
            "ARCHITECTURE_SIGNAL_ONLY_CLEAN_GATE_BLOCKED"
            if architecture_signal
            else "NO_GO_REVISE_FACTORIZED_MODEL"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

