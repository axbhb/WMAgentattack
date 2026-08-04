"""Summarize and gate the frozen Stage 3 Markov-sufficiency experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from wmagentattack.hybrid_semantic_world_model import EVIDENCE_DELTA_TARGETS
from wmagentattack.markov_sufficiency import (
    FROZEN_SUFFICIENCY_VARIANTS,
    evaluate_sufficiency_gate,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _task_map(
    rows: Sequence[Mapping[str, Any]],
    *,
    prediction_type: str,
    variant: str,
    seed: int,
    split: str,
    metric: str,
) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if (
            row["prediction_type"] == prediction_type
            and row["variant"] == variant
            and int(row["training_seed"]) == seed
            and row["split"] == split
        ):
            grouped[str(row["task_id"])].append(float(row[metric]))
    if not grouped:
        raise ValueError(
            f"no rows for {prediction_type}/{variant}/seed{seed}/{split}/{metric}"
        )
    return {task: float(np.mean(values)) for task, values in sorted(grouped.items())}


def _mean_task_metric(
    rows: Sequence[Mapping[str, Any]],
    *,
    prediction_type: str,
    variant: str,
    seed: int,
    split: str,
    metric: str,
) -> float:
    return float(
        np.mean(
            list(
                _task_map(
                    rows,
                    prediction_type=prediction_type,
                    variant=variant,
                    seed=seed,
                    split=split,
                    metric=metric,
                ).values()
            )
        )
    )


def _paired_task_gains(
    rows: Sequence[Mapping[str, Any]],
    *,
    prediction_type: str,
    baseline: str,
    candidate: str,
    seeds: Sequence[int],
    split: str,
    metric: str,
) -> dict[str, float]:
    baseline_maps = [
        _task_map(
            rows,
            prediction_type=prediction_type,
            variant=baseline,
            seed=seed,
            split=split,
            metric=metric,
        )
        for seed in seeds
    ]
    candidate_maps = [
        _task_map(
            rows,
            prediction_type=prediction_type,
            variant=candidate,
            seed=seed,
            split=split,
            metric=metric,
        )
        for seed in seeds
    ]
    tasks = set(baseline_maps[0])
    if any(set(row) != tasks for row in (*baseline_maps, *candidate_maps)):
        raise ValueError("paired task surfaces differ")
    return {
        task: float(
            np.mean([row[task] for row in baseline_maps])
            - np.mean([row[task] for row in candidate_maps])
        )
        for task in sorted(tasks)
    }


def _bootstrap(values: Sequence[float], *, draws: int, seed: int) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    sampled = rng.choice(array, size=(draws, len(array)), replace=True).mean(axis=1)
    return {
        "mean": float(array.mean()),
        "ci95_low": float(np.quantile(sampled, 0.025)),
        "ci95_high": float(np.quantile(sampled, 0.975)),
    }


def _sign_test(values: Sequence[float]) -> dict[str, Any]:
    wins = sum(value > 0.0 for value in values)
    losses = sum(value < 0.0 for value in values)
    ties = len(values) - wins - losses
    n = wins + losses
    if n == 0:
        p_value = 1.0
    else:
        tail = sum(math.comb(n, index) for index in range(min(wins, losses) + 1))
        p_value = min(1.0, 2.0 * tail / (2**n))
    return {"wins": wins, "losses": losses, "ties": ties, "p_value": p_value}


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _markdown(summary: Mapping[str, Any]) -> str:
    confirmation = summary["confirmation"]
    comparison = summary["comparison"]
    lines = [
        "# Stage 3 Markov-sufficiency results",
        "",
        f"Decision: `{summary['decision']}`",
        "",
        "## Confirmation task-macro means across three seeds",
        "",
        "| Representation | Action NLL | Action accuracy | Evidence BCE | Evidence Brier |",
        "|---|---:|---:|---:|---:|",
    ]
    for variant in FROZEN_SUFFICIENCY_VARIANTS:
        arm = confirmation["arms"][variant]
        lines.append(
            f"| {variant} | {arm['action_nll']:.6f} | {arm['action_accuracy']:.6f} | "
            f"{arm['evidence_bce']:.6f} | {arm['evidence_brier']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Frozen comparisons",
            "",
            f"- Structured minus Semantic action-NLL gain: {comparison['structured_vs_semantic']['action_mean_gain']:.6f}",
            f"- Structured minus Semantic evidence-BCE gain: {comparison['structured_vs_semantic']['evidence_mean_gain']:.6f}",
            f"- Structured minus Full-History action-NLL gap: {comparison['structured_vs_full_history']['action_nll_gap']:.6f}",
            f"- Structured minus Full-History evidence-BCE gap: {comparison['structured_vs_full_history']['evidence_bce_gap']:.6f}",
            "",
            "## Gate checks",
            "",
        ]
    )
    for name, passed in summary["gate_checks"].items():
        lines.append(f"- `{name}`: {'PASS' if passed else 'FAIL'}")
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "This is a clean-only 48-task synthetic AgentDojo representation test. "
            "The confirmation identities were task-disjoint from training but had been "
            "examined in earlier July studies. A GO authorizes only the preregistered "
            "small paired sandbox pilot; a NO-GO forbids attack generation and large "
            "Dreamer/value training.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--run-metrics", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    run_metrics = json.loads(args.run_metrics.read_text(encoding="utf-8"))
    rows = _read_jsonl(args.predictions)
    seeds = [int(seed) for seed in protocol["training"]["training_seeds"]]
    if tuple(run_metrics["variants"]) != FROZEN_SUFFICIENCY_VARIANTS:
        raise ValueError("run variants differ from the frozen protocol")
    if run_metrics["training_seeds"] != seeds:
        raise ValueError("run seeds differ from the frozen protocol")
    if len(run_metrics["runs"]) != int(protocol["fixed_budget"]["training_runs"]):
        raise ValueError("training run budget is incomplete")
    expected_predictions = int(protocol["fixed_budget"]["training_runs"]) * (
        int(protocol["fixed_budget"]["prefixes"])
        + int(protocol["fixed_budget"]["evidence_transitions"])
    )
    if len(rows) != expected_predictions:
        raise ValueError(
            f"prediction row count mismatch: {len(rows)} != {expected_predictions}"
        )

    split_summaries: dict[str, Any] = {}
    for split in ("training", "calibration", "confirmation"):
        arms = {}
        for variant in FROZEN_SUFFICIENCY_VARIANTS:
            action_nll = [
                _mean_task_metric(
                    rows,
                    prediction_type="dynamics",
                    variant=variant,
                    seed=seed,
                    split=split,
                    metric="action_nll",
                )
                for seed in seeds
            ]
            action_accuracy = [
                _mean_task_metric(
                    rows,
                    prediction_type="dynamics",
                    variant=variant,
                    seed=seed,
                    split=split,
                    metric="action_correct",
                )
                for seed in seeds
            ]
            evidence_bce = [
                _mean_task_metric(
                    rows,
                    prediction_type="evidence",
                    variant=variant,
                    seed=seed,
                    split=split,
                    metric="evidence_bce",
                )
                for seed in seeds
            ]
            evidence_brier = [
                _mean_task_metric(
                    rows,
                    prediction_type="evidence",
                    variant=variant,
                    seed=seed,
                    split=split,
                    metric="evidence_brier",
                )
                for seed in seeds
            ]
            arms[variant] = {
                "action_nll": float(np.mean(action_nll)),
                "action_nll_by_seed": dict(zip(map(str, seeds), action_nll)),
                "action_accuracy": float(np.mean(action_accuracy)),
                "evidence_bce": float(np.mean(evidence_bce)),
                "evidence_bce_by_seed": dict(zip(map(str, seeds), evidence_bce)),
                "evidence_brier": float(np.mean(evidence_brier)),
            }
        split_summaries[split] = {"arms": arms}

    split = str(protocol["acceptance_gate"]["split"])
    confirmation = split_summaries[split]
    semantic = confirmation["arms"]["semantic_markov"]
    structured = confirmation["arms"]["structured_markov_v3"]
    full = confirmation["arms"]["full_history_diagnostic"]
    action_seed_gains = [
        semantic["action_nll_by_seed"][str(seed)]
        - structured["action_nll_by_seed"][str(seed)]
        for seed in seeds
    ]
    evidence_seed_gains = [
        semantic["evidence_bce_by_seed"][str(seed)]
        - structured["evidence_bce_by_seed"][str(seed)]
        for seed in seeds
    ]
    action_task_gains = _paired_task_gains(
        rows,
        prediction_type="dynamics",
        baseline="semantic_markov",
        candidate="structured_markov_v3",
        seeds=seeds,
        split=split,
        metric="action_nll",
    )
    evidence_task_gains = _paired_task_gains(
        rows,
        prediction_type="evidence",
        baseline="semantic_markov",
        candidate="structured_markov_v3",
        seeds=seeds,
        split=split,
        metric="evidence_bce",
    )
    action_full_gap = structured["action_nll"] - full["action_nll"]
    evidence_full_gap = structured["evidence_bce"] - full["evidence_bce"]
    gate_checks = evaluate_sufficiency_gate(
        action_seed_gains=action_seed_gains,
        evidence_seed_gains=evidence_seed_gains,
        action_task_gains=list(action_task_gains.values()),
        evidence_task_gains=list(evidence_task_gains.values()),
        structured_minus_full_action_nll=action_full_gap,
        structured_minus_full_evidence_bce=evidence_full_gap,
        gates=protocol["acceptance_gate"],
    )

    draws = int(protocol["uncertainty"]["paired_task_bootstrap_draws"])
    bootstrap_seed = int(protocol["uncertainty"]["bootstrap_seed"])
    target_diagnostics = {}
    for target_index, name in enumerate(EVIDENCE_DELTA_TARGETS):
        metric = f"bce_{name}"
        gains = _paired_task_gains(
            rows,
            prediction_type="evidence",
            baseline="semantic_markov",
            candidate="structured_markov_v3",
            seeds=seeds,
            split=split,
            metric=metric,
        )
        prevalence_rows = [
            row
            for row in rows
            if row["prediction_type"] == "evidence"
            and row["variant"] == "semantic_markov"
            and int(row["training_seed"]) == seeds[0]
            and row["split"] == split
        ]
        target_diagnostics[name] = {
            "positive_rows": int(sum(row[f"target_{name}"] for row in prevalence_rows)),
            "rows": len(prevalence_rows),
            "paired_task_gain": _bootstrap(
                list(gains.values()),
                draws=draws,
                seed=bootstrap_seed + 100 + target_index,
            ),
            "paired_sign_test": _sign_test(list(gains.values())),
        }

    decision = (
        "GO__STRUCTURED_MARKOV_V3_SUFFICIENT_FOR_PAIRED_PILOT"
        if all(gate_checks.values())
        else "NO_GO__STRUCTURED_MARKOV_V3_SUFFICIENCY_NOT_ESTABLISHED"
    )
    summary = {
        "protocol_sha256": _sha256(args.protocol),
        "run_metrics_sha256": _sha256(args.run_metrics),
        "predictions_sha256": _sha256(args.predictions),
        "prediction_rows": len(rows),
        "training_runs": len(run_metrics["runs"]),
        "training_seeds": seeds,
        **split_summaries,
        "comparison": {
            "structured_vs_semantic": {
                "action_mean_gain": float(np.mean(action_seed_gains)),
                "action_seed_gains": dict(zip(map(str, seeds), action_seed_gains)),
                "action_positive_tasks": sum(
                    value > 0.0 for value in action_task_gains.values()
                ),
                "action_task_gains": action_task_gains,
                "action_bootstrap": _bootstrap(
                    list(action_task_gains.values()),
                    draws=draws,
                    seed=bootstrap_seed,
                ),
                "action_sign_test": _sign_test(list(action_task_gains.values())),
                "evidence_mean_gain": float(np.mean(evidence_seed_gains)),
                "evidence_seed_gains": dict(zip(map(str, seeds), evidence_seed_gains)),
                "evidence_positive_tasks": sum(
                    value > 0.0 for value in evidence_task_gains.values()
                ),
                "evidence_task_gains": evidence_task_gains,
                "evidence_bootstrap": _bootstrap(
                    list(evidence_task_gains.values()),
                    draws=draws,
                    seed=bootstrap_seed + 1,
                ),
                "evidence_sign_test": _sign_test(
                    list(evidence_task_gains.values())
                ),
            },
            "structured_vs_full_history": {
                "action_nll_gap": action_full_gap,
                "evidence_bce_gap": evidence_full_gap,
            },
        },
        "evidence_target_diagnostics": target_diagnostics,
        "gate_checks": gate_checks,
        "decision": decision,
        "stage4_authorized": decision.startswith("GO__"),
        "claim_boundary": protocol["source"]["confirmation_caveat"],
    }
    _write_json(args.output, summary)
    args.report.write_text(_markdown(summary), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
