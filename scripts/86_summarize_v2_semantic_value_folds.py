"""Aggregate frozen semantic-value OOF fold results without retuning."""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PROBE_PATH = ROOT / "scripts" / "85_probe_v2_semantic_configuration_value.py"
SPEC = importlib.util.spec_from_file_location("semantic_probe", PROBE_PATH)
PROBE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(PROBE)


def _exact_sign_flip(differences: list[float]) -> dict[str, Any]:
    """Exact one-sided paired randomization test for a positive mean delta."""
    values = np.asarray(differences, dtype=np.float64)
    nonzero = values[np.abs(values) > 1e-12]
    if len(nonzero) > 24:
        raise ValueError("Exact sign-flip test is limited to 24 non-zero pairs")
    observed = float(values.mean())
    exceedances = 0
    total = 2 ** len(nonzero)
    for signs in itertools.product((-1.0, 1.0), repeat=len(nonzero)):
        randomized = float(np.dot(nonzero, np.asarray(signs)) / len(values))
        exceedances += randomized + 1e-12 >= observed
    return {
        "observed_mean_delta": observed,
        "nonzero_pair_count": int(len(nonzero)),
        "randomization_count": int(total),
        "one_sided_p_delta_at_least_observed": float(exceedances / total),
    }


def _paired_comparison(
    semantic: dict[str, Any],
    baseline_summary: dict[str, Any],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    baseline = baseline_summary["prospective_oof"]["pct100"]["1"]["by_task"]
    semantic_tasks = semantic["per_task"]
    if set(semantic_tasks) != set(baseline):
        missing_semantic = sorted(set(baseline) - set(semantic_tasks))
        missing_baseline = sorted(set(semantic_tasks) - set(baseline))
        raise ValueError(
            "Semantic and baseline tasks differ: "
            f"missing_semantic={missing_semantic}, missing_baseline={missing_baseline}"
        )
    metric_fields = {
        "ASR": ("selected_ASR", "ASR"),
        "BUP": ("selected_BUP", "BUP"),
        "ASR_plus_BUP": ("selected_observed", "ASR_plus_BUP"),
    }
    tasks = sorted(semantic_tasks)
    deltas = np.zeros((len(tasks), len(metric_fields)), dtype=np.float64)
    paired_rows = []
    for task_index, task in enumerate(tasks):
        row = {"task_key": task, "domain": task.split("|", 1)[0]}
        for metric_index, (metric, (semantic_field, baseline_field)) in enumerate(
            metric_fields.items()
        ):
            semantic_value = float(semantic_tasks[task][semantic_field])
            baseline_value = float(baseline[task][baseline_field])
            delta = semantic_value - baseline_value
            deltas[task_index, metric_index] = delta
            row[metric] = {
                "semantic": semantic_value,
                "baseline": baseline_value,
                "delta": delta,
            }
        paired_rows.append(row)

    rng = np.random.default_rng(bootstrap_seed)
    bootstrap = np.empty((bootstrap_samples, len(metric_fields)), dtype=np.float64)
    chunk_size = 10_000
    for start in range(0, bootstrap_samples, chunk_size):
        stop = min(start + chunk_size, bootstrap_samples)
        indices = rng.integers(0, len(tasks), size=(stop - start, len(tasks)))
        bootstrap[start:stop] = deltas[indices].mean(axis=1)

    metrics = {}
    for metric_index, metric in enumerate(metric_fields):
        metric_delta = deltas[:, metric_index]
        distribution = bootstrap[:, metric_index]
        metrics[metric] = {
            "semantic_mean": float(
                np.mean([row[metric]["semantic"] for row in paired_rows])
            ),
            "baseline_mean": float(
                np.mean([row[metric]["baseline"] for row in paired_rows])
            ),
            "mean_delta": float(metric_delta.mean()),
            "paired_bootstrap_ci95_low": float(np.quantile(distribution, 0.025)),
            "paired_bootstrap_ci95_high": float(np.quantile(distribution, 0.975)),
            "paired_bootstrap_probability_delta_positive": float(
                np.mean(distribution > 0.0)
            ),
            "exact_sign_flip": _exact_sign_flip(metric_delta.tolist()),
        }

    per_domain = {}
    for domain in sorted({row["domain"] for row in paired_rows}):
        domain_rows = [row for row in paired_rows if row["domain"] == domain]
        per_domain[domain] = {
            metric: {
                "semantic_mean": float(
                    np.mean([row[metric]["semantic"] for row in domain_rows])
                ),
                "baseline_mean": float(
                    np.mean([row[metric]["baseline"] for row in domain_rows])
                ),
                "mean_delta": float(
                    np.mean([row[metric]["delta"] for row in domain_rows])
                ),
            }
            for metric in metric_fields
        }

    engineering_checks = {
        "joint_delta_at_least_0.05": metrics["ASR_plus_BUP"]["mean_delta"]
        >= 0.05 - 1e-12,
        "BUP_delta_nonnegative": metrics["BUP"]["mean_delta"] >= -1e-12,
        "no_domain_joint_delta_below_minus_0.10": min(
            row["ASR_plus_BUP"]["mean_delta"] for row in per_domain.values()
        )
        >= -0.10 - 1e-12,
        "all_top1_selections_unique": semantic["unique_top1_rate"] >= 1.0 - 1e-12,
    }
    confirmatory_checks = {
        "joint_bootstrap_ci95_excludes_zero": metrics["ASR_plus_BUP"][
            "paired_bootstrap_ci95_low"
        ]
        > 0.0,
        "joint_exact_sign_flip_p_at_most_0.05": metrics["ASR_plus_BUP"][
            "exact_sign_flip"
        ]["one_sided_p_delta_at_least_observed"]
        <= 0.05,
    }
    engineering_pass = all(engineering_checks.values())
    confirmatory_pass = all(confirmatory_checks.values())
    if engineering_pass and confirmatory_pass:
        decision = "GO"
    elif engineering_pass:
        decision = "PILOT_ONLY_UNCONFIRMED"
    else:
        decision = "NO_GO"
    return {
        "baseline": "formal grouped Dreamer OOF pct100 Top-1",
        "task_count": len(tasks),
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": bootstrap_seed,
        "metrics": metrics,
        "per_domain": per_domain,
        "per_task": paired_rows,
        "integration_gate": {
            "engineering_checks": engineering_checks,
            "confirmatory_checks": confirmatory_checks,
            "engineering_pass": engineering_pass,
            "confirmatory_pass": confirmatory_pass,
            "decision": decision,
        },
    }


def summarize(
    paths: list[Path],
    *,
    baseline_summary: Path | None = None,
    bootstrap_samples: int = 200_000,
    bootstrap_seed: int = 20260715,
) -> dict[str, Any]:
    fold_results = []
    candidate_rows = []
    task_to_fold: dict[str, int] = {}
    frozen_candidates = set()
    for fold, path in enumerate(paths):
        payload = json.loads(path.read_text(encoding="utf-8"))
        frozen = payload["frozen_method"]
        frozen_candidates.add(json.dumps(frozen["frozen_candidate"], sort_keys=True))
        fold_results.append(
            {
                "fold": fold,
                "path": str(path.resolve()),
                "test": frozen["test"],
            }
        )
        for row in frozen["test_candidate_scores"]:
            task = str(row["task_key"])
            previous = task_to_fold.setdefault(task, fold)
            if previous != fold:
                raise ValueError(f"Held-out task appears in multiple folds: {task}")
            candidate_rows.append(row)
    if len(frozen_candidates) != 1:
        raise ValueError("Fold results used different frozen methods")
    expected_tasks = sum(item["test"]["task_count"] for item in fold_results)
    if len(task_to_fold) != expected_tasks:
        raise ValueError("Fold task counts do not form a disjoint OOF partition")
    rank_scores = np.asarray([float(row["rank_score"]) for row in candidate_rows])
    predictions = np.asarray([float(row["prediction"]) for row in candidate_rows])
    aggregate = PROBE._evaluate(
        candidate_rows, rank_scores=rank_scores, predictions=predictions
    )
    result = {
        "scope": "five-fold frozen E5 plus structured configuration-value OOF",
        "protocol": {
            "fold_count": len(paths),
            "method_frozen_before_fold_replication": True,
            "test_retuning": False,
            "frozen_candidate": json.loads(next(iter(frozen_candidates))),
        },
        "counts": {
            "tasks": len(task_to_fold),
            "configurations": len(candidate_rows),
        },
        "per_fold": fold_results,
        "oof_aggregate": aggregate,
    }
    if baseline_summary is not None:
        baseline_payload = json.loads(baseline_summary.read_text(encoding="utf-8"))
        result["paired_baseline_comparison"] = _paired_comparison(
            aggregate,
            baseline_payload,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed,
        )
        result["baseline_summary_path"] = str(baseline_summary.resolve())
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold-results", type=Path, nargs="+", required=True)
    parser.add_argument("--baseline-summary", type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=200_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260715)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(
        args.fold_results,
        baseline_summary=args.baseline_summary,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
