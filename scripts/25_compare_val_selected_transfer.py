"""Compare validation-selected Pareto configurations on a frozen test split.

The per-split Pareto sweep intentionally reports diagnostic best rows.  This
script adds the leakage-safe cross-split layer used for final comparisons:

1. average validation metrics over independently trained model seeds;
2. select one shared hyperparameter configuration on validation;
3. copy each model seed's numeric validation threshold to test;
4. report the mean, sample standard deviation, and per-seed test outcomes.

It also reports a same-quantile transfer, a validation-selected weighted-score
baseline, a random baseline, and candidate-level AUC diagnostics.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[1]


def _load_pareto_module():
    path = ROOT / "scripts" / "18_pareto_utility_selection.py"
    spec = importlib.util.spec_from_file_location("pareto_utility_selection", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import Pareto selector from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PARETO = _load_pareto_module()


def _parse_seeds(value: str) -> list[int]:
    seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not seeds:
        raise ValueError("At least one seed is required")
    return seeds


def _parse_key_priority(value: str) -> list[str]:
    keys = [item.strip() for item in value.split(",") if item.strip()]
    if len(keys) != len(set(keys)):
        raise ValueError("--utility-key-priority contains duplicates")
    return keys


def _config_key(row: dict[str, Any]) -> tuple[int, str, str, float]:
    return (
        int(row["top_k"]),
        str(row["utility_key"]),
        str(row["threshold_mode"]),
        float(row["threshold_value"]),
    )


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows]))


def _aggregate_configs(
    report_root: Path,
    seeds: list[int],
    split: str,
    method: str,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str, str, float], list[dict[str, Any]]] = defaultdict(list)
    for seed in seeds:
        payload = json.loads(
            (report_root / f"seed{seed}_{split}_pareto.json").read_text(
                encoding="utf-8"
            )
        )
        for row in payload["aggregate"]:
            if row["method"] == method:
                grouped[_config_key(row)].append(row)

    output = []
    for config, rows in grouped.items():
        if len(rows) != len(seeds):
            continue
        output.append(
            {
                "top_k": config[0],
                "utility_key": config[1],
                "threshold_mode": config[2],
                "threshold_value": config[3],
                "model_seed_count": len(rows),
                "observed_asr": _mean(rows, "observed_asr_mean"),
                "observed_bup": _mean(rows, "observed_bup_mean"),
                "objective_asr_plus_bup": _mean(
                    rows, "objective_asr_plus_bup_mean"
                ),
                "conditional_coverage": _mean(rows, "conditional_coverage_mean"),
                "conditional_observed_asr": _mean(
                    rows, "conditional_observed_asr_mean"
                ),
                "conditional_observed_bup": _mean(
                    rows, "conditional_observed_bup_mean"
                ),
                "conditional_asr_plus_bup": _mean(
                    rows, "conditional_asr_plus_bup_mean"
                ),
            }
        )
    if not output:
        raise RuntimeError(f"No common {method} configurations found for {split}")
    return output


def _select_validation_config(
    rows: list[dict[str, Any]],
    min_conditional_coverage: float,
    utility_key_priority: list[str] | None = None,
) -> dict[str, Any]:
    eligible = [
        row
        for row in rows
        if row["conditional_coverage"] >= min_conditional_coverage
    ]
    if not eligible:
        raise RuntimeError(
            "No validation configuration satisfies min conditional coverage "
            f"{min_conditional_coverage}"
        )
    priority = {
        key: len(utility_key_priority) - index
        for index, key in enumerate(utility_key_priority or [])
    }
    return max(
        eligible,
        key=lambda row: (
            row["objective_asr_plus_bup"],
            row["observed_bup"],
            row["observed_asr"],
            priority.get(str(row["utility_key"]), 0),
            -row["top_k"],
        ),
    )


def _matching_aggregate(
    report_root: Path,
    seed: int,
    split: str,
    method: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    payload = json.loads(
        (report_root / f"seed{seed}_{split}_pareto.json").read_text(
            encoding="utf-8"
        )
    )
    expected = (
        int(config["top_k"]),
        str(config["utility_key"]),
        str(config["threshold_mode"]),
        float(config["threshold_value"]),
    )
    return next(
        row
        for row in payload["aggregate"]
        if row["method"] == method and _config_key(row) == expected
    )


def _load_candidates(
    report_root: Path,
    seed: int,
    split: str,
    clean_rates: dict[tuple[str, str], float],
    min_base_success_rate: float,
) -> list[dict[str, Any]]:
    payload = json.loads(
        (report_root / f"seed{seed}_{split}_candidates.json").read_text(
            encoding="utf-8"
        )
    )
    return PARETO._annotate_clean_rates(
        payload["candidates"],
        clean_rates,
        min_base_success_rate=min_base_success_rate,
    )


SUMMARY_KEYS = (
    "observed_asr",
    "observed_bup",
    "objective_asr_plus_bup",
    "conditional_eval_count",
    "conditional_coverage",
    "conditional_observed_asr",
    "conditional_observed_bup",
    "conditional_asr_plus_bup",
    "feasible_rate",
)


def _summarize_seed_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    aggregate: dict[str, Any] = {"model_seed_count": len(rows), "per_seed": rows}
    for key in SUMMARY_KEYS:
        values = [float(row[key]) for row in rows]
        aggregate[key] = float(np.mean(values))
        aggregate[f"{key}_std"] = (
            float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        )
    return aggregate


def _transfer_pareto(
    report_root: Path,
    seeds: list[int],
    config: dict[str, Any],
    clean_rates: dict[tuple[str, str], float],
    min_base_success_rate: float,
    *,
    freeze_numeric_threshold: bool,
) -> dict[str, Any]:
    rows = []
    for seed in seeds:
        validation = _matching_aggregate(
            report_root,
            seed,
            "val",
            "pareto_utility_constraint",
            config,
        )
        candidates = _load_candidates(
            report_root,
            seed,
            "test",
            clean_rates,
            min_base_success_rate,
        )
        if freeze_numeric_threshold:
            threshold = float(validation["threshold"])
        else:
            threshold = PARETO._threshold(
                candidates,
                config["utility_key"],
                config["threshold_mode"],
                config["threshold_value"],
            )
        selected = PARETO._select_pareto(
            candidates,
            top_k=config["top_k"],
            utility_key=config["utility_key"],
            threshold=threshold,
            max_per_user_task=2,
        )
        rows.append(
            {
                "seed": seed,
                "threshold": threshold,
                **PARETO._summarize(selected),
            }
        )
    return _summarize_seed_rows(rows)


def _transfer_weighted(
    report_root: Path,
    seeds: list[int],
    top_k: int,
    clean_rates: dict[tuple[str, str], float],
    min_base_success_rate: float,
) -> dict[str, Any]:
    rows = []
    for seed in seeds:
        candidates = _load_candidates(
            report_root,
            seed,
            "test",
            clean_rates,
            min_base_success_rate,
        )
        selected = PARETO._select_weighted(
            candidates, top_k=top_k, max_per_user_task=2
        )
        rows.append({"seed": seed, **PARETO._summarize(selected)})
    return _summarize_seed_rows(rows)


def _safe_auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    return float(roc_auc_score(labels, scores)) if len(set(labels.tolist())) > 1 else None


def _candidate_diagnostics(
    report_root: Path,
    seeds: list[int],
    clean_rates: dict[tuple[str, str], float],
    min_base_success_rate: float,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for split in ("val", "test"):
        per_seed = []
        for seed in seeds:
            candidates = _load_candidates(
                report_root,
                seed,
                split,
                clean_rates,
                min_base_success_rate,
            )
            utility = np.asarray(
                [float(bool(row["observed_utility"])) for row in candidates]
            )
            security = np.asarray(
                [float(bool(row["observed_security"])) for row in candidates]
            )
            eligible = np.asarray(
                [bool(row.get("preservation_eval_eligible", True)) for row in candidates]
            )
            row: dict[str, Any] = {
                "seed": seed,
                "candidate_count": len(candidates),
                "observed_asr_rate": float(security.mean()),
                "observed_bup_rate": float(utility.mean()),
                "conditional_eligible_rate": float(eligible.mean()),
            }
            score_keys = (
                "risk_score",
                "utility_score",
                "preservation_score",
                "expected_attacked_utility_score",
                "final_utility_score",
                "value_score",
                "selection_score",
                "candidate_risk_score",
                "candidate_utility_score",
                "candidate_preservation_score",
                "candidate_expected_utility_score",
                "candidate_joint_score",
            )
            for key in score_keys:
                if not all(key in candidate for candidate in candidates):
                    continue
                scores = np.asarray([float(candidate[key]) for candidate in candidates])
                row[f"{key}_asr_auc"] = _safe_auc(security, scores)
                row[f"{key}_bup_auc"] = _safe_auc(utility, scores)
                row[f"{key}_conditional_bup_auc"] = (
                    _safe_auc(utility[eligible], scores[eligible])
                    if eligible.any()
                    else None
                )
            per_seed.append(row)

        numeric_keys = sorted(
            {
                key
                for row in per_seed
                for key, value in row.items()
                if key != "seed" and isinstance(value, (int, float)) and value is not None
            }
        )
        means = {}
        for key in numeric_keys:
            values = [float(row[key]) for row in per_seed if row.get(key) is not None]
            if values and all(math.isfinite(value) for value in values):
                means[key] = float(np.mean(values))
                means[f"{key}_std"] = (
                    float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
                )
        output[split] = {"mean": means, "per_seed": per_seed}
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-root", type=Path, required=True)
    parser.add_argument("--seeds", default="7,13,21")
    parser.add_argument("--clean-solvability-json", type=Path, required=True)
    parser.add_argument("--min-base-success-rate", type=float, default=0.5)
    parser.add_argument("--min-conditional-coverage", type=float, default=0.0)
    parser.add_argument(
        "--utility-key-priority",
        default="",
        help=(
            "Validation-only tie-break priority, highest first. It is used "
            "only after ASR+BUP, BUP, and ASR are exactly tied."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    seeds = _parse_seeds(args.seeds)
    utility_key_priority = _parse_key_priority(args.utility_key_priority)
    clean_rates = PARETO._load_clean_rates(args.clean_solvability_json)

    validation_pareto = _aggregate_configs(
        args.report_root, seeds, "val", "pareto_utility_constraint"
    )
    selected_pareto = _select_validation_config(
        validation_pareto,
        args.min_conditional_coverage,
        utility_key_priority,
    )
    validation_best_by_utility_key = {}
    for utility_key in sorted({row["utility_key"] for row in validation_pareto}):
        key_rows = [
            row for row in validation_pareto if row["utility_key"] == utility_key
        ]
        eligible_key_rows = [
            row
            for row in key_rows
            if row["conditional_coverage"] >= args.min_conditional_coverage
        ]
        if not eligible_key_rows:
            validation_best_by_utility_key[utility_key] = {
                "status": "no_configuration_meets_min_conditional_coverage",
                "max_conditional_coverage": max(
                    row["conditional_coverage"] for row in key_rows
                ),
            }
            continue
        selected = _select_validation_config(
            eligible_key_rows, args.min_conditional_coverage
        )
        validation_best_by_utility_key[utility_key] = {
            "status": "selected",
            "selected_validation_config": selected,
            "test_frozen_validation_threshold": _transfer_pareto(
                args.report_root,
                seeds,
                selected,
                clean_rates,
                args.min_base_success_rate,
                freeze_numeric_threshold=True,
            ),
        }
    validation_weighted = _aggregate_configs(
        args.report_root, seeds, "val", "weighted_baseline"
    )
    selected_weighted = _select_validation_config(
        validation_weighted, args.min_conditional_coverage
    )
    test_random = _aggregate_configs(
        args.report_root, seeds, "test", "random"
    )
    selected_k_random = next(
        row for row in test_random if row["top_k"] == selected_pareto["top_k"]
    )

    output = {
        "scope": "validation_selected_frozen_test_transfer",
        "report_root": str(args.report_root.resolve()),
        "seeds": seeds,
        "clean_solvability_json": str(args.clean_solvability_json.resolve()),
        "min_conditional_coverage": args.min_conditional_coverage,
        "utility_key_priority": utility_key_priority,
        "selection_rule": (
            "maximize validation mean ASR+BUP, then BUP, then ASR, then the "
            "optional predeclared utility-key priority; copy each model seed's "
            "numeric validation threshold to test after enforcing the minimum "
            "conditional coverage"
        ),
        "pareto": {
            "selected_validation_config": selected_pareto,
            "validation_best_by_utility_key": validation_best_by_utility_key,
            "test_same_quantile": _transfer_pareto(
                args.report_root,
                seeds,
                selected_pareto,
                clean_rates,
                args.min_base_success_rate,
                freeze_numeric_threshold=False,
            ),
            "test_frozen_validation_threshold": _transfer_pareto(
                args.report_root,
                seeds,
                selected_pareto,
                clean_rates,
                args.min_base_success_rate,
                freeze_numeric_threshold=True,
            ),
        },
        "weighted_baseline": {
            "selected_validation_config": selected_weighted,
            "test": _transfer_weighted(
                args.report_root,
                seeds,
                int(selected_weighted["top_k"]),
                clean_rates,
                args.min_base_success_rate,
            ),
        },
        "random_baseline_at_selected_k": selected_k_random,
        "candidate_diagnostics": _candidate_diagnostics(
            args.report_root,
            seeds,
            clean_rates,
            args.min_base_success_rate,
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
