"""Post-hoc cross-check of grouped-train hybrid heads on 0713 fresh outcomes.

The 0713 outcomes were already inspected, so this script cannot produce a
confirmation claim.  It tests whether the validation diagnosis (world features
help attack ordering while text helps utility ordering) is directionally
consistent under five-repeat AgentDojo outcomes.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]


def _load_base_module():
    path = ROOT / "scripts" / "49_evaluate_grouped_train_hybrid.py"
    spec = importlib.util.spec_from_file_location("grouped_hybrid_base", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = _load_base_module()


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row["suite"]),
        str(row["user_task_id"]),
        str(row["injection_task_id"]),
    )


def _method_from_rows(
    rows: list[dict[str, Any]], method_name: str
) -> dict[str, np.ndarray]:
    fields = (
        "attack_rank",
        "utility_rank",
        "attack_probability",
        "utility_probability",
    )
    return {
        field: np.asarray(
            [row["all_methods"][method_name][field] for row in rows], dtype=float
        )
        for field in fields
    }


def _headwise(
    attack_source: dict[str, np.ndarray],
    utility_source: dict[str, np.ndarray],
    probability_source: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    return BASE._method(
        attack_source["attack_rank"],
        utility_source["utility_rank"],
        probability_source["attack_probability"],
        probability_source["utility_probability"],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-predictions", type=Path, required=True)
    parser.add_argument("--fresh-predictions", type=Path, required=True)
    parser.add_argument("--continuous-test-steps", type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=20000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260718)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    candidate_payload = _load(args.candidate_predictions)
    candidate_rows = candidate_payload.get("pairs")
    fresh_rows = _load(args.fresh_predictions).get("pairs")
    if not isinstance(candidate_rows, list) or not isinstance(fresh_rows, list):
        raise ValueError("Candidate or fresh pair list is missing")
    candidate_mapping = {_key(row): row for row in candidate_rows}
    if len(candidate_mapping) != len(candidate_rows):
        raise ValueError("Duplicate candidate prediction key")
    aligned = []
    for fresh in fresh_rows:
        candidate = candidate_mapping.get(_key(fresh))
        if candidate is None:
            raise ValueError(f"Fresh pair missing from candidate scores: {_key(fresh)}")
        aligned.append({**candidate, "fresh": fresh})

    rows = [
        {
            "suite": row["suite"],
            "user_task_id": row["user_task_id"],
            "injection_task_id": row["injection_task_id"],
        }
        for row in aligned
    ]
    attack_rates = np.asarray(
        [row["fresh"]["observed_attack_probability"] for row in aligned],
        dtype=float,
    )
    utility_rates = np.asarray(
        [row["fresh"]["observed_utility_probability"] for row in aligned],
        dtype=float,
    )
    method_names = sorted(aligned[0]["all_methods"])
    methods = {
        name: _method_from_rows(aligned, name) for name in method_names
    }
    world_name = "world_pairwise_c0p03"
    text_name = "text_pointwise"
    if world_name not in methods or text_name not in methods:
        raise ValueError("Required frozen diagnostic heads are missing")
    diagnostic_name = "world_attack_text_utility_text_probability"
    methods[diagnostic_name] = _headwise(
        methods[world_name], methods[text_name], methods[text_name]
    )
    combined_probability_name = "world_attack_text_utility_combined_probability"
    methods[combined_probability_name] = _headwise(
        methods[world_name], methods[text_name], methods[world_name]
    )
    world_rank = BASE._within_task_rank(
        rows, methods[world_name]["attack_rank"]
    )
    text_rank = BASE._within_task_rank(rows, methods[text_name]["attack_rank"])
    for alpha in (0.25, 0.5, 0.75):
        name = f"world_text_attack_rank_alpha_{str(alpha).replace('.', 'p')}"
        methods[name] = BASE._method(
            alpha * world_rank + (1.0 - alpha) * text_rank,
            methods[text_name]["utility_rank"],
            methods[text_name]["attack_probability"],
            methods[text_name]["utility_probability"],
        )

    results = {
        name: BASE._method_metrics(
            rows, attack_rates, utility_rates, method
        )
        for name, method in methods.items()
    }
    comparisons = {}
    comparison_methods = (
        diagnostic_name,
        world_name,
        "world_text_attack_rank_alpha_0p5",
        "world_text_attack_rank_alpha_0p75",
    )
    for method_index, method_name in enumerate(comparison_methods):
        for reference_index, reference in enumerate((text_name, "clean_raw")):
            comparisons[f"{method_name}__minus__{reference}"] = (
                BASE._bootstrap_difference(
                    rows,
                    methods[method_name],
                    methods[reference],
                    attack_rates,
                    utility_rates,
                    samples=args.bootstrap_samples,
                    seed=args.bootstrap_seed + 10 * method_index + reference_index,
                )
            )

    # Search only for a diagnostic ceiling.  These combinations are explicitly
    # post-hoc and are never exported as a frozen selection protocol.
    combinations = []
    for attack_name in method_names:
        for utility_name in method_names:
            candidate = _headwise(
                methods[attack_name], methods[utility_name], methods[text_name]
            )
            metrics = BASE._method_metrics(
                rows, attack_rates, utility_rates, candidate
            )
            combinations.append(
                {
                    "attack_head": attack_name,
                    "utility_head": utility_name,
                    "primary_mean_within_task_pairwise_accuracy": metrics[
                        "primary_mean_within_task_pairwise_accuracy"
                    ],
                    "attack_pairwise_accuracy": metrics["attack"]["within_task"][
                        "pairwise_accuracy"
                    ],
                    "utility_pairwise_accuracy": metrics["utility"]["within_task"][
                        "pairwise_accuracy"
                    ],
                }
            )
    combinations.sort(
        key=lambda row: (
            -row["primary_mean_within_task_pairwise_accuracy"],
            row["attack_head"],
            row["utility_head"],
        )
    )
    continuous_utility_diagnostic = None
    if args.continuous_test_steps is not None:
        soft_mapping = BASE._soft_targets(args.continuous_test_steps)
        missing = [
            row["trajectory_id"]
            for row in aligned
            if row["trajectory_id"] not in soft_mapping
        ]
        if missing:
            raise ValueError(
                f"Continuous utility targets missing for {len(missing)} fresh pairs"
            )
        soft_utility = np.asarray(
            [soft_mapping[row["trajectory_id"]][0] for row in aligned], dtype=float
        )
        correlation = spearmanr(soft_utility, utility_rates).statistic
        soft_pairwise = BASE.METRICS._within_task_metrics(
            rows, utility_rates, soft_utility
        )
        continuous_utility_diagnostic = {
            "soft_target_mean": float(soft_utility.mean()),
            "fresh_utility_mean": float(utility_rates.mean()),
            "soft_target_brier_against_fresh_rate": float(
                np.mean((soft_utility - utility_rates) ** 2)
            ),
            "soft_target_fresh_spearman": (
                float(correlation) if np.isfinite(correlation) else None
            ),
            "soft_target_fresh_within_task_pairwise_accuracy": soft_pairwise[
                "pairwise_accuracy"
            ],
            "informative_task_count": sum(
                value["pairwise_accuracy"] is not None
                for value in soft_pairwise["per_task"].values()
            ),
            "interpretation": (
                "This measures target drift only; the already inspected fresh "
                "outcomes cannot tune a replacement label."
            ),
        }
    output = {
        "scope": "posthoc_0713_fresh_headwise_crosscheck",
        "fresh_confirmation_claim": False,
        "reason": "0713 five-repeat outcomes had already been inspected",
        "pair_count": len(rows),
        "task_count": len({BASE._task_key(row) for row in rows}),
        "fixed_diagnostic_method": diagnostic_name,
        "results": results,
        "comparisons": comparisons,
        "continuous_utility_label_diagnostic": continuous_utility_diagnostic,
        "posthoc_oracle_top_10": combinations[:10],
        "interpretation_rule": (
            "Only directional agreement across grouped validation and fresh outcomes "
            "may motivate a new frozen replay; oracle combinations are diagnostic only."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
