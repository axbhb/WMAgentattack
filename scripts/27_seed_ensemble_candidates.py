"""Average independently trained model-seed candidate scores.

The candidate identities and observed labels must match exactly across seeds.
Only an explicit allow-list of model scores is averaged; identifiers, labels,
and trajectory structures are preserved from the first seed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


AVERAGE_KEYS = (
    "risk_score",
    "rollout_mean_risk_score",
    "utility_score",
    "selection_utility_score",
    "preservation_score",
    "min_utility_score",
    "final_utility_score",
    "value_score",
    "reward_score",
    "target_skill_probability",
    "rollout_mean_target_skill_probability",
    "rollout_target_reached",
    "selection_score",
    "base_selection_score",
    "candidate_risk_score",
    "candidate_utility_score",
    "candidate_preservation_score",
    "candidate_expected_utility_score",
    "candidate_marginal_sum_score",
    "candidate_joint_score",
    "candidate_conservative_joint_score",
    "candidate_objective_score",
)
LABEL_KEYS = ("observed_security", "observed_utility")


def _parse_seeds(value: str) -> list[int]:
    seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not seeds:
        raise ValueError("At least one seed is required")
    return seeds


def _pair_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row["suite"]),
        str(row["user_task_id"]),
        str(row["injection_task_id"]),
    )


def _average_candidates(
    candidate_sets: list[list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    if not candidate_sets:
        raise ValueError("No candidate sets provided")
    primary = candidate_sets[0]
    primary_keys = [_pair_key(row) for row in primary]
    if len(primary_keys) != len(set(primary_keys)):
        raise ValueError("Duplicate pair in primary candidate set")

    mappings = []
    for seed_index, rows in enumerate(candidate_sets):
        mapping = {_pair_key(row): row for row in rows}
        if len(mapping) != len(rows):
            raise ValueError(f"Duplicate pair in candidate set {seed_index}")
        if set(mapping) != set(primary_keys):
            raise ValueError(f"Candidate identity mismatch in set {seed_index}")
        mappings.append(mapping)

    output = []
    for key, reference in zip(primary_keys, primary, strict=True):
        rows = [mapping[key] for mapping in mappings]
        if any(
            bool(row[label]) != bool(reference[label])
            for row in rows
            for label in LABEL_KEYS
        ):
            raise ValueError(f"Observed label mismatch at {key}")
        averaged = dict(reference)
        for score_key in AVERAGE_KEYS:
            if all(
                score_key in row
                and isinstance(row[score_key], (int, float))
                and not isinstance(row[score_key], bool)
                for row in rows
            ):
                averaged[score_key] = float(
                    np.mean([float(row[score_key]) for row in rows])
                )
        averaged["candidate_ranker_fold"] = -2
        output.append(averaged)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--seeds", default="7,13,21")
    parser.add_argument("--output-seed", type=int, default=7)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    seeds = _parse_seeds(args.seeds)
    args.output_root.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "scope": "candidate_model_seed_ensemble",
        "input_root": str(args.input_root.resolve()),
        "model_seeds": seeds,
        "output_seed": args.output_seed,
        "averaged_keys": list(AVERAGE_KEYS),
        "splits": {},
    }
    for split in ("val", "test"):
        payloads = [
            json.loads(
                (args.input_root / f"seed{seed}_{split}_candidates.json").read_text(
                    encoding="utf-8"
                )
            )
            for seed in seeds
        ]
        candidates = _average_candidates(
            [payload["candidates"] for payload in payloads]
        )
        output = {
            **payloads[0],
            "candidate_seed_ensemble": {
                "model_seeds": seeds,
                "score_aggregation": "arithmetic_mean",
                "labels_and_identifiers": "preserved_after_exact_match_validation",
            },
            "candidates": candidates,
        }
        path = args.output_root / f"seed{args.output_seed}_{split}_candidates.json"
        path.write_text(json.dumps(output, indent=2), encoding="utf-8")
        summary["splits"][split] = {
            "candidate_count": len(candidates),
            "output": str(path.resolve()),
        }

    args.output_root.joinpath("seed_ensemble_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
