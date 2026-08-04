"""Freeze a label-blind grouped-task fresh-replay confirmation set.

The selector aligns clean-prefix and injection-conditioned Dreamer scores from
three training seeds.  It deliberately never copies or reads outcome labels
when ranking tasks or injections.  Two stress-test tasks per suite and four
injections per task are selected before any fresh replay is collected.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


SUITES = ("banking", "slack", "travel", "workspace")
SEEDS = (7, 13, 21)
MODES = ("clean", "injection")
MODEL_NAMES = (
    "clean_view",
    "injection_view",
    "dual_view",
    "symmetric_shrinkage_alpha_0_5",
)
REMOVED_LABEL_KEYS = (
    "observed_security",
    "observed_utility",
    "security",
    "utility",
    "task_success",
    "attack_success",
)
REPLAY_IDENTITY_KEYS = (
    "suite",
    "user_task_id",
    "injection_task_id",
    "attack",
    "trajectory_id",
    "target_skill",
    "source_trace",
)
EPSILON = 1e-5


def _pair_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row["suite"]),
        str(row["user_task_id"]),
        str(row["injection_task_id"]),
    )


def _task_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row["suite"]), str(row["user_task_id"])


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_sources(values: list[str]) -> dict[str, Path]:
    sources: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected NAME=PATH, got {value!r}")
        name, raw_path = value.split("=", 1)
        if name in sources or not name:
            raise ValueError(f"Duplicate or empty source name: {name!r}")
        sources[name] = Path(raw_path).expanduser()
    expected = {f"{mode}_seed{seed}" for mode in MODES for seed in SEEDS}
    if set(sources) != expected:
        raise ValueError(
            f"Expected sources {sorted(expected)}, got {sorted(sources)}"
        )
    return sources


def _load_candidates(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("candidates")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Candidate list missing or empty in {path}")
    keys = [_pair_key(row) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError(f"Duplicate candidate pair in {path}")
    return rows


def _trajectory_tasks(path: Path) -> set[tuple[str, str]]:
    tasks = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            tasks.add((str(row["domain"]), str(row["task_id"])))
    if not tasks:
        raise ValueError(f"No trajectory tasks in {path}")
    return tasks


def _align_sources(
    sources: dict[str, Path],
) -> tuple[
    list[dict[str, Any]],
    dict[str, dict[tuple[str, str, str], dict[str, Any]]],
]:
    rows_by_source = {
        name: _load_candidates(path) for name, path in sources.items()
    }
    mappings = {
        name: {_pair_key(row): row for row in rows}
        for name, rows in rows_by_source.items()
    }
    reference = rows_by_source["clean_seed7"]
    reference_keys = {_pair_key(row) for row in reference}
    for name, mapping in mappings.items():
        if set(mapping) != reference_keys:
            raise ValueError(
                f"Candidate identity mismatch for {name}: "
                f"missing={len(reference_keys - set(mapping))}, "
                f"extra={len(set(mapping) - reference_keys)}"
            )
    return reference, mappings


def _probability(value: Any, *, field: str, key: tuple[str, str, str]) -> float:
    if not isinstance(value, (bool, int, float, np.number)):
        raise ValueError(f"Non-numeric {field} at {key}: {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"Non-finite {field} at {key}: {number}")
    return float(np.clip(number, EPSILON, 1.0 - EPSILON))


def _view_values(
    mappings: dict[str, dict[tuple[str, str, str], dict[str, Any]]],
    key: tuple[str, str, str],
    mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    attack = np.asarray(
        [
            _probability(
                mappings[f"{mode}_seed{seed}"][key].get("risk_score"),
                field=f"{mode}_seed{seed}.risk_score",
                key=key,
            )
            for seed in SEEDS
        ],
        dtype=float,
    )
    utility = np.asarray(
        [
            _probability(
                mappings[f"{mode}_seed{seed}"][key].get("utility_score"),
                field=f"{mode}_seed{seed}.utility_score",
                key=key,
            )
            for seed in SEEDS
        ],
        dtype=float,
    )
    return attack, utility


def _models(attack_clean: float, utility_clean: float,
            attack_injection: float, utility_injection: float) -> dict[str, dict[str, float]]:
    return {
        "clean_view": {
            "attack_probability": attack_clean,
            "utility_probability": utility_clean,
        },
        "injection_view": {
            "attack_probability": attack_injection,
            "utility_probability": utility_injection,
        },
        "dual_view": {
            "attack_probability": attack_injection,
            "utility_probability": utility_clean,
        },
        "symmetric_shrinkage_alpha_0_5": {
            "attack_probability": 0.5 * (attack_clean + attack_injection),
            "utility_probability": 0.5 * (utility_clean + utility_injection),
        },
    }


def _annotate_candidates(
    reference: list[dict[str, Any]],
    mappings: dict[str, dict[tuple[str, str, str], dict[str, Any]]],
) -> list[dict[str, Any]]:
    output = []
    for source_row in reference:
        key = _pair_key(source_row)
        clean_attack, clean_utility = _view_values(
            mappings, key, "clean"
        )
        injection_attack, injection_utility = _view_values(
            mappings, key, "injection"
        )
        identity = {
            field: source_row[field]
            for field in REPLAY_IDENTITY_KEYS
            if field in source_row
        }
        for field in ("suite", "user_task_id", "injection_task_id", "source_trace"):
            if field not in identity:
                raise ValueError(f"Missing replay identity {field} at {key}")
        ensemble = _models(
            float(clean_attack.mean()),
            float(clean_utility.mean()),
            float(injection_attack.mean()),
            float(injection_utility.mean()),
        )
        per_seed = {
            str(seed): _models(
                float(clean_attack[index]),
                float(clean_utility[index]),
                float(injection_attack[index]),
                float(injection_utility[index]),
            )
            for index, seed in enumerate(SEEDS)
        }
        uncertainty = {
            "clean_attack_seed_std": float(clean_attack.std()),
            "clean_utility_seed_std": float(clean_utility.std()),
            "injection_attack_seed_std": float(injection_attack.std()),
            "injection_utility_seed_std": float(injection_utility.std()),
        }
        output.append(
            {
                **identity,
                "confirmation_predictions": ensemble,
                "confirmation_seed_predictions": per_seed,
                "confirmation_prediction_uncertainty": uncertainty,
                "confirmation_view_gap": {
                    "attack": float(
                        ensemble["injection_view"]["attack_probability"]
                        - ensemble["clean_view"]["attack_probability"]
                    ),
                    "utility": float(
                        ensemble["injection_view"]["utility_probability"]
                        - ensemble["clean_view"]["utility_probability"]
                    ),
                },
            }
        )
    return output


def _task_statistics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    injection_attack = np.asarray(
        [
            row["confirmation_predictions"]["injection_view"]
            ["attack_probability"]
            for row in rows
        ],
        dtype=float,
    )
    injection_utility = np.asarray(
        [
            row["confirmation_predictions"]["injection_view"]
            ["utility_probability"]
            for row in rows
        ],
        dtype=float,
    )
    view_gap = np.asarray(
        [
            abs(float(row["confirmation_view_gap"]["attack"]))
            + abs(float(row["confirmation_view_gap"]["utility"]))
            for row in rows
        ],
        dtype=float,
    )
    seed_disagreement = np.asarray(
        [
            sum(row["confirmation_prediction_uncertainty"].values())
            for row in rows
        ],
        dtype=float,
    )
    attack_span = float(np.ptp(injection_attack))
    utility_span = float(np.ptp(injection_utility))
    return {
        "candidate_count": len(rows),
        "injection_attack_span": attack_span,
        "injection_utility_span": utility_span,
        "joint_score_span": attack_span + utility_span,
        "mean_absolute_view_gap": float(view_gap.mean()),
        "mean_seed_disagreement": float(seed_disagreement.mean()),
        "disagreement_score": float(
            view_gap.mean() + seed_disagreement.mean()
        ),
    }


def _choose_tasks(
    grouped: dict[tuple[str, str], list[dict[str, Any]]]
) -> dict[tuple[str, str], tuple[str, dict[str, Any]]]:
    chosen: dict[tuple[str, str], tuple[str, dict[str, Any]]] = {}
    for suite in SUITES:
        eligible = [
            (task, _task_statistics(rows))
            for task, rows in grouped.items()
            if task[0] == suite and len(rows) >= 4
        ]
        if len(eligible) < 2:
            raise RuntimeError(
                f"Suite {suite} has only {len(eligible)} tasks with >=4 pairs"
            )
        high_span = max(
            eligible,
            key=lambda item: (
                item[1]["joint_score_span"],
                item[1]["disagreement_score"],
                item[0],
            ),
        )
        high_disagreement = max(
            (item for item in eligible if item[0] != high_span[0]),
            key=lambda item: (
                item[1]["disagreement_score"],
                item[1]["joint_score_span"],
                item[0],
            ),
        )
        chosen[high_span[0]] = ("high_prediction_span", high_span[1])
        chosen[high_disagreement[0]] = (
            "high_view_seed_disagreement",
            high_disagreement[1],
        )
    return chosen


def _selection_vector(row: dict[str, Any]) -> np.ndarray:
    models = row["confirmation_predictions"]
    uncertainty = row["confirmation_prediction_uncertainty"]
    clean = models["clean_view"]
    injection = models["injection_view"]
    return np.asarray(
        [
            injection["attack_probability"],
            injection["utility_probability"],
            injection["attack_probability"] - clean["attack_probability"],
            injection["utility_probability"] - clean["utility_probability"],
            uncertainty["injection_attack_seed_std"],
            uncertainty["injection_utility_seed_std"],
        ],
        dtype=float,
    )


def _farthest_pairs(
    rows: list[dict[str, Any]], count: int = 4
) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=_pair_key)
    if len(ordered) < count:
        raise ValueError(f"Need {count} pairs, found {len(ordered)}")
    matrix = np.asarray([_selection_vector(row) for row in ordered])
    scale = matrix.std(axis=0)
    scale[scale < 1e-8] = 1.0
    normalized = (matrix - matrix.mean(axis=0)) / scale
    first = int(np.argmin(matrix[:, 0]))
    selected = [first]
    while len(selected) < count:
        available = [index for index in range(len(ordered)) if index not in selected]
        scored = []
        for index in available:
            distance = min(
                float(np.linalg.norm(normalized[index] - normalized[other]))
                for other in selected
            )
            scored.append((distance, _pair_key(ordered[index]), index))
        selected.append(max(scored)[2])
    return [ordered[index] for index in selected]


def _select(
    candidates: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        grouped[_task_key(row)].append(row)
    chosen = _choose_tasks(grouped)
    selected = []
    metadata = []
    for task in sorted(chosen):
        stratum, statistics = chosen[task]
        rows = _farthest_pairs(grouped[task])
        annotated = [
            {
                **row,
                "contrast_task_stratum": stratum,
                "confirmation_task_stratum": stratum,
                "confirmation_injection_slot": slot,
                "confirmation_replay_chunk": slot,
                "source_split": "grouped_test",
            }
            for slot, row in enumerate(rows)
        ]
        selected.extend(annotated)
        metadata.append(
            {
                "suite": task[0],
                "user_task_id": task[1],
                "stratum": stratum,
                **statistics,
                "selected_injection_task_ids": [
                    row["injection_task_id"] for row in annotated
                ],
            }
        )
    return selected, metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", action="append", required=True)
    parser.add_argument("--train-trajectories", type=Path, required=True)
    parser.add_argument("--test-trajectories", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    sources = _parse_sources(args.source)
    reference, mappings = _align_sources(sources)
    candidates = _annotate_candidates(reference, mappings)
    selected, metadata = _select(candidates)
    if len(selected) != 32 or len(metadata) != 8:
        raise AssertionError("Frozen design requires 8 tasks and 32 pairs")
    task_counts = Counter(_task_key(row) for row in selected)
    if set(task_counts.values()) != {4}:
        raise AssertionError("Each task must contribute four pairs")
    chunks = {
        f"grouped_task_confirmation_chunk{chunk}": [
            row
            for row in selected
            if int(row["confirmation_replay_chunk"]) == chunk
        ]
        for chunk in range(4)
    }
    if any(len(rows) != 8 for rows in chunks.values()):
        raise AssertionError("Each replay chunk must contain eight pairs")
    if any(
        label in row for row in selected for label in REMOVED_LABEL_KEYS
    ):
        raise AssertionError("An observed label leaked into frozen selection")
    train_tasks = _trajectory_tasks(args.train_trajectories)
    test_tasks = _trajectory_tasks(args.test_trajectories)
    selected_tasks = {_task_key(row) for row in selected}
    if selected_tasks & train_tasks:
        raise AssertionError("Selected confirmation task overlaps grouped train")
    if not selected_tasks <= test_tasks:
        raise AssertionError("Selected confirmation task is absent from grouped test")

    output = {
        "scope": "label_blind_grouped_unseen_task_fresh_replay_confirmation",
        "source_split": "grouped_test",
        "selection_uses_observed_labels": False,
        "fresh_outcomes_frozen_before_collection": True,
        "world_model_user_task_held_out": True,
        "historical_pair_labels_ignored": True,
        "grouped_split_audit": {
            "train_trajectories": str(args.train_trajectories.resolve()),
            "test_trajectories": str(args.test_trajectories.resolve()),
            "train_trajectories_sha256": _sha256(args.train_trajectories),
            "test_trajectories_sha256": _sha256(args.test_trajectories),
            "train_task_count": len(train_tasks),
            "test_task_count": len(test_tasks),
            "selected_train_task_overlap": 0,
            "selected_tasks_subset_of_test": True,
        },
        "removed_label_keys": list(REMOVED_LABEL_KEYS),
        "score_sources": {
            name: str(path.resolve()) for name, path in sources.items()
        },
        "score_source_sha256": {
            name: _sha256(path) for name, path in sources.items()
        },
        "model_seeds": list(SEEDS),
        "fixed_models": list(MODEL_NAMES),
        "design": {
            "suites": list(SUITES),
            "tasks_per_suite": 2,
            "pairs_per_task": 4,
            "fresh_repeats_per_pair": 5,
            "task_count": 8,
            "pair_count": 32,
            "fresh_outcome_budget": 160,
            "task_strata_per_suite": {
                "high_prediction_span": 1,
                "high_view_seed_disagreement": 1,
            },
            "pair_selection": (
                "minimum injection-risk anchor plus deterministic farthest-point "
                "coverage in fixed clean/injection prediction space"
            ),
        },
        "suite_task_counts": dict(Counter(row["suite"] for row in metadata)),
        "task_stratum_counts": dict(
            Counter(row["stratum"] for row in metadata)
        ),
        "task_metadata": metadata,
        "limitations_declared_before_outcomes": [
            "Method design was informed by an earlier single-outcome exploratory analysis on this grouped test split.",
            "Pair identities may have historical outcomes, but those labels are neither read by the selector nor used to fit the fixed four-view models.",
            "Score-span enrichment is a stress-test sample and is not prevalence-representative of all AgentDojo tasks.",
        ],
        "selections": {
            "grouped_task_confirmation": selected,
            **chunks,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {key: value for key, value in output.items() if key != "selections"},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
