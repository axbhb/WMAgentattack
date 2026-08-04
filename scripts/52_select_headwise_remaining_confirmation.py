"""Freeze a label-blind confirmation set on the seven remaining grouped tasks."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import rankdata


ALPHA = 0.75
SELECTED_MODEL = "headwise_world_text_attack_clean_utility_alpha_0p75"
FIXED_MODELS = (
    "clean_raw",
    "text_pointwise",
    "world_attack_clean_utility_text_probability",
    SELECTED_MODEL,
)
PROHIBITED_LABEL_KEYS = {
    "observed_security",
    "observed_utility",
    "security",
    "utility",
    "attack_success",
    "task_success",
    "observed_attack_target",
    "observed_utility_target",
}


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _task_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row["suite"]), str(row["user_task_id"])


def _pair_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (*_task_key(row), str(row["injection_task_id"]))


def _trajectory_tasks(path: Path) -> set[tuple[str, str]]:
    tasks = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                tasks.add((str(row["domain"]), str(row["task_id"])))
    if not tasks:
        raise ValueError(f"No trajectory tasks found in {path}")
    return tasks


def _contains_prohibited_label(value: Any) -> bool:
    if isinstance(value, dict):
        return bool(PROHIBITED_LABEL_KEYS & set(value)) or any(
            _contains_prohibited_label(item) for item in value.values()
        )
    if isinstance(value, list):
        return any(_contains_prohibited_label(item) for item in value)
    return False


def _normalized_rank(values: np.ndarray) -> np.ndarray:
    denominator = max(len(values) - 1, 1)
    return (rankdata(values, method="average") - 1) / denominator


def _attack_name(row: dict[str, Any]) -> str:
    value = row.get("attack")
    if value:
        return str(value)
    source = Path(str(row["source_trace"]))
    if not source.is_file():
        raise ValueError(f"Cannot recover attack identity from {source}")
    raw = json.loads(source.read_text(encoding="utf-8"))
    value = raw.get("attack_type")
    if not value:
        raise ValueError(f"attack_type is missing in {source}")
    return str(value)


def _annotate_task(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=_pair_key)
    world_attack = np.asarray(
        [
            row["all_methods"]["world_pairwise_c0p03"]["attack_rank"]
            for row in ordered
        ],
        dtype=float,
    )
    text_attack = np.asarray(
        [
            row["all_methods"]["text_pointwise"]["attack_rank"]
            for row in ordered
        ],
        dtype=float,
    )
    clean_utility = np.asarray(
        [row["all_methods"]["clean_raw"]["utility_rank"] for row in ordered],
        dtype=float,
    )
    world_rank = _normalized_rank(world_attack)
    text_rank = _normalized_rank(text_attack)
    clean_utility_rank = _normalized_rank(clean_utility)
    blended_attack = ALPHA * world_rank + (1.0 - ALPHA) * text_rank
    output = []
    for index, row in enumerate(ordered):
        clean = row["all_methods"]["clean_raw"]
        text = row["all_methods"]["text_pointwise"]
        world = row["all_methods"]["world_pairwise_c0p03"]
        models = {
            "clean_raw": dict(clean),
            "text_pointwise": dict(text),
            "world_attack_clean_utility_text_probability": {
                "attack_rank": float(world_attack[index]),
                "utility_rank": float(clean_utility[index]),
                "attack_probability": float(text["attack_probability"]),
                "utility_probability": float(text["utility_probability"]),
            },
            SELECTED_MODEL: {
                "attack_rank": float(blended_attack[index]),
                "utility_rank": float(clean_utility[index]),
                "attack_probability": float(text["attack_probability"]),
                "utility_probability": float(text["utility_probability"]),
            },
        }
        output.append(
            {
                "suite": row["suite"],
                "user_task_id": row["user_task_id"],
                "injection_task_id": row["injection_task_id"],
                "attack": _attack_name(row),
                "trajectory_id": row["trajectory_id"],
                "source_trace": row["source_trace"],
                "confirmation_models": models,
                "selection_features": {
                    "world_attack_rank": float(world_rank[index]),
                    "text_attack_rank": float(text_rank[index]),
                    "blended_attack_rank": float(blended_attack[index]),
                    "clean_utility_rank": float(clean_utility_rank[index]),
                    "world_text_attack_rank_gap": float(
                        world_rank[index] - text_rank[index]
                    ),
                    "text_attack_probability": float(
                        text["attack_probability"]
                    ),
                    "text_utility_probability": float(
                        text["utility_probability"]
                    ),
                },
            }
        )
    return output


def _selection_vector(row: dict[str, Any]) -> np.ndarray:
    feature = row["selection_features"]
    return np.asarray(
        [
            feature["blended_attack_rank"],
            feature["clean_utility_rank"],
            0.5
            * (
                feature["blended_attack_rank"]
                + feature["clean_utility_rank"]
            ),
            feature["world_text_attack_rank_gap"],
            feature["text_attack_probability"],
            feature["text_utility_probability"],
        ],
        dtype=float,
    )


def _farthest_coverage(
    rows: list[dict[str, Any]], count: int = 4
) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=_pair_key)
    if len(ordered) < count:
        raise ValueError(f"Need at least {count} pairs, found {len(ordered)}")
    matrix = np.asarray([_selection_vector(row) for row in ordered])
    scale = matrix.std(axis=0)
    scale[scale < 1e-8] = 1.0
    normalized = (matrix - matrix.mean(axis=0)) / scale
    first = min(
        range(len(ordered)),
        key=lambda index: (
            matrix[index, 2],
            _pair_key(ordered[index]),
        ),
    )
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


def _select_remaining(
    candidates: list[dict[str, Any]],
    excluded_tasks: set[tuple[str, str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        task = _task_key(row)
        if task not in excluded_tasks:
            grouped[task].append(row)
    if len(grouped) != 7:
        raise ValueError(f"Expected seven remaining tasks, found {len(grouped)}")
    selected = []
    metadata = []
    for task, task_rows in sorted(grouped.items()):
        annotated = _annotate_task(task_rows)
        chosen = _farthest_coverage(annotated)
        for slot, row in enumerate(chosen):
            selected.append(
                {
                    **row,
                    "contrast_task_stratum": "remaining_grouped_test_task",
                    "confirmation_task_stratum": "remaining_grouped_test_task",
                    "confirmation_injection_slot": slot,
                    "confirmation_replay_chunk": slot,
                    "source_split": "grouped_test_remaining",
                }
            )
        metadata.append(
            {
                "suite": task[0],
                "user_task_id": task[1],
                "candidate_count": len(task_rows),
                "selected_injection_task_ids": [
                    row["injection_task_id"] for row in chosen
                ],
            }
        )
    return selected, metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-predictions", type=Path, required=True)
    parser.add_argument("--previous-selection", type=Path, required=True)
    parser.add_argument("--train-trajectories", type=Path, required=True)
    parser.add_argument("--test-trajectories", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    candidate_payload = _load(args.candidate_predictions)
    if candidate_payload.get("labels_included") is not False:
        raise ValueError("Candidate predictions must explicitly exclude labels")
    candidates = candidate_payload.get("pairs")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("Candidate prediction rows are missing")
    if any(_contains_prohibited_label(row) for row in candidates):
        raise ValueError("An outcome label leaked into candidate predictions")
    previous_rows = _load(args.previous_selection).get("selections", {}).get(
        "grouped_task_confirmation"
    )
    if not isinstance(previous_rows, list) or len(previous_rows) != 32:
        raise ValueError("Previous 32-pair selection is missing")
    excluded_tasks = {_task_key(row) for row in previous_rows}
    selected, metadata = _select_remaining(candidates, excluded_tasks)
    if len(selected) != 28 or len(metadata) != 7:
        raise AssertionError("Frozen design requires 7 tasks and 28 pairs")
    if set(Counter(_task_key(row) for row in selected).values()) != {4}:
        raise AssertionError("Every remaining task must contribute four pairs")
    if any(_contains_prohibited_label(row) for row in selected):
        raise AssertionError("A prohibited outcome label leaked into selection")

    train_tasks = _trajectory_tasks(args.train_trajectories)
    test_tasks = _trajectory_tasks(args.test_trajectories)
    selected_tasks = {_task_key(row) for row in selected}
    if selected_tasks & train_tasks:
        raise AssertionError("Selected task overlaps grouped training")
    if not selected_tasks <= test_tasks:
        raise AssertionError("Selected task is outside grouped test")
    if selected_tasks & excluded_tasks:
        raise AssertionError("Selected task already had 0713 fresh outcomes")
    if selected_tasks | excluded_tasks != test_tasks:
        raise AssertionError("Old and new selections do not cover grouped test tasks")

    chunks = {
        f"headwise_remaining_confirmation_chunk{chunk}": [
            row
            for row in selected
            if int(row["confirmation_replay_chunk"]) == chunk
        ]
        for chunk in range(4)
    }
    if set(len(rows) for rows in chunks.values()) != {7}:
        raise AssertionError("Every replay chunk must contain seven pairs")
    output = {
        "scope": "label_blind_headwise_remaining_grouped_task_confirmation",
        "selection_uses_observed_labels": False,
        "fresh_outcomes_frozen_before_collection": True,
        "historical_pair_labels_ignored": True,
        "candidate_prediction_protocol_sha256": candidate_payload.get(
            "protocol_sha256"
        ),
        "candidate_predictions": str(args.candidate_predictions.resolve()),
        "candidate_predictions_sha256": _sha256(args.candidate_predictions),
        "previous_selection": str(args.previous_selection.resolve()),
        "previous_selection_sha256": _sha256(args.previous_selection),
        "excluded_prior_fresh_task_count": len(excluded_tasks),
        "fixed_models": list(FIXED_MODELS),
        "selected_model": SELECTED_MODEL,
        "method": {
            "attack_ordering": (
                "0.75 within-task percentile rank of grouped-train world pairwise "
                "C=0.03 plus 0.25 text-pointwise percentile rank"
            ),
            "utility_ordering": "three-seed mean clean-view Dreamer utility score",
            "reported_probabilities": "grouped-train text-pointwise attack and utility",
            "rank_probability_decoupled": True,
        },
        "design": {
            "task_count": 7,
            "pairs_per_task": 4,
            "fresh_repeats_per_pair": 5,
            "fresh_outcome_budget": 140,
            "task_selection": "all grouped-test tasks not used in the prior 0713 replay",
            "pair_selection": (
                "minimum joint-rank anchor plus deterministic farthest-point "
                "coverage in fixed label-blind headwise score space"
            ),
        },
        "grouped_split_audit": {
            "train_task_count": len(train_tasks),
            "test_task_count": len(test_tasks),
            "selected_train_task_overlap": 0,
            "selected_prior_fresh_task_overlap": 0,
            "selected_plus_prior_cover_test_tasks": True,
            "train_trajectories_sha256": _sha256(args.train_trajectories),
            "test_trajectories_sha256": _sha256(args.test_trajectories),
        },
        "suite_task_counts": dict(Counter(row["suite"] for row in metadata)),
        "task_metadata": metadata,
        "limitations_declared_before_outcomes": [
            "The method was designed after inspecting validation and the prior eight-task fresh replay, so only these seven tasks provide new stochastic outcomes.",
            "Historical single outcomes for candidate identities existed but are absent from the selector input and are not used for selection.",
            "The remaining task set is suite-imbalanced: one banking, one slack, one travel, and four workspace tasks.",
            "Farthest-point pair coverage is a contrast stress test, not a prevalence estimate.",
        ],
        "selections": {
            "headwise_remaining_confirmation": selected,
            **chunks,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
