"""Select a label-blind within-task injection contrast dataset.

The selection uses only frozen world-model scores. It chooses six user tasks
per AgentDojo suite: two high score-span tasks, two high model-disagreement
tasks, and two low-span hard controls. Four injections per selected task are
chosen by deterministic farthest-point sampling in score space.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


SUITES = ("banking", "slack", "travel", "workspace")
SCORE_KEYS = (
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
)
SELECTION_FEATURE_KEYS = (
    "risk_score",
    "final_utility_score",
    "preservation_score",
    "target_skill_probability",
)
REMOVED_LABEL_KEYS = (
    "observed_security",
    "observed_utility",
    "security",
    "utility",
)


def _parse_named_paths(values: Iterable[str]) -> dict[str, Path]:
    output: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected NAME=PATH, got {value!r}")
        name, raw_path = value.split("=", 1)
        name = name.strip()
        if not name or name in output:
            raise ValueError(f"Invalid or duplicate source name: {name!r}")
        output[name] = Path(raw_path).expanduser()
    if not output:
        raise ValueError("At least one score source is required")
    return output


def _pair_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row["suite"]),
        str(row["user_task_id"]),
        str(row["injection_task_id"]),
    )


def _task_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row["suite"]), str(row["user_task_id"])


def _safe_float(value: Any) -> float:
    if isinstance(value, (bool, int, float, np.number)):
        number = float(value)
        if math.isfinite(number):
            return number
    return float("nan")


def _load_candidates(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("candidates")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Candidate list missing or empty in {path}")
    keys = [_pair_key(row) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError(f"Duplicate candidate pair in {path}")
    return rows


def _align_sources(
    sources: dict[str, Path],
    primary_source: str,
) -> tuple[list[dict[str, Any]], dict[str, dict[tuple[str, str, str], dict[str, Any]]]]:
    if primary_source not in sources:
        raise ValueError(f"Unknown primary source: {primary_source}")
    rows_by_source = {name: _load_candidates(path) for name, path in sources.items()}
    mappings = {
        name: {_pair_key(row): row for row in rows}
        for name, rows in rows_by_source.items()
    }
    primary = rows_by_source[primary_source]
    primary_keys = {_pair_key(row) for row in primary}
    for name, mapping in mappings.items():
        if set(mapping) != primary_keys:
            raise ValueError(
                f"Candidate identity mismatch for {name}: "
                f"missing={len(primary_keys - set(mapping))} "
                f"extra={len(set(mapping) - primary_keys)}"
            )
    return primary, mappings


def _load_clean_rates(path: Path | None) -> dict[tuple[str, str], float]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        (str(row["suite"]), str(row["user_task_id"])): float(
            row["base_success_rate"]
        )
        for row in payload.get("tasks", [])
    }


def _load_excluded_pairs(paths: list[Path]) -> set[tuple[str, str, str]]:
    output: set[tuple[str, str, str]] = set()
    for path in paths:
        output.update(_pair_key(row) for row in _load_candidates(path))
    return output


def _aggregate_rows(
    primary: list[dict[str, Any]],
    mappings: dict[str, dict[tuple[str, str, str], dict[str, Any]]],
    clean_rates: dict[tuple[str, str], float],
) -> list[dict[str, Any]]:
    output = []
    for reference in primary:
        key = _pair_key(reference)
        clean = {
            field: value
            for field, value in reference.items()
            if field not in REMOVED_LABEL_KEYS
        }
        views: dict[str, dict[str, float]] = {}
        for name, mapping in mappings.items():
            row = mapping[key]
            views[name] = {
                score_key: value
                for score_key in SCORE_KEYS
                if math.isfinite(value := _safe_float(row.get(score_key)))
            }
        annotated = {
            **clean,
            "contrast_score_views": views,
            "contrast_clean_solvability": clean_rates.get(_task_key(reference)),
        }
        for score_key in SCORE_KEYS:
            values = np.asarray(
                [
                    view[score_key]
                    for view in views.values()
                    if score_key in view
                ],
                dtype=float,
            )
            if not len(values):
                continue
            annotated[f"contrast_{score_key}_mean"] = float(values.mean())
            annotated[f"contrast_{score_key}_std"] = float(values.std())
            annotated[f"contrast_{score_key}_range"] = float(
                values.max() - values.min()
            )
        for required in SELECTION_FEATURE_KEYS:
            if f"contrast_{required}_mean" not in annotated:
                raise ValueError(f"Required score {required} is missing at {key}")
        output.append(annotated)
    return output


def _task_statistics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def values(key: str) -> np.ndarray:
        return np.asarray(
            [float(row[f"contrast_{key}_mean"]) for row in rows],
            dtype=float,
        )

    risk_span = float(np.ptp(values("risk_score")))
    final_utility_span = float(np.ptp(values("final_utility_score")))
    preservation_span = float(np.ptp(values("preservation_score")))
    disagreement = float(
        np.mean(
            [
                float(row.get("contrast_risk_score_std", 0.0))
                + float(row.get("contrast_final_utility_score_std", 0.0))
                + float(row.get("contrast_preservation_score_std", 0.0))
                for row in rows
            ]
        )
    )
    target_diversity = len({str(row.get("target_skill")) for row in rows}) / len(
        rows
    )
    score_span = risk_span + 0.5 * (
        final_utility_span + preservation_span
    )
    return {
        "candidate_count": len(rows),
        "risk_span": risk_span,
        "final_utility_span": final_utility_span,
        "preservation_span": preservation_span,
        "score_span": score_span,
        "mean_model_disagreement": disagreement,
        "target_skill_diversity": target_diversity,
    }


def _choose_tasks(
    grouped: dict[tuple[str, str], list[dict[str, Any]]],
    *,
    tasks_per_suite: int,
) -> dict[tuple[str, str], tuple[str, dict[str, Any]]]:
    if tasks_per_suite != 6:
        raise ValueError("The frozen design requires exactly six tasks per suite")
    chosen: dict[tuple[str, str], tuple[str, dict[str, Any]]] = {}
    for suite in SUITES:
        eligible = []
        for task, rows in grouped.items():
            if task[0] != suite or len(rows) < 4:
                continue
            eligible.append((task, _task_statistics(rows)))
        if len(eligible) < tasks_per_suite:
            raise RuntimeError(
                f"Suite {suite} has only {len(eligible)} tasks with >=4 injections"
            )

        strata = (
            (
                "high_score_span",
                sorted(
                    eligible,
                    key=lambda item: (
                        item[1]["score_span"],
                        item[1]["mean_model_disagreement"],
                        item[0],
                    ),
                    reverse=True,
                ),
            ),
            (
                "high_model_disagreement",
                sorted(
                    eligible,
                    key=lambda item: (
                        item[1]["mean_model_disagreement"],
                        item[1]["score_span"],
                        item[0],
                    ),
                    reverse=True,
                ),
            ),
            (
                "low_score_span_hard_control",
                sorted(
                    eligible,
                    key=lambda item: (
                        item[1]["score_span"],
                        -item[1]["target_skill_diversity"],
                        item[0],
                    ),
                ),
            ),
        )
        for stratum, ranked in strata:
            added = 0
            for task, statistics in ranked:
                if task in chosen:
                    continue
                chosen[task] = (stratum, statistics)
                added += 1
                if added == 2:
                    break
            if added != 2:
                raise RuntimeError(f"Unable to fill {suite}/{stratum}: {added}/2")
    return chosen


def _farthest_injections(
    rows: list[dict[str, Any]],
    *,
    count: int,
) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=_pair_key)
    if len(ordered) < count:
        raise ValueError(f"Need {count} candidates, found {len(ordered)}")
    matrix = np.asarray(
        [
            [
                float(row[f"contrast_{key}_mean"])
                for key in SELECTION_FEATURE_KEYS
            ]
            + [
                float(row.get("contrast_risk_score_std", 0.0)),
                float(row.get("contrast_final_utility_score_std", 0.0)),
            ]
            for row in ordered
        ],
        dtype=float,
    )
    scale = matrix.std(axis=0)
    scale[scale < 1e-8] = 1.0
    normalized = (matrix - matrix.mean(axis=0)) / scale
    first = int(np.argmin(matrix[:, 0]))
    selected = [first]
    while len(selected) < count:
        candidates = [index for index in range(len(ordered)) if index not in selected]
        distances = []
        for index in candidates:
            minimum = min(
                float(np.linalg.norm(normalized[index] - normalized[other]))
                for other in selected
            )
            distances.append((minimum, _pair_key(ordered[index]), index))
        _, _, selected_index = max(distances)
        selected.append(selected_index)
    return [ordered[index] for index in selected]


def _select_contrast(
    candidates: list[dict[str, Any]],
    *,
    tasks_per_suite: int,
    pairs_per_task: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        grouped[_task_key(row)].append(row)
    chosen_tasks = _choose_tasks(grouped, tasks_per_suite=tasks_per_suite)

    selected = []
    task_metadata = []
    for task in sorted(chosen_tasks):
        stratum, statistics = chosen_tasks[task]
        task_rows = _farthest_injections(
            grouped[task],
            count=pairs_per_task,
        )
        annotated = [
            {
                **row,
                "contrast_task_stratum": stratum,
                "contrast_injection_slot": slot,
            }
            for slot, row in enumerate(task_rows)
        ]
        selected.extend(annotated)
        task_metadata.append(
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
    return selected, task_metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", action="append", required=True)
    parser.add_argument("--primary-source", default="continuous_seed7")
    parser.add_argument("--clean-solvability-json", type=Path)
    parser.add_argument(
        "--exclude-candidates", action="append", type=Path, default=[]
    )
    parser.add_argument("--tasks-per-suite", type=int, default=6)
    parser.add_argument("--pairs-per-task", type=int, default=4)
    parser.add_argument("--chunks", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.pairs_per_task != 4 or args.chunks != 4:
        raise ValueError("The frozen design requires four pairs and four chunks")
    sources = _parse_named_paths(args.source)
    primary, mappings = _align_sources(sources, args.primary_source)
    clean_rates = _load_clean_rates(args.clean_solvability_json)
    candidates = _aggregate_rows(primary, mappings, clean_rates)
    excluded = _load_excluded_pairs(args.exclude_candidates)
    overlap = {_pair_key(row) for row in candidates} & excluded
    if overlap:
        raise ValueError(f"Train candidates overlap excluded val/test pairs: {len(overlap)}")

    selected, task_metadata = _select_contrast(
        candidates,
        tasks_per_suite=args.tasks_per_suite,
        pairs_per_task=args.pairs_per_task,
    )
    expected_tasks = args.tasks_per_suite * len(SUITES)
    expected_pairs = expected_tasks * args.pairs_per_task
    if len(selected) != expected_pairs:
        raise AssertionError(f"Selected {len(selected)}/{expected_pairs} pairs")
    task_counts = Counter(_task_key(row) for row in selected)
    if set(task_counts.values()) != {args.pairs_per_task}:
        raise AssertionError("Each selected task must have exactly four pairs")

    chunks = {
        f"within_task_contrast_chunk{chunk}": [
            row
            for row in selected
            if int(row["contrast_injection_slot"]) == chunk
        ]
        for chunk in range(args.chunks)
    }
    if any(len(rows) != expected_tasks for rows in chunks.values()):
        raise AssertionError("Each chunk must contain one injection per task")
    output = {
        "scope": "label_blind_within_task_injection_contrast",
        "selection_uses_observed_labels": False,
        "removed_label_keys": list(REMOVED_LABEL_KEYS),
        "score_sources": {
            name: str(path.resolve()) for name, path in sources.items()
        },
        "primary_source": args.primary_source,
        "clean_solvability_json": (
            str(args.clean_solvability_json.resolve())
            if args.clean_solvability_json
            else None
        ),
        "excluded_candidate_paths": [
            str(path.resolve()) for path in args.exclude_candidates
        ],
        "excluded_pair_overlap": 0,
        "design": {
            "suites": list(SUITES),
            "tasks_per_suite": args.tasks_per_suite,
            "pairs_per_task": args.pairs_per_task,
            "task_count": expected_tasks,
            "pair_count": expected_pairs,
            "task_strata_per_suite": {
                "high_score_span": 2,
                "high_model_disagreement": 2,
                "low_score_span_hard_control": 2,
            },
            "injection_selection": "lowest-risk anchor plus farthest-point score coverage",
        },
        "task_stratum_counts": dict(
            Counter(row["stratum"] for row in task_metadata)
        ),
        "suite_task_counts": dict(
            Counter(row["suite"] for row in task_metadata)
        ),
        "task_metadata": task_metadata,
        "selections": {
            "within_task_contrast": selected,
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
