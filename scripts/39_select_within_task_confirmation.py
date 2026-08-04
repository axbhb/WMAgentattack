"""Freeze label-blind confirmation tasks disjoint from contrast-model tasks."""

from __future__ import annotations

import argparse
import importlib.util
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def _load_contrast_module():
    path = ROOT / "scripts" / "36_select_within_task_contrast.py"
    spec = importlib.util.spec_from_file_location("contrast_selector", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import contrast selector from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CONTRAST = _load_contrast_module()


def _choose_confirmation_tasks(
    grouped: dict[tuple[str, str], list[dict[str, Any]]],
    excluded_tasks: set[tuple[str, str]],
) -> dict[tuple[str, str], tuple[str, dict[str, Any]]]:
    chosen = {}
    for suite in CONTRAST.SUITES:
        eligible = [
            (task, CONTRAST._task_statistics(rows))
            for task, rows in grouped.items()
            if task[0] == suite and task not in excluded_tasks and len(rows) >= 4
        ]
        if len(eligible) < 2:
            raise RuntimeError(f"Suite {suite} has only {len(eligible)} holdout tasks")
        high_span = max(
            eligible,
            key=lambda item: (
                item[1]["score_span"],
                item[1]["mean_model_disagreement"],
                item[0],
            ),
        )
        chosen[high_span[0]] = ("confirmation_high_score_span", high_span[1])
        high_disagreement = max(
            (item for item in eligible if item[0] not in chosen),
            key=lambda item: (
                item[1]["mean_model_disagreement"],
                item[1]["score_span"],
                item[0],
            ),
        )
        chosen[high_disagreement[0]] = (
            "confirmation_high_model_disagreement",
            high_disagreement[1],
        )
    return chosen


def _select_confirmation(
    candidates: list[dict[str, Any]],
    excluded_tasks: set[tuple[str, str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        grouped[CONTRAST._task_key(row)].append(row)
    chosen = _choose_confirmation_tasks(grouped, excluded_tasks)
    selected = []
    metadata = []
    for task in sorted(chosen):
        stratum, statistics = chosen[task]
        rows = CONTRAST._farthest_injections(grouped[task], count=4)
        annotated = [
            {
                **row,
                "contrast_task_stratum": stratum,
                "contrast_injection_slot": slot,
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
    parser.add_argument("--primary-source", default="continuous_seed7")
    parser.add_argument("--clean-solvability-json", type=Path)
    parser.add_argument("--training-selection-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    sources = CONTRAST._parse_named_paths(args.source)
    primary, mappings = CONTRAST._align_sources(sources, args.primary_source)
    candidates = CONTRAST._aggregate_rows(
        primary,
        mappings,
        CONTRAST._load_clean_rates(args.clean_solvability_json),
    )
    training_payload = json.loads(
        args.training_selection_json.read_text(encoding="utf-8")
    )
    training_rows = training_payload["selections"]["within_task_contrast"]
    excluded_tasks = {CONTRAST._task_key(row) for row in training_rows}
    selected, metadata = _select_confirmation(candidates, excluded_tasks)
    if len(selected) != 32 or len(metadata) != 8:
        raise AssertionError("Frozen confirmation design requires 8 tasks/32 pairs")
    if {_task for _task in (CONTRAST._task_key(row) for row in selected)} & excluded_tasks:
        raise AssertionError("Confirmation task overlaps training task")
    chunks = {
        f"within_task_confirmation_chunk{chunk}": [
            row
            for row in selected
            if int(row["contrast_injection_slot"]) == chunk
        ]
        for chunk in range(4)
    }
    if any(len(rows) != 8 for rows in chunks.values()):
        raise AssertionError("Each confirmation chunk must have eight pairs")
    output = {
        "scope": "label_blind_within_task_confirmation",
        "selection_uses_observed_labels": False,
        "frozen_before_model_outcomes": True,
        "score_sources": {
            name: str(path.resolve()) for name, path in sources.items()
        },
        "training_selection_json": str(
            args.training_selection_json.resolve()
        ),
        "excluded_training_task_count": len(excluded_tasks),
        "design": {
            "tasks_per_suite": 2,
            "pairs_per_task": 4,
            "task_count": 8,
            "pair_count": 32,
            "task_strata_per_suite": {
                "confirmation_high_score_span": 1,
                "confirmation_high_model_disagreement": 1,
            },
        },
        "suite_task_counts": dict(Counter(row["suite"] for row in metadata)),
        "task_stratum_counts": dict(
            Counter(row["stratum"] for row in metadata)
        ),
        "task_metadata": metadata,
        "selections": {
            "within_task_confirmation": selected,
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
