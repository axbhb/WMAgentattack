"""Freeze a balanced, label-blind external test contrast set."""

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


def _choose_suite_tasks(
    grouped: dict[tuple[str, str], list[dict[str, Any]]],
    suite: str,
    excluded_tasks: set[tuple[str, str]],
) -> list[tuple[tuple[str, str], str, dict[str, Any]]]:
    eligible = [
        (task, CONTRAST._task_statistics(rows))
        for task, rows in grouped.items()
        if task[0] == suite and len(rows) >= 2 and task not in excluded_tasks
    ]
    if len(eligible) < 2:
        raise RuntimeError(f"Suite {suite} has only {len(eligible)} two-pair tasks")
    span = max(
        eligible,
        key=lambda item: (
            item[1]["score_span"],
            item[1]["mean_model_disagreement"],
            item[0],
        ),
    )
    disagreement = max(
        (item for item in eligible if item[0] != span[0]),
        key=lambda item: (
            item[1]["mean_model_disagreement"],
            item[1]["score_span"],
            item[0],
        ),
    )
    return [
        (span[0], "external_high_score_span", span[1]),
        (
            disagreement[0],
            "external_high_model_disagreement",
            disagreement[1],
        ),
    ]


def _select_external(
    candidates: list[dict[str, Any]],
    excluded_tasks: set[tuple[str, str]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    excluded_tasks = excluded_tasks or set()
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        grouped[CONTRAST._task_key(row)].append(row)
    selected = []
    metadata = []
    for replay_chunk, suite in enumerate(CONTRAST.SUITES):
        for task, stratum, statistics in _choose_suite_tasks(
            grouped, suite, excluded_tasks
        ):
            rows = CONTRAST._farthest_injections(grouped[task], count=2)
            annotated = [
                {
                    **row,
                    "contrast_task_stratum": stratum,
                    "contrast_injection_slot": slot,
                    "contrast_replay_chunk": replay_chunk,
                    "source_split": "test",
                }
                for slot, row in enumerate(rows)
            ]
            selected.extend(annotated)
            metadata.append(
                {
                    "suite": task[0],
                    "user_task_id": task[1],
                    "stratum": stratum,
                    "replay_chunk": replay_chunk,
                    **statistics,
                    "selected_injection_task_ids": [
                        row["injection_task_id"] for row in annotated
                    ],
                }
            )
    return selected, metadata


def _excluded_tasks(paths: list[Path]) -> set[tuple[str, str]]:
    output = set()
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for rows in payload.get("selections", {}).values():
            if isinstance(rows, list):
                output.update(
                    CONTRAST._task_key(row)
                    for row in rows
                    if isinstance(row, dict)
                    and "suite" in row
                    and "user_task_id" in row
                )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", action="append", required=True)
    parser.add_argument("--primary-source", default="continuous_seed7")
    parser.add_argument("--clean-solvability-json", type=Path)
    parser.add_argument(
        "--exclude-selection-json", type=Path, action="append", default=[]
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    sources = CONTRAST._parse_named_paths(args.source)
    primary, mappings = CONTRAST._align_sources(sources, args.primary_source)
    candidates = CONTRAST._aggregate_rows(
        primary,
        mappings,
        CONTRAST._load_clean_rates(args.clean_solvability_json),
    )
    excluded_tasks = _excluded_tasks(args.exclude_selection_json)
    selected, metadata = _select_external(candidates, excluded_tasks)
    if len(selected) != 16 or len(metadata) != 8:
        raise AssertionError("External design requires 8 tasks and 16 pairs")
    chunks = {
        f"within_task_external_test_chunk{chunk}": [
            row
            for row in selected
            if int(row["contrast_replay_chunk"]) == chunk
        ]
        for chunk in range(4)
    }
    if any(len(rows) != 4 for rows in chunks.values()):
        raise AssertionError("Each external test chunk must have four pairs")
    output = {
        "scope": "label_blind_pair_heldout_calibrator_task_disjoint_test",
        "source_split": "test",
        "selection_uses_observed_labels": False,
        "frozen_before_main_model_results": True,
        "world_model_pair_held_out_by_original_split": True,
        "world_model_user_task_held_out": False,
        "calibrator_user_task_disjoint": True,
        "excluded_calibrator_task_count": len(excluded_tasks),
        "limitation": (
            "The original trajectory-level split places every test user task "
            "in the world-model training split; this tests held-out pairs and "
            "calibrator task transfer, not world-model unseen-task transfer."
        ),
        "removed_label_keys": list(CONTRAST.REMOVED_LABEL_KEYS),
        "score_sources": {
            name: str(path.resolve()) for name, path in sources.items()
        },
        "design": {
            "tasks_per_suite": 2,
            "pairs_per_task": 2,
            "task_count": 8,
            "pair_count": 16,
            "replay_budget_if_activated": 80,
        },
        "suite_task_counts": dict(Counter(row["suite"] for row in metadata)),
        "task_stratum_counts": dict(
            Counter(row["stratum"] for row in metadata)
        ),
        "task_metadata": metadata,
        "selections": {
            "within_task_external_test": selected,
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
