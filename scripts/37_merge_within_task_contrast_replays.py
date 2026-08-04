"""Merge chunked AgentDojo replays into a five-replicate probability dataset."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


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

def _parse_seeds(value: str) -> list[int]:
    seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not seeds:
        raise ValueError("At least one base seed is required")
    return seeds


def _pair_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row["suite"]),
        str(row["user_task_id"]),
        str(row["injection_task_id"]),
    )


def _task_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row["suite"]), str(row["user_task_id"])


def _parse_named_paths(values: list[str]) -> dict[str, Path]:
    output = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected NAME=PATH, got {value!r}")
        name, raw_path = value.split("=", 1)
        if not name or name in output:
            raise ValueError(f"Invalid or duplicate source name: {name!r}")
        output[name] = Path(raw_path)
    return output


def _safe_float(value: Any) -> float:
    if isinstance(value, (bool, int, float, np.number)):
        number = float(value)
        if math.isfinite(number):
            return number
    return float("nan")


def _load_injection_score_mappings(
    sources: dict[str, Path],
) -> dict[str, dict[tuple[str, str, str], dict[str, Any]]]:
    output = {}
    for name, path in sources.items():
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload.get("candidates")
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"Candidates missing in injection source {path}")
        mapping = {_pair_key(row): row for row in rows}
        if len(mapping) != len(rows):
            raise ValueError(f"Duplicate pairs in injection source {path}")
        output[name] = mapping
    return output


def _annotate_injection_scores(
    row: dict[str, Any],
    mappings: dict[str, dict[tuple[str, str, str], dict[str, Any]]],
) -> dict[str, Any]:
    if not mappings:
        return row
    key = _pair_key(row)
    views = {}
    for name, mapping in mappings.items():
        if key not in mapping:
            raise ValueError(f"Selected pair missing from injection source {name}: {key}")
        candidate = mapping[key]
        views[name] = {
            score_key: value
            for score_key in SCORE_KEYS
            if math.isfinite(value := _safe_float(candidate.get(score_key)))
        }
    annotated = {**row, "injection_conditioned_score_views": views}
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
        annotated[f"injection_contrast_{score_key}_mean"] = float(values.mean())
        annotated[f"injection_contrast_{score_key}_std"] = float(values.std())
        annotated[f"injection_contrast_{score_key}_range"] = float(
            values.max() - values.min()
        )
    return annotated


def _rates(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    count = len(rows)
    attack = sum(bool(row["security"]) for row in rows)
    utility = sum(bool(row["utility"]) for row in rows)
    return {
        "attempt_count": count,
        "observed_asr": attack / count if count else 0.0,
        "observed_bup": utility / count if count else 0.0,
        "asr_plus_bup": (attack + utility) / count if count else 0.0,
    }


def _load_injection(row: dict[str, Any]) -> tuple[list[str], str]:
    path = Path(str(row["source_trace"]))
    raw = json.loads(path.read_text(encoding="utf-8"))
    injections = raw.get("injections")
    if isinstance(injections, dict):
        locations = sorted(str(key) for key in injections)
        text = "\n".join(
            f"{key}: {injections[key]}" for key in locations
        )
    elif isinstance(injections, list):
        locations = [f"item_{index}" for index in range(len(injections))]
        text = "\n".join(str(item) for item in injections)
    else:
        locations = []
        text = str(injections or "")
    if not text:
        raise ValueError(f"Injection text missing at {_pair_key(row)}")
    return locations, text


def _load_chunk(
    path: Path,
    *,
    base_seed: int,
    chunk: int,
    expected_rows: list[dict[str, Any]],
    selection_prefix: str = "within_task_contrast",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected_actual_seed = base_seed + 1000 * chunk
    if int(payload.get("seed")) != expected_actual_seed:
        raise ValueError(
            f"Unexpected actual seed in {path}: "
            f"{payload.get('seed')} != {expected_actual_seed}"
        )
    name = f"{selection_prefix}_chunk{chunk}"
    rows = payload.get("results", {}).get(name, {}).get("rows")
    if not isinstance(rows, list):
        raise ValueError(f"Selection {name} missing in {path}")
    expected_keys = {_pair_key(row) for row in expected_rows}
    row_keys = {_pair_key(row) for row in rows}
    if len(rows) != len(row_keys) or row_keys != expected_keys:
        raise ValueError(
            f"Chunk pair mismatch in {path}: "
            f"missing={len(expected_keys - row_keys)} "
            f"extra={len(row_keys - expected_keys)}"
        )
    annotated = [
        {
            **row,
            "contrast_base_seed": base_seed,
            "contrast_actual_seed": expected_actual_seed,
            "contrast_chunk": chunk,
        }
        for row in rows
    ]
    metadata = {
        "path": str(path.resolve()),
        "base_seed": base_seed,
        "actual_seed": expected_actual_seed,
        "chunk": chunk,
        "do_sample": payload.get("do_sample"),
        "temperature": payload.get("temperature"),
        "top_p": payload.get("top_p"),
    }
    return annotated, metadata


def _merge(
    selection_payload: dict[str, Any],
    replay_root: Path,
    *,
    base_seeds: list[int],
    chunks: int,
    output_dir: Path,
    injection_score_mappings: (
        dict[str, dict[tuple[str, str, str], dict[str, Any]]] | None
    ) = None,
    selection_prefix: str = "within_task_contrast",
) -> tuple[dict[str, Any], dict[str, Any]]:
    selections = selection_payload.get("selections", {})
    full_selection = selections.get(selection_prefix)
    if not isinstance(full_selection, list) or not full_selection:
        raise ValueError(f"Full {selection_prefix} selection is missing")
    injection_score_mappings = injection_score_mappings or {}
    selected_by_key = {
        _pair_key(row): _annotate_injection_scores(
            row, injection_score_mappings
        )
        for row in full_selection
    }
    if len(selected_by_key) != len(full_selection):
        raise ValueError("Full selection contains duplicate pairs")

    outcomes: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    replay_metadata = []
    merged_paths = {}
    merged_dir = output_dir / "merged_replays"
    merged_dir.mkdir(parents=True, exist_ok=True)
    for base_seed in base_seeds:
        merged_rows = []
        substream_seeds = {}
        for chunk in range(chunks):
            expected_rows = selections.get(f"{selection_prefix}_chunk{chunk}")
            if not isinstance(expected_rows, list):
                raise ValueError(f"Selection chunk {chunk} is missing")
            path = (
                replay_root
                / f"base_seed{base_seed}"
                / f"chunk{chunk}"
                / "replay.json"
            )
            rows, metadata = _load_chunk(
                path,
                base_seed=base_seed,
                chunk=chunk,
                expected_rows=expected_rows,
                selection_prefix=selection_prefix,
            )
            replay_metadata.append(metadata)
            substream_seeds[str(chunk)] = metadata["actual_seed"]
            merged_rows.extend(rows)
            for row in rows:
                outcomes[_pair_key(row)].append(
                    {
                        "base_seed": base_seed,
                        "actual_seed": metadata["actual_seed"],
                        "chunk": chunk,
                        "security": bool(row["security"]),
                        "utility": bool(row["utility"]),
                    }
                )
        if {_pair_key(row) for row in merged_rows} != set(selected_by_key):
            raise ValueError(f"Merged pair mismatch for base seed {base_seed}")
        payload = {
            "scope": f"merged_{selection_prefix}_replay",
            "seed": base_seed,
            "base_seed": base_seed,
            "actual_substream_seeds": substream_seeds,
            "selection_pair_count": len(full_selection),
            "unique_pair_count": len(full_selection),
            "results": {
                selection_prefix: {
                    "aggregate": _rates(merged_rows),
                    "rows": merged_rows,
                }
            },
        }
        path = merged_dir / f"base_seed{base_seed}_replay.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        merged_paths[str(base_seed)] = str(path.resolve())

    pair_rows = []
    for key, selected in selected_by_key.items():
        pair_outcomes = sorted(outcomes[key], key=lambda row: row["base_seed"])
        if len(pair_outcomes) != len(base_seeds):
            raise ValueError(
                f"Expected {len(base_seeds)} outcomes at {key}, "
                f"found {len(pair_outcomes)}"
            )
        attack_count = sum(row["security"] for row in pair_outcomes)
        utility_count = sum(row["utility"] for row in pair_outcomes)
        joint_counts = Counter(
            f"attack{int(row['security'])}_utility{int(row['utility'])}"
            for row in pair_outcomes
        )
        locations, injection_text = _load_injection(selected)
        pair_rows.append(
            {
                **selected,
                "injection_locations": locations,
                "injection_text": injection_text,
                "replay_attempt_count": len(pair_outcomes),
                "attack_success_count": attack_count,
                "utility_success_count": utility_count,
                "observed_attack_probability": attack_count / len(pair_outcomes),
                "observed_utility_probability": utility_count / len(pair_outcomes),
                "joint_outcome_counts": {
                    name: int(joint_counts.get(name, 0))
                    for name in (
                        "attack0_utility0",
                        "attack0_utility1",
                        "attack1_utility0",
                        "attack1_utility1",
                    )
                },
                "outcomes": pair_outcomes,
            }
        )

    by_suite = defaultdict(list)
    by_stratum = defaultdict(list)
    for row in pair_rows:
        by_suite[str(row["suite"])].append(row)
        by_stratum[str(row["contrast_task_stratum"])].append(row)

    def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
        attempts = sum(int(row["replay_attempt_count"]) for row in rows)
        attack = sum(int(row["attack_success_count"]) for row in rows)
        utility = sum(int(row["utility_success_count"]) for row in rows)
        return {
            "pair_count": len(rows),
            "attempt_count": attempts,
            "observed_asr": attack / attempts,
            "observed_bup": utility / attempts,
            "variable_attack_pair_count": sum(
                0 < int(row["attack_success_count"]) < int(row["replay_attempt_count"])
                for row in rows
            ),
            "variable_utility_pair_count": sum(
                0 < int(row["utility_success_count"]) < int(row["replay_attempt_count"])
                for row in rows
            ),
        }

    dataset = {
        "scope": f"{selection_prefix}_probability_dataset",
        "selection_uses_observed_labels": False,
        "base_replicate_seeds": base_seeds,
        "substream_seed_rule": "actual_seed = base_seed + 1000 * chunk",
        "injection_conditioned_score_views": list(
            injection_score_mappings
        ),
        "task_grouping_unit": "suite_and_user_task_id",
        "pair_count": len(pair_rows),
        "attempt_count": sum(
            int(row["replay_attempt_count"]) for row in pair_rows
        ),
        "pairs": pair_rows,
    }
    summary = {
        "scope": f"{selection_prefix}_replay_summary",
        "base_replicate_seeds": base_seeds,
        "replays": replay_metadata,
        "merged_replays": merged_paths,
        "overall": summarize(pair_rows),
        "by_suite": {
            name: summarize(rows) for name, rows in sorted(by_suite.items())
        },
        "by_task_stratum": {
            name: summarize(rows)
            for name, rows in sorted(by_stratum.items())
        },
        "task_count": len({_task_key(row) for row in pair_rows}),
        "pairs_per_task": [
            {
                "suite": task[0],
                "user_task_id": task[1],
                "pair_count": count,
            }
            for task, count in sorted(
                Counter(_task_key(row) for row in pair_rows).items()
            )
        ],
    }
    return dataset, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-json", type=Path, required=True)
    parser.add_argument("--replay-root", type=Path, required=True)
    parser.add_argument("--base-seeds", default="51,57,63,69,75")
    parser.add_argument("--chunks", type=int, default=4)
    parser.add_argument(
        "--selection-prefix", default="within_task_contrast"
    )
    parser.add_argument("--output-stem", default="within_task_contrast")
    parser.add_argument("--injection-source", action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    selection_payload = json.loads(
        args.selection_json.read_text(encoding="utf-8")
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    injection_sources = _parse_named_paths(args.injection_source)
    dataset, summary = _merge(
        selection_payload,
        args.replay_root,
        base_seeds=_parse_seeds(args.base_seeds),
        chunks=args.chunks,
        output_dir=args.output_dir,
        injection_score_mappings=_load_injection_score_mappings(
            injection_sources
        ),
        selection_prefix=args.selection_prefix,
    )
    args.output_dir.joinpath(f"{args.output_stem}_probability_dataset.json").write_text(
        json.dumps(dataset, indent=2), encoding="utf-8"
    )
    args.output_dir.joinpath(f"{args.output_stem}_replay_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
