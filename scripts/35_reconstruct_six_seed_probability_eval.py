"""Reconstruct old-seed probability selections from cached and retrofit runs.

The probability selector was frozen after the first three-seed test replay.
Fifteen of its sixteen pairs already exist in the historic union of the old
world-model and sklearn selections. After executing the one missing pair, this
script reconstructs complete probability-selector rows for the historic seeds
without rerunning shared candidates.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _pair_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row["suite"]),
        str(row["user_task_id"]),
        str(row["injection_task_id"]),
    )


def _rates(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    count = len(rows)
    security = sum(bool(row["security"]) for row in rows)
    utility = sum(bool(row["utility"]) for row in rows)
    return {
        "count": count,
        "ASR": security / count if count else 0.0,
        "BUP": utility / count if count else 0.0,
    }


def _row_cache(
    payloads: list[dict[str, Any]],
) -> dict[tuple[str, str, str], dict[str, Any]]:
    cache: dict[tuple[str, str, str], dict[str, Any]] = {}
    for payload in payloads:
        for result in payload.get("results", {}).values():
            for row in result.get("rows", []):
                key = _pair_key(row)
                if key in cache:
                    existing = cache[key]
                    if (
                        bool(existing["security"]) != bool(row["security"])
                        or bool(existing["utility"]) != bool(row["utility"])
                    ):
                        raise ValueError(
                            f"Conflicting shared replay outcome at {key}"
                        )
                else:
                    cache[key] = row
    return cache


def _reconstruct_seed(
    historic: dict[str, Any],
    retrofit: dict[str, Any],
    selections: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    if str(historic.get("seed")) != str(retrofit.get("seed")):
        raise ValueError("Historic and retrofit replay seeds do not match")
    cache = _row_cache([historic, retrofit])
    results: dict[str, Any] = {}
    for name, selected in selections.items():
        rows = []
        for candidate in selected:
            key = _pair_key(candidate)
            if key not in cache:
                raise ValueError(
                    f"Missing reconstructed replay outcome for {name}: {key}"
                )
            rows.append(cache[key])
        results[name] = {"aggregate": _rates(rows), "rows": rows}
    return {
        "scope": "reconstructed_frozen_probability_six_seed_replay",
        "seed": historic.get("seed"),
        "do_sample": historic.get("do_sample"),
        "temperature": historic.get("temperature"),
        "top_p": historic.get("top_p"),
        "historic_replay": historic.get("selection"),
        "retrofit_replay": retrofit.get("selection"),
        "selection_pair_count": sum(len(rows) for rows in selections.values()),
        "unique_pair_count": len(
            {
                _pair_key(row)
                for rows in selections.values()
                for row in rows
            }
        ),
        "results": results,
    }


def _load_by_seed(paths: list[Path]) -> dict[str, dict[str, Any]]:
    output = {}
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        seed = str(payload.get("seed"))
        if seed in output:
            raise ValueError(f"Duplicate replay seed: {seed}")
        output[seed] = payload
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-json", type=Path, required=True)
    parser.add_argument(
        "--historic-replay", action="append", type=Path, required=True
    )
    parser.add_argument(
        "--retrofit-replay", action="append", type=Path, required=True
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    selection_payload = json.loads(
        args.selection_json.read_text(encoding="utf-8")
    )
    selections = selection_payload.get("selections", {})
    if not selections:
        raise ValueError("No frozen selections found")
    historic = _load_by_seed(args.historic_replay)
    retrofit = _load_by_seed(args.retrofit_replay)
    if set(historic) != set(retrofit):
        raise ValueError(
            f"Seed mismatch: historic={sorted(historic)} "
            f"retrofit={sorted(retrofit)}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for seed in sorted(historic, key=int):
        payload = _reconstruct_seed(historic[seed], retrofit[seed], selections)
        path = args.output_dir / f"seed{seed}_reconstructed.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        outputs[seed] = str(path.resolve())
    summary = {
        "scope": "historic_probability_replay_reconstruction",
        "selection_json": str(args.selection_json.resolve()),
        "seeds": sorted(historic, key=int),
        "selection_names": list(selections),
        "outputs": outputs,
    }
    args.output_dir.joinpath("reconstruction_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
