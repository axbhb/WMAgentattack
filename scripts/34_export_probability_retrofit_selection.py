"""Export target candidates absent from previously executed selections."""

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


def _missing_target_rows(
    selections: dict[str, list[dict[str, Any]]],
    target: str,
    references: list[str],
) -> list[dict[str, Any]]:
    if target not in selections:
        raise ValueError(f"Unknown target selection: {target}")
    reference_keys: set[tuple[str, str, str]] = set()
    for name in references:
        if name not in selections:
            raise ValueError(f"Unknown reference selection: {name}")
        reference_keys.update(_pair_key(row) for row in selections[name])
    return [
        row for row in selections[target] if _pair_key(row) not in reference_keys
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-json", type=Path, required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--reference", action="append", required=True)
    parser.add_argument("--output-name", default="replay_probability_retrofit")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    payload = json.loads(args.selection_json.read_text(encoding="utf-8"))
    rows = _missing_target_rows(
        payload.get("selections", {}),
        args.target,
        args.reference,
    )
    if not rows:
        raise ValueError("No missing target candidates remain")
    output = {
        "scope": "missing_probability_selection_retrofit",
        "source_selection_json": str(args.selection_json.resolve()),
        "target": args.target,
        "references": args.reference,
        "selection_pair_count": len(rows),
        "selections": {args.output_name: rows},
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
