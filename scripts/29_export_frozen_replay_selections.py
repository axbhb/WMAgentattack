"""Export strict frozen-threshold selections for real AgentDojo replay."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def _load_pareto_module():
    path = ROOT / "scripts" / "18_pareto_utility_selection.py"
    spec = importlib.util.spec_from_file_location("pareto_utility_selection", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import Pareto selector from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PARETO = _load_pareto_module()


def _select_frozen(
    candidate_payload: dict[str, Any],
    strict_payload: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    config = strict_payload["pareto"]["selected_validation_config"]
    per_seed = strict_payload["pareto"]["test_frozen_validation_threshold"][
        "per_seed"
    ]
    if len(per_seed) != 1:
        raise ValueError(
            "Replay export requires a single deployed model or score ensemble"
        )
    threshold = float(per_seed[0]["threshold"])
    selected = PARETO._select_pareto(
        candidate_payload["candidates"],
        top_k=int(config["top_k"]),
        utility_key=str(config["utility_key"]),
        threshold=threshold,
        max_per_user_task=2,
    )
    if len(selected) != int(config["top_k"]):
        raise AssertionError("Frozen selector did not produce the requested top-k")
    metadata = {
        "top_k": int(config["top_k"]),
        "utility_key": str(config["utility_key"]),
        "threshold_mode": str(config["threshold_mode"]),
        "threshold_value": float(config["threshold_value"]),
        "frozen_numeric_threshold": threshold,
        "cached_observed_asr": sum(
            bool(row["observed_security"]) for row in selected
        )
        / len(selected),
        "cached_observed_bup": sum(bool(row["observed_utility"]) for row in selected)
        / len(selected),
    }
    metadata["cached_asr_plus_bup"] = (
        metadata["cached_observed_asr"] + metadata["cached_observed_bup"]
    )
    return selected, metadata


def _pair_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row["suite"]),
        str(row["user_task_id"]),
        str(row["injection_task_id"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--selection",
        action="append",
        nargs=3,
        metavar=("NAME", "CANDIDATES_JSON", "STRICT_JSON"),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    output: dict[str, Any] = {
        "scope": "frozen_validation_selected_agentdojo_replay",
        "selections": {},
        "metadata": {},
        "overlap": {},
    }
    for name, candidate_path_raw, strict_path_raw in args.selection:
        if name in output["selections"]:
            raise ValueError(f"Duplicate selection name: {name}")
        candidate_path = Path(candidate_path_raw)
        strict_path = Path(strict_path_raw)
        candidate_payload = json.loads(candidate_path.read_text(encoding="utf-8"))
        strict_payload = json.loads(strict_path.read_text(encoding="utf-8"))
        selected, metadata = _select_frozen(candidate_payload, strict_payload)
        output["selections"][name] = selected
        output["metadata"][name] = {
            **metadata,
            "candidate_json": str(candidate_path.resolve()),
            "strict_json": str(strict_path.resolve()),
        }

    names = list(output["selections"])
    for left_index, left_name in enumerate(names):
        left = {_pair_key(row) for row in output["selections"][left_name]}
        for right_name in names[left_index + 1 :]:
            right = {_pair_key(row) for row in output["selections"][right_name]}
            output["overlap"][f"{left_name}__{right_name}"] = len(left & right)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output["metadata"], indent=2))


if __name__ == "__main__":
    main()
