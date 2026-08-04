"""Summarize held-task metrics for outer-crossfit Dreamer checkpoints."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np


PATH_PATTERN = re.compile(r"fold(?P<fold>\d+)[/\\]seed(?P<seed>\d+)")


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def summarize(score_root: Path, expected_checkpoints: int) -> dict[str, Any]:
    metric_paths = sorted(score_root.glob("fold*/seed*/held_metrics.json"))
    if len(metric_paths) != expected_checkpoints:
        raise ValueError(
            f"Expected {expected_checkpoints} checkpoints, found {len(metric_paths)}"
        )
    checkpoints = []
    metric_values: dict[str, list[float]] = {}
    parameter_counts = []
    reached_epoch_budget = []
    for metric_path in metric_paths:
        match = PATH_PATTERN.search(str(metric_path))
        if match is None:
            raise ValueError(f"Unable to parse fold/seed from {metric_path}")
        metadata_path = metric_path.with_name("model_metadata.json")
        metrics_payload = _load(metric_path)
        metadata = _load(metadata_path)
        metrics = metrics_payload.get("metrics")
        if not isinstance(metrics, dict):
            raise ValueError(f"Metrics missing in {metric_path}")
        numeric_metrics = {}
        for name, value in metrics.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            number = float(value)
            if not math.isfinite(number):
                raise ValueError(f"Non-finite {name} in {metric_path}")
            numeric_metrics[name] = number
            metric_values.setdefault(name, []).append(number)
        history = metadata.get("training_history")
        config = metadata.get("config")
        if not isinstance(history, list) or not history or not isinstance(config, dict):
            raise ValueError(f"Training provenance missing in {metadata_path}")
        reached = int(history[-1]["epoch"]) == int(config["epochs"])
        reached_epoch_budget.append(reached)
        metadata_model_info = metadata.get("model_info")
        evaluation_model_info = metrics_payload.get("model_info")
        if isinstance(metadata_model_info, dict):
            parameter_count = int(metadata_model_info["parameter_count"])
        elif "parameter_count" in metadata:
            parameter_count = int(metadata["parameter_count"])
        elif isinstance(evaluation_model_info, dict):
            parameter_count = int(evaluation_model_info["parameter_count"])
        else:
            raise ValueError(f"Parameter count missing for {metric_path}")
        parameter_counts.append(parameter_count)
        checkpoints.append(
            {
                "fold": int(match.group("fold")),
                "seed": int(match.group("seed")),
                "held_step_count": int(metrics_payload["test_steps"]),
                "reached_epoch_budget": reached,
                "parameter_count": parameter_count,
                "metrics": numeric_metrics,
            }
        )
    aggregates = {}
    for name, values in sorted(metric_values.items()):
        array = np.asarray(values, dtype=float)
        aggregates[name] = {
            "count": len(values),
            "mean": float(array.mean()),
            "std": float(array.std()),
            "min": float(array.min()),
            "max": float(array.max()),
        }
    return {
        "scope": "outer_crossfit_checkpoint_held_metric_summary",
        "checkpoint_count": len(checkpoints),
        "all_checkpoints_reached_epoch_budget": all(reached_epoch_budget),
        "parameter_count_identical": len(set(parameter_counts)) == 1,
        "parameter_count": parameter_counts[0] if len(set(parameter_counts)) == 1 else None,
        "aggregate_metrics": aggregates,
        "checkpoints": checkpoints,
        "interpretation_constraint": (
            "Held-step prediction metrics diagnose training health; they do not "
            "establish candidate-level attack or utility ranking quality."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--score-root", type=Path, required=True)
    parser.add_argument("--expected-checkpoints", type=int, default=12)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = summarize(args.score_root, args.expected_checkpoints)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
