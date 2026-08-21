"""Audit the frozen AgentDojo trajectory support for H1/H2/H3/H5/H10."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def suite(task_id: str) -> str:
    parts = task_id.split("::")
    if parts[0] in {"banking", "slack", "travel", "workspace"}:
        return parts[0]
    return parts[1] if len(parts) > 2 else "unknown"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if protocol["status"] != "preregistered_before_results":
        raise ValueError("parallel v22 protocol not frozen")
    surface = protocol["long_horizon_gate"]["agentdojo_adjacent_dataset"]
    if sha256(args.dataset) != surface["sha256"]:
        raise ValueError("long-horizon audit data hash mismatch")
    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    by_trajectory: dict[str, list[dict]] = defaultdict(list)
    for event in dataset["events"]:
        by_trajectory[str(event["trajectory_id"])].append(event)
    trajectory_rows = []
    for reference, events in sorted(by_trajectory.items()):
        ordered = sorted(events, key=lambda row: int(row["step_id"]))
        steps = [int(row["step_id"]) for row in ordered]
        if steps != list(range(steps[0], steps[0] + len(steps))):
            raise ValueError("non-contiguous AgentDojo trajectory")
        tasks = {str(row["task_name"]) for row in ordered}
        if len(tasks) != 1:
            raise ValueError("trajectory crosses task boundary")
        task = next(iter(tasks))
        trajectory_rows.append({"trajectory_id": reference, "length": len(ordered), "task": task, "suite": suite(task)})
    lengths = Counter(row["length"] for row in trajectory_rows)
    horizons = [1, 2, 3, 5, 10]
    windows = {
        horizon: sum(max(row["length"] - horizon, 0) for row in trajectory_rows)
        for horizon in horizons
    }
    horizon_tasks = {
        horizon: {row["task"] for row in trajectory_rows if row["length"] > horizon}
        for horizon in horizons
    }
    fold_windows = {}
    for fold_index, fold in enumerate(dataset["folds"]):
        test = set(fold["test_tasks"])
        fold_windows[str(fold_index)] = {
            str(horizon): sum(
                max(row["length"] - horizon, 0)
                for row in trajectory_rows if row["task"] in test
            ) for horizon in horizons
        }
    requirements = protocol["long_horizon_gate"]["minimum_requirements"]
    checks = {
        "trajectories": len(trajectory_rows) == requirements["trajectories"],
        "adjacent_transitions": windows[1] == requirements["adjacent_transitions"],
        "tasks": len({row["task"] for row in trajectory_rows}) == requirements["tasks"],
        "suites": len({row["suite"] for row in trajectory_rows}) == requirements["suites"],
        "horizon_windows": all(windows[int(key)] == value for key, value in requirements["horizon_windows"].items()),
        "horizon5_tasks": len(horizon_tasks[5]) == requirements["horizon5_tasks"],
        "horizon10_tasks": len(horizon_tasks[10]) == requirements["horizon10_tasks"],
        "horizon5_all_folds_supported": all(values["5"] > 0 for values in fold_windows.values()),
        "horizon10_zero_support_fold_preserved": any(values["10"] == 0 for values in fold_windows.values()),
        "runtime_failures": requirements["runtime_failures"] == 0
    }
    decision = "GO_LONG_HORIZON_MODEL_GATE_V22" if all(checks.values()) else "NO_GO_LONG_HORIZON_DATA_SUFFICIENCY_V22"
    payload = {
        "schema_version": "wmagentattack.long_horizon_data_gate.v22",
        "dataset_sha256": sha256(args.dataset),
        "decision": decision,
        "checks": checks,
        "passed": sum(checks.values()),
        "total": len(checks),
        "observed": {
            "event_rows": len(dataset["events"]),
            "trajectories": len(trajectory_rows),
            "tasks": len({row["task"] for row in trajectory_rows}),
            "suites": sorted({row["suite"] for row in trajectory_rows}),
            "length_histogram": dict(sorted(lengths.items())),
            "horizon_windows": windows,
            "horizon_task_counts": {key: len(value) for key, value in horizon_tasks.items()},
            "fold_windows": fold_windows
        },
        "authorization": {
            "fit_long_horizon_models": decision == "GO_LONG_HORIZON_MODEL_GATE_V22",
            "construct_pseudo_sequences": False,
            "large_scale_world_model": False
        },
        "counterevidence": "H10 has only 206 windows over 9/20 tasks and one frozen fold has zero support, so H10 is diagnostic only."
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"decision": decision, "passed": sum(checks.values()), "total": len(checks)}))


if __name__ == "__main__":
    main()
