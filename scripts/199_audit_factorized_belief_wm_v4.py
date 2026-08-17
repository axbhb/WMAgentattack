"""Label-blind integrity audit for the frozen v4 representation and horizons."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wmagentattack.factorized_belief_world_model import (
    TYPED_STATE_NODES,
    stack_typed_state_nodes,
)
from wmagentattack.multisource_suitability import file_sha256


FORBIDDEN_INPUT_KEYS = {
    "attack_success",
    "decision",
    "future_action",
    "future_observation",
    "next_action",
    "reward",
    "security",
    "target",
    "task_success",
    "utility",
}


def _sha256_array(value: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(str(value.shape).encode("ascii"))
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    source_audit = json.loads(args.audit.read_text(encoding="utf-8"))
    events = dataset["events"]
    dimension = int(protocol["training"]["typed_hash_dimension"])
    first = stack_typed_state_nodes(events, hash_dimension=dimension)
    second = stack_typed_state_nodes(events, hash_dimension=dimension)
    first_hash = _sha256_array(first)
    second_hash = _sha256_array(second)
    forbidden = sorted(
        {
            str(key).lower()
            for event in events
            for key in event["causal_model_input"]
            if str(key).lower() in FORBIDDEN_INPUT_KEYS
        }
    )
    trajectories: dict[str, list[int]] = defaultdict(list)
    for event in events:
        trajectories[str(event["trajectory_id"])].append(int(event["step_id"]))
    horizon_counts = Counter()
    contiguous = True
    for steps in trajectories.values():
        ordered = sorted(steps)
        contiguous &= ordered == list(range(len(ordered)))
        for horizon in range(1, int(protocol["training"]["maximum_horizon"]) + 1):
            horizon_counts[horizon] += max(0, len(ordered) - horizon)
    checks = {
        "source_dataset_audit_passed": bool(source_audit["passed"]),
        "dataset_hash_frozen": file_sha256(args.dataset)
        == protocol["frozen_dataset"]["sha256"],
        "source_audit_hash_frozen": file_sha256(args.audit)
        == protocol["frozen_dataset"]["audit_sha256"],
        "typed_builds_byte_identical": first_hash == second_hash,
        "typed_shape_exact": first.shape
        == (
            int(protocol["frozen_dataset"]["event_rows"]),
            len(TYPED_STATE_NODES),
            dimension + 8,
        ),
        "typed_values_finite": bool(np.isfinite(first).all()),
        "forbidden_inputs_absent": not forbidden,
        "trajectory_steps_contiguous": contiguous,
        "horizon_one_matches_adjacent_transitions": horizon_counts[1]
        == int(protocol["frozen_dataset"]["adjacent_transitions"]),
        "all_horizons_nonempty_and_monotone": all(
            horizon_counts[horizon] > 0
            and (
                horizon == 1
                or horizon_counts[horizon] <= horizon_counts[horizon - 1]
            )
            for horizon in range(1, int(protocol["training"]["maximum_horizon"]) + 1)
        ),
        "current_actions_legal": all(
            event["current_action_candidate_id"]
            in event["current_legal_candidate_ids"]
            for event in events
        ),
        "next_targets_legal": all(
            event["next_target_candidate_id"] is None
            or event["next_target_candidate_id"] in event["next_legal_candidate_ids"]
            for event in events
        ),
    }
    result: dict[str, Any] = {
        "protocol_id": protocol["protocol_id"],
        "passed": all(checks.values()),
        "checks": checks,
        "typed_node_names": list(TYPED_STATE_NODES),
        "typed_array_shape": list(first.shape),
        "typed_array_sha256_a": first_hash,
        "typed_array_sha256_b": second_hash,
        "horizon_transition_counts": {
            str(key): value for key, value in sorted(horizon_counts.items())
        },
        "forbidden_input_keys": forbidden,
        "label_blind": True,
        "new_llm_calls": 0,
        "new_tool_executions": 0,
        "real_external_endpoint_calls": 0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    if not result["passed"]:
        raise SystemExit("v4 representation audit failed")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
