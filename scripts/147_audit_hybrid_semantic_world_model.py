"""Audit the frozen exact-transition plus learned-head hybrid architecture."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from wmagentattack.hybrid_semantic_world_model import (
    EVIDENCE_DELTA_TARGETS,
    ExactObservedSemanticTransition,
    HybridSemanticWorldModel,
    assert_no_planning_or_value_heads,
    evidence_delta_target,
    semantic_state_v3_feature_vector,
    tool_candidate_vector,
)
from wmagentattack.semantic_state_v3 import (
    StructuredSemanticStateV3,
    find_semantic_state_v3_leakage,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    if protocol.get("status") != "preregistered_before_architecture_audit":
        raise ValueError("Stage 2 protocol was not frozen before its audit")
    dataset_hash = _sha256(args.dataset)
    if dataset_hash != protocol["source_dataset_sha256"]:
        raise ValueError("Stage 2 source dataset hash mismatch")
    hash_dimension = int(protocol["architecture"]["hash_dimension"])

    states: list[StructuredSemanticStateV3] = []
    prefixes: list[dict[str, Any]] = []
    transition_rows: list[dict[str, Any]] = []
    leakage_failures: list[str] = []
    terminal_rows = 0
    exact = ExactObservedSemanticTransition()

    for episode in dataset["episodes"]:
        episode_states = []
        for prefix in episode["prefixes"]:
            payload = prefix["features"]["semantic_state_v3"]
            findings = find_semantic_state_v3_leakage(payload)
            if findings:
                leakage_failures.append(
                    f"{episode['episode_id']}::p{prefix['prefix_index']}::{findings}"
                )
            state = StructuredSemanticStateV3.model_validate(payload)
            episode_states.append(state)
            states.append(state)
            prefixes.append(
                {
                    "episode_id": episode["episode_id"],
                    "task_id": episode["task_id"],
                    "split": episode["split"],
                    "prefix": prefix,
                    "state": state,
                }
            )
        for index, (prefix, state) in enumerate(
            zip(episode["prefixes"], episode_states)
        ):
            action = str(prefix["targets"]["next_action"])
            if action == "STOP":
                terminal_rows += 1
                if index != len(episode_states) - 1:
                    raise ValueError("STOP appears before the final episode prefix")
                continue
            if index + 1 >= len(episode_states):
                raise ValueError("non-STOP target has no observed next prefix")
            following, transition_audit = exact.advance(
                state,
                episode_states[index + 1],
                executed_action_id=action,
            )
            transition_rows.append(
                {
                    "episode_id": episode["episode_id"],
                    "state": state,
                    "following": following,
                    "action": action,
                    "audit": transition_audit,
                    "delta": evidence_delta_target(state, following),
                }
            )

    candidates = sorted(dataset["tool_catalog"])
    candidate_index = {name: index for index, name in enumerate(candidates)}
    argument_keys = list(dataset["argument_key_vocab"])
    argument_index = {name: index for index, name in enumerate(argument_keys)}
    state_inputs = np.stack(
        [
            semantic_state_v3_feature_vector(
                row["state"], hash_dimension=hash_dimension
            )
            for row in prefixes
        ]
    )
    candidate_inputs = np.stack(
        [
            tool_candidate_vector(
                dataset["tool_catalog"][candidate],
                hash_dimension=hash_dimension,
            )
            for candidate in candidates
        ]
    )
    legal_masks = np.zeros((len(prefixes), len(candidates)), dtype=bool)
    action_targets = np.zeros(len(prefixes), dtype=np.int64)
    argument_targets = np.zeros(
        (len(prefixes), len(argument_keys)), dtype=np.float32
    )
    for row_index, row in enumerate(prefixes):
        for candidate in row["state"].legal_actions:
            legal_masks[row_index, candidate_index[candidate]] = True
        target = str(row["prefix"]["targets"]["next_action"])
        action_targets[row_index] = candidate_index[target]
        if not legal_masks[row_index, action_targets[row_index]]:
            raise ValueError("target victim action is not legal")
        for key in row["prefix"]["targets"]["argument_keys"]:
            argument_targets[row_index, argument_index[key]] = 1.0

    _set_seed(int(protocol["architecture"]["smoke_seed"]))
    model = HybridSemanticWorldModel(
        state_size=state_inputs.shape[1],
        candidate_size=candidate_inputs.shape[1],
        argument_keys=len(argument_keys),
        hidden_size=int(protocol["architecture"]["hidden_size"]),
        dropout=0.0,
    )
    assert_no_planning_or_value_heads(model)
    x = torch.as_tensor(state_inputs, dtype=torch.float32)
    c = torch.as_tensor(candidate_inputs, dtype=torch.float32)
    legal = torch.as_tensor(legal_masks, dtype=torch.bool)
    action = torch.as_tensor(action_targets, dtype=torch.long)
    arguments = torch.as_tensor(argument_targets, dtype=torch.float32)
    action_logits, argument_logits, evidence_logits = model(x, c)
    masked = action_logits.masked_fill(~legal, torch.finfo(action_logits.dtype).min)
    action_loss = F.cross_entropy(masked, action)
    argument_loss = F.binary_cross_entropy_with_logits(
        argument_logits, arguments
    )

    prefix_lookup = {
        (row["episode_id"], int(row["prefix"]["prefix_index"])): index
        for index, row in enumerate(prefixes)
    }
    evidence_row_indices = []
    evidence_candidate_indices = []
    evidence_targets = []
    for row in transition_rows:
        evidence_row_indices.append(
            prefix_lookup[(row["episode_id"], row["state"].step_index)]
        )
        evidence_candidate_indices.append(candidate_index[row["action"]])
        evidence_targets.append(row["delta"])
    selected_evidence_logits = evidence_logits[
        torch.as_tensor(evidence_row_indices, dtype=torch.long),
        torch.as_tensor(evidence_candidate_indices, dtype=torch.long),
    ]
    evidence_tensor = torch.as_tensor(
        np.stack(evidence_targets), dtype=torch.float32
    )
    evidence_loss = F.binary_cross_entropy_with_logits(
        selected_evidence_logits, evidence_tensor
    )
    smoke_loss = action_loss + argument_loss + evidence_loss
    smoke_loss.backward()
    finite_gradient_parameters = sum(
        parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
    )
    parameter_tensors = sum(1 for _ in model.parameters())
    probabilities = model.action_probabilities(x, c, legal)
    illegal_probability_mass = float(probabilities[~legal].sum().detach())

    delta_counts = Counter()
    for row in evidence_targets:
        for name, value in zip(EVIDENCE_DELTA_TARGETS, row):
            delta_counts[name] += int(value)
    gate = protocol["acceptance_gate"]
    checks = {
        "episodes": len(dataset["episodes"]) == int(gate["expected_episodes"]),
        "states": len(states) == int(gate["expected_states"]),
        "exact_transitions": len(transition_rows)
        == int(gate["expected_exact_transitions"]),
        "terminal_rows": terminal_rows == int(gate["expected_terminal_rows"]),
        "zero_leakage": not leakage_failures,
        "finite_all_head_gradients": finite_gradient_parameters == parameter_tensors,
        "zero_illegal_probability_mass": illegal_probability_mass == 0.0,
        "no_planning_or_value_heads": True,
    }
    audit = {
        "protocol_sha256": _sha256(args.protocol),
        "dataset_sha256": dataset_hash,
        "episodes": len(dataset["episodes"]),
        "states": len(states),
        "exact_observed_transitions": len(transition_rows),
        "terminal_rows": terminal_rows,
        "candidate_count": len(candidates),
        "argument_key_count": len(argument_keys),
        "state_feature_size": int(state_inputs.shape[1]),
        "candidate_feature_size": int(candidate_inputs.shape[1]),
        "evidence_delta_targets": list(EVIDENCE_DELTA_TARGETS),
        "evidence_delta_positive_counts": dict(sorted(delta_counts.items())),
        "leakage_failures": leakage_failures,
        "model_parameter_count": sum(
            int(parameter.numel()) for parameter in model.parameters()
        ),
        "gradient_parameter_tensors": parameter_tensors,
        "finite_gradient_parameter_tensors": finite_gradient_parameters,
        "illegal_action_probability_mass": illegal_probability_mass,
        "smoke_losses": {
            "action": float(action_loss.detach()),
            "argument": float(argument_loss.detach()),
            "evidence": float(evidence_loss.detach()),
            "total": float(smoke_loss.detach()),
        },
        "gate_checks": checks,
        "decision": (
            "GO__FREEZE_HYBRID_ARCHITECTURE"
            if all(checks.values())
            else "NO_GO__HYBRID_ARCHITECTURE_AUDIT_FAILED"
        ),
    }
    _write_json(args.output, audit)
    print(json.dumps(audit, ensure_ascii=False, sort_keys=True))
    if not all(checks.values()):
        raise SystemExit("Stage 2 hybrid architecture gate failed")


if __name__ == "__main__":
    main()
