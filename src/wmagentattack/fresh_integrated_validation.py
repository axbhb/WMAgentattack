"""Fresh-task integrated semantic world-model validation.

The learned model shares one Structured Semantic State v3 encoder and one
candidate encoder across current-action and adjacent-transition objectives.
Only the final current-action scorer can be source-specific.  The module has
no reward, utility, security, planning, or attack-generation head.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor, nn

from .adjacent_transition import OBSERVED_OUTCOME_TARGETS
from .multisource_suitability import TEXT_ACTION, candidate_id, stable_hash


INTEGRATED_VALIDATION_SCHEMA_VERSION = (
    "wmagentattack.fresh_integrated_semantic_validation.v1"
)
FROZEN_SOURCES = ("agentdojo", "injecagent", "tool_sandbox")
STOP = TEXT_ACTION

_FORBIDDEN_CAUSAL_KEYS = {
    "attack_action",
    "attack_success",
    "decision",
    "execution",
    "future_action",
    "next_action",
    "policy_violation",
    "reward",
    "security",
    "target",
    "task_success",
    "utility",
}


def _task_name(row: Mapping[str, Any]) -> str:
    domain = row.get("suite", row.get("domain"))
    task = row.get("user_task_id", row.get("task_id"))
    if domain is None or task is None:
        raise ValueError("fresh metadata lacks domain/task identity")
    return f"{domain}|{task}"


def build_fresh_action_and_transition_rows(
    *,
    steps: Sequence[Mapping[str, Any]],
    metadata: Sequence[Mapping[str, Any]],
    historical_catalog: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any]]:
    """Convert sealed clean traces into action rows and adjacent events."""

    metadata_by_trajectory = {str(row["trajectory_id"]): row for row in metadata}
    if len(metadata_by_trajectory) != len(metadata):
        raise ValueError("duplicate fresh trajectory metadata")
    by_trajectory: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for step in steps:
        by_trajectory[str(step["trajectory_id"])].append(step)
    if set(by_trajectory) != set(metadata_by_trajectory):
        raise ValueError("fresh step and metadata trajectory sets differ")

    catalog = {key: dict(value) for key, value in historical_catalog.items()}
    action_rows: list[dict[str, Any]] = []
    action_by_trajectory_step: dict[tuple[str, int], dict[str, Any]] = {}
    all_contiguous = True
    target_legal = True
    clean_only = True
    task_names: set[str] = set()

    for trajectory_id, unsorted in sorted(by_trajectory.items()):
        trajectory_steps = sorted(unsorted, key=lambda row: int(row["step_id"]))
        indices = [int(row["step_id"]) for row in trajectory_steps]
        all_contiguous &= indices == list(range(len(indices)))
        meta = metadata_by_trajectory[trajectory_id]
        task_name = _task_name(meta)
        task_names.add(task_name)
        clean_only &= str(meta.get("attack_family", "clean")) == "clean"
        clean_only &= not bool(meta.get("security"))
        for step in trajectory_steps:
            if _task_name(step) != task_name:
                raise ValueError(f"fresh step identity mismatch: {trajectory_id}")
            descriptions = step["candidate_skill_descriptions"]
            legal: list[str] = []
            schemas: list[dict[str, Any]] = []
            for raw_skill in step["candidate_skills"]:
                skill = str(raw_skill)
                if skill == "finish":
                    continue
                schema = {
                    "type": "function",
                    "function": {
                        "name": skill,
                        "description": str(descriptions.get(skill, "")),
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
                key = candidate_id("agentdojo", skill, schema)
                descriptor = {
                    "source": "agentdojo",
                    "kind": "tool",
                    "function": schema["function"],
                }
                previous = catalog.setdefault(key, descriptor)
                if stable_hash(previous) != stable_hash(descriptor):
                    raise ValueError(f"fresh candidate schema conflict: {key}")
                legal.append(key)
                schemas.append(schema)
            text_key = candidate_id("agentdojo", STOP)
            catalog.setdefault(
                text_key,
                {
                    "source": "agentdojo",
                    "kind": "text_or_stop",
                    "function": {
                        "name": STOP,
                        "description": "Return a final textual response without another skill.",
                        "parameters": {"type": "object", "properties": {}},
                    },
                },
            )
            legal.append(text_key)
            selected = str(step["selected_skill"])
            if selected == "finish":
                target = text_key
            else:
                matches = [
                    key
                    for key, schema in zip(legal[:-1], schemas)
                    if schema["function"]["name"] == selected
                ]
                if len(matches) != 1:
                    raise ValueError(f"fresh target is not uniquely legal: {selected}")
                target = matches[0]
            target_legal &= target in legal
            previous_skills = list(step.get("previous_skills", []))
            causal = {
                "source": "agentdojo",
                "trusted_goal": str(step["user_goal"]),
                "track": f"agentdojo:{step['domain']}",
                "tool_schemas": schemas,
                "legal_tool_names": [schema["function"]["name"] for schema in schemas],
                "visible_observation": str(step.get("current_observation", "")),
                "visible_prior_tool": (
                    str(previous_skills[-1]) if previous_skills else "<START>"
                ),
            }
            step_id = int(step["step_id"])
            row = {
                "row_id": f"fresh::agentdojo::{trajectory_id}::step{step_id}",
                "source": "agentdojo",
                "task_key": stable_hash({"source": "agentdojo", "task": task_name}),
                "task_name": task_name,
                "task_cohort": "fresh_custom_confirmation_v3",
                "group_id": f"{meta['row_id']}::step{step_id}",
                "repeat_id": trajectory_id,
                "variant": "clean",
                "causal_model_input": causal,
                "causal_input_fingerprint": stable_hash(causal),
                "legal_candidate_ids": legal,
                "target_candidate_id": target,
                "target_is_tool": selected != "finish",
                "exact_outcome": {
                    "available": False,
                    "execution_error": False,
                    "state_changed": False,
                    "output_nonempty": False,
                },
            }
            action_rows.append(row)
            action_by_trajectory_step[(trajectory_id, step_id)] = row

    transition_events: list[dict[str, Any]] = []
    next_target_legal = True
    for trajectory_id, unsorted in sorted(by_trajectory.items()):
        trajectory_steps = sorted(unsorted, key=lambda row: int(row["step_id"]))
        for index, step in enumerate(trajectory_steps):
            current = action_by_trajectory_step[(trajectory_id, index)]
            following = (
                action_by_trajectory_step[(trajectory_id, index + 1)]
                if index + 1 < len(trajectory_steps)
                else None
            )
            if following is not None:
                next_target_legal &= (
                    following["target_candidate_id"] in following["legal_candidate_ids"]
                )
            transition_events.append(
                {
                    "event_id": f"{current['row_id']}::observed_transition",
                    "task_key": current["task_key"],
                    "task_name": current["task_name"],
                    "task_cohort": current["task_cohort"],
                    "group_id": str(current["group_id"]),
                    "trajectory_id": trajectory_id,
                    "step_id": index,
                    "variant": "clean",
                    "causal_model_input": current["causal_model_input"],
                    "causal_input_fingerprint": current["causal_input_fingerprint"],
                    "current_action_candidate_id": current["target_candidate_id"],
                    "current_legal_candidate_ids": current["legal_candidate_ids"],
                    "next_target_candidate_id": (
                        following["target_candidate_id"] if following is not None else None
                    ),
                    "next_legal_candidate_ids": (
                        following["legal_candidate_ids"]
                        if following is not None
                        else current["legal_candidate_ids"]
                    ),
                    "observed_outcome": {
                        "execution_error": bool(step.get("tool_error")),
                        "output_nonempty": bool(step.get("skill_output")),
                        "trajectory_continues": following is not None,
                    },
                }
            )

    forbidden = sorted(
        {
            key
            for row in (*action_rows, *transition_events)
            for key in row["causal_model_input"]
            if str(key).lower() in _FORBIDDEN_CAUSAL_KEYS
        }
    )
    audit = {
        "fresh_action_rows": len(action_rows),
        "fresh_transition_events": len(transition_events),
        "fresh_adjacent_transitions": sum(
            bool(row["observed_outcome"]["trajectory_continues"])
            for row in transition_events
        ),
        "fresh_trajectories": len(by_trajectory),
        "fresh_tasks": len(task_names),
        "all_step_indices_contiguous": all_contiguous,
        "current_targets_legal": target_legal,
        "next_targets_legal": next_target_legal,
        "clean_only": clean_only,
        "forbidden_causal_keys": forbidden,
        "candidate_count_after_fresh_merge": len(catalog),
    }
    audit["passed"] = (
        audit["fresh_trajectories"] == 36
        and audit["fresh_tasks"] == 12
        and all_contiguous
        and target_legal
        and next_target_legal
        and clean_only
        and not forbidden
    )
    return action_rows, transition_events, dict(sorted(catalog.items())), audit


class FreshIntegratedSemanticWorldModel(nn.Module):
    """Shared representation with current-action and observed-transition heads."""

    def __init__(
        self,
        *,
        state_size: int,
        candidate_size: int,
        hidden_size: int,
        source_count: int,
        source_specific_action_head: bool,
        dropout: float,
    ) -> None:
        super().__init__()
        if source_count <= 0:
            raise ValueError("source_count must be positive")
        self.source_specific_action_head = bool(source_specific_action_head)
        self.state_encoder = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
        )
        self.candidate_encoder = nn.Sequential(
            nn.Linear(candidate_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
        )
        head_count = source_count if source_specific_action_head else 1
        self.current_action_heads = nn.ModuleList(
            [nn.Linear(hidden_size, 1) for _ in range(head_count)]
        )
        self.transition_action_encoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.LayerNorm(hidden_size), nn.GELU()
        )
        self.next_action_head = nn.Linear(hidden_size, 1)
        self.outcome_head = nn.Linear(hidden_size, len(OBSERVED_OUTCOME_TARGETS))

    def _encoded(self, states: Tensor, candidates: Tensor) -> tuple[Tensor, Tensor]:
        return self.state_encoder(states), self.candidate_encoder(candidates)

    def current_action_logits(
        self, states: Tensor, candidates: Tensor, row_source_indices: Tensor
    ) -> Tensor:
        state, candidate = self._encoded(states, candidates)
        joint = torch.tanh(state[:, None, :] + candidate[None, :, :])
        logits = torch.empty(joint.shape[:2], dtype=joint.dtype, device=joint.device)
        if self.source_specific_action_head:
            for source_index, head in enumerate(self.current_action_heads):
                mask = row_source_indices == source_index
                if bool(mask.any()):
                    logits[mask] = head(joint[mask]).squeeze(-1)
        else:
            logits = self.current_action_heads[0](joint).squeeze(-1)
        return logits

    def transition_logits(
        self, states: Tensor, selected_candidate_inputs: Tensor, candidates: Tensor
    ) -> tuple[Tensor, Tensor]:
        state, candidate = self._encoded(states, candidates)
        selected = self.candidate_encoder(selected_candidate_inputs)
        context = torch.tanh(state + self.transition_action_encoder(selected))
        joint = torch.tanh(context[:, None, :] + candidate[None, :, :])
        return self.next_action_head(joint).squeeze(-1), self.outcome_head(context)

    @staticmethod
    def probabilities(logits: Tensor, legal_mask: Tensor) -> Tensor:
        if logits.shape != legal_mask.shape:
            raise ValueError("legal mask shape differs from logits")
        if not bool(torch.all(legal_mask.any(dim=1))):
            raise ValueError("every row requires at least one legal action")
        masked = logits.masked_fill(~legal_mask, torch.finfo(logits.dtype).min)
        return torch.softmax(masked, dim=1)


def assert_no_unauthorized_heads(model: nn.Module) -> None:
    forbidden = ("reward", "utility", "security", "planner", "attack", "value")
    found = [
        name
        for name, _ in model.named_modules()
        if name and any(token in name.casefold() for token in forbidden)
    ]
    if found:
        raise AssertionError(f"unauthorized model heads: {found}")
