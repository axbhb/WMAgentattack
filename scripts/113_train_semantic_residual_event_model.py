"""Train and evaluate the semantic residual victim-event model.

The script is intentionally diagnostic: it uses existing AgentDojo-v2 sandbox
trajectories, builds every vocabulary from the training split, compares against
a candidate-aware hierarchical Markov baseline, and reports teacher-forced and
free-running metrics separately.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from wmagentattack.event_world_model import JOINT_OUTCOME_ORDER
from wmagentattack.io_utils import read_jsonl, write_jsonl
from wmagentattack.schema import TrajectoryRecord
from wmagentattack.semantic_residual_event_model import (
    SemanticResidualEventConfig,
    SemanticResidualEventWorldModel,
    build_skill_token_incidence,
)


PAD = "<PAD>"
UNK = "<UNK>"
BOS = "<BOS>"
CLEAN = "<CLEAN>"
SEMANTIC_FIELDS = (
    "attack_family",
    "attack_role",
    "trigger_stage",
    "payload_position",
    "knowledge_level",
    "endpoint_policy",
    "required_tool_depth_bucket",
)


def _vocab(values: Iterable[str], *, specials: Iterable[str]) -> dict[str, int]:
    ordered: list[str] = []
    seen: set[str] = set()
    for value in [*specials, *sorted(set(values))]:
        if value not in seen:
            seen.add(value)
            ordered.append(value)
    return {token: index for index, token in enumerate(ordered)}


def _names_by_id(vocab: dict[str, int]) -> list[str]:
    names = [""] * len(vocab)
    for name, index in vocab.items():
        names[index] = name
    return names


def _argument_signature(arguments: dict[str, Any]) -> str:
    return "+".join(sorted(str(key) for key in arguments)) or "<NO_ARGS>"


def _depth_bucket(value: Any) -> str:
    try:
        depth = int(value)
    except (TypeError, ValueError):
        return "depth_unknown"
    if depth <= 1:
        return "depth_1"
    if depth == 2:
        return "depth_2"
    if depth <= 4:
        return "depth_3_4"
    return "depth_5_plus"


def _semantic_values(
    trajectory: TrajectoryRecord, metadata: dict[str, Any]
) -> tuple[str, ...]:
    if trajectory.steps[0].attack_action is None:
        return tuple(CLEAN for _ in SEMANTIC_FIELDS)
    values = []
    for field in SEMANTIC_FIELDS:
        if field == "required_tool_depth_bucket":
            values.append(_depth_bucket(metadata.get("required_tool_depth")))
        else:
            raw = metadata.get(field)
            values.append(str(raw) if raw not in (None, "") else "unknown")
    return tuple(values)


def _groups(trajectories: Iterable[TrajectoryRecord]) -> set[tuple[str, str]]:
    return {(item.domain, item.task_id) for item in trajectories}


def _overlap(left: set[tuple[str, str]], right: set[tuple[str, str]]) -> list[str]:
    return [f"{suite}|{task}" for suite, task in sorted(left & right)]


def _metadata_map(path: Path, trajectories: list[TrajectoryRecord]) -> dict[str, dict]:
    rows = read_jsonl(path)
    mapping: dict[str, dict] = {}
    for row in rows:
        trajectory_id = str(row.get("trajectory_id", ""))
        if not trajectory_id:
            raise ValueError(f"metadata row in {path} has no trajectory_id")
        if trajectory_id in mapping:
            raise ValueError(f"duplicate metadata trajectory_id: {trajectory_id}")
        mapping[trajectory_id] = row
    expected = {item.trajectory_id for item in trajectories}
    missing = sorted(expected - set(mapping))
    if missing:
        raise ValueError(f"metadata missing {len(missing)} trajectories; first={missing[0]}")
    return {key: mapping[key] for key in expected}


def _joint_evidence(trajectory: TrajectoryRecord) -> tuple[list[float], float]:
    first = trajectory.steps[0]
    counts = first.joint_outcome_counts
    attacked = first.attack_action is not None
    if counts is None and attacked:
        key = (
            f"attack{int(trajectory.final_attack_success)}_"
            f"utility{int(trajectory.final_task_success)}"
        )
        counts = {name: int(name == key) for name in JOINT_OUTCOME_ORDER}
    values = [float((counts or {}).get(name, 0)) for name in JOINT_OUTCOME_ORDER]
    trials = float(first.multiseed_trials or 1)
    return values, (1.0 / trials if counts is not None else 0.0)


class SemanticEventDataset(Dataset):
    def __init__(
        self,
        trajectories: list[TrajectoryRecord],
        metadata: dict[str, dict],
        vocabs: dict[str, Any],
    ) -> None:
        self.trajectories = trajectories
        self.metadata = metadata
        self.vocabs = vocabs
        skill_catalog = set(vocabs["skills"])
        self.audit = {
            "trajectory_count": len(trajectories),
            "event_count": sum(len(item.steps) for item in trajectories),
            "selected_skill_oov_events": sum(
                step.selected_skill not in skill_catalog
                for item in trajectories
                for step in item.steps
            ),
            "candidate_skill_oov_mentions": sum(
                candidate not in skill_catalog
                for item in trajectories
                for step in item.steps
                for candidate in step.candidate_skills
            ),
            "future_selected_not_in_initial_candidates": sum(
                step.selected_skill not in set(item.steps[0].candidate_skills)
                for item in trajectories
                for step in item.steps
            ),
            "semantic_unknown_trajectories": 0,
            "semantic_unknown_field_mentions": 0,
        }
        for item in trajectories:
            values = _semantic_values(item, metadata[item.trajectory_id])
            unknowns = sum(
                value not in vocabs["semantics"][index]
                for index, value in enumerate(values)
            )
            self.audit["semantic_unknown_field_mentions"] += unknowns
            self.audit["semantic_unknown_trajectories"] += int(unknowns > 0)

    def __len__(self) -> int:
        return len(self.trajectories)

    def __getitem__(self, index: int) -> dict[str, Any]:
        trajectory = self.trajectories[index]
        skill_vocab = self.vocabs["skills"]
        argument_vocab = self.vocabs["arguments"]
        skills = [step.selected_skill for step in trajectory.steps]
        true_skill_ids = [
            skill_vocab.get(skill, skill_vocab[UNK]) for skill in skills
        ]
        # The final observed skill is included in the prefix used by the value
        # head.  Event losses occupy only the preceding prediction positions.
        input_skill_ids = [skill_vocab[BOS], *true_skill_ids]
        target_skill_ids = [*true_skill_ids, skill_vocab[PAD]]
        argument_targets = [
            argument_vocab.get(
                _argument_signature(step.skill_arguments), argument_vocab[UNK]
            )
            for step in trajectory.steps
        ] + [argument_vocab[PAD]]
        stop_targets = [skill == "finish" for skill in skills] + [False]

        candidate_ids: list[list[int]] = []
        for step in trajectory.steps:
            ids = {
                skill_vocab.get(candidate, skill_vocab[UNK])
                for candidate in step.candidate_skills
            }
            ids.discard(skill_vocab[PAD])
            ids.discard(skill_vocab[BOS])
            candidate_ids.append(sorted(ids))
        initial_candidate_ids = candidate_ids[0]
        # No event loss is attached to the final prefix token, but a non-empty
        # mask avoids undefined all-masked logits in diagnostics.
        candidate_ids.append(initial_candidate_ids)

        semantic_values = _semantic_values(
            trajectory, self.metadata[trajectory.trajectory_id]
        )
        semantic_ids = [
            self.vocabs["semantics"][field_index].get(
                value, self.vocabs["semantics"][field_index][UNK]
            )
            for field_index, value in enumerate(semantic_values)
        ]
        context_has_unknown = any(
            semantic_id == self.vocabs["semantics"][field_index][UNK]
            for field_index, semantic_id in enumerate(semantic_ids)
        )
        counts, joint_weight = _joint_evidence(trajectory)
        return {
            "trajectory_id": trajectory.trajectory_id,
            "domain": trajectory.domain,
            "skill_ids": input_skill_ids,
            "skill_targets": target_skill_ids,
            "argument_targets": argument_targets,
            "stop_targets": stop_targets,
            "candidate_ids": candidate_ids,
            "initial_candidate_ids": initial_candidate_ids,
            "event_length": len(skills),
            "true_skill_ids": true_skill_ids,
            "semantic_ids": semantic_ids,
            "context_has_unknown": context_has_unknown,
            "domain_id": self.vocabs["domains"].get(
                trajectory.domain, self.vocabs["domains"][UNK]
            ),
            "clean_prior": float(
                trajectory.steps[0].base_task_success_rate
                if trajectory.steps[0].base_task_success_rate is not None
                else 0.5
            ),
            "joint_counts": counts,
            "joint_weight": joint_weight,
        }


def _collate(rows: list[dict[str, Any]], *, num_skills: int) -> dict[str, Any]:
    batch = len(rows)
    length = max(len(row["skill_ids"]) for row in rows)
    skill_ids = torch.zeros(batch, length, dtype=torch.long)
    skill_targets = torch.zeros(batch, length, dtype=torch.long)
    argument_targets = torch.zeros(batch, length, dtype=torch.long)
    stop_targets = torch.zeros(batch, length, dtype=torch.float32)
    attention_mask = torch.zeros(batch, length, dtype=torch.bool)
    event_mask = torch.zeros(batch, length, dtype=torch.bool)
    candidate_mask = torch.zeros(batch, length, num_skills, dtype=torch.bool)
    initial_candidate_mask = torch.zeros(batch, num_skills, dtype=torch.bool)
    for row_index, row in enumerate(rows):
        size = len(row["skill_ids"])
        event_length = row["event_length"]
        skill_ids[row_index, :size] = torch.tensor(row["skill_ids"])
        skill_targets[row_index, :size] = torch.tensor(row["skill_targets"])
        argument_targets[row_index, :size] = torch.tensor(row["argument_targets"])
        stop_targets[row_index, :size] = torch.tensor(row["stop_targets"])
        attention_mask[row_index, :size] = True
        event_mask[row_index, :event_length] = True
        for time_index, candidates in enumerate(row["candidate_ids"]):
            candidate_mask[row_index, time_index, candidates] = True
        initial_candidate_mask[row_index, row["initial_candidate_ids"]] = True
    return {
        "trajectory_ids": [row["trajectory_id"] for row in rows],
        "domains": [row["domain"] for row in rows],
        "true_skill_ids": [row["true_skill_ids"] for row in rows],
        "event_lengths": torch.tensor([row["event_length"] for row in rows]),
        "skill_ids": skill_ids,
        "skill_targets": skill_targets,
        "argument_targets": argument_targets,
        "stop_targets": stop_targets,
        "attention_mask": attention_mask,
        "event_mask": event_mask,
        "candidate_mask": candidate_mask,
        "initial_candidate_mask": initial_candidate_mask,
        "semantic_ids": torch.tensor([row["semantic_ids"] for row in rows]),
        "context_has_unknown": torch.tensor(
            [row["context_has_unknown"] for row in rows], dtype=torch.bool
        ),
        "domain_ids": torch.tensor([row["domain_id"] for row in rows]),
        "clean_prior": torch.tensor([row["clean_prior"] for row in rows]),
        "joint_counts": torch.tensor([row["joint_counts"] for row in rows]),
        "joint_weight": torch.tensor([row["joint_weight"] for row in rows]),
    }


def _tensor_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: (value.to(device) if isinstance(value, Tensor) else value)
        for key, value in batch.items()
    }


def _levenshtein(left: list[int], right: list[int]) -> int:
    previous = list(range(len(right) + 1))
    for left_index, left_value in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_value in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + int(left_value != right_value),
                )
            )
        previous = current
    return previous[-1]


def _sequence_statistics(pairs: list[tuple[list[int], list[int]]], finish_id: int) -> dict:
    if not pairs:
        raise ValueError("sequence metric input cannot be empty")
    exact = []
    normalized_edit = []
    prefix_lengths = []
    first_event = []
    finish = []
    length_errors = []
    for truth, prediction in pairs:
        exact.append(float(truth == prediction))
        distance = _levenshtein(truth, prediction)
        normalized_edit.append(distance / max(len(truth), len(prediction), 1))
        prefix = 0
        for expected, observed in zip(truth, prediction):
            if expected != observed:
                break
            prefix += 1
        prefix_lengths.append(float(prefix))
        first_event.append(float(bool(truth and prediction and truth[0] == prediction[0])))
        finish.append(float(finish_id in prediction))
        length_errors.append(abs(len(truth) - len(prediction)))
    return {
        "trajectory_count": len(pairs),
        "exact_sequence_accuracy": statistics.fmean(exact),
        "normalized_edit_distance": statistics.fmean(normalized_edit),
        "mean_correct_prefix_length": statistics.fmean(prefix_lengths),
        "first_event_accuracy": statistics.fmean(first_event),
        "finish_rate": statistics.fmean(finish),
        "mean_absolute_length_error": statistics.fmean(length_errors),
    }


class CandidateHierarchicalMarkov:
    """Domain/backoff first-order baseline restricted to legal candidates."""

    def __init__(self, num_skills: int, *, global_alpha: float = 0.5) -> None:
        self.num_skills = num_skills
        self.global_alpha = global_alpha
        self.global_counts = torch.zeros(num_skills, dtype=torch.float64)
        self.domain_counts: dict[str, Tensor] = defaultdict(
            lambda: torch.zeros(num_skills, dtype=torch.float64)
        )
        self.context_counts: dict[tuple[str, int], Tensor] = defaultdict(
            lambda: torch.zeros(num_skills, dtype=torch.float64)
        )

    def fit(
        self,
        trajectories: list[TrajectoryRecord],
        skill_vocab: dict[str, int],
    ) -> None:
        for trajectory in trajectories:
            previous = skill_vocab[BOS]
            for step in trajectory.steps:
                target = skill_vocab.get(step.selected_skill, skill_vocab[UNK])
                self.global_counts[target] += 1
                self.domain_counts[trajectory.domain][target] += 1
                self.context_counts[(trajectory.domain, previous)][target] += 1
                previous = target

    def probabilities(
        self, domain: str, previous: int, allowed: Tensor
    ) -> Tensor:
        global_probability = self.global_counts + self.global_alpha
        global_probability = global_probability / global_probability.sum()
        domain_counts = self.domain_counts.get(domain)
        if domain_counts is None:
            domain_probability = global_probability
        else:
            domain_probability = domain_counts + 5.0 * global_probability
            domain_probability = domain_probability / domain_probability.sum()
        context_counts = self.context_counts.get((domain, previous))
        if context_counts is None:
            probability = domain_probability
        else:
            probability = context_counts + 3.0 * domain_probability
            probability = probability / probability.sum()
        probability = probability * allowed.to(torch.float64)
        if probability.sum() <= 0:
            probability = allowed.to(torch.float64)
        return probability / probability.sum()

    def teacher_metrics(
        self,
        trajectories: list[TrajectoryRecord],
        skill_vocab: dict[str, int],
    ) -> dict:
        nll = 0.0
        correct = 0
        events = 0
        for trajectory in trajectories:
            previous = skill_vocab[BOS]
            for step in trajectory.steps:
                allowed = torch.zeros(self.num_skills, dtype=torch.bool)
                for candidate in step.candidate_skills:
                    allowed[skill_vocab.get(candidate, skill_vocab[UNK])] = True
                target = skill_vocab.get(step.selected_skill, skill_vocab[UNK])
                probability = self.probabilities(trajectory.domain, previous, allowed)
                nll -= math.log(max(float(probability[target]), 1e-12))
                correct += int(int(probability.argmax()) == target)
                events += 1
                previous = target
        return {
            "event_count": events,
            "next_skill_nll": nll / max(events, 1),
            "next_skill_accuracy": correct / max(events, 1),
        }

    def free_metrics(
        self,
        trajectories: list[TrajectoryRecord],
        skill_vocab: dict[str, int],
        max_generation_steps: int,
    ) -> dict:
        finish_id = skill_vocab.get("finish", -1)
        pairs = []
        for trajectory in trajectories:
            allowed = torch.zeros(self.num_skills, dtype=torch.bool)
            for candidate in trajectory.steps[0].candidate_skills:
                allowed[skill_vocab.get(candidate, skill_vocab[UNK])] = True
            previous = skill_vocab[BOS]
            prediction: list[int] = []
            for _ in range(max_generation_steps):
                probability = self.probabilities(trajectory.domain, previous, allowed)
                selected = int(probability.argmax())
                prediction.append(selected)
                previous = selected
                if selected == finish_id:
                    break
            truth = [
                skill_vocab.get(step.selected_skill, skill_vocab[UNK])
                for step in trajectory.steps
            ]
            pairs.append((truth, prediction))
        return _sequence_statistics(pairs, finish_id)


@torch.no_grad()
def _teacher_evaluate(
    model: SemanticResidualEventWorldModel,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    model.eval()
    event_count = 0
    correct = 0
    nll_sum = 0.0
    static_concentrations = []
    dynamic_concentrations = []
    counts = []
    weights = []
    for raw in loader:
        batch = _tensor_batch(raw, device)
        outputs = model(
            batch["skill_ids"],
            batch["semantic_ids"],
            batch["domain_ids"],
            batch["clean_prior"],
            batch["attention_mask"],
            batch["candidate_mask"],
        )
        log_probabilities = outputs["next_skill_logits"].log_softmax(-1)
        selected_log_probability = log_probabilities.gather(
            -1, batch["skill_targets"].unsqueeze(-1)
        ).squeeze(-1)
        mask = batch["event_mask"]
        events = int(mask.sum())
        event_count += events
        nll_sum -= float(selected_log_probability[mask].sum())
        predictions = outputs["next_skill_logits"].argmax(-1)
        correct += int(((predictions == batch["skill_targets"]) & mask).sum())
        static_concentrations.append(outputs["static_joint_concentration"])
        dynamic_concentrations.append(outputs["dynamic_joint_concentration"])
        counts.append(batch["joint_counts"])
        weights.append(batch["joint_weight"])
    all_counts = torch.cat(counts)
    all_weights = torch.cat(weights)
    static_nll = model.dirichlet_multinomial_nll(
        torch.cat(static_concentrations), all_counts, all_weights
    )
    dynamic_nll = model.dirichlet_multinomial_nll(
        torch.cat(dynamic_concentrations), all_counts, all_weights
    )
    return {
        "event_count": event_count,
        "next_skill_nll": nll_sum / max(event_count, 1),
        "next_skill_accuracy": correct / max(event_count, 1),
        "static_joint_count_nll": float(static_nll),
        "dynamic_joint_count_nll": float(dynamic_nll),
        "dynamic_minus_static_joint_nll": float(dynamic_nll - static_nll),
    }


@torch.no_grad()
def _free_evaluate(
    model: SemanticResidualEventWorldModel,
    loader: DataLoader,
    device: torch.device,
    skill_names: list[str],
    *,
    max_generation_steps: int,
    minimum_support_probability: float,
) -> tuple[dict, list[dict]]:
    model.eval()
    skill_vocab = {name: index for index, name in enumerate(skill_names)}
    bos_id = skill_vocab[BOS]
    pad_id = skill_vocab[PAD]
    finish_id = skill_vocab.get("finish", -1)
    pairs: list[tuple[list[int], list[int]]] = []
    records: list[dict] = []
    free_concentrations = []
    counts = []
    weights = []
    max_probabilities: list[float] = []
    entropies: list[float] = []
    would_truncate = 0
    unknown_context = 0
    low_support = 0

    for raw in loader:
        batch = _tensor_batch(raw, device)
        batch_size = batch["domain_ids"].shape[0]
        sequence = torch.full((batch_size, 1), bos_id, dtype=torch.long, device=device)
        attention = torch.ones((batch_size, 1), dtype=torch.bool, device=device)
        active = torch.ones(batch_size, dtype=torch.bool, device=device)
        generated: list[list[int]] = [[] for _ in range(batch_size)]
        row_low_support = [False] * batch_size
        row_max_probabilities: list[list[float]] = [[] for _ in range(batch_size)]

        for _ in range(max_generation_steps):
            candidate_mask = batch["initial_candidate_mask"].unsqueeze(1).expand(
                -1, sequence.shape[1], -1
            )
            outputs = model(
                sequence,
                batch["semantic_ids"],
                batch["domain_ids"],
                batch["clean_prior"],
                attention,
                candidate_mask,
            )
            last_indices = attention.long().sum(1) - 1
            current_logits = outputs["next_skill_logits"][
                torch.arange(batch_size, device=device), last_indices
            ]
            probability = current_logits.softmax(-1)
            maximum, selected = probability.max(-1)
            entropy = -(probability * probability.clamp_min(1e-12).log()).sum(-1)
            append_mask = active.clone()
            for row_index in range(batch_size):
                if not bool(append_mask[row_index]):
                    continue
                selected_id = int(selected[row_index])
                probability_value = float(maximum[row_index])
                generated[row_index].append(selected_id)
                row_max_probabilities[row_index].append(probability_value)
                max_probabilities.append(probability_value)
                entropies.append(float(entropy[row_index]))
                if probability_value < minimum_support_probability:
                    row_low_support[row_index] = True
            active = active & selected.ne(finish_id)
            appended = torch.where(
                append_mask, selected, torch.full_like(selected, pad_id)
            ).unsqueeze(1)
            sequence = torch.cat([sequence, appended], dim=1)
            attention = torch.cat([attention, append_mask.unsqueeze(1)], dim=1)
            if not torch.any(active):
                break

        final_candidate_mask = batch["initial_candidate_mask"].unsqueeze(1).expand(
            -1, sequence.shape[1], -1
        )
        final_outputs = model(
            sequence,
            batch["semantic_ids"],
            batch["domain_ids"],
            batch["clean_prior"],
            attention,
            final_candidate_mask,
        )
        free_concentrations.append(final_outputs["dynamic_joint_concentration"])
        counts.append(batch["joint_counts"])
        weights.append(batch["joint_weight"])
        joint_probability = model.outcome_probabilities(
            final_outputs["dynamic_joint_concentration"]
        )["joint"]

        for row_index, trajectory_id in enumerate(batch["trajectory_ids"]):
            truth = list(batch["true_skill_ids"][row_index])
            prediction = generated[row_index]
            pairs.append((truth, prediction))
            has_unknown = bool(batch["context_has_unknown"][row_index])
            row_truncated = has_unknown or row_low_support[row_index]
            unknown_context += int(has_unknown)
            low_support += int(row_low_support[row_index])
            would_truncate += int(row_truncated)
            records.append(
                {
                    "trajectory_id": trajectory_id,
                    "true_skill_path": [skill_names[item] for item in truth],
                    "generated_skill_path": [skill_names[item] for item in prediction],
                    "context_has_unknown": has_unknown,
                    "low_support_event": row_low_support[row_index],
                    "conservative_rollout_would_truncate": row_truncated,
                    "minimum_generated_step_probability": (
                        min(row_max_probabilities[row_index])
                        if row_max_probabilities[row_index]
                        else None
                    ),
                    "free_joint_probability": {
                        name: float(joint_probability[row_index, outcome_index])
                        for outcome_index, name in enumerate(JOINT_OUTCOME_ORDER)
                    },
                }
            )

    free_joint_nll = model.dirichlet_multinomial_nll(
        torch.cat(free_concentrations), torch.cat(counts), torch.cat(weights)
    )
    metrics = _sequence_statistics(pairs, finish_id)
    metrics.update(
        {
            "free_dynamic_joint_count_nll": float(free_joint_nll),
            "mean_generated_step_max_probability": (
                statistics.fmean(max_probabilities) if max_probabilities else 0.0
            ),
            "mean_generated_step_entropy": (
                statistics.fmean(entropies) if entropies else 0.0
            ),
            "unknown_context_fraction": unknown_context / max(len(pairs), 1),
            "low_support_fraction": low_support / max(len(pairs), 1),
            "conservative_truncation_fraction": would_truncate / max(len(pairs), 1),
            "minimum_support_probability": minimum_support_probability,
        }
    )
    return metrics, records


def _constant_joint_nll(dataset: SemanticEventDataset, concentration: Tensor) -> float:
    rows = [dataset[index] for index in range(len(dataset))]
    counts = torch.tensor([row["joint_counts"] for row in rows], dtype=torch.float32)
    weights = torch.tensor([row["joint_weight"] for row in rows], dtype=torch.float32)
    expanded = concentration.reshape(1, 4).expand(len(rows), 4)
    return float(
        SemanticResidualEventWorldModel.dirichlet_multinomial_nll(
            expanded, counts, weights
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--train-metadata", type=Path, required=True)
    parser.add_argument("--validation-metadata", type=Path, required=True)
    parser.add_argument("--test-metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-generation-steps", type=int, default=16)
    parser.add_argument("--minimum-support-probability", type=float, default=0.35)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if not 0.0 < args.minimum_support_probability < 1.0:
        raise ValueError("minimum support probability must lie in (0, 1)")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    trajectories = {
        "train": [TrajectoryRecord.model_validate(row) for row in read_jsonl(args.train)],
        "validation": [
            TrajectoryRecord.model_validate(row) for row in read_jsonl(args.validation)
        ],
        "test": [TrajectoryRecord.model_validate(row) for row in read_jsonl(args.test)],
    }
    metadata = {
        "train": _metadata_map(args.train_metadata, trajectories["train"]),
        "validation": _metadata_map(
            args.validation_metadata, trajectories["validation"]
        ),
        "test": _metadata_map(args.test_metadata, trajectories["test"]),
    }
    overlaps = {
        "train_validation": _overlap(
            _groups(trajectories["train"]), _groups(trajectories["validation"])
        ),
        "train_test": _overlap(
            _groups(trajectories["train"]), _groups(trajectories["test"])
        ),
        "validation_test": _overlap(
            _groups(trajectories["validation"]), _groups(trajectories["test"])
        ),
    }
    if any(overlaps.values()):
        raise ValueError("task-group overlap detected; grouped splits are required")

    skill_candidates = [
        candidate
        for trajectory in trajectories["train"]
        for step in trajectory.steps
        for candidate in step.candidate_skills
    ]
    skill_vocab = _vocab(skill_candidates, specials=[PAD, UNK, BOS])
    skill_names = _names_by_id(skill_vocab)
    skill_token_vocab, skill_token_incidence = build_skill_token_incidence(skill_names)
    argument_vocab = _vocab(
        [
            _argument_signature(step.skill_arguments)
            for trajectory in trajectories["train"]
            for step in trajectory.steps
        ],
        specials=[PAD, UNK],
    )
    domain_vocab = _vocab(
        [item.domain for item in trajectories["train"]], specials=[UNK]
    )
    train_semantics = [
        _semantic_values(item, metadata["train"][item.trajectory_id])
        for item in trajectories["train"]
    ]
    semantic_vocabs = [
        _vocab((row[index] for row in train_semantics), specials=[UNK, CLEAN])
        for index in range(len(SEMANTIC_FIELDS))
    ]
    vocabs: dict[str, Any] = {
        "skills": skill_vocab,
        "skill_tokens": skill_token_vocab,
        "arguments": argument_vocab,
        "domains": domain_vocab,
        "semantics": semantic_vocabs,
        "semantic_fields": list(SEMANTIC_FIELDS),
    }
    datasets = {
        split: SemanticEventDataset(trajectories[split], metadata[split], vocabs)
        for split in ("train", "validation", "test")
    }
    collate = lambda rows: _collate(rows, num_skills=len(skill_vocab))
    generator = torch.Generator().manual_seed(args.seed)
    loaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate,
            generator=generator,
        ),
        "train_evaluation": DataLoader(
            datasets["train"], batch_size=args.batch_size, collate_fn=collate
        ),
        "validation": DataLoader(
            datasets["validation"], batch_size=args.batch_size, collate_fn=collate
        ),
        "test": DataLoader(
            datasets["test"], batch_size=args.batch_size, collate_fn=collate
        ),
    }

    config = SemanticResidualEventConfig(
        num_skills=len(skill_vocab),
        num_skill_tokens=len(skill_token_vocab),
        semantic_cardinalities=tuple(len(vocab) for vocab in semantic_vocabs),
        num_domains=len(domain_vocab),
        num_argument_signatures=len(argument_vocab),
        hidden_size=args.hidden_size,
        num_layers=args.layers,
        num_heads=args.heads,
        feedforward_size=2 * args.hidden_size,
        max_sequence_length=max(
            len(item.steps) + 1
            for split in trajectories.values()
            for item in split
        ),
        pad_skill_id=skill_vocab[PAD],
    )
    device = torch.device(args.device)
    model = SemanticResidualEventWorldModel(config, skill_token_incidence).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    history = []
    best_state = None
    best_score = math.inf
    best_epoch = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        training_loss = 0.0
        batches = 0
        for raw in loaders["train"]:
            batch = _tensor_batch(raw, device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(
                batch["skill_ids"],
                batch["semantic_ids"],
                batch["domain_ids"],
                batch["clean_prior"],
                batch["attention_mask"],
                batch["candidate_mask"],
            )
            losses = model.loss(
                outputs,
                event_loss_mask=batch["event_mask"],
                next_skill_targets=batch["skill_targets"],
                candidate_mask=batch["candidate_mask"],
                argument_signature_targets=batch["argument_targets"],
                stop_targets=batch["stop_targets"],
                joint_outcome_counts=batch["joint_counts"],
                joint_sample_weight=batch["joint_weight"],
            )
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            training_loss += float(losses["total"].detach())
            batches += 1
        validation = _teacher_evaluate(model, loaders["validation"], device)
        selection_score = (
            validation["next_skill_nll"]
            + 0.25 * validation["dynamic_joint_count_nll"]
        )
        history.append(
            {
                "epoch": epoch,
                "train_batch_loss": training_loss / max(batches, 1),
                "validation": validation,
                "selection_score": selection_score,
            }
        )
        if selection_score < best_score:
            best_score = selection_score
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
    if best_state is None:
        raise RuntimeError("training produced no checkpoint")
    model.load_state_dict(best_state)

    teacher_metrics = {
        "train": _teacher_evaluate(model, loaders["train_evaluation"], device),
        "validation": _teacher_evaluate(model, loaders["validation"], device),
        "test": _teacher_evaluate(model, loaders["test"], device),
    }
    free_metrics = {}
    prediction_records = {}
    for split in ("validation", "test"):
        free_metrics[split], prediction_records[split] = _free_evaluate(
            model,
            loaders[split],
            device,
            skill_names,
            max_generation_steps=args.max_generation_steps,
            minimum_support_probability=args.minimum_support_probability,
        )

    markov = CandidateHierarchicalMarkov(len(skill_vocab))
    markov.fit(trajectories["train"], skill_vocab)
    markov_metrics = {
        split: {
            "teacher": markov.teacher_metrics(trajectories[split], skill_vocab),
            "free": markov.free_metrics(
                trajectories[split], skill_vocab, args.max_generation_steps
            ),
        }
        for split in ("validation", "test")
    }

    weighted_counts = torch.zeros(4, dtype=torch.float32)
    for index in range(len(datasets["train"])):
        row = datasets["train"][index]
        weighted_counts += (
            torch.tensor(row["joint_counts"], dtype=torch.float32)
            * float(row["joint_weight"])
        )
    constant_concentration = weighted_counts + 0.5
    constant_nll = {
        split: _constant_joint_nll(dataset, constant_concentration)
        for split, dataset in datasets.items()
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split, records in prediction_records.items():
        write_jsonl(args.output_dir / f"{split}_free_predictions.jsonl", records)
    checkpoint_path = args.output_dir / "semantic_residual_event_model.pt"
    torch.save(
        {
            "model_state": best_state,
            "config": model.export_config(),
            "vocabs": vocabs,
            "skill_token_incidence": skill_token_incidence,
        },
        checkpoint_path,
    )
    report = {
        "scope": "existing AgentDojo-v2 synthetic sandbox trajectories only",
        "confirmatory": False,
        "reason_non_confirmatory": (
            "The frozen unseen-seed clean eligibility gate remains NO-GO; this "
            "run tests architecture behavior and cannot authorize new attack data."
        ),
        "known_transition_model": "exact AgentDojo simulator; not learned here",
        "evaluation_boundary": {
            "teacher_forced": "per-step candidate set from the observed exact state",
            "free_running": "initial label-blind candidate set; no future-state candidate leakage",
            "free_running_is_not_h2_simulation": True,
        },
        "semantic_fields": list(SEMANTIC_FIELDS),
        "config": model.export_config(),
        "training": {
            "seed": args.seed,
            "epochs": args.epochs,
            "best_epoch": best_epoch,
            "best_validation_selection_score": best_score,
            "selection_rule_frozen": "next_skill_nll + 0.25 * dynamic_joint_count_nll",
            "history": history,
        },
        "data": {
            "split_sizes": {key: len(value) for key, value in trajectories.items()},
            "task_group_overlap": overlaps,
            "vocabulary_source": "training candidate_skills only",
            "dataset_audit": {key: value.audit for key, value in datasets.items()},
            "skill_vocabulary": skill_names,
            "skill_token_vocabulary": _names_by_id(skill_token_vocab),
        },
        "metrics": {
            split: {
                "teacher": teacher_metrics[split],
                **({"free": free_metrics[split]} if split in free_metrics else {}),
            }
            for split in ("train", "validation", "test")
        },
        "baselines": {
            "candidate_hierarchical_markov": {
                "global_alpha": markov.global_alpha,
                "domain_prior_strength": 5.0,
                "context_prior_strength": 3.0,
                "metrics": markov_metrics,
            },
            "joint_constant": {
                "training_concentration": constant_concentration.tolist(),
                "count_nll": constant_nll,
            },
        },
        "checkpoint": str(checkpoint_path.resolve()),
    }
    metrics_path = args.output_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
