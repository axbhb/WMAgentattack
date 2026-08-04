"""Task-balanced, task-macro ablations for the semantic event model.

This is a development-only experiment on the frozen AgentDojo-v2 archive.  It
keeps the candidate ontology fixed and isolates three learned conditions:

* ``length_semantic``: semantic context and position/length, no event content;
* ``event_no_attack_semantics``: event history, domain, and clean prior, with
  all attack-semantic fields masked;
* ``semantic_event``: the complete semantic residual event model.

The full condition additionally evaluates label-blind, length-matched prefix
controls for the joint utility/security head.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from wmagentattack.io_utils import read_jsonl, write_jsonl
from wmagentattack.schema import TrajectoryRecord
from wmagentattack.semantic_residual_event_model import (
    SemanticResidualEventConfig,
    SemanticResidualEventWorldModel,
    build_skill_token_incidence,
)


BASE_SPEC = importlib.util.spec_from_file_location(
    "semantic_event_base_113",
    ROOT / "scripts" / "113_train_semantic_residual_event_model.py",
)
BASE = importlib.util.module_from_spec(BASE_SPEC)
assert BASE_SPEC.loader is not None
BASE_SPEC.loader.exec_module(BASE)


VARIANTS = ("length_semantic", "event_no_attack_semantics", "semantic_event")


def _task_key(trajectory: TrajectoryRecord) -> str:
    return f"{trajectory.domain}|{trajectory.task_id}"


class TaskBalancedDataset(Dataset):
    def __init__(self, trajectories, metadata, vocabs) -> None:
        self.base = BASE.SemanticEventDataset(trajectories, metadata, vocabs)
        self.trajectories = trajectories
        task_trajectories: dict[str, int] = defaultdict(int)
        task_events: dict[str, int] = defaultdict(int)
        task_outcome_support: dict[str, float] = defaultdict(float)
        for trajectory in trajectories:
            key = _task_key(trajectory)
            task_trajectories[key] += 1
            task_events[key] += len(trajectory.steps)
            _, joint_weight = BASE._joint_evidence(trajectory)
            task_outcome_support[key] += float(joint_weight)
        self.task_trajectories = dict(task_trajectories)
        self.task_events = dict(task_events)
        self.task_outcome_support = dict(task_outcome_support)
        self.task_count = len(task_trajectories)
        self.audit = {
            **self.base.audit,
            "task_count": self.task_count,
            "trajectories_per_task": dict(sorted(task_trajectories.items())),
            "events_per_task": dict(sorted(task_events.items())),
            "joint_support_weight_per_task": dict(
                sorted(task_outcome_support.items())
            ),
        }

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        row = dict(self.base[index])
        trajectory = self.trajectories[index]
        key = _task_key(trajectory)
        row["task_group"] = key
        row["event_task_weight"] = 1.0 / (
            self.task_count * self.task_events[key]
        )
        row["trajectory_task_weight"] = 1.0 / (
            self.task_count * self.task_trajectories[key]
        )
        support = self.task_outcome_support[key]
        row["outcome_task_weight"] = (
            1.0 / (self.task_count * support) if support > 0 else 0.0
        )
        row["sampling_population_size"] = len(self)
        return row


def _collate(rows, *, num_skills: int):
    batch = BASE._collate(rows, num_skills=num_skills)
    event_weights = torch.zeros_like(batch["event_mask"], dtype=torch.float32)
    for index, row in enumerate(rows):
        event_weights[index, : row["event_length"]] = row["event_task_weight"]
    batch.update(
        {
            "task_groups": [row["task_group"] for row in rows],
            "event_task_weights": event_weights,
            "trajectory_task_weights": torch.tensor(
                [row["trajectory_task_weight"] for row in rows],
                dtype=torch.float32,
            ),
            "outcome_task_weights": torch.tensor(
                [row["outcome_task_weight"] for row in rows],
                dtype=torch.float32,
            ),
            "sampling_population_size": rows[0]["sampling_population_size"],
        }
    )
    return batch


def _variant_inputs(
    batch: dict[str, Any],
    variant: str,
    *,
    bos_id: int,
    semantic_unknown_ids: Tensor,
) -> tuple[Tensor, Tensor]:
    skill_ids = batch["skill_ids"]
    semantic_ids = batch["semantic_ids"]
    if variant == "length_semantic":
        skill_ids = torch.where(
            batch["attention_mask"], torch.full_like(skill_ids, bos_id), skill_ids
        )
    elif variant == "event_no_attack_semantics":
        semantic_ids = semantic_unknown_ids.reshape(1, -1).expand_as(semantic_ids)
    elif variant != "semantic_event":
        raise ValueError(f"unknown variant: {variant}")
    return skill_ids, semantic_ids


class VariantAdapter(nn.Module):
    def __init__(self, model, variant, bos_id, semantic_unknown_ids) -> None:
        super().__init__()
        self.model = model
        self.variant = variant
        self.bos_id = bos_id
        self.register_buffer("semantic_unknown_ids", semantic_unknown_ids.clone())

    def forward(
        self,
        skill_ids,
        semantic_ids,
        domain_ids,
        clean_prior,
        attention_mask=None,
        candidate_mask=None,
    ):
        batch = {
            "skill_ids": skill_ids,
            "semantic_ids": semantic_ids,
            "attention_mask": attention_mask,
        }
        transformed_skills, transformed_semantics = _variant_inputs(
            batch,
            self.variant,
            bos_id=self.bos_id,
            semantic_unknown_ids=self.semantic_unknown_ids,
        )
        return self.model(
            transformed_skills,
            transformed_semantics,
            domain_ids,
            clean_prior,
            attention_mask,
            candidate_mask,
        )

    @staticmethod
    def dirichlet_multinomial_nll(concentration, counts, sample_weight=None):
        return SemanticResidualEventWorldModel.dirichlet_multinomial_nll(
            concentration, counts, sample_weight
        )

    @staticmethod
    def outcome_probabilities(concentration):
        return SemanticResidualEventWorldModel.outcome_probabilities(concentration)


def _dm_nll_rows(concentration: Tensor, counts: Tensor) -> tuple[Tensor, Tensor]:
    counts = counts.to(concentration.dtype)
    trials = counts.sum(-1)
    valid = trials > 0
    safe_counts = torch.where(valid.unsqueeze(-1), counts, torch.zeros_like(counts))
    safe_trials = torch.where(valid, trials, torch.ones_like(trials))
    log_probability = torch.lgamma(safe_trials + 1.0)
    log_probability = log_probability - torch.lgamma(safe_counts + 1.0).sum(-1)
    log_probability = log_probability + torch.lgamma(concentration.sum(-1))
    log_probability = log_probability - torch.lgamma(
        concentration.sum(-1) + safe_trials
    )
    log_probability = log_probability + (
        torch.lgamma(concentration + safe_counts) - torch.lgamma(concentration)
    ).sum(-1)
    return -log_probability, valid


def _joint_log_score_rows(concentration: Tensor, counts: Tensor) -> tuple[Tensor, Tensor]:
    trials = counts.sum(-1)
    valid = trials > 0
    empirical = counts / trials.clamp_min(1.0).unsqueeze(-1)
    probability = concentration / concentration.sum(-1, keepdim=True)
    score = -(empirical * probability.clamp_min(1e-12).log()).sum(-1)
    return score, valid


def _unbiased_task_macro(values: Tensor, weights: Tensor, batch: dict) -> Tensor:
    """Estimate an equal-task objective under uniform trajectory sampling."""

    batch_size = int(batch["trajectory_task_weights"].shape[0])
    scale = float(batch["sampling_population_size"]) / max(batch_size, 1)
    return scale * (values * weights).sum()


def _task_balanced_loss(model, outputs, batch) -> dict[str, Tensor]:
    event_mask = batch["event_mask"]
    event_weights = batch["event_task_weights"] * event_mask
    target_allowed = batch["candidate_mask"].gather(
        -1, batch["skill_targets"].unsqueeze(-1)
    ).squeeze(-1)
    if torch.any(event_mask & ~target_allowed):
        raise ValueError("target missing from current candidate set")
    skill_rows = F.cross_entropy(
        outputs["next_skill_logits"].transpose(1, 2),
        batch["skill_targets"],
        reduction="none",
    )
    argument_rows = F.cross_entropy(
        outputs["argument_signature_logits"].transpose(1, 2),
        batch["argument_targets"],
        reduction="none",
    )
    stop_rows = F.binary_cross_entropy_with_logits(
        outputs["stop_logits"], batch["stop_targets"], reduction="none"
    )
    static_rows, valid = _dm_nll_rows(
        outputs["static_joint_concentration"], batch["joint_counts"]
    )
    dynamic_rows, _ = _dm_nll_rows(
        outputs["dynamic_joint_concentration"], batch["joint_counts"]
    )
    outcome_weights = batch["joint_weight"] * batch["outcome_task_weights"] * valid
    residual_rows = outputs["dynamic_joint_logit_residual"].square().mean(-1)
    trajectory_weights = batch["trajectory_task_weights"]
    components = {
        "skill": _unbiased_task_macro(skill_rows, event_weights, batch),
        "argument": _unbiased_task_macro(argument_rows, event_weights, batch),
        "stop": _unbiased_task_macro(stop_rows, event_weights, batch),
        "static_joint": _unbiased_task_macro(static_rows, outcome_weights, batch),
        "dynamic_joint": _unbiased_task_macro(dynamic_rows, outcome_weights, batch),
        "dynamic_residual_penalty": _unbiased_task_macro(
            residual_rows, trajectory_weights, batch
        ),
    }
    total = (
        components["skill"]
        + 0.1 * components["argument"]
        + 0.1 * components["stop"]
        + 0.25 * components["static_joint"]
        + 0.5 * components["dynamic_joint"]
        + 0.01 * components["dynamic_residual_penalty"]
    )
    return {"total": total, **components}


def _aggregate_task_rows(task_rows: dict[str, dict]) -> dict:
    per_task = {}
    for task, row in sorted(task_rows.items()):
        event_count = row["event_count"]
        outcome_weight = row["outcome_weight"]
        per_task[task] = {
            "event_count": event_count,
            "next_skill_nll": row["event_nll"] / max(event_count, 1),
            "next_skill_accuracy": row["event_correct"] / max(event_count, 1),
            "static_joint_count_nll": row["static_nll"] / max(outcome_weight, 1e-12),
            "dynamic_joint_count_nll": row["dynamic_nll"] / max(outcome_weight, 1e-12),
            "static_joint_log_score": row["static_log_score"]
            / max(outcome_weight, 1e-12),
            "dynamic_joint_log_score": row["dynamic_log_score"]
            / max(outcome_weight, 1e-12),
        }
        per_task[task]["dynamic_minus_static_joint_nll"] = (
            per_task[task]["dynamic_joint_count_nll"]
            - per_task[task]["static_joint_count_nll"]
        )
    macro_keys = (
        "next_skill_nll",
        "next_skill_accuracy",
        "static_joint_count_nll",
        "dynamic_joint_count_nll",
        "static_joint_log_score",
        "dynamic_joint_log_score",
        "dynamic_minus_static_joint_nll",
    )
    return {
        "task_count": len(per_task),
        "task_macro": {
            key: statistics.fmean(row[key] for row in per_task.values())
            for key in macro_keys
        },
        "per_task": per_task,
    }


@torch.no_grad()
def _teacher_task_macro(model, loader, device, variant, bos_id, semantic_unknown_ids):
    model.eval()
    task_rows: dict[str, dict] = defaultdict(
        lambda: {
            "event_count": 0,
            "event_nll": 0.0,
            "event_correct": 0,
            "outcome_weight": 0.0,
            "static_nll": 0.0,
            "dynamic_nll": 0.0,
            "static_log_score": 0.0,
            "dynamic_log_score": 0.0,
        }
    )
    micro_events = micro_correct = 0
    micro_nll = 0.0
    for raw in loader:
        batch = BASE._tensor_batch(raw, device)
        skill_ids, semantic_ids = _variant_inputs(
            batch,
            variant,
            bos_id=bos_id,
            semantic_unknown_ids=semantic_unknown_ids,
        )
        outputs = model(
            skill_ids,
            semantic_ids,
            batch["domain_ids"],
            batch["clean_prior"],
            batch["attention_mask"],
            batch["candidate_mask"],
        )
        log_probabilities = outputs["next_skill_logits"].log_softmax(-1)
        selected_logp = log_probabilities.gather(
            -1, batch["skill_targets"].unsqueeze(-1)
        ).squeeze(-1)
        predictions = outputs["next_skill_logits"].argmax(-1)
        static_nll, valid = _dm_nll_rows(
            outputs["static_joint_concentration"], batch["joint_counts"]
        )
        dynamic_nll, _ = _dm_nll_rows(
            outputs["dynamic_joint_concentration"], batch["joint_counts"]
        )
        static_score, _ = _joint_log_score_rows(
            outputs["static_joint_concentration"], batch["joint_counts"]
        )
        dynamic_score, _ = _joint_log_score_rows(
            outputs["dynamic_joint_concentration"], batch["joint_counts"]
        )
        for index, task in enumerate(batch["task_groups"]):
            mask = batch["event_mask"][index]
            events = int(mask.sum())
            nll = -float(selected_logp[index][mask].sum())
            correct = int((predictions[index][mask] == batch["skill_targets"][index][mask]).sum())
            weight = float(batch["joint_weight"][index] * valid[index])
            row = task_rows[task]
            row["event_count"] += events
            row["event_nll"] += nll
            row["event_correct"] += correct
            row["outcome_weight"] += weight
            row["static_nll"] += float(static_nll[index]) * weight
            row["dynamic_nll"] += float(dynamic_nll[index]) * weight
            row["static_log_score"] += float(static_score[index]) * weight
            row["dynamic_log_score"] += float(dynamic_score[index]) * weight
            micro_events += events
            micro_nll += nll
            micro_correct += correct
    report = _aggregate_task_rows(task_rows)
    report["micro"] = {
        "event_count": micro_events,
        "next_skill_nll": micro_nll / max(micro_events, 1),
        "next_skill_accuracy": micro_correct / max(micro_events, 1),
    }
    return report


def _sequence_task_macro(records, task_by_trajectory):
    task_rows: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        truth = record["true_skill_path"]
        prediction = record["generated_skill_path"]
        task_rows[task_by_trajectory[record["trajectory_id"]]].append(
            {
                "exact": float(truth == prediction),
                "edit": BASE._levenshtein(truth, prediction)
                / max(len(truth), len(prediction), 1),
            }
        )
    per_task = {
        task: {
            "trajectory_count": len(rows),
            "exact_sequence_accuracy": statistics.fmean(row["exact"] for row in rows),
            "normalized_edit_distance": statistics.fmean(row["edit"] for row in rows),
        }
        for task, rows in sorted(task_rows.items())
    }
    return {
        "task_count": len(per_task),
        "task_macro": {
            "exact_sequence_accuracy": statistics.fmean(
                row["exact_sequence_accuracy"] for row in per_task.values()
            ),
            "normalized_edit_distance": statistics.fmean(
                row["normalized_edit_distance"] for row in per_task.values()
            ),
        },
        "per_task": per_task,
    }


class CandidateFactorizedSemanticMarkov:
    """Field-factorized semantic backoff over the candidate Markov baseline."""

    def __init__(self, num_skills: int) -> None:
        self.base = BASE.CandidateHierarchicalMarkov(num_skills)
        self.num_skills = num_skills
        self.field_counts: dict[tuple[int, str], Tensor] = defaultdict(
            lambda: torch.zeros(num_skills, dtype=torch.float64)
        )
        self.field_context_counts: dict[tuple[int, str, int], Tensor] = defaultdict(
            lambda: torch.zeros(num_skills, dtype=torch.float64)
        )

    def fit(self, trajectories, semantic_contexts, skill_vocab) -> None:
        self.base.fit(trajectories, skill_vocab)
        field_scale = 1.0 / len(BASE.SEMANTIC_FIELDS)
        for trajectory in trajectories:
            semantics = semantic_contexts[trajectory.trajectory_id]
            previous = skill_vocab[BASE.BOS]
            for step in trajectory.steps:
                target = skill_vocab.get(step.selected_skill, skill_vocab[BASE.UNK])
                for field_index, value in enumerate(semantics):
                    self.field_counts[(field_index, value)][target] += field_scale
                    self.field_context_counts[
                        (field_index, value, previous)
                    ][target] += field_scale
                previous = target

    def probabilities(self, domain, semantics, previous, allowed) -> Tensor:
        base_probability = self.base.probabilities(domain, previous, allowed)
        semantic_counts = torch.zeros(self.num_skills, dtype=torch.float64)
        context_counts = torch.zeros(self.num_skills, dtype=torch.float64)
        for field_index, value in enumerate(semantics):
            semantic_counts += self.field_counts.get(
                (field_index, value), torch.zeros_like(semantic_counts)
            )
            context_counts += self.field_context_counts.get(
                (field_index, value, previous), torch.zeros_like(context_counts)
            )
        semantic_probability = semantic_counts + 5.0 * base_probability
        semantic_probability = semantic_probability / semantic_probability.sum()
        probability = context_counts + 3.0 * semantic_probability
        probability = probability * allowed.to(torch.float64)
        if probability.sum() <= 0:
            probability = base_probability
        return probability / probability.sum()


def _baseline_probability(
    markov,
    trajectory,
    previous,
    allowed,
    semantic_contexts=None,
):
    if isinstance(markov, CandidateFactorizedSemanticMarkov):
        if semantic_contexts is None:
            raise ValueError("semantic Markov requires label-blind semantic context")
        return markov.probabilities(
            trajectory.domain,
            semantic_contexts[trajectory.trajectory_id],
            previous,
            allowed,
        )
    return markov.probabilities(trajectory.domain, previous, allowed)


def _baseline_task_metrics(
    trajectories,
    skill_vocab,
    markov=None,
    semantic_contexts=None,
):
    rows = defaultdict(lambda: {"nll": 0.0, "correct": 0, "events": 0})
    for trajectory in trajectories:
        task = _task_key(trajectory)
        previous = skill_vocab[BASE.BOS]
        for step in trajectory.steps:
            allowed = torch.zeros(len(skill_vocab), dtype=torch.bool)
            for candidate in step.candidate_skills:
                allowed[skill_vocab.get(candidate, skill_vocab[BASE.UNK])] = True
            target = skill_vocab.get(step.selected_skill, skill_vocab[BASE.UNK])
            if markov is None:
                probability = allowed.to(torch.float64)
                probability = probability / probability.sum()
            else:
                probability = _baseline_probability(
                    markov,
                    trajectory,
                    previous,
                    allowed,
                    semantic_contexts,
                )
            rows[task]["nll"] -= math.log(max(float(probability[target]), 1e-12))
            rows[task]["correct"] += int(int(probability.argmax()) == target)
            rows[task]["events"] += 1
            previous = target
    per_task = {
        task: {
            "event_count": row["events"],
            "next_skill_nll": row["nll"] / row["events"],
            "next_skill_accuracy": row["correct"] / row["events"],
        }
        for task, row in sorted(rows.items())
    }
    total_events = sum(row["events"] for row in rows.values())
    return {
        "task_count": len(per_task),
        "task_macro": {
            "next_skill_nll": statistics.fmean(row["next_skill_nll"] for row in per_task.values()),
            "next_skill_accuracy": statistics.fmean(
                row["next_skill_accuracy"] for row in per_task.values()
            ),
        },
        "micro": {
            "event_count": total_events,
            "next_skill_nll": sum(row["nll"] for row in rows.values())
            / max(total_events, 1),
            "next_skill_accuracy": sum(row["correct"] for row in rows.values())
            / max(total_events, 1),
        },
        "per_task": per_task,
    }


def _baseline_free_task_metrics(
    trajectories,
    skill_vocab,
    *,
    max_generation_steps,
    markov=None,
    semantic_contexts=None,
):
    finish_id = skill_vocab.get("finish", -1)
    num_skills = len(skill_vocab)
    pairs = []
    records = []
    task_map = {}
    for trajectory in trajectories:
        allowed = torch.zeros(num_skills, dtype=torch.bool)
        for candidate in trajectory.steps[0].candidate_skills:
            allowed[skill_vocab.get(candidate, skill_vocab[BASE.UNK])] = True
        previous = skill_vocab[BASE.BOS]
        generated = []
        for _ in range(max_generation_steps):
            if markov is None:
                probability = allowed.to(torch.float64)
                probability = probability / probability.sum()
            else:
                probability = _baseline_probability(
                    markov,
                    trajectory,
                    previous,
                    allowed,
                    semantic_contexts,
                )
            previous = int(probability.argmax())
            generated.append(previous)
            if previous == finish_id:
                break
        truth = [
            skill_vocab.get(step.selected_skill, skill_vocab[BASE.UNK])
            for step in trajectory.steps
        ]
        pairs.append((truth, generated))
        records.append(
            {
                "trajectory_id": trajectory.trajectory_id,
                "true_skill_path": truth,
                "generated_skill_path": generated,
            }
        )
        task_map[trajectory.trajectory_id] = _task_key(trajectory)
    return {
        "micro": BASE._sequence_statistics(pairs, finish_id),
        **_sequence_task_macro(records, task_map),
    }


def _markov_length_matched_prefix(
    trajectory,
    markov,
    skill_vocab,
    semantic_contexts=None,
):
    previous = skill_vocab[BASE.BOS]
    allowed = torch.zeros(len(skill_vocab), dtype=torch.bool)
    for candidate in trajectory.steps[0].candidate_skills:
        allowed[skill_vocab.get(candidate, skill_vocab[BASE.UNK])] = True
    generated = []
    for _ in trajectory.steps:
        probability = _baseline_probability(
            markov,
            trajectory,
            previous,
            allowed,
            semantic_contexts,
        )
        previous = int(probability.argmax())
        generated.append(previous)
    return generated


def _controlled_prefix(
    batch,
    mode,
    *,
    bos_id,
    trajectory_rows,
    markov_prefixes,
    semantic_markov_prefixes=None,
):
    skill_ids = batch["skill_ids"].clone()
    if mode == "observed":
        return skill_ids
    for row_index, trajectory_id in enumerate(batch["trajectory_ids"]):
        length = int(batch["attention_mask"][row_index].sum())
        event_length = length - 1
        if mode == "length_only":
            replacement = [bos_id] * event_length
        elif mode == "shuffled":
            replacement = skill_ids[row_index, 1:length].tolist()
            seed = int(hashlib.sha256(trajectory_id.encode()).hexdigest()[:16], 16)
            random.Random(seed).shuffle(replacement)
        elif mode == "random_length_matched":
            candidates = torch.where(batch["initial_candidate_mask"][row_index])[0].tolist()
            seed = int(hashlib.sha256((trajectory_id + mode).encode()).hexdigest()[:16], 16)
            generator = random.Random(seed)
            replacement = [generator.choice(candidates) for _ in range(event_length)]
        elif mode == "markov_length_matched":
            replacement = markov_prefixes[trajectory_id]
        elif mode == "semantic_markov_length_matched":
            if semantic_markov_prefixes is None:
                raise ValueError("semantic Markov prefix map is required")
            replacement = semantic_markov_prefixes[trajectory_id]
        else:
            raise ValueError(f"unknown control mode: {mode}")
        skill_ids[row_index, 0] = bos_id
        skill_ids[row_index, 1:length] = torch.tensor(
            replacement, device=skill_ids.device, dtype=skill_ids.dtype
        )
    return skill_ids


@torch.no_grad()
def _prefix_value_controls(
    model,
    loader,
    device,
    *,
    bos_id,
    trajectory_rows,
    markov_prefixes,
    semantic_markov_prefixes,
):
    model.eval()
    modes = (
        "observed",
        "shuffled",
        "length_only",
        "random_length_matched",
        "markov_length_matched",
        "semantic_markov_length_matched",
    )
    accumulators = {
        mode: defaultdict(lambda: {"weight": 0.0, "nll": 0.0, "score": 0.0})
        for mode in modes
    }
    static_accumulator = defaultdict(
        lambda: {"weight": 0.0, "nll": 0.0, "score": 0.0}
    )
    for raw in loader:
        batch = BASE._tensor_batch(raw, device)
        for mode in modes:
            controlled = _controlled_prefix(
                batch,
                mode,
                bos_id=bos_id,
                trajectory_rows=trajectory_rows,
                markov_prefixes=markov_prefixes,
                semantic_markov_prefixes=semantic_markov_prefixes,
            )
            outputs = model(
                controlled,
                batch["semantic_ids"],
                batch["domain_ids"],
                batch["clean_prior"],
                batch["attention_mask"],
                batch["candidate_mask"],
            )
            dynamic_nll, valid = _dm_nll_rows(
                outputs["dynamic_joint_concentration"], batch["joint_counts"]
            )
            dynamic_score, _ = _joint_log_score_rows(
                outputs["dynamic_joint_concentration"], batch["joint_counts"]
            )
            if mode == "observed":
                static_nll, _ = _dm_nll_rows(
                    outputs["static_joint_concentration"], batch["joint_counts"]
                )
                static_score, _ = _joint_log_score_rows(
                    outputs["static_joint_concentration"], batch["joint_counts"]
                )
            for index, task in enumerate(batch["task_groups"]):
                weight = float(batch["joint_weight"][index] * valid[index])
                row = accumulators[mode][task]
                row["weight"] += weight
                row["nll"] += float(dynamic_nll[index]) * weight
                row["score"] += float(dynamic_score[index]) * weight
                if mode == "observed":
                    static = static_accumulator[task]
                    static["weight"] += weight
                    static["nll"] += float(static_nll[index]) * weight
                    static["score"] += float(static_score[index]) * weight

    def summarize(rows):
        per_task = {
            task: {
                "joint_count_nll": row["nll"] / max(row["weight"], 1e-12),
                "joint_probability_log_score": row["score"]
                / max(row["weight"], 1e-12),
            }
            for task, row in sorted(rows.items())
        }
        return {
            "task_macro_joint_count_nll": statistics.fmean(
                row["joint_count_nll"] for row in per_task.values()
            ),
            "task_macro_joint_probability_log_score": statistics.fmean(
                row["joint_probability_log_score"] for row in per_task.values()
            ),
            "per_task": per_task,
        }

    return {
        "static_semantic": summarize(static_accumulator),
        **{mode: summarize(rows) for mode, rows in accumulators.items()},
        "interpretation": (
            "All controls are length matched. Shuffled preserves the event multiset; "
            "length_only removes event identity; random uses the initial candidate set; "
            "Markov uses a candidate-constrained first-order generated prefix; "
            "semantic Markov adds field-factorized attack-context backoff."
        ),
    }


def _prepare(args):
    trajectories = {
        "train": [TrajectoryRecord.model_validate(row) for row in read_jsonl(args.train)],
        "validation": [
            TrajectoryRecord.model_validate(row) for row in read_jsonl(args.validation)
        ],
        "test": [TrajectoryRecord.model_validate(row) for row in read_jsonl(args.test)],
    }
    metadata = {
        "train": BASE._metadata_map(args.train_metadata, trajectories["train"]),
        "validation": BASE._metadata_map(
            args.validation_metadata, trajectories["validation"]
        ),
        "test": BASE._metadata_map(args.test_metadata, trajectories["test"]),
    }
    overlaps = {
        "train_validation": BASE._overlap(
            BASE._groups(trajectories["train"]), BASE._groups(trajectories["validation"])
        ),
        "train_test": BASE._overlap(
            BASE._groups(trajectories["train"]), BASE._groups(trajectories["test"])
        ),
        "validation_test": BASE._overlap(
            BASE._groups(trajectories["validation"]), BASE._groups(trajectories["test"])
        ),
    }
    if any(overlaps.values()):
        raise ValueError("task-group overlap detected")
    skill_candidates = [
        candidate
        for trajectory in trajectories["train"]
        for step in trajectory.steps
        for candidate in step.candidate_skills
    ]
    skill_vocab = BASE._vocab(skill_candidates, specials=[BASE.PAD, BASE.UNK, BASE.BOS])
    skill_names = BASE._names_by_id(skill_vocab)
    skill_token_vocab, incidence = build_skill_token_incidence(skill_names)
    argument_vocab = BASE._vocab(
        [
            BASE._argument_signature(step.skill_arguments)
            for trajectory in trajectories["train"]
            for step in trajectory.steps
        ],
        specials=[BASE.PAD, BASE.UNK],
    )
    domain_vocab = BASE._vocab(
        [item.domain for item in trajectories["train"]], specials=[BASE.UNK]
    )
    train_semantics = [
        BASE._semantic_values(item, metadata["train"][item.trajectory_id])
        for item in trajectories["train"]
    ]
    semantic_vocabs = [
        BASE._vocab(
            (row[index] for row in train_semantics), specials=[BASE.UNK, BASE.CLEAN]
        )
        for index in range(len(BASE.SEMANTIC_FIELDS))
    ]
    vocabs = {
        "skills": skill_vocab,
        "skill_tokens": skill_token_vocab,
        "arguments": argument_vocab,
        "domains": domain_vocab,
        "semantics": semantic_vocabs,
        "semantic_fields": list(BASE.SEMANTIC_FIELDS),
    }
    datasets = {
        split: TaskBalancedDataset(trajectories[split], metadata[split], vocabs)
        for split in ("train", "validation", "test")
    }
    return trajectories, metadata, overlaps, vocabs, incidence, datasets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--train-metadata", type=Path, required=True)
    parser.add_argument("--validation-metadata", type=Path, required=True)
    parser.add_argument("--test-metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-generation-steps", type=int, default=16)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--run-prefix-controls", action="store_true")
    args = parser.parse_args()
    if args.run_prefix_controls and args.variant != "semantic_event":
        raise ValueError("prefix controls are defined only for semantic_event")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    trajectories, metadata, overlaps, vocabs, incidence, datasets = _prepare(args)
    skill_vocab = vocabs["skills"]
    skill_names = BASE._names_by_id(skill_vocab)
    bos_id = skill_vocab[BASE.BOS]
    semantic_unknown_ids = torch.tensor(
        [vocab[BASE.UNK] for vocab in vocabs["semantics"]], dtype=torch.long
    )
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
        num_skill_tokens=incidence.shape[1],
        semantic_cardinalities=tuple(len(vocab) for vocab in vocabs["semantics"]),
        num_domains=len(vocabs["domains"]),
        num_argument_signatures=len(vocabs["arguments"]),
        hidden_size=args.hidden_size,
        num_layers=args.layers,
        num_heads=args.heads,
        feedforward_size=2 * args.hidden_size,
        max_sequence_length=max(
            args.max_generation_steps + 1,
            max(
                len(item.steps) + 1
                for rows in trajectories.values()
                for item in rows
            ),
        ),
        pad_skill_id=skill_vocab[BASE.PAD],
    )
    device = torch.device(args.device)
    semantic_unknown_ids = semantic_unknown_ids.to(device)
    model = SemanticResidualEventWorldModel(config, incidence).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    best_state = None
    best_score = math.inf
    best_epoch = None
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        batches = 0
        for raw in loaders["train"]:
            batch = BASE._tensor_batch(raw, device)
            skill_ids, semantic_ids = _variant_inputs(
                batch,
                args.variant,
                bos_id=bos_id,
                semantic_unknown_ids=semantic_unknown_ids,
            )
            optimizer.zero_grad(set_to_none=True)
            outputs = model(
                skill_ids,
                semantic_ids,
                batch["domain_ids"],
                batch["clean_prior"],
                batch["attention_mask"],
                batch["candidate_mask"],
            )
            losses = _task_balanced_loss(model, outputs, batch)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(losses["total"].detach())
            batches += 1
        validation = _teacher_task_macro(
            model,
            loaders["validation"],
            device,
            args.variant,
            bos_id,
            semantic_unknown_ids,
        )
        score = (
            validation["task_macro"]["next_skill_nll"]
            + 0.25 * validation["task_macro"]["dynamic_joint_count_nll"]
        )
        history.append(
            {
                "epoch": epoch,
                "training_batch_loss": total / max(batches, 1),
                "validation_task_macro": validation["task_macro"],
                "selection_score": score,
            }
        )
        if score < best_score:
            best_score = score
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
    if best_state is None:
        raise RuntimeError("no checkpoint selected")
    model.load_state_dict(best_state)

    teacher = {
        "train": _teacher_task_macro(
            model, loaders["train_evaluation"], device, args.variant, bos_id, semantic_unknown_ids
        ),
        "validation": _teacher_task_macro(
            model, loaders["validation"], device, args.variant, bos_id, semantic_unknown_ids
        ),
        "test": _teacher_task_macro(
            model, loaders["test"], device, args.variant, bos_id, semantic_unknown_ids
        ),
    }
    adapter = VariantAdapter(
        model, args.variant, bos_id, semantic_unknown_ids
    ).to(device)
    free = {}
    free_records = {}
    for split in ("validation", "test"):
        micro, records = BASE._free_evaluate(
            adapter,
            loaders[split],
            device,
            skill_names,
            max_generation_steps=args.max_generation_steps,
            minimum_support_probability=0.35,
        )
        task_map = {
            item.trajectory_id: _task_key(item) for item in trajectories[split]
        }
        free[split] = {
            "micro": micro,
            **_sequence_task_macro(records, task_map),
        }
        free_records[split] = records

    semantic_contexts = {
        split: {
            item.trajectory_id: BASE._semantic_values(
                item, metadata[split][item.trajectory_id]
            )
            for item in trajectories[split]
        }
        for split in ("train", "validation", "test")
    }
    markov = BASE.CandidateHierarchicalMarkov(len(skill_vocab))
    markov.fit(trajectories["train"], skill_vocab)
    semantic_markov = CandidateFactorizedSemanticMarkov(len(skill_vocab))
    semantic_markov.fit(
        trajectories["train"], semantic_contexts["train"], skill_vocab
    )
    baselines = {
        split: {
            "candidate_uniform": {
                "teacher": _baseline_task_metrics(
                    trajectories[split], skill_vocab, None
                ),
                "free": _baseline_free_task_metrics(
                    trajectories[split],
                    skill_vocab,
                    max_generation_steps=args.max_generation_steps,
                    markov=None,
                ),
            },
            "candidate_hierarchical_markov": {
                "teacher": _baseline_task_metrics(
                    trajectories[split], skill_vocab, markov
                ),
                "free": _baseline_free_task_metrics(
                    trajectories[split],
                    skill_vocab,
                    max_generation_steps=args.max_generation_steps,
                    markov=markov,
                ),
            },
            "candidate_factorized_semantic_markov": {
                "teacher": _baseline_task_metrics(
                    trajectories[split],
                    skill_vocab,
                    semantic_markov,
                    semantic_contexts[split],
                ),
                "free": _baseline_free_task_metrics(
                    trajectories[split],
                    skill_vocab,
                    max_generation_steps=args.max_generation_steps,
                    markov=semantic_markov,
                    semantic_contexts=semantic_contexts[split],
                ),
            },
        }
        for split in ("validation", "test")
    }
    prefix_controls = None
    if args.run_prefix_controls:
        prefix_controls = {}
        for split in ("validation", "test"):
            trajectory_rows = {
                item.trajectory_id: item for item in trajectories[split]
            }
            markov_prefixes = {
                item.trajectory_id: _markov_length_matched_prefix(
                    item, markov, skill_vocab
                )
                for item in trajectories[split]
            }
            semantic_markov_prefixes = {
                item.trajectory_id: _markov_length_matched_prefix(
                    item,
                    semantic_markov,
                    skill_vocab,
                    semantic_contexts[split],
                )
                for item in trajectories[split]
            }
            prefix_controls[split] = _prefix_value_controls(
                model,
                loaders[split],
                device,
                bos_id=bos_id,
                trajectory_rows=trajectory_rows,
                markov_prefixes=markov_prefixes,
                semantic_markov_prefixes=semantic_markov_prefixes,
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split, records in free_records.items():
        write_jsonl(args.output_dir / f"{split}_free_predictions.jsonl", records)
    checkpoint = args.output_dir / "task_macro_event_model.pt"
    torch.save(
        {
            "model_state": best_state,
            "config": model.export_config(),
            "vocabs": vocabs,
            "skill_token_incidence": incidence,
            "variant": args.variant,
        },
        checkpoint,
    )
    report = {
        "scope": "development-only task-balanced ablation on frozen AgentDojo-v2",
        "variant": args.variant,
        "confirmatory": False,
        "clean_eligibility_gate": False,
        "task_balancing": {
            "event_loss_weight": "inverse total event count in base task",
            "outcome_loss_weight": "inverse trajectory count in base task",
            "early_stopping": "task-macro next_skill_nll + 0.25 * task-macro dynamic_joint_count_nll",
        },
        "config": model.export_config(),
        "training": {
            "seed": args.seed,
            "epochs": args.epochs,
            "best_epoch": best_epoch,
            "best_selection_score": best_score,
            "history": history,
        },
        "data": {
            "task_group_overlap": overlaps,
            "dataset_audit": {split: dataset.audit for split, dataset in datasets.items()},
            "skill_vocabulary": skill_names,
            "vocabulary_source": "training candidate_skills only",
        },
        "metrics": {
            split: {
                "teacher": teacher[split],
                **({"free": free[split]} if split in free else {}),
            }
            for split in ("train", "validation", "test")
        },
        "baselines": baselines,
        "prefix_value_controls": prefix_controls,
        "checkpoint": str(checkpoint.resolve()),
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
