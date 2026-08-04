"""Train the factorized victim Event Transformer on grouped trajectory files."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
from torch.utils.data import DataLoader, Dataset

from wmagentattack.event_world_model import (
    JOINT_OUTCOME_ORDER,
    EventWorldModelConfig,
    FactorizedEventWorldModel,
)
from wmagentattack.io_utils import read_jsonl
from wmagentattack.schema import TrajectoryRecord


PAD = "<PAD>"
UNK = "<UNK>"
BOS = "<BOS>"


def _vocab(values, *, specials):
    ordered = list(specials) + sorted(set(values) - set(specials))
    return {token: index for index, token in enumerate(ordered)}


def _argument_signature(arguments: dict[str, Any]) -> str:
    return "+".join(sorted(str(key) for key in arguments)) or "<NO_ARGS>"


def _attack_context(trajectory: TrajectoryRecord) -> str:
    first = trajectory.steps[0]
    return first.attack_action or "clean"


def _groups(trajectories):
    return {(item.domain, item.task_id) for item in trajectories}


def _overlap(left, right):
    return [f"{suite}|{task}" for suite, task in sorted(left & right)]


class EventDataset(Dataset):
    def __init__(self, trajectories, vocabs):
        self.rows = trajectories
        self.vocabs = vocabs

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        trajectory = self.rows[index]
        skills = [step.selected_skill for step in trajectory.steps]
        tool_vocab = self.vocabs["tools"]
        arg_vocab = self.vocabs["arguments"]
        input_tools = [BOS, *skills[:-1]]
        first = trajectory.steps[0]
        counts = first.joint_outcome_counts
        attacked = first.attack_action is not None
        if counts is None and attacked:
            key = (
                f"attack{int(trajectory.final_attack_success)}_"
                f"utility{int(trajectory.final_task_success)}"
            )
            counts = {name: int(name == key) for name in JOINT_OUTCOME_ORDER}
        count_values = [float((counts or {}).get(name, 0)) for name in JOINT_OUTCOME_ORDER]
        repeated_trials = float(first.multiseed_trials or 1)
        joint_weight = 1.0 / repeated_trials if counts is not None else 0.0
        return {
            "tool_ids": [tool_vocab.get(item, tool_vocab[UNK]) for item in input_tools],
            "tool_targets": [tool_vocab.get(item, tool_vocab[UNK]) for item in skills],
            "argument_targets": [
                arg_vocab.get(_argument_signature(step.skill_arguments), arg_vocab[UNK])
                for step in trajectory.steps
            ],
            "stop_targets": [item == "finish" for item in skills],
            "attack_id": self.vocabs["attacks"].get(
                _attack_context(trajectory), self.vocabs["attacks"][UNK]
            ),
            "domain_id": self.vocabs["domains"].get(
                trajectory.domain, self.vocabs["domains"][UNK]
            ),
            "clean_prior": float(
                first.base_task_success_rate
                if first.base_task_success_rate is not None
                else 0.5
            ),
            "joint_counts": count_values,
            "joint_weight": joint_weight,
            "utility_residual": (
                float(first.attack_utility_logit_residual_target)
                if first.attack_utility_logit_residual_target is not None
                else math.nan
            ),
        }


def _collate(rows):
    batch = len(rows)
    length = max(len(row["tool_ids"]) for row in rows)
    tool_ids = torch.zeros(batch, length, dtype=torch.long)
    tool_targets = torch.zeros(batch, length, dtype=torch.long)
    argument_targets = torch.zeros(batch, length, dtype=torch.long)
    stop_targets = torch.zeros(batch, length, dtype=torch.float32)
    mask = torch.zeros(batch, length, dtype=torch.bool)
    for index, row in enumerate(rows):
        size = len(row["tool_ids"])
        tool_ids[index, :size] = torch.tensor(row["tool_ids"])
        tool_targets[index, :size] = torch.tensor(row["tool_targets"])
        argument_targets[index, :size] = torch.tensor(row["argument_targets"])
        stop_targets[index, :size] = torch.tensor(row["stop_targets"])
        mask[index, :size] = True
    return {
        "tool_ids": tool_ids,
        "tool_targets": tool_targets,
        "argument_targets": argument_targets,
        "stop_targets": stop_targets,
        "attention_mask": mask,
        "attack_ids": torch.tensor([row["attack_id"] for row in rows]),
        "domain_ids": torch.tensor([row["domain_id"] for row in rows]),
        "clean_prior": torch.tensor([row["clean_prior"] for row in rows]),
        "joint_counts": torch.tensor([row["joint_counts"] for row in rows]),
        "joint_weight": torch.tensor([row["joint_weight"] for row in rows]),
        "utility_residual": torch.tensor([row["utility_residual"] for row in rows]),
    }


def _move(batch, device):
    return {key: value.to(device) for key, value in batch.items()}


@torch.no_grad()
def _evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_events = 0
    correct = 0
    joint_concentrations = []
    joint_counts = []
    joint_weights = []
    for raw in loader:
        batch = _move(raw, device)
        outputs = model(
            batch["tool_ids"],
            batch["attack_ids"],
            batch["domain_ids"],
            batch["clean_prior"],
            batch["attention_mask"],
        )
        losses = model.loss(
            outputs,
            attention_mask=batch["attention_mask"],
            next_tool_targets=batch["tool_targets"],
            argument_signature_targets=batch["argument_targets"],
            stop_targets=batch["stop_targets"],
            joint_outcome_counts=batch["joint_counts"],
            joint_sample_weight=batch["joint_weight"],
            utility_residual_targets=batch["utility_residual"],
        )
        events = int(batch["attention_mask"].sum())
        total_loss += float(losses["total"]) * events
        total_events += events
        predicted = outputs["next_tool_logits"].argmax(-1)
        correct += int(
            ((predicted == batch["tool_targets"]) & batch["attention_mask"]).sum()
        )
        joint_concentrations.append(outputs["joint_concentration"].detach())
        joint_counts.append(batch["joint_counts"])
        joint_weights.append(batch["joint_weight"])
    global_joint_nll = model.dirichlet_multinomial_nll(
        torch.cat(joint_concentrations),
        torch.cat(joint_counts),
        torch.cat(joint_weights),
    )
    return {
        "event_count": total_events,
        "loss_per_event": total_loss / max(total_events, 1),
        "next_tool_accuracy": correct / max(total_events, 1),
        "joint_count_nll": float(global_joint_nll),
    }


def _constant_joint_nll(dataset, concentration):
    rows = [dataset[index] for index in range(len(dataset))]
    counts = torch.tensor([row["joint_counts"] for row in rows], dtype=torch.float32)
    weights = torch.tensor([row["joint_weight"] for row in rows], dtype=torch.float32)
    expanded = concentration.reshape(1, 4).expand(len(rows), 4)
    return float(
        FactorizedEventWorldModel.dirichlet_multinomial_nll(
            expanded, counts, weights
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--test", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--smoke-allow-task-overlap", action="store_true")
    args = parser.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    train = [TrajectoryRecord.model_validate(row) for row in read_jsonl(args.train)]
    validation = [
        TrajectoryRecord.model_validate(row) for row in read_jsonl(args.validation)
    ]
    test = (
        [TrajectoryRecord.model_validate(row) for row in read_jsonl(args.test)]
        if args.test
        else []
    )
    overlaps = {
        "train_validation": _overlap(_groups(train), _groups(validation)),
        "train_test": _overlap(_groups(train), _groups(test)),
        "validation_test": _overlap(_groups(validation), _groups(test)),
    }
    if any(overlaps.values()) and not args.smoke_allow_task_overlap:
        raise ValueError("Task-group overlap detected; grouped split is required")

    vocabs = {
        "tools": _vocab(
            [step.selected_skill for item in train for step in item.steps],
            specials=[PAD, UNK, BOS],
        ),
        "arguments": _vocab(
            [_argument_signature(step.skill_arguments) for item in train for step in item.steps],
            specials=[PAD, UNK],
        ),
        "attacks": _vocab([_attack_context(item) for item in train], specials=[UNK]),
        "domains": _vocab([item.domain for item in train], specials=[UNK]),
    }
    datasets = {
        "train": EventDataset(train, vocabs),
        "validation": EventDataset(validation, vocabs),
        "test": EventDataset(test, vocabs) if test else None,
    }
    weighted_train_counts = torch.zeros(4, dtype=torch.float32)
    for index in range(len(datasets["train"])):
        row = datasets["train"][index]
        weighted_train_counts += (
            torch.tensor(row["joint_counts"], dtype=torch.float32)
            * float(row["joint_weight"])
        )
    constant_joint_concentration = weighted_train_counts + 0.5
    constant_joint_baseline = {
        name: _constant_joint_nll(dataset, constant_joint_concentration)
        for name, dataset in datasets.items()
        if dataset is not None
    }
    generator = torch.Generator().manual_seed(args.seed)
    loaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=_collate,
            generator=generator,
        ),
        "validation": DataLoader(
            datasets["validation"], batch_size=args.batch_size, collate_fn=_collate
        ),
    }
    if datasets["test"] is not None:
        loaders["test"] = DataLoader(
            datasets["test"], batch_size=args.batch_size, collate_fn=_collate
        )

    config = EventWorldModelConfig(
        num_tools=len(vocabs["tools"]),
        num_attack_contexts=len(vocabs["attacks"]),
        num_domains=len(vocabs["domains"]),
        num_argument_signatures=len(vocabs["arguments"]),
        hidden_size=args.hidden_size,
        num_layers=args.layers,
        num_heads=args.heads,
        feedforward_size=2 * args.hidden_size,
        max_sequence_length=max(
            len(item.steps) for item in [*train, *validation, *test]
        ),
        pad_tool_id=vocabs["tools"][PAD],
    )
    device = torch.device(args.device)
    model = FactorizedEventWorldModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    history = []
    best_state = None
    best_validation = math.inf
    for epoch in range(1, args.epochs + 1):
        model.train()
        training_total = 0.0
        batches = 0
        for raw in loaders["train"]:
            batch = _move(raw, device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(
                batch["tool_ids"],
                batch["attack_ids"],
                batch["domain_ids"],
                batch["clean_prior"],
                batch["attention_mask"],
            )
            losses = model.loss(
                outputs,
                attention_mask=batch["attention_mask"],
                next_tool_targets=batch["tool_targets"],
                argument_signature_targets=batch["argument_targets"],
                stop_targets=batch["stop_targets"],
                joint_outcome_counts=batch["joint_counts"],
                joint_sample_weight=batch["joint_weight"],
                utility_residual_targets=batch["utility_residual"],
            )
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            training_total += float(losses["total"].detach())
            batches += 1
        validation_metrics = _evaluate(model, loaders["validation"], device)
        history.append(
            {
                "epoch": epoch,
                "train_batch_loss": training_total / max(batches, 1),
                "validation": validation_metrics,
            }
        )
        if validation_metrics["loss_per_event"] < best_validation:
            best_validation = validation_metrics["loss_per_event"]
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    assert best_state is not None
    model.load_state_dict(best_state)
    final_metrics = {
        name: _evaluate(model, loader, device)
        for name, loader in loaders.items()
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "event_world_model.pt"
    torch.save(
        {
            "model_state": best_state,
            "config": asdict(config),
            "vocabs": vocabs,
        },
        checkpoint_path,
    )
    report = {
        "scope": "factorized victim event dynamics; AgentDojo sandbox only",
        "confirmatory": not any(overlaps.values()),
        "known_transition_model": "exact AgentDojo simulator (not learned)",
        "config": asdict(config),
        "data": {
            "train": len(train),
            "validation": len(validation),
            "test": len(test),
            "task_group_overlap": overlaps,
        },
        "metrics": final_metrics,
        "joint_constant_baseline": {
            "training_concentration": constant_joint_concentration.tolist(),
            "count_nll": constant_joint_baseline,
            "validation_model_minus_constant_nll": (
                final_metrics["validation"]["joint_count_nll"]
                - constant_joint_baseline["validation"]
            ),
        },
        "history": history,
        "checkpoint": str(checkpoint_path.resolve()),
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
