"""Train parameter-matched v11 factorized semantic-transition dynamics."""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wmagentattack.factorized_transition_dynamics import (
    FactorizedSemanticTransitionDynamics,
    trainable_parameter_count,
)
from wmagentattack.factorized_transition_labels import FACTOR_CLASSES
from wmagentattack.joint_outcome_auxiliary import JOINT_OUTCOME_CLASSES
from wmagentattack.multisource_suitability import file_sha256


def _load_script(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


v5 = _load_script("v5", "201_train_structured_joint_outcome_v5.py")

ARMS = ("capacity_control_f11", "factorized_predicted_f11")
FACTOR_NAMES = tuple(FACTOR_CLASSES)


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _append(path: Path, rows: list[dict]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")


def _seed(value: int) -> None:
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)


def _surfaces(events, arrays, maximum_horizon: int = 5):
    by_trajectory = defaultdict(list)
    for index, event in enumerate(events):
        by_trajectory[event["trajectory_id"]].append(index)
    output = {}
    for horizon in range(1, maximum_horizon + 1):
        starts, paths, legal, targets, future, sequences = [], [], [], [], [], []
        for indices in by_trajectory.values():
            indices = sorted(indices, key=lambda index: events[index]["step_id"])
            for position in range(len(indices) - horizon):
                sequence = indices[position : position + horizon + 1]
                starts.append(sequence[0])
                paths.append([arrays["selected"][index] for index in sequence[:-1]])
                legal.append([arrays["legal"][index] for index in sequence[:-1]])
                targets.append(arrays["selected"][sequence[-1]])
                future.append(sequence[-2])
                sequences.append(sequence)
        output[horizon] = {
            "starts": np.asarray(starts),
            "paths": np.asarray(paths),
            "legal": np.asarray(legal),
            "targets": np.asarray(targets),
            "future": np.asarray(future),
            "sequences": np.asarray(sequences),
        }
    return output


def _factor_arrays(events, factor_dataset):
    rows = {row["source_event_id"]: row for row in factor_dataset["rows"]}
    values = np.full((len(events), len(FACTOR_NAMES)), -1, dtype=np.int64)
    for index, event in enumerate(events):
        row = rows.get(event["event_id"])
        if row is None:
            continue
        for column, name in enumerate(FACTOR_NAMES):
            values[index, column] = FACTOR_CLASSES[name].index(row["labels"][name])
    return values


def _factor_priors(events, factor_values):
    training = np.asarray([
        index for index, event in enumerate(events)
        if event["split"] == "training" and np.all(factor_values[index] >= 0)
    ])
    priors = {}
    for column, (name, classes) in enumerate(FACTOR_CLASSES.items()):
        counts = np.bincount(factor_values[training, column], minlength=len(classes)).astype(float)
        priors[name] = ((counts + 1.0) / (counts.sum() + len(classes))).tolist()
    return priors


def _train_model(
    arm, teacher, events, arrays, surfaces, factor_values, protocol, training_seed, device
):
    cfg = protocol["stage_f2_model"]["training"]
    states = torch.tensor(arrays["states"], dtype=torch.float32, device=device)
    candidates = torch.tensor(arrays["candidate_inputs"], dtype=torch.float32, device=device)
    selected = torch.tensor(arrays["selected"], dtype=torch.long, device=device)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    with torch.no_grad():
        context = teacher.encode_context(states, candidates[selected])
        teacher_logits = teacher.score_candidates(context, candidates)

    _seed(training_seed * 22103)
    model = FactorizedSemanticTransitionDynamics(
        candidate_size=candidates.shape[1],
        hidden_size=cfg["hidden_size"],
        dropout=cfg["dropout"],
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"]
    )
    history = []
    for epoch in range(cfg["epochs"]):
        _seed(training_seed * 22103 + epoch)
        model.train()
        surface = surfaces[1]
        keep = np.asarray([events[index]["split"] == "training" for index in surface["starts"]])
        starts_np = surface["starts"][keep]
        starts = torch.tensor(starts_np, device=device)
        targets = torch.tensor(surface["targets"][keep], device=device)
        legal = torch.tensor(surface["legal"][keep, -1], dtype=torch.bool, device=device)
        hidden, factor_logits = model.initial_hidden(context[starts])
        logits = (teacher_logits[starts] + model.one_step_delta_logits(hidden, candidates)).masked_fill(
            ~legal, torch.finfo(torch.float32).min
        )
        base = teacher_logits[starts].masked_fill(~legal, torch.finfo(torch.float32).min)
        weights = torch.tensor(
            v5._task_weights([events[index] for index in starts_np]), device=device
        )
        action_ce = (F.cross_entropy(logits, targets, reduction="none") * weights).sum() / weights.sum()
        base_probability = F.softmax(base, dim=1)
        kl = (
            base_probability * (F.log_softmax(base, dim=1) - F.log_softmax(logits, dim=1))
        ).sum(1)
        kl = (kl * weights).sum() / weights.sum()
        total = cfg["h1_ce_weight"] * action_ce + cfg["h1_kl_weight"] * kl

        factor_loss = torch.zeros((), device=device)
        if arm == "factorized_predicted_f11":
            factor_targets = torch.tensor(factor_values[starts_np], dtype=torch.long, device=device)
            for column, name in enumerate(FACTOR_NAMES):
                per_row = F.cross_entropy(factor_logits[name], factor_targets[:, column], reduction="none")
                factor_loss = factor_loss + (per_row * weights).sum() / weights.sum()
            factor_loss = factor_loss / len(FACTOR_NAMES)
            total = total + cfg["factor_loss_weight"] * factor_loss

        parts = {"h1_ce": action_ce, "h1_kl": kl, "factor_ce": factor_loss}
        for horizon in range(2, 6):
            surface = surfaces[horizon]
            keep = np.asarray([events[index]["split"] == "training" for index in surface["starts"]])
            starts_np = surface["starts"][keep]
            starts = torch.tensor(starts_np, device=device)
            paths = torch.tensor(surface["paths"][keep], device=device)
            hidden = model.initial_hidden(context[starts])[0]
            for step in range(1, horizon):
                hidden = model.advance(hidden, candidates[paths[:, step]])
            legal = torch.tensor(surface["legal"][keep, -1], dtype=torch.bool, device=device)
            targets = torch.tensor(surface["targets"][keep], device=device)
            rollout_logits = model.rollout_logits(hidden, candidates).masked_fill(
                ~legal, torch.finfo(torch.float32).min
            )
            weights = torch.tensor(
                v5._task_weights([events[index] for index in starts_np]), device=device
            )
            horizon_ce = (
                F.cross_entropy(rollout_logits, targets, reduction="none") * weights
            ).sum() / weights.sum()
            future = torch.tensor(surface["future"][keep], device=device)
            latent = 1 - F.cosine_similarity(model.projected_context(hidden), context[future], dim=1)
            latent = (latent * weights).sum() / weights.sum()
            trainable = np.asarray([events[index]["joint_outcome_trainable"] for index in starts_np])
            positions = np.flatnonzero(trainable)
            joint_loss = torch.zeros((), device=device)
            if len(positions):
                position_tensor = torch.tensor(positions, device=device)
                joint_targets = torch.tensor(
                    np.stack([
                        [events[starts_np[position]]["joint_outcome_target"][name] for name in JOINT_OUTCOME_CLASSES]
                        for position in positions
                    ]),
                    dtype=torch.float32,
                    device=device,
                )
                joint_loss = -(
                    joint_targets * F.log_softmax(model.joint_logits(hidden[position_tensor]), dim=1)
                ).sum(1).mean()
            total = (
                total
                + cfg["horizon_weights"][str(horizon)] * horizon_ce
                + cfg["latent_weight"] * latent
                + cfg["future_joint_weight"] * joint_loss
            )
            parts[f"h{horizon}_ce"] = horizon_ce

        optimizer.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
        optimizer.step()
        if epoch in (0, cfg["epochs"] - 1):
            history.append({
                "epoch": epoch,
                "total": float(total.detach()),
                **{name: float(value.detach()) for name, value in parts.items()},
                "factor_gate": float(torch.tanh(model.factor_gate).detach()),
            })
    return model, context, teacher_logits, history


def _condition(model, context, starts, factor_values, oracle: bool):
    if not oracle:
        return model.initial_hidden(context[starts])[0]
    labels = torch.tensor(factor_values[starts.cpu().numpy()], dtype=torch.long, device=context.device)
    return model.condition_oracle(context[starts], labels)


def _evaluate(
    model, teacher, context, teacher_logits, events, arrays, surfaces, factor_values,
    factor_priors, arm, fold, training_seed, device, oracle=False
):
    candidates = torch.tensor(arrays["candidate_inputs"], dtype=torch.float32, device=device)
    rows = []
    model.eval()
    with torch.no_grad():
        for horizon in range(1, 6):
            surface = surfaces[horizon]
            keep = np.asarray([events[index]["split"] == "confirmation" for index in surface["starts"]])
            starts_np = surface["starts"][keep]
            starts = torch.tensor(starts_np, device=device)
            legal_np = surface["legal"][keep]
            hidden = _condition(model, context, starts, factor_values, oracle)
            initial_factor_logits = model.factor_logits(context[starts])
            if horizon == 1:
                logits = teacher_logits[starts] + model.one_step_delta_logits(hidden, candidates)
                probabilities = F.softmax(
                    logits.masked_fill(
                        ~torch.tensor(legal_np[:, -1], dtype=torch.bool, device=device),
                        torch.finfo(torch.float32).min,
                    ), dim=1
                )
            else:
                probabilities = F.softmax(
                    teacher_logits[starts].masked_fill(
                        ~torch.tensor(legal_np[:, 0], dtype=torch.bool, device=device),
                        torch.finfo(torch.float32).min,
                    ), dim=1
                )
                for step in range(1, horizon):
                    if oracle:
                        hidden = model.advance_base(hidden, probabilities @ candidates)
                        source_indices = surface["sequences"][keep, step]
                        labels = torch.tensor(
                            factor_values[source_indices], dtype=torch.long, device=device
                        )
                        hidden = model.condition_oracle(hidden, labels)
                    else:
                        hidden = model.advance(hidden, probabilities @ candidates)
                    logits = model.rollout_logits(hidden, candidates)
                    probabilities = F.softmax(
                        logits.masked_fill(
                            ~torch.tensor(legal_np[:, step], dtype=torch.bool, device=device),
                            torch.finfo(torch.float32).min,
                        ), dim=1
                    )
            probabilities_np = probabilities.cpu().numpy()
            targets = surface["targets"][keep]
            joint_probability = torch.softmax(model.joint_logits(hidden), dim=1).cpu().numpy()
            factor_probabilities = {
                name: torch.softmax(value, dim=1).cpu().numpy()
                for name, value in initial_factor_logits.items()
            }
            for position, event_index in enumerate(starts_np):
                event = events[event_index]
                target = int(targets[position])
                prediction = int(probabilities_np[position].argmax())
                joint_target = None
                if event["joint_outcome_trainable"]:
                    joint_target = np.asarray([
                        event["joint_outcome_target"][name] for name in JOINT_OUTCOME_CLASSES
                    ])
                row = {
                    "arm": arm,
                    "fold": fold,
                    "training_seed": training_seed,
                    "horizon": horizon,
                    "event_id": event["event_id"],
                    "task_name": event["task_name"],
                    "trajectory_id": event["trajectory_id"],
                    "joint_group_id": event["joint_outcome_group_id"],
                    "action_nll": float(-math.log(max(probabilities_np[position, target], 1e-12))),
                    "action_correct": float(prediction == target),
                    "legal_prediction": float(legal_np[position, -1, prediction]),
                    "joint_trainable": float(joint_target is not None),
                    "joint_ce": float(-(
                        joint_target * np.log(np.clip(joint_probability[position], 1e-12, 1))
                    ).sum()) if joint_target is not None else None,
                }
                if horizon == 1:
                    for column, name in enumerate(FACTOR_NAMES):
                        factor_target = int(factor_values[event_index, column])
                        factor_probability = factor_probabilities[name][position]
                        prior = np.asarray(factor_priors[name])
                        row[f"{name}_nll"] = float(-math.log(max(factor_probability[factor_target], 1e-12)))
                        row[f"{name}_correct"] = float(factor_probability.argmax() == factor_target)
                        row[f"{name}_prior_nll"] = float(-math.log(max(prior[factor_target], 1e-12)))
                        row[f"{name}_prior_correct"] = float(prior.argmax() == factor_target)
                rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--factor-dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    if protocol["status"] != "label_gate_passed_stage_f2_frozen_before_training":
        raise ValueError("Stage F2 protocol is not frozen")
    if file_sha256(args.dataset) != protocol["frozen_dataset"]["sha256"]:
        raise ValueError("source dataset hash mismatch")
    if file_sha256(args.factor_dataset) != protocol["stage_f1_result"]["dataset_sha256"]:
        raise ValueError("factor dataset hash mismatch")
    source = json.loads(args.dataset.read_text())
    factor_dataset = json.loads(args.factor_dataset.read_text())
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else (
        "cpu" if args.device == "auto" else args.device
    )
    if device == "cpu":
        torch.set_num_threads(8)
    folds = [0] if args.smoke else list(range(protocol["stage_f2_model"]["folds"]))
    seeds = [protocol["stage_f2_model"]["seeds"][0]] if args.smoke else protocol["stage_f2_model"]["seeds"]
    run_protocol = copy.deepcopy(protocol)
    if args.smoke:
        run_protocol["teacher_training_protocol"]["training"]["fixed_epochs"] = 1
        run_protocol["stage_f2_model"]["training"]["epochs"] = 1
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = args.output_dir / "predictions.jsonl"
    prediction_path.write_text("", encoding="utf-8")
    runs = []
    parameter_counts = {}
    for fold in folds:
        events = v5._fold(source, fold)
        arrays = v5._arrays(events, source["candidate_catalog"], 128)
        surfaces = _surfaces(events, arrays)
        factor_values = _factor_arrays(events, factor_dataset)
        priors = _factor_priors(events, factor_values)
        for training_seed in seeds:
            _seed(training_seed)
            teacher_values = v5._train(
                "structured_joint_aux", events, arrays,
                run_protocol["teacher_training_protocol"], training_seed, device,
                return_model=True,
            )
            teacher = teacher_values[4]
            for arm in ARMS:
                model, context, teacher_logits, history = _train_model(
                    arm, teacher, events, arrays, surfaces, factor_values,
                    run_protocol, training_seed, device,
                )
                parameter_counts[arm] = trainable_parameter_count(model)
                rows = _evaluate(
                    model, teacher, context, teacher_logits, events, arrays, surfaces,
                    factor_values, priors, arm, fold, training_seed, device,
                )
                _append(prediction_path, rows)
                runs.append({
                    "fold": fold, "seed": training_seed, "arm": arm,
                    "rows": len(rows), "history": history,
                    "factor_priors": priors,
                })
                if arm == "factorized_predicted_f11":
                    oracle_rows = _evaluate(
                        model, teacher, context, teacher_logits, events, arrays, surfaces,
                        factor_values, priors, "factorized_oracle_diagnostic_f11",
                        fold, training_seed, device, oracle=True,
                    )
                    _append(prediction_path, oracle_rows)
    expected_runs = len(folds) * len(seeds) * len(ARMS)
    if len(runs) != expected_runs:
        raise ValueError("incomplete fixed fit budget")
    if len(set(parameter_counts.values())) != 1:
        raise ValueError("capacity arms are not parameter matched")
    _write(args.output_dir / "run_metrics.json", {
        "smoke": args.smoke,
        "device": device,
        "training_units": len(runs),
        "parameter_counts": parameter_counts,
        "parameter_match": True,
        "runtime_failures": 0,
        "runs": runs,
        "predictions_sha256": file_sha256(prediction_path),
    })


if __name__ == "__main__":
    main()
