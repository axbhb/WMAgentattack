"""Train equal-capacity unsupervised and supervised predicted-event-graph dynamics."""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wmagentattack.multisource_suitability import file_sha256
from wmagentattack.predicted_event_graph_dynamics import (
    PredictedEventGraphDynamics,
    trainable_parameter_count,
)


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


v5 = _load("v5", "201_train_structured_joint_outcome_v5.py")
v12 = _load("v12", "224_train_action_event_graph_oracle_v12.py")

ARMS = ("unsupervised_graph_capacity_control_v13", "predicted_event_graph_v13")


def _seed(value: int) -> None:
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")


def _append(path: Path, rows: list[dict]) -> None:
    with path.open("a") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")


def _weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return (values * weights).sum() / weights.sum()


def _graph_training_statistics(events, surface, graphs, smoothing: float, cap: float):
    keep = np.asarray([events[index]["split"] == "training" for index in surface["starts"]])
    targets = graphs[surface["sequences"][keep, 1]]
    positive = targets.sum(axis=0)
    count = len(targets)
    prior = (positive + smoothing) / (count + 2 * smoothing)
    positive_weight = np.minimum((count - positive) / np.maximum(positive, 1), cap)
    return prior.astype(np.float32), positive_weight.astype(np.float32)


def _train_model(arm, teacher, events, arrays, surfaces, graphs, protocol, training_seed, device):
    cfg = protocol["stage_m2"]["training"]
    states = torch.tensor(arrays["states"], dtype=torch.float32, device=device)
    candidates = torch.tensor(arrays["candidate_inputs"], dtype=torch.float32, device=device)
    selected = torch.tensor(arrays["selected"], dtype=torch.long, device=device)
    graph_tensor = torch.tensor(graphs, dtype=torch.float32, device=device)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    with torch.no_grad():
        context = teacher.encode_context(states, candidates[selected])
        teacher_logits = teacher.score_candidates(context, candidates)

    _seed(training_seed * 22703)
    model = PredictedEventGraphDynamics(
        graph_size=graphs.shape[1],
        candidate_size=candidates.shape[1],
        hidden_size=cfg["hidden_size"],
        dropout=cfg["dropout"],
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"]
    )
    supervised = arm == "predicted_event_graph_v13"
    observed_graph = graph_tensor if supervised else torch.zeros_like(graph_tensor)
    prior, positive_weight = _graph_training_statistics(
        events,
        surfaces[1],
        graphs,
        cfg["graph_prior_smoothing"],
        cfg["graph_positive_weight_cap"],
    )
    pos_weight = torch.tensor(positive_weight, dtype=torch.float32, device=device)
    history = []

    for epoch in range(cfg["epochs"]):
        _seed(training_seed * 22703 + epoch)
        model.train()
        surface = surfaces[1]
        keep = np.asarray([events[index]["split"] == "training" for index in surface["starts"]])
        starts_np = surface["starts"][keep]
        starts = torch.tensor(starts_np, device=device)
        targets = torch.tensor(surface["targets"][keep], device=device)
        legal = torch.tensor(surface["legal"][keep, -1], dtype=torch.bool, device=device)
        weights = torch.tensor(v5._task_weights([events[index] for index in starts_np]), device=device)
        hidden = model.condition(context[starts], observed_graph[starts])
        logits = (teacher_logits[starts] + model.one_step_delta_logits(hidden, candidates)).masked_fill(
            ~legal, torch.finfo(torch.float32).min
        )
        base = teacher_logits[starts].masked_fill(~legal, torch.finfo(torch.float32).min)
        action_ce = _weighted_mean(F.cross_entropy(logits, targets, reduction="none"), weights)
        base_probability = F.softmax(base, dim=1)
        kl = (base_probability * (F.log_softmax(base, dim=1) - F.log_softmax(logits, dim=1))).sum(1)
        kl = _weighted_mean(kl, weights)

        next_actions = torch.tensor(surface["targets"][keep], device=device)
        next_graphs = graph_tensor[torch.tensor(surface["sequences"][keep, 1], device=device)]
        graph_logits = model.predict_graph_logits(hidden, candidates[next_actions])
        per_graph = F.binary_cross_entropy_with_logits(
            graph_logits, next_graphs, reduction="none", pos_weight=pos_weight
        ).mean(1)
        graph_loss = _weighted_mean(per_graph, weights) if supervised else torch.zeros((), device=device)
        total = (
            cfg["h1_ce_weight"] * action_ce
            + cfg["h1_kl_weight"] * kl
            + cfg["graph_loss_weight"] * graph_loss
        )
        parts = {"h1_ce": action_ce, "h1_kl": kl, "graph_bce": graph_loss}

        for horizon in range(2, 6):
            surface = surfaces[horizon]
            keep = np.asarray([events[index]["split"] == "training" for index in surface["starts"]])
            starts_np = surface["starts"][keep]
            starts = torch.tensor(starts_np, device=device)
            paths = torch.tensor(surface["paths"][keep], device=device)
            hidden = model.condition(context[starts], observed_graph[starts])
            for step in range(1, horizon):
                hidden, _ = model.advance_predicted(hidden, candidates[paths[:, step]])
            legal = torch.tensor(surface["legal"][keep, -1], dtype=torch.bool, device=device)
            targets = torch.tensor(surface["targets"][keep], device=device)
            rollout = model.rollout_logits(hidden, candidates).masked_fill(
                ~legal, torch.finfo(torch.float32).min
            )
            weights = torch.tensor(v5._task_weights([events[index] for index in starts_np]), device=device)
            horizon_ce = _weighted_mean(F.cross_entropy(rollout, targets, reduction="none"), weights)
            future = torch.tensor(surface["future"][keep], device=device)
            latent = 1 - F.cosine_similarity(model.projected_context(hidden), context[future], dim=1)
            latent = _weighted_mean(latent, weights)
            total = total + cfg["horizon_weights"][str(horizon)] * horizon_ce + cfg["latent_weight"] * latent
            parts[f"h{horizon}_ce"] = horizon_ce

        optimizer.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
        optimizer.step()
        if epoch in (0, cfg["epochs"] - 1):
            history.append(
                {
                    "epoch": epoch,
                    "total": float(total.detach()),
                    "graph_gate": float(torch.tanh(model.graph_gate).detach()),
                    **{name: float(value.detach()) for name, value in parts.items()},
                }
            )
    return model, context, teacher_logits, history, prior


def _bce(probability: np.ndarray, target: np.ndarray) -> np.ndarray:
    probability = np.clip(probability, 1e-12, 1 - 1e-12)
    return -(target * np.log(probability) + (1 - target) * np.log(1 - probability)).mean(axis=1)


def _evaluate(model, context, teacher_logits, events, arrays, surfaces, graphs, arm, fold, seed, prior, device):
    candidates = torch.tensor(arrays["candidate_inputs"], dtype=torch.float32, device=device)
    graph_tensor = torch.tensor(graphs, dtype=torch.float32, device=device)
    observed_graph = graph_tensor if arm == "predicted_event_graph_v13" else torch.zeros_like(graph_tensor)
    rows = []
    model.eval()
    with torch.no_grad():
        for horizon in range(1, 6):
            surface = surfaces[horizon]
            keep = np.asarray([events[index]["split"] == "confirmation" for index in surface["starts"]])
            starts_np = surface["starts"][keep]
            starts = torch.tensor(starts_np, device=device)
            legal_np = surface["legal"][keep]
            hidden = model.condition(context[starts], observed_graph[starts])
            if horizon == 1:
                logits = teacher_logits[starts] + model.one_step_delta_logits(hidden, candidates)
                legal = torch.tensor(legal_np[:, -1], dtype=torch.bool, device=device)
                probabilities = F.softmax(logits.masked_fill(~legal, torch.finfo(torch.float32).min), dim=1)
                next_actions = torch.tensor(surface["targets"][keep], device=device)
                graph_logits = model.predict_graph_logits(hidden, candidates[next_actions])
                graph_probability = torch.sigmoid(graph_logits).cpu().numpy()
                graph_target = graphs[surface["sequences"][keep, 1]]
                graph_bce = _bce(graph_probability, graph_target)
                graph_prior_bce = _bce(np.broadcast_to(prior, graph_target.shape), graph_target)
            else:
                legal = torch.tensor(legal_np[:, 0], dtype=torch.bool, device=device)
                probabilities = F.softmax(
                    teacher_logits[starts].masked_fill(~legal, torch.finfo(torch.float32).min), dim=1
                )
                for step in range(1, horizon):
                    hidden, _ = model.advance_predicted(hidden, probabilities @ candidates)
                    logits = model.rollout_logits(hidden, candidates)
                    legal = torch.tensor(legal_np[:, step], dtype=torch.bool, device=device)
                    probabilities = F.softmax(
                        logits.masked_fill(~legal, torch.finfo(torch.float32).min), dim=1
                    )
                graph_bce = graph_prior_bce = None

            probability = probabilities.cpu().numpy()
            targets = surface["targets"][keep]
            for position, event_index in enumerate(starts_np):
                event = events[event_index]
                target = int(targets[position])
                prediction = int(probability[position].argmax())
                rows.append(
                    {
                        "arm": arm,
                        "fold": fold,
                        "training_seed": seed,
                        "horizon": horizon,
                        "event_id": event["event_id"],
                        "task_name": event["task_name"],
                        "trajectory_id": event["trajectory_id"],
                        "joint_group_id": event["joint_outcome_group_id"],
                        "action_nll": float(-math.log(max(probability[position, target], 1e-12))),
                        "action_correct": float(prediction == target),
                        "legal_prediction": float(legal_np[position, -1, prediction]),
                        "graph_bce": float(graph_bce[position]) if graph_bce is not None else None,
                        "graph_prior_bce": float(graph_prior_bce[position]) if graph_prior_bce is not None else None,
                    }
                )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--graph-dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    if protocol["status"] != "stage_m1_passed_stage_m2_frozen_before_training":
        raise ValueError("v13 M2 protocol not frozen")
    stage = protocol["stage_m2"]
    if file_sha256(args.events) != stage["events_sha256"]:
        raise ValueError("event hash mismatch")
    if file_sha256(args.graph_dataset) != stage["event_graph_sha256"]:
        raise ValueError("graph hash mismatch")
    source = json.loads(args.events.read_text())
    graph_dataset = json.loads(args.graph_dataset.read_text())
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    if device == "cpu":
        torch.set_num_threads(8)
    folds = [0] if args.smoke else list(range(stage["folds"]))
    seeds = [stage["seeds"][0]] if args.smoke else stage["seeds"]
    run_protocol = copy.deepcopy(protocol)
    if args.smoke:
        run_protocol["teacher_training_protocol"]["training"]["fixed_epochs"] = 1
        run_protocol["stage_m2"]["training"]["epochs"] = 1
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions = args.output_dir / "predictions.jsonl"
    predictions.write_text("")
    runs = []
    parameter_counts = {}
    teacher_fits = 0
    for fold in folds:
        events = v5._fold(source, fold)
        arrays = v5._arrays(events, source["candidate_catalog"], 128)
        surfaces = v12._surfaces(events, arrays)
        graph_array = v12._graph_array(events, graph_dataset)
        for training_seed in seeds:
            _seed(training_seed)
            teacher_values = v5._train(
                "structured_joint_aux",
                events,
                arrays,
                run_protocol["teacher_training_protocol"],
                training_seed,
                device,
                return_model=True,
            )
            teacher = teacher_values[4]
            teacher_fits += 1
            for arm in ARMS:
                model, context, teacher_logits, history, prior = _train_model(
                    arm, teacher, events, arrays, surfaces, graph_array, run_protocol, training_seed, device
                )
                parameter_counts[arm] = trainable_parameter_count(model)
                rows = _evaluate(
                    model,
                    context,
                    teacher_logits,
                    events,
                    arrays,
                    surfaces,
                    graph_array,
                    arm,
                    fold,
                    training_seed,
                    prior,
                    device,
                )
                _append(predictions, rows)
                runs.append({"fold": fold, "seed": training_seed, "arm": arm, "rows": len(rows), "history": history})
    expected = len(folds) * len(seeds) * len(ARMS)
    if len(runs) != expected:
        raise ValueError("incomplete fit budget")
    if len(set(parameter_counts.values())) != 1:
        raise ValueError("parameter mismatch")
    _write(
        args.output_dir / "run_metrics.json",
        {
            "smoke": args.smoke,
            "device": device,
            "training_units": len(runs),
            "teacher_fits": teacher_fits,
            "runtime_failures": 0,
            "parameter_counts": parameter_counts,
            "parameter_match": True,
            "runs": runs,
            "predictions_sha256": file_sha256(predictions),
        },
    )


if __name__ == "__main__":
    main()
