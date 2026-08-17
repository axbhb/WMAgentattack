"""Stage A: paired v6 replication versus zero-gated relational-slot residual."""
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
from wmagentattack.joint_outcome_auxiliary import JOINT_OUTCOME_CLASSES
from wmagentattack.multisource_suitability import file_sha256
from wmagentattack.relational_slot_latent import (
    SlotAugmentedResidualDynamics,
    stack_relational_slot_states,
)


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


v5 = _load("v5", ROOT / "scripts/201_train_structured_joint_outcome_v5.py")
v6 = _load("v6", ROOT / "scripts/203_train_structured_residual_v6.py")


def write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def append(path: Path, rows: list[dict]) -> None:
    with path.open("a") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def restore_rng(state) -> None:
    random.setstate(state[0]); np.random.set_state(state[1]); torch.set_rng_state(state[2])


def train_slot(teacher, events, arrays, surfaces, slots, protocol, training_seed, device):
    cfg = protocol["stage_a"]["residual"]
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    states = torch.tensor(arrays["states"], dtype=torch.float32, device=device)
    candidates = torch.tensor(arrays["candidate_inputs"], dtype=torch.float32, device=device)
    selected = torch.tensor(arrays["selected"], dtype=torch.long, device=device)
    slot_features = torch.tensor(slots["features"], dtype=torch.float32, device=device)
    slot_types = torch.tensor(slots["node_types"], dtype=torch.long, device=device)
    slot_relations = torch.tensor(slots["relations"], dtype=torch.long, device=device)
    slot_mask = torch.tensor(slots["mask"], dtype=torch.bool, device=device)
    with torch.no_grad():
        teacher_context = teacher.encode_context(states, candidates[selected])
        teacher_logits = teacher.score_candidates(teacher_context, candidates)
    model = SlotAugmentedResidualDynamics(
        candidate_size=candidates.shape[1], slot_feature_size=slot_features.shape[2],
        hidden_size=cfg["hidden_size"], slot_layers=protocol["stage_a"]["slot_builder"]["message_layers"],
        dropout=cfg["dropout"],
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    history = []
    for epoch in range(cfg["epochs"]):
        v6.seed(training_seed * 2003 + epoch); model.train()
        initial, _ = model.initial_hidden(teacher_context, slot_features, slot_types, slot_relations, slot_mask)
        surface = surfaces[1]
        keep = np.asarray([events[i]["split"] == "training" for i in surface["starts"]])
        index = torch.tensor(surface["starts"][keep], device=device)
        target = torch.tensor(surface["targets"][keep], device=device)
        legal = torch.tensor(surface["legals"][keep, -1], dtype=torch.bool, device=device)
        base = teacher_logits[index].masked_fill(~legal, torch.finfo(torch.float32).min)
        logits = (teacher_logits[index] + model.one_step_delta_logits(initial[index], candidates)).masked_fill(~legal, torch.finfo(torch.float32).min)
        weights = torch.tensor(v5._task_weights([events[i] for i in surface["starts"][keep]]), device=device)
        h1_ce = (F.cross_entropy(logits, target, reduction="none") * weights).sum() / weights.sum()
        base_probability = F.softmax(base, 1)
        h1_kl = (base_probability * (F.log_softmax(base, 1) - F.log_softmax(logits, 1))).sum(1)
        h1_kl = (h1_kl * weights).sum() / weights.sum()
        total = cfg["h1_ce_weight"] * h1_ce + cfg["h1_kl_weight"] * h1_kl
        parts = {"h1_ce": h1_ce, "h1_kl": h1_kl}
        for horizon in range(2, 6):
            surface = surfaces[horizon]
            keep = np.asarray([events[i]["split"] == "training" for i in surface["starts"]])
            starts = torch.tensor(surface["starts"][keep], device=device)
            paths = torch.tensor(surface["paths"][keep], device=device)
            hidden = initial[starts]
            for step in range(1, horizon):
                hidden = model.advance(hidden, candidates[paths[:, step]])
            legal = torch.tensor(surface["legals"][keep, -1], dtype=torch.bool, device=device)
            target = torch.tensor(surface["targets"][keep], device=device)
            logits = model.rollout_logits(hidden, candidates).masked_fill(~legal, torch.finfo(torch.float32).min)
            weights = torch.tensor(v5._task_weights([events[i] for i in surface["starts"][keep]]), device=device)
            horizon_ce = (F.cross_entropy(logits, target, reduction="none") * weights).sum() / weights.sum()
            future = torch.tensor(surface["future"][keep], device=device)
            latent = 1 - F.cosine_similarity(model.projected_context(hidden), teacher_context[future], dim=1)
            latent = (latent * weights).sum() / weights.sum()
            trainable = np.asarray([events[i]["joint_outcome_trainable"] for i in surface["starts"][keep]])
            joint_indices = torch.tensor(np.flatnonzero(trainable), device=device)
            joint_loss = torch.zeros((), device=device)
            if len(joint_indices):
                y = torch.tensor(np.stack([[events[surface["starts"][keep][i]]["joint_outcome_target"][name] for name in JOINT_OUTCOME_CLASSES] for i in np.flatnonzero(trainable)]), dtype=torch.float32, device=device)
                joint_loss = -(y * F.log_softmax(model.joint_logits(hidden[joint_indices]), 1)).sum(1).mean()
            total = total + cfg["horizon_weights"][str(horizon)] * horizon_ce + cfg["latent_weight"] * latent + cfg["future_joint_weight"] * joint_loss
            parts[f"h{horizon}_ce"] = horizon_ce
        optimizer.zero_grad(set_to_none=True); total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10); optimizer.step()
        if epoch in (0, cfg["epochs"] - 1):
            history.append({"epoch": epoch, "total": float(total.detach()), "slot_gate": float(torch.tanh(model.slot_gate).detach()), **{key: float(value.detach()) for key, value in parts.items()}})
    return model, teacher_context, teacher_logits, (slot_features, slot_types, slot_relations, slot_mask), history


def evaluate_slot(model, teacher, teacher_context, teacher_logits, slot_tensors, events, arrays, surfaces, diagnostics, fold, training_seed, device):
    candidates = torch.tensor(arrays["candidate_inputs"], dtype=torch.float32, device=device)
    model.eval(); rows = []; prior = np.asarray(diagnostics["joint_prior"])
    with torch.no_grad():
        initial, _ = model.initial_hidden(teacher_context, *slot_tensors)
        for horizon in range(1, 6):
            surface = surfaces[horizon]
            keep = np.asarray([events[i]["split"] == "confirmation" for i in surface["starts"]])
            starts_np = surface["starts"][keep]; starts = torch.tensor(starts_np, device=device)
            legal_np = surface["legals"][keep]
            if horizon == 1:
                logits = teacher_logits[starts] + model.one_step_delta_logits(initial[starts], candidates)
                probability = F.softmax(logits.masked_fill(~torch.tensor(legal_np[:, -1], dtype=torch.bool, device=device), torch.finfo(torch.float32).min), 1)
                hidden = initial[starts]
            else:
                hidden = initial[starts]
                probability = F.softmax(teacher_logits[starts].masked_fill(~torch.tensor(legal_np[:, 0], dtype=torch.bool, device=device), torch.finfo(torch.float32).min), 1)
                for step in range(1, horizon):
                    hidden = model.advance(hidden, probability @ candidates)
                    logits = model.rollout_logits(hidden, candidates)
                    probability = F.softmax(logits.masked_fill(~torch.tensor(legal_np[:, step], dtype=torch.bool, device=device), torch.finfo(torch.float32).min), 1)
            probability = probability.cpu().numpy(); targets = surface["targets"][keep]
            joint_probability = torch.softmax(teacher.joint_outcome_head(hidden), 1).cpu().numpy() if horizon == 1 else torch.softmax(model.joint_logits(hidden), 1).cpu().numpy()
            for offset, event_index in enumerate(starts_np):
                event = events[event_index]; target = int(targets[offset])
                y = np.asarray([event["joint_outcome_target"][name] for name in JOINT_OUTCOME_CLASSES]) if event["joint_outcome_trainable"] else None
                rows.append({
                    "arm": "relational_slot_stage_a", "fold": fold, "training_seed": training_seed,
                    "horizon": horizon, "event_id": event["event_id"], "task_name": event["task_name"],
                    "trajectory_id": event["trajectory_id"], "joint_group_id": event["joint_outcome_group_id"],
                    "action_nll": float(-math.log(max(probability[offset, target], 1e-12))),
                    "action_correct": float(probability[offset].argmax() == target),
                    "legal_prediction": float(legal_np[offset, -1, probability[offset].argmax()]),
                    "joint_trainable": float(y is not None),
                    "joint_ce": float(-(y * np.log(np.clip(joint_probability[offset], 1e-12, 1))).sum()) if y is not None else None,
                    "joint_prior_ce": float(-(y * np.log(np.clip(prior, 1e-12, 1))).sum()) if y is not None else None,
                })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True); parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True); parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(); protocol = json.loads(args.protocol.read_text()); data = json.loads(args.dataset.read_text())
    if protocol["status"] != "preregistered_stage_a_before_training": raise ValueError("protocol not frozen")
    if file_sha256(args.dataset) != protocol["frozen_dataset"]["sha256"] or file_sha256(args.audit) != protocol["frozen_dataset"]["audit_sha256"]: raise ValueError("frozen data mismatch")
    torch.set_num_threads(8); device = "cpu"; args.output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = args.output_dir / "predictions.jsonl"; prediction_path.write_text("")
    runs = []; slot_audits = []
    teacher_protocol = {"training": protocol["stage_a"]["teacher"]}
    for fold in range(5):
        events = v5._fold(data, fold); arrays = v5._arrays(events, data["candidate_catalog"], 128); surfaces = v6.horizons(events, arrays)
        slots = stack_relational_slot_states(events, hash_dimension=protocol["stage_a"]["slot_builder"]["hash_dimension"], max_nodes=protocol["stage_a"]["slot_builder"]["max_nodes"])
        slot_audits.extend(slots["audit"])
        for training_seed in protocol["research_budget"]["seeds"]:
            v6.seed(training_seed)
            teacher_values = v5._train("structured_joint_aux", events, arrays, teacher_protocol, training_seed, device, return_model=True)
            teacher = teacher_values[4]
            rng = (random.getstate(), np.random.get_state(), torch.get_rng_state())
            baseline, context, logits, baseline_history = v6.train_residual(teacher, events, arrays, surfaces, {"residual_training": protocol["stage_a"]["residual"]}, training_seed, device)
            baseline_rows = [row for row in v6.evaluate(baseline, teacher, context, logits, events, arrays, surfaces, teacher_values[3], fold, training_seed, device) if row["arm"] == "structured_residual_v6"]
            for row in baseline_rows: row["arm"] = "v6_replication"
            append(prediction_path, baseline_rows)
            restore_rng(rng)
            candidate = train_slot(teacher, events, arrays, surfaces, slots, protocol, training_seed, device)
            candidate_rows = evaluate_slot(candidate[0], teacher, candidate[1], candidate[2], candidate[3], events, arrays, surfaces, teacher_values[3], fold, training_seed, device)
            append(prediction_path, candidate_rows)
            runs.append({"fold": fold, "seed": training_seed, "v6_history": baseline_history, "slot_history": candidate[4], "rows_per_arm": len(candidate_rows)})
    metrics = {
        "training_units": len(runs), "teacher_fits": len(runs), "v6_residual_fits": len(runs), "slot_residual_fits": len(runs),
        "runtime_failures": 0, "runs": runs, "slot_audit": {
            "rows": len(slot_audits), "truncated_rows": sum(row["truncated"] for row in slot_audits),
            "raw_values_encoded": any(row["raw_values_encoded"] for row in slot_audits),
            "maximum_nodes": max(row["node_count"] for row in slot_audits),
        }, "predictions_sha256": file_sha256(prediction_path),
    }
    if len(runs) != 15: raise ValueError("fixed budget incomplete")
    write(args.output_dir / "run_metrics.json", metrics)


if __name__ == "__main__":
    main()
