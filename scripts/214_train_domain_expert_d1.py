"""Stage D1: parameter-matched dense versus deterministic domain experts."""
from __future__ import annotations

import argparse
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
from wmagentattack.domain_expert_latent import (
    DOMAIN_NAMES,
    DenseCapacityAffordanceResidual,
    DomainExpertAffordanceResidual,
    routed_parameter_gap_fraction,
    stack_domain_indices,
    trainable_parameter_count,
)
from wmagentattack.joint_outcome_auxiliary import JOINT_OUTCOME_CLASSES
from wmagentattack.multisource_suitability import file_sha256
from wmagentattack.relational_slot_latent import stack_interface_affordance_states


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


v5 = load("v5", ROOT / "scripts/201_train_structured_joint_outcome_v5.py")
v6 = load("v6", ROOT / "scripts/203_train_structured_residual_v6.py")


def write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def append(path: Path, rows: list[dict]) -> None:
    with path.open("a") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def restore_rng(state) -> None:
    random.setstate(state[0]); np.random.set_state(state[1]); torch.set_rng_state(state[2])


def tensors(arrays, slots, domains, device):
    return (
        torch.tensor(arrays["states"], dtype=torch.float32, device=device),
        torch.tensor(arrays["candidate_inputs"], dtype=torch.float32, device=device),
        torch.tensor(arrays["selected"], dtype=torch.long, device=device),
        torch.tensor(slots["features"], dtype=torch.float32, device=device),
        torch.tensor(slots["node_types"], dtype=torch.long, device=device),
        torch.tensor(slots["relations"], dtype=torch.long, device=device),
        torch.tensor(slots["mask"], dtype=torch.bool, device=device),
        torch.tensor(domains, dtype=torch.long, device=device),
    )


def build_model(arm, candidates, slot_features, protocol, device):
    cfg = protocol["stage_d1"]
    common = dict(
        candidate_size=candidates.shape[1], slot_feature_size=slot_features.shape[2],
        hidden_size=cfg["training"]["hidden_size"],
        slot_layers=cfg["affordance_builder"]["message_layers"],
        dropout=cfg["training"]["dropout"],
    )
    if arm == "dense_capacity_control":
        return DenseCapacityAffordanceResidual(
            **common, dense_bottleneck_size=cfg["dense_control"]["bottleneck_size"]
        ).to(device)
    if arm == "domain_expert_d1":
        return DomainExpertAffordanceResidual(
            **common, expert_bottleneck_size=cfg["domain_experts"]["bottleneck_size"]
        ).to(device)
    raise ValueError(arm)


def train_arm(arm, teacher, events, arrays, surfaces, slots, domains, protocol, training_seed, device):
    cfg = protocol["stage_d1"]["training"]
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    states, candidates, selected, sf, st, sr, sm, domain = tensors(arrays, slots, domains, device)
    with torch.no_grad():
        teacher_context = teacher.encode_context(states, candidates[selected])
        teacher_logits = teacher.score_candidates(teacher_context, candidates)
    model = build_model(arm, candidates, sf, protocol, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"]
    )
    discounts = {h: cfg["successor_discount"] ** (h - 2) for h in range(2, 6)}
    discount_total = sum(discounts.values())
    history = []
    for epoch in range(cfg["epochs"]):
        v6.seed(training_seed * 5003 + epoch)
        model.train()
        initial, _ = model.initial_hidden(teacher_context, sf, st, sr, sm, domain)
        surface = surfaces[1]
        keep = np.asarray([events[i]["split"] == "training" for i in surface["starts"]])
        index = torch.tensor(surface["starts"][keep], device=device)
        target = torch.tensor(surface["targets"][keep], device=device)
        legal = torch.tensor(surface["legals"][keep, -1], dtype=torch.bool, device=device)
        base = teacher_logits[index].masked_fill(~legal, torch.finfo(torch.float32).min)
        logits = (
            teacher_logits[index] + model.one_step_delta_logits(initial[index], candidates)
        ).masked_fill(~legal, torch.finfo(torch.float32).min)
        weights = torch.tensor(
            v5._task_weights([events[i] for i in surface["starts"][keep]]), device=device
        )
        h1_ce = (F.cross_entropy(logits, target, reduction="none") * weights).sum() / weights.sum()
        base_probability = F.softmax(base, 1)
        h1_kl = (
            base_probability * (F.log_softmax(base, 1) - F.log_softmax(logits, 1))
        ).sum(1)
        h1_kl = (h1_kl * weights).sum() / weights.sum()
        total = cfg["h1_ce_weight"] * h1_ce + cfg["h1_kl_weight"] * h1_kl
        successor_total = torch.zeros((), device=device)
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
            logits = model.rollout_logits(hidden, candidates).masked_fill(
                ~legal, torch.finfo(torch.float32).min
            )
            weights = torch.tensor(
                v5._task_weights([events[i] for i in surface["starts"][keep]]), device=device
            )
            horizon_ce = (
                F.cross_entropy(logits, target, reduction="none") * weights
            ).sum() / weights.sum()
            future = torch.tensor(surface["future"][keep], device=device)
            latent = 1 - F.cosine_similarity(
                model.projected_context(hidden), teacher_context[future], dim=1
            )
            latent = (latent * weights).sum() / weights.sum()
            trainable = np.asarray([
                events[i]["joint_outcome_trainable"] for i in surface["starts"][keep]
            ])
            joint_indices = torch.tensor(np.flatnonzero(trainable), device=device)
            joint_loss = torch.zeros((), device=device)
            if len(joint_indices):
                y = torch.tensor(np.stack([
                    [
                        events[surface["starts"][keep][i]]["joint_outcome_target"][name]
                        for name in JOINT_OUTCOME_CLASSES
                    ] for i in np.flatnonzero(trainable)
                ]), dtype=torch.float32, device=device)
                joint_loss = -(
                    y * F.log_softmax(model.joint_logits(hidden[joint_indices]), 1)
                ).sum(1).mean()
            successor_logits = model.successor_logits(initial[starts], candidates).masked_fill(
                ~legal, torch.finfo(torch.float32).min
            )
            successor = (
                F.cross_entropy(successor_logits, target, reduction="none") * weights
            ).sum() / weights.sum()
            successor_total = successor_total + discounts[horizon] * successor / discount_total
            total = (
                total + cfg["horizon_weights"][str(horizon)] * horizon_ce
                + cfg["latent_weight"] * latent
                + cfg["future_joint_weight"] * joint_loss
            )
            parts[f"h{horizon}_ce"] = horizon_ce
            parts[f"h{horizon}_successor"] = successor
        total = total + cfg["successor_weight"] * successor_total
        parts["successor_total"] = successor_total
        optimizer.zero_grad(set_to_none=True); total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10); optimizer.step()
        if epoch in (0, cfg["epochs"] - 1):
            gate = (
                [float(value) for value in torch.tanh(model.expert_gates).detach().cpu()]
                if hasattr(model, "expert_gates")
                else [float(torch.tanh(model.adapter_gate).detach().cpu())]
            )
            history.append({
                "epoch": epoch, "total": float(total.detach()), "gates": gate,
                **{key: float(value.detach()) for key, value in parts.items()},
            })
    return model, teacher_context, teacher_logits, (sf, st, sr, sm, domain), history


def evaluate(model, teacher, teacher_context, teacher_logits, slot_tensors, events, arrays, surfaces, diagnostics, fold, training_seed, device, arm):
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
                probability = F.softmax(logits.masked_fill(
                    ~torch.tensor(legal_np[:, -1], dtype=torch.bool, device=device),
                    torch.finfo(torch.float32).min,
                ), 1)
                hidden = initial[starts]
            else:
                hidden = initial[starts]
                probability = F.softmax(teacher_logits[starts].masked_fill(
                    ~torch.tensor(legal_np[:, 0], dtype=torch.bool, device=device),
                    torch.finfo(torch.float32).min,
                ), 1)
                for step in range(1, horizon):
                    hidden = model.advance(hidden, probability @ candidates)
                    probability = F.softmax(model.rollout_logits(hidden, candidates).masked_fill(
                        ~torch.tensor(legal_np[:, step], dtype=torch.bool, device=device),
                        torch.finfo(torch.float32).min,
                    ), 1)
            probability_np = probability.cpu().numpy(); targets = surface["targets"][keep]
            joint_probability = (
                torch.softmax(teacher.joint_outcome_head(hidden), 1).cpu().numpy()
                if horizon == 1 else torch.softmax(model.joint_logits(hidden), 1).cpu().numpy()
            )
            for offset, event_index in enumerate(starts_np):
                event = events[event_index]; target = int(targets[offset])
                y = (
                    np.asarray([event["joint_outcome_target"][name] for name in JOINT_OUTCOME_CLASSES])
                    if event["joint_outcome_trainable"] else None
                )
                rows.append({
                    "arm": arm, "fold": fold, "training_seed": training_seed,
                    "horizon": horizon, "event_id": event["event_id"],
                    "task_name": event["task_name"], "trajectory_id": event["trajectory_id"],
                    "joint_group_id": event["joint_outcome_group_id"],
                    "action_nll": float(-math.log(max(probability_np[offset, target], 1e-12))),
                    "action_correct": float(probability_np[offset].argmax() == target),
                    "legal_prediction": float(legal_np[offset, -1, probability_np[offset].argmax()]),
                    "joint_trainable": float(y is not None),
                    "joint_ce": float(-(y * np.log(np.clip(joint_probability[offset], 1e-12, 1))).sum()) if y is not None else None,
                    "joint_prior_ce": float(-(y * np.log(np.clip(prior, 1e-12, 1))).sum()) if y is not None else None,
                })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(); protocol = json.loads(args.protocol.read_text())
    data = json.loads(args.dataset.read_text())
    if protocol["status"] != "preregistered_stage_d1_before_training":
        raise ValueError("D1 protocol not frozen")
    if file_sha256(args.dataset) != protocol["frozen_dataset"]["sha256"] or file_sha256(args.audit) != protocol["frozen_dataset"]["audit_sha256"]:
        raise ValueError("frozen data mismatch")
    torch.set_num_threads(8); device = "cpu"; args.output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = args.output_dir / "predictions.jsonl"; prediction_path.write_text("")
    runs = []; all_audits = []; route_counts = np.zeros(len(DOMAIN_NAMES), dtype=np.int64)
    parameter_diagnostics = None; builder = protocol["stage_d1"]["affordance_builder"]
    teacher_protocol = {"training": protocol["teacher"]}
    for fold in range(protocol["research_budget"]["folds"]):
        events = v5._fold(data, fold); arrays = v5._arrays(events, data["candidate_catalog"], 128)
        surfaces = v6.horizons(events, arrays)
        slots = stack_interface_affordance_states(
            events, hash_dimension=builder["hash_dimension"],
            max_nodes=builder["max_nodes"], max_concepts=builder["max_concepts"],
        )
        domains = stack_domain_indices(events); route_counts += np.bincount(domains, minlength=len(DOMAIN_NAMES))
        all_audits.extend(slots["audit"])
        for training_seed in protocol["research_budget"]["seeds"]:
            v6.seed(training_seed)
            values = v5._train(
                "structured_joint_aux", events, arrays, teacher_protocol,
                training_seed, device, return_model=True,
            )
            rng = (random.getstate(), np.random.get_state(), torch.get_rng_state())
            dense = train_arm(
                "dense_capacity_control", values[4], events, arrays, surfaces,
                slots, domains, protocol, training_seed, device,
            )
            append(prediction_path, evaluate(
                dense[0], values[4], dense[1], dense[2], dense[3], events, arrays,
                surfaces, values[3], fold, training_seed, device, "dense_capacity_control",
            ))
            restore_rng(rng)
            expert = train_arm(
                "domain_expert_d1", values[4], events, arrays, surfaces,
                slots, domains, protocol, training_seed, device,
            )
            append(prediction_path, evaluate(
                expert[0], values[4], expert[1], expert[2], expert[3], events, arrays,
                surfaces, values[3], fold, training_seed, device, "domain_expert_d1",
            ))
            if parameter_diagnostics is None:
                parameter_diagnostics = {
                    "dense": trainable_parameter_count(dense[0]),
                    "expert": trainable_parameter_count(expert[0]),
                    "gap_fraction": routed_parameter_gap_fraction(dense[0], expert[0]),
                }
            runs.append({
                "fold": fold, "seed": training_seed,
                "dense_history": dense[4], "expert_history": expert[4],
            })
    metrics = {
        "training_units": len(runs), "teacher_fits": len(runs),
        "dense_control_fits": len(runs), "domain_expert_fits": len(runs),
        "runtime_failures": 0, "runs": runs,
        "parameter_diagnostics": parameter_diagnostics,
        "routing": {
            "source": "causal_model_input.track", "task_id_used": False,
            "counts": {name: int(route_counts[index]) for index, name in enumerate(DOMAIN_NAMES)},
        },
        "slot_audit": {
            "rows": len(all_audits),
            "raw_values_encoded": any(row["raw_values_encoded"] for row in all_audits),
            "interface_only_lexical_encoding": all(row["interface_only_lexical_encoding"] for row in all_audits),
            "unmatched_text_tokens_encoded": sum(row["unmatched_text_tokens_encoded"] for row in all_audits),
            "truncated_rows": sum(row["truncated"] for row in all_audits),
            "concept_truncated_rows": sum(row["concepts_truncated"] for row in all_audits),
            "maximum_nodes": max(row["node_count"] for row in all_audits),
        },
        "predictions_sha256": file_sha256(prediction_path),
    }
    if len(runs) != 15:
        raise ValueError("fixed D1 budget incomplete")
    write(args.output_dir / "run_metrics.json", metrics)


if __name__ == "__main__":
    main()
