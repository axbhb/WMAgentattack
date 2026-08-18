"""Stage E1: non-lexical relational signature, dense versus sparse routing."""
from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from wmagentattack.relational_router_residual import (
    DenseRelationalSignatureResidual,
    SparseRelationalSignatureResidual,
    parameter_gap_fraction,
    stack_relation_signature_features,
    trainable_parameter_count,
)
from wmagentattack.relational_slot_latent import stack_interface_affordance_states
from wmagentattack.multisource_suitability import file_sha256


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


d1 = load("d1", ROOT / "scripts/214_train_domain_expert_d1.py")
v5 = d1.v5
v6 = d1.v6


def tensors(arrays, slots, unused, device):
    del unused
    rows = len(arrays["states"])
    return (
        torch.tensor(arrays["states"], dtype=torch.float32, device=device),
        torch.tensor(arrays["candidate_inputs"], dtype=torch.float32, device=device),
        torch.tensor(arrays["selected"], dtype=torch.long, device=device),
        torch.tensor(slots["signature"], dtype=torch.float32, device=device),
        torch.zeros(rows, dtype=torch.long, device=device),
        torch.zeros(rows, dtype=torch.long, device=device),
        torch.ones(rows, dtype=torch.bool, device=device),
        torch.zeros(rows, dtype=torch.long, device=device),
    )


def build_model(arm, candidates, signatures, protocol, device):
    cfg = protocol["stage_e1"]
    common = dict(
        candidate_size=candidates.shape[1], route_feature_size=signatures.shape[1],
        hidden_size=cfg["training"]["hidden_size"], dropout=cfg["training"]["dropout"],
    )
    if arm == "dense_relation_control":
        return DenseRelationalSignatureResidual(
            **common, dense_bottleneck_size=cfg["dense_control"]["bottleneck_size"]
        ).to(device)
    if arm == "sparse_relation_e1":
        return SparseRelationalSignatureResidual(
            **common, experts=cfg["sparse_router"]["experts"],
            active_experts=cfg["sparse_router"]["active_experts"],
            expert_bottleneck_size=cfg["sparse_router"]["expert_bottleneck_size"],
            router_hidden_size=cfg["sparse_router"]["router_hidden_size"],
        ).to(device)
    raise ValueError(arm)


d1.tensors = tensors
d1.build_model = build_model


def restore_rng(state) -> None:
    random.setstate(state[0]); np.random.set_state(state[1]); torch.set_rng_state(state[2])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(); protocol = json.loads(args.protocol.read_text())
    if protocol["status"] != "preregistered_stage_e1_before_training":
        raise ValueError("E1 protocol not frozen")
    frozen = protocol["frozen_dataset"]
    if file_sha256(args.dataset) != frozen["sha256"] or file_sha256(args.audit) != frozen["audit_sha256"]:
        raise ValueError("frozen data mismatch")
    data = json.loads(args.dataset.read_text()); torch.set_num_threads(8); device = "cpu"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = args.output_dir / "predictions.jsonl"; prediction_path.write_text("")
    runs = []; audits = []; route_totals = np.zeros(4); hard_totals = np.zeros(4, dtype=np.int64)
    entropies = []; active_counts = []; parameter_diagnostics = None
    builder = protocol["stage_e1"]["relation_signature"]
    teacher_protocol = {"training": protocol["teacher"]}
    training_protocol = dict(protocol)
    training_protocol["stage_d1"] = protocol["stage_e1"]
    for fold in range(protocol["research_budget"]["folds"]):
        events = v5._fold(data, fold); arrays = v5._arrays(events, data["candidate_catalog"], 128)
        surfaces = v6.horizons(events, arrays)
        graph = stack_interface_affordance_states(
            events, hash_dimension=builder["removed_hash_dimension"],
            max_nodes=builder["max_nodes"], max_concepts=builder["max_concepts"],
        )
        signature = stack_relation_signature_features(
            graph, hash_dimension=builder["removed_hash_dimension"]
        )
        slots = {"signature": signature}; audits.extend(graph["audit"])
        unused = np.zeros(len(events), dtype=np.int64)
        for training_seed in protocol["research_budget"]["seeds"]:
            v6.seed(training_seed)
            values = v5._train(
                "structured_joint_aux", events, arrays, teacher_protocol,
                training_seed, device, return_model=True,
            )
            rng = (random.getstate(), np.random.get_state(), torch.get_rng_state())
            dense = d1.train_arm(
                "dense_relation_control", values[4], events, arrays, surfaces,
                slots, unused, training_protocol, training_seed, device,
            )
            d1.append(prediction_path, d1.evaluate(
                dense[0], values[4], dense[1], dense[2], dense[3], events, arrays,
                surfaces, values[3], fold, training_seed, device, "dense_relation_control",
            ))
            restore_rng(rng)
            sparse = d1.train_arm(
                "sparse_relation_e1", values[4], events, arrays, surfaces,
                slots, unused, training_protocol, training_seed, device,
            )
            d1.append(prediction_path, d1.evaluate(
                sparse[0], values[4], sparse[1], sparse[2], sparse[3], events, arrays,
                surfaces, values[3], fold, training_seed, device, "sparse_relation_e1",
            ))
            with torch.no_grad():
                weights = sparse[0].routing_weights(
                    torch.tensor(signature, dtype=torch.float32, device=device)
                ).cpu().numpy()
            route_totals += weights.sum(0); hard_totals += (weights > 0).sum(0)
            active_counts.extend((weights > 0).sum(1).tolist())
            entropy = -(weights * np.log(np.clip(weights, 1e-12, 1))).sum(1) / np.log(2)
            entropies.extend(entropy.tolist())
            if parameter_diagnostics is None:
                parameter_diagnostics = {
                    "dense": trainable_parameter_count(dense[0]),
                    "sparse": trainable_parameter_count(sparse[0]),
                    "gap_fraction": parameter_gap_fraction(dense[0], sparse[0]),
                }
            runs.append({
                "fold": fold, "seed": training_seed,
                "dense_history": dense[4], "sparse_history": sparse[4],
            })
    total_weight = route_totals.sum()
    metrics = {
        "training_units": len(runs), "teacher_fits": len(runs),
        "dense_control_fits": len(runs), "sparse_relation_fits": len(runs),
        "runtime_failures": 0, "runs": runs,
        "parameter_diagnostics": parameter_diagnostics,
        "routing": {
            "source": "nonlexical_node_relation_numeric_signature",
            "task_id_used": False, "track_used": False, "label_used": False,
            "soft_load": (route_totals / total_weight).tolist(),
            "hard_counts": hard_totals.tolist(),
            "mean_active_experts": float(np.mean(active_counts)),
            "mean_normalized_topk_entropy": float(np.mean(entropies)),
            "maximum_soft_load": float((route_totals / total_weight).max()),
        },
        "signature_audit": {
            "rows": len(audits), "signature_dimension": int(signature.shape[1]),
            "lexical_hash_coordinates_used": False, "raw_values_encoded": False,
            "unmatched_text_tokens_encoded": sum(row["unmatched_text_tokens_encoded"] for row in audits),
            "truncated_rows": sum(row["truncated"] for row in audits),
            "concept_truncated_rows": sum(row["concepts_truncated"] for row in audits),
        },
        "predictions_sha256": file_sha256(prediction_path),
    }
    if len(runs) != 15:
        raise ValueError("fixed E1 budget incomplete")
    d1.write(args.output_dir / "run_metrics.json", metrics)


if __name__ == "__main__":
    main()
