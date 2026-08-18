"""Train equal-capacity full, exact-protocol, and stochastic-evidence graph oracles."""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wmagentattack.action_event_graph_dynamics import trainable_parameter_count
from wmagentattack.multisource_suitability import file_sha256


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


v5 = _load("v5", "201_train_structured_joint_outcome_v5.py")
v12 = _load("v12", "224_train_action_event_graph_oracle_v12.py")

ARMS = (
    "full_graph_modular_oracle_v14",
    "exact_protocol_oracle_v14",
    "stochastic_evidence_oracle_v14",
)


def _seed(value):
    random.seed(value); np.random.seed(value); torch.manual_seed(value)


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")


def _append(path, rows):
    with path.open("a") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")


def _masks(graph_dataset, partition_dataset):
    catalog = graph_dataset["feature_catalog"]
    if catalog != partition_dataset["full_feature_catalog"]:
        raise ValueError("full catalog mismatch")
    exact = set(partition_dataset["exact_feature_catalog"])
    evidence = set(partition_dataset["evidence_feature_catalog"])
    return {
        "full_graph_modular_oracle_v14": np.ones(len(catalog), dtype=np.float32),
        "exact_protocol_oracle_v14": np.asarray([feature in exact for feature in catalog], dtype=np.float32),
        "stochastic_evidence_oracle_v14": np.asarray([feature in evidence for feature in catalog], dtype=np.float32),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--graph-dataset", type=Path, required=True)
    parser.add_argument("--partition-dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    if protocol["status"] != "partition_data_gate_passed_oracle_attribution_frozen_before_training":
        raise ValueError("v14 oracle protocol not frozen")
    if file_sha256(args.events) != protocol["sources"]["events_sha256"]:
        raise ValueError("events hash mismatch")
    if file_sha256(args.graph_dataset) != protocol["sources"]["event_graph_sha256"]:
        raise ValueError("graph hash mismatch")
    if file_sha256(args.partition_dataset) != protocol["data_gate_result"]["dataset_sha256"]:
        raise ValueError("partition hash mismatch")
    source = json.loads(args.events.read_text())
    graph_dataset = json.loads(args.graph_dataset.read_text())
    partition_dataset = json.loads(args.partition_dataset.read_text())
    masks = _masks(graph_dataset, partition_dataset)
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    if device == "cpu": torch.set_num_threads(8)
    stage = protocol["oracle_attribution_stage"]
    folds = [0] if args.smoke else list(range(stage["folds"]))
    seeds = [stage["seeds"][0]] if args.smoke else stage["seeds"]
    run_protocol = copy.deepcopy(protocol)
    run_protocol["oracle_sufficiency_stage"] = copy.deepcopy(stage)
    if args.smoke:
        run_protocol["teacher_training_protocol"]["training"]["fixed_epochs"] = 1
        run_protocol["oracle_sufficiency_stage"]["training"]["epochs"] = 1
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions = args.output_dir / "predictions.jsonl"
    predictions.write_text("")
    runs = []; parameter_counts = {}; teacher_fits = 0
    for fold in folds:
        events = v5._fold(source, fold)
        arrays = v5._arrays(events, source["candidate_catalog"], 128)
        surfaces = v12._surfaces(events, arrays)
        full_graph = v12._graph_array(events, graph_dataset)
        for training_seed in seeds:
            _seed(training_seed)
            teacher_values = v5._train(
                "structured_joint_aux", events, arrays, run_protocol["teacher_training_protocol"],
                training_seed, device, return_model=True,
            )
            teacher = teacher_values[4]; teacher_fits += 1
            for arm in ARMS:
                graph = full_graph * masks[arm][None, :]
                model, context, teacher_logits, history = v12._train_model(
                    "true_event_graph_oracle_v12", teacher, events, arrays, surfaces, graph,
                    run_protocol, training_seed, device,
                )
                parameter_counts[arm] = trainable_parameter_count(model)
                rows = v12._evaluate(
                    model, teacher, context, teacher_logits, events, arrays, surfaces, graph,
                    "true_event_graph_oracle_v12", fold, training_seed, device,
                )
                for row in rows: row["arm"] = arm
                _append(predictions, rows)
                runs.append({"fold": fold, "seed": training_seed, "arm": arm, "rows": len(rows), "history": history})
    expected = len(folds) * len(seeds) * len(ARMS)
    if len(runs) != expected: raise ValueError("incomplete fit budget")
    if len(set(parameter_counts.values())) != 1: raise ValueError("parameter mismatch")
    _write(args.output_dir / "run_metrics.json", {
        "smoke": args.smoke, "device": device, "training_units": len(runs),
        "teacher_fits": teacher_fits, "runtime_failures": 0,
        "parameter_counts": parameter_counts, "parameter_match": True,
        "runs": runs, "predictions_sha256": file_sha256(predictions),
    })


if __name__ == "__main__": main()
