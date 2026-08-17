"""Stage C1: interface-aligned affordance graph on the frozen v6 residual."""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from wmagentattack.multisource_suitability import file_sha256
from wmagentattack.relational_slot_latent import stack_interface_affordance_states


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


v5 = load("v5", ROOT / "scripts/201_train_structured_joint_outcome_v5.py")
v6 = load("v6", ROOT / "scripts/203_train_structured_residual_v6.py")
stage_a = load("stage_a", ROOT / "scripts/205_train_relational_slot_stage_a.py")


def write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def append(path: Path, rows: list[dict]) -> None:
    with path.open("a") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    data = json.loads(args.dataset.read_text())
    if protocol["status"] != "preregistered_stage_c1_before_training":
        raise ValueError("C1 protocol is not frozen")
    if file_sha256(args.dataset) != protocol["frozen_dataset"]["sha256"]:
        raise ValueError("frozen dataset mismatch")
    if file_sha256(args.audit) != protocol["frozen_dataset"]["audit_sha256"]:
        raise ValueError("frozen audit mismatch")
    preaudit = Path(protocol["label_blind_preaudit"]["path"])
    if file_sha256(preaudit) != protocol["label_blind_preaudit"]["sha256"]:
        raise ValueError("frozen affordance preaudit mismatch")

    torch.set_num_threads(8); device = "cpu"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = args.output_dir / "predictions.jsonl"
    prediction_path.write_text("")
    runs = []; all_audits = []
    builder = protocol["stage_c1"]["affordance_builder"]
    proxy = {
        "stage_a": {
            "residual": protocol["stage_c1"]["residual"],
            "slot_builder": {"message_layers": builder["message_layers"]},
        }
    }
    teacher_protocol = {"training": protocol["teacher"]}
    for fold in range(protocol["research_budget"]["folds"]):
        events = v5._fold(data, fold)
        arrays = v5._arrays(events, data["candidate_catalog"], 128)
        surfaces = v6.horizons(events, arrays)
        slots = stack_interface_affordance_states(
            events, hash_dimension=builder["hash_dimension"],
            max_nodes=builder["max_nodes"], max_concepts=builder["max_concepts"],
        )
        all_audits.extend(slots["audit"])
        for training_seed in protocol["research_budget"]["seeds"]:
            v6.seed(training_seed)
            teacher_values = v5._train(
                "structured_joint_aux", events, arrays, teacher_protocol,
                training_seed, device, return_model=True,
            )
            candidate = stage_a.train_slot(
                teacher_values[4], events, arrays, surfaces, slots,
                proxy, training_seed, device,
            )
            rows = stage_a.evaluate_slot(
                candidate[0], teacher_values[4], candidate[1], candidate[2], candidate[3],
                events, arrays, surfaces, teacher_values[3], fold, training_seed, device,
                arm="interface_affordance_c1",
            )
            append(prediction_path, rows)
            runs.append({
                "fold": fold, "seed": training_seed,
                "history": candidate[4], "prediction_rows": len(rows),
            })
    metrics = {
        "training_units": len(runs), "teacher_fits": len(runs),
        "affordance_residual_fits": len(runs), "runtime_failures": 0,
        "runs": runs,
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
        raise ValueError("fixed C1 budget incomplete")
    write(args.output_dir / "run_metrics.json", metrics)


if __name__ == "__main__":
    main()
