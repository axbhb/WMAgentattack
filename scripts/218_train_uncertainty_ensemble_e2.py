"""Stage E2: uniform probabilistic ensembles of frozen-architecture v6 models."""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from wmagentattack.joint_outcome_auxiliary import JOINT_OUTCOME_CLASSES
from wmagentattack.multisource_suitability import file_sha256
from wmagentattack.uncertainty_ensemble import uniform_categorical_ensemble


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


v6 = load("v6", ROOT / "scripts/203_train_structured_residual_v6.py")
v5 = v6.v5


def write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def append(path: Path, rows: list[dict]) -> None:
    with path.open("a") as handle:
        for row in rows: handle.write(json.dumps(row, sort_keys=True) + "\n")


def member_predictions(model, teacher, context, teacher_logits, events, arrays, surfaces, device):
    candidates = torch.tensor(arrays["candidate_inputs"], dtype=torch.float32, device=device)
    output = {}
    model.eval(); teacher.eval()
    with torch.no_grad():
        for horizon in range(1, 6):
            surface = surfaces[horizon]
            keep = np.asarray([events[i]["split"] == "confirmation" for i in surface["starts"]])
            starts_np = surface["starts"][keep]; starts = torch.tensor(starts_np, device=device)
            legal_np = surface["legals"][keep]
            if horizon == 1:
                hidden = context[starts]
                logits = teacher_logits[starts] + model.one_step_delta_logits(hidden, candidates)
                probability = F.softmax(logits.masked_fill(
                    ~torch.tensor(legal_np[:, -1], dtype=torch.bool, device=device),
                    torch.finfo(torch.float32).min,
                ), 1)
                joint = torch.softmax(teacher.joint_outcome_head(hidden), 1)
            else:
                hidden = context[starts]
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
                joint = torch.softmax(model.joint_logits(hidden), 1)
            probabilities = probability.cpu().numpy(); joint_probabilities = joint.cpu().numpy()
            targets = surface["targets"][keep]
            for offset, event_index in enumerate(starts_np):
                event = events[event_index]
                output[(horizon, event["event_id"])] = {
                    "event": event, "target": int(targets[offset]),
                    "legal": legal_np[offset, -1],
                    "probability": probabilities[offset],
                    "joint_probability": joint_probabilities[offset],
                }
    return output


def result_row(record, probability, joint_probability, *, fold, group, arm, member_seed=None, uncertainty=None):
    event = record["event"]; target = record["target"]
    y = np.asarray([event["joint_outcome_target"][name] for name in JOINT_OUTCOME_CLASSES]) if event["joint_outcome_trainable"] else None
    row = {
        "arm": arm, "fold": fold, "ensemble_group": group,
        "member_seed": member_seed, "horizon": int(record["horizon"]),
        "event_id": event["event_id"], "task_name": event["task_name"],
        "trajectory_id": event["trajectory_id"], "joint_group_id": event["joint_outcome_group_id"],
        "action_nll": float(-math.log(max(float(probability[target]), 1e-12))),
        "action_correct": float(probability.argmax() == target),
        "legal_prediction": float(record["legal"][probability.argmax()]),
        "joint_trainable": float(y is not None),
        "joint_ce": float(-(y * np.log(np.clip(joint_probability, 1e-12, 1))).sum()) if y is not None else None,
        "predictive_entropy": None, "expected_member_entropy": None, "epistemic_mi": None,
    }
    if uncertainty is not None:
        row.update({
            "predictive_entropy": float(uncertainty[0]),
            "expected_member_entropy": float(uncertainty[1]),
            "epistemic_mi": float(uncertainty[2]),
        })
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(); protocol = json.loads(args.protocol.read_text())
    if protocol["status"] != "stage_e1_no_go_stage_e2_frozen_before_training":
        raise ValueError("E2 not authorized and frozen")
    frozen = protocol["frozen_dataset"]
    if file_sha256(args.dataset) != frozen["sha256"] or file_sha256(args.audit) != frozen["audit_sha256"]:
        raise ValueError("frozen data mismatch")
    frozen_v6 = protocol["stage_e2_direction_switch"]["frozen_v6_protocol"]
    if file_sha256(ROOT / frozen_v6["path"]) != frozen_v6["sha256"]:
        raise ValueError("frozen v6 protocol mismatch")
    data = json.loads(args.dataset.read_text()); device = "cpu"; torch.set_num_threads(8)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = args.output_dir / "predictions.jsonl"; prediction_path.write_text("")
    stage = protocol["stage_e2_direction_switch"]
    teacher_protocol = {"training": protocol["teacher"]}
    residual_protocol = {"residual_training": stage["residual_training"]}
    runs = []; fit_count = 0
    for fold in range(protocol["research_budget"]["folds"]):
        events = v5._fold(data, fold); arrays = v5._arrays(events, data["candidate_catalog"], 128)
        surfaces = v6.horizons(events, arrays)
        for group, member_seeds in stage["ensemble_groups"].items():
            members = []
            for member_seed in member_seeds:
                v6.seed(member_seed)
                values = v5._train(
                    "structured_joint_aux", events, arrays, teacher_protocol,
                    member_seed, device, return_model=True,
                )
                model, context, teacher_logits, history = v6.train_residual(
                    values[4], events, arrays, surfaces, residual_protocol,
                    member_seed, device,
                )
                members.append(member_predictions(
                    model, values[4], context, teacher_logits, events, arrays, surfaces, device
                ))
                runs.append({"fold": fold, "ensemble_group": group, "member_seed": member_seed, "history": history})
                fit_count += 1
            keys = list(members[0])
            if any(set(member) != set(keys) for member in members[1:]):
                raise ValueError("ensemble member surface mismatch")
            rows = []
            for key in keys:
                records = [member[key] for member in members]
                probabilities = np.stack([record["probability"] for record in records])
                joints = np.stack([record["joint_probability"] for record in records])
                mixture, predictive, expected, epistemic = uniform_categorical_ensemble(probabilities)
                joint_mixture = joints.mean(0)
                base = dict(records[0]); base["horizon"] = key[0]
                member_rows = [
                    result_row(
                        {**record, "horizon": key[0]}, record["probability"], record["joint_probability"],
                        fold=fold, group=int(group), arm="member_v6", member_seed=int(seed),
                    ) for record, seed in zip(records, member_seeds)
                ]
                mean_member = dict(member_rows[0])
                mean_member.update({
                    "arm": "mean_member_v6", "member_seed": None,
                    "action_nll": float(np.mean([row["action_nll"] for row in member_rows])),
                    "action_correct": float(np.mean([row["action_correct"] for row in member_rows])),
                    "legal_prediction": float(min(row["legal_prediction"] for row in member_rows)),
                    "joint_ce": float(np.mean([row["joint_ce"] for row in member_rows])) if member_rows[0]["joint_ce"] is not None else None,
                })
                rows.extend(member_rows); rows.append(mean_member)
                rows.append(result_row(
                    base, mixture, joint_mixture, fold=fold, group=int(group),
                    arm="uncertainty_ensemble_e2",
                    uncertainty=(float(predictive), float(expected), float(epistemic)),
                ))
            append(prediction_path, rows)
    metrics = {
        "training_units": len(runs), "teacher_fits": fit_count,
        "residual_fits": fit_count, "ensemble_evaluations": 15,
        "runtime_failures": 0, "runs": runs,
        "uniform_member_weights": True, "confirmation_tuned_parameters": 0,
        "predictions_sha256": file_sha256(prediction_path),
    }
    if fit_count != 45:
        raise ValueError("fixed E2 fit budget incomplete")
    write(args.output_dir / "run_metrics.json", metrics)


if __name__ == "__main__":
    main()
