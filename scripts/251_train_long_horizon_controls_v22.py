"""Train frozen v6 teacher/residual fits and expose repeated/teacher/free rollouts."""

from __future__ import annotations

import argparse
import copy
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

from wmagentattack.multisource_suitability import file_sha256


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


v6 = load("structured_residual_v6", ROOT / "scripts" / "203_train_structured_residual_v6.py")
v5 = v6.v5


def append(path: Path, rows: list[dict]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def evaluate_controls(
    model,
    teacher,
    context,
    teacher_logits,
    events,
    arrays,
    surfaces,
    fold: int,
    seed: int,
    device: str,
) -> list[dict]:
    candidates = torch.tensor(arrays["candidate_inputs"], dtype=torch.float32, device=device)
    rows = []
    model.eval()
    teacher.eval()
    with torch.no_grad():
        for horizon in (1, 2, 3, 5, 10):
            surface = surfaces[horizon]
            keep = np.asarray([
                events[int(index)]["split"] == "confirmation"
                for index in surface["starts"]
            ])
            if not keep.any():
                continue
            starts_np = surface["starts"][keep]
            future_np = surface["future"][keep]
            paths_np = surface["paths"][keep]
            legal_np = surface["legals"][keep]
            targets = surface["targets"][keep]
            starts = torch.tensor(starts_np, dtype=torch.long, device=device)
            future = torch.tensor(future_np, dtype=torch.long, device=device)
            legal_final = torch.tensor(legal_np[:, -1], dtype=torch.bool, device=device)

            repeated_logits = teacher_logits[future].masked_fill(
                ~legal_final, torch.finfo(torch.float32).min
            )
            repeated = F.softmax(repeated_logits, dim=1)

            if horizon == 1:
                residual_logits = teacher_logits[starts] + model.one_step_delta_logits(
                    context[starts], candidates
                )
                residual_logits = residual_logits.masked_fill(
                    ~legal_final, torch.finfo(torch.float32).min
                )
                teacher_forced = F.softmax(residual_logits, dim=1)
                free = teacher_forced
            else:
                teacher_hidden = context[starts]
                true_paths = torch.tensor(paths_np, dtype=torch.long, device=device)
                for step in range(1, horizon):
                    teacher_hidden = model.advance(
                        teacher_hidden, candidates[true_paths[:, step]]
                    )
                teacher_forced = F.softmax(
                    model.rollout_logits(teacher_hidden, candidates).masked_fill(
                        ~legal_final, torch.finfo(torch.float32).min
                    ), dim=1,
                )

                free_hidden = context[starts]
                first_legal = torch.tensor(legal_np[:, 0], dtype=torch.bool, device=device)
                free_probability = F.softmax(
                    teacher_logits[starts].masked_fill(
                        ~first_legal, torch.finfo(torch.float32).min
                    ), dim=1,
                )
                for step in range(1, horizon):
                    free_hidden = model.advance(free_hidden, free_probability @ candidates)
                    step_legal = torch.tensor(
                        legal_np[:, step], dtype=torch.bool, device=device
                    )
                    free_probability = F.softmax(
                        model.rollout_logits(free_hidden, candidates).masked_fill(
                            ~step_legal, torch.finfo(torch.float32).min
                        ), dim=1,
                    )
                free = free_probability

            for control, probabilities in (
                ("one_step_repeated", repeated),
                ("teacher_forced_residual", teacher_forced),
                ("free_latent_residual", free),
            ):
                values = probabilities.cpu().numpy()
                for local, start in enumerate(starts_np):
                    event = events[int(start)]
                    target = int(targets[local])
                    predicted = int(values[local].argmax())
                    target_probability = max(float(values[local, target]), 1e-12)
                    rows.append({
                        "control": control,
                        "fold": fold,
                        "training_seed": seed,
                        "horizon": horizon,
                        "event_id": event["event_id"],
                        "task_name": event["task_name"],
                        "trajectory_id": event["trajectory_id"],
                        "step_id": event["step_id"],
                        "action_nll": float(-math.log(target_probability)),
                        "action_correct": float(predicted == target),
                        "legal_prediction": float(bool(legal_np[local, -1, predicted])),
                    })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if protocol["status"] != "preregistered_before_results":
        raise ValueError("v22 protocol is not frozen")
    cfg = protocol["long_horizon_gate"]
    if file_sha256(args.dataset) != cfg["agentdojo_adjacent_dataset"]["sha256"]:
        raise ValueError("long-horizon dataset hash mismatch")
    if file_sha256(args.audit) != cfg["agentdojo_adjacent_dataset"]["audit_sha256"]:
        raise ValueError("long-horizon audit hash mismatch")
    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    device = "cpu"
    torch.set_num_threads(8)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions = args.output_dir / "predictions.jsonl"
    predictions.write_text("", encoding="utf-8")
    runs = []
    for fold in range(5):
        events = v5._fold(dataset, fold)
        arrays = v5._arrays(events, dataset["candidate_catalog"], 128)
        surfaces = v6.horizons(events, arrays, max_h=10)
        for seed in cfg["training_seeds"]:
            v6.seed(int(seed))
            teacher_values = v5._train(
                "structured_joint_aux", events, arrays,
                copy.deepcopy(cfg["teacher_training_protocol"]),
                int(seed), device, return_model=True,
            )
            teacher = teacher_values[4]
            model, context, logits, history = v6.train_residual(
                teacher, events, arrays, surfaces, cfg, int(seed), device
            )
            rows = evaluate_controls(
                model, teacher, context, logits, events, arrays, surfaces,
                fold, int(seed), device,
            )
            append(predictions, rows)
            runs.append({
                "fold": fold,
                "seed": int(seed),
                "prediction_rows": len(rows),
                "teacher_history": teacher_values[3]["history"],
                "residual_history": history,
            })
    expected = int(cfg["fixed_budget_if_data_go"]["agentdojo_residual_fits"])
    if len(runs) != expected:
        raise RuntimeError(f"long-horizon fixed budget incomplete: {len(runs)} != {expected}")
    metrics = {
        "schema_version": "wmagentattack.long_horizon_controls.v22",
        "dataset_sha256": file_sha256(args.dataset),
        "predictions_sha256": file_sha256(predictions),
        "teacher_fits": expected,
        "residual_fits": expected,
        "completed_paired_fit_units": len(runs),
        "runtime_failures": 0,
        "device": device,
        "runs": runs,
    }
    (args.output_dir / "run_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"paired_fit_units": len(runs), "runtime_failures": 0}))


if __name__ == "__main__":
    main()
