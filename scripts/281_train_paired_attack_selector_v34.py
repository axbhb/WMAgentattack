"""Train the frozen pre-execution v34 attack selectors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch
from sklearn.feature_extraction import DictVectorizer
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wmagentattack.attack_conditioned_ranker import (
    AttackConditionedResidualRanker,
    pairwise_logistic_loss,
    task_balanced_weights,
    within_task_pairs,
)
from wmagentattack.joint_outcome_auxiliary import JOINT_OUTCOME_CLASSES
from wmagentattack.paired_attack_intervention import (
    CONFIRMATION_TASKS,
    FactorizedStateAttackSelector,
    aggregate_paired_results,
    split_preexecution_features,
    targets_from_legacy_labels,
)


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def _prepare_vectors(train_rows: list[dict], test_rows: list[dict]):
    train_parts = [split_preexecution_features(row) for row in train_rows]
    test_parts = [split_preexecution_features(row) for row in test_rows]
    state_vectorizer = DictVectorizer(sparse=False, sort=True)
    action_vectorizer = DictVectorizer(sparse=False, sort=True)
    state_train = state_vectorizer.fit_transform([state for state, _ in train_parts]).astype(np.float32)
    state_test = state_vectorizer.transform([state for state, _ in test_parts]).astype(np.float32)
    action_train = action_vectorizer.fit_transform([action for _, action in train_parts]).astype(np.float32)
    action_test = action_vectorizer.transform([action for _, action in test_parts]).astype(np.float32)
    return state_train, action_train, state_test, action_test


def _train_one(
    *,
    arm: str,
    train_rows: list[dict],
    test_rows: list[dict],
    config: dict,
    seed: int,
) -> tuple[list[dict], dict]:
    state_train, action_train, state_test, action_test = _prepare_vectors(train_rows, test_rows)
    target = np.asarray([row["target"] for row in train_rows], dtype=np.float32)
    weights = task_balanced_weights(train_rows)
    pairs = within_task_pairs(train_rows, minimum_target_gap=float(config["minimum_pair_target_gap"]))
    prior = torch.tensor(np.average(target, axis=0, weights=weights), dtype=torch.float32)
    train_base = prior[None, :].repeat(len(train_rows), 1)
    test_base = prior[None, :].repeat(len(test_rows), 1)

    _seed(seed)
    if arm == "structured_preexecution":
        train_x = np.concatenate([state_train, action_train], axis=1)
        test_x = np.concatenate([state_test, action_test], axis=1)
        model = AttackConditionedResidualRanker(
            input_size=train_x.shape[1],
            hidden_size=int(config["hidden_size"]),
            dropout=float(config["dropout"]),
        )
    elif arm == "factorized_state_attack":
        train_x = None
        test_x = None
        model = FactorizedStateAttackSelector(
            state_size=state_train.shape[1],
            action_size=action_train.shape[1],
            hidden_size=int(config["hidden_size"]),
        )
    else:
        raise ValueError(f"unknown v34 arm: {arm}")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    y = torch.tensor(target)
    w = torch.tensor(weights)
    pair_tensor = torch.tensor(pairs)
    state_tensor = torch.tensor(state_train)
    action_tensor = torch.tensor(action_train)
    history = []
    for epoch in range(int(config["epochs"])):
        _seed(seed * 1009 + epoch)
        model.train()
        logits = (
            model(torch.tensor(train_x), train_base)
            if train_x is not None
            else model(state_tensor, action_tensor, train_base)
        )
        per_row = -(y * F.log_softmax(logits, dim=1)).sum(dim=1)
        cell_loss = (per_row * w).sum() / w.sum()
        p11 = torch.softmax(logits, dim=1)[:, 3]
        rank_loss = pairwise_logistic_loss(p11, pair_tensor)
        residual = logits - torch.log(train_base.clamp_min(1e-8))
        anchor_loss = residual.square().mean()
        loss = (
            cell_loss
            + float(config["pairwise_loss_weight"]) * rank_loss
            + float(config["zero_start_anchor_weight"]) * anchor_loss
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["gradient_clip"]))
        optimizer.step()
        if epoch in (0, int(config["epochs"]) - 1):
            history.append(
                {
                    "epoch": epoch,
                    "loss": float(loss.detach()),
                    "four_cell_loss": float(cell_loss.detach()),
                    "rank_loss": float(rank_loss.detach()),
                    "anchor_loss": float(anchor_loss.detach()),
                }
            )
    model.eval()
    with torch.no_grad():
        logits = (
            model(torch.tensor(test_x), test_base)
            if test_x is not None
            else model(torch.tensor(state_test), torch.tensor(action_test), test_base)
        )
        probabilities = torch.softmax(logits, dim=1).numpy()
    predictions = []
    for row, probability in zip(test_rows, probabilities, strict=True):
        predictions.append(
            {
                "arm": arm,
                "seed": seed,
                "row_id": row["row_id"],
                "task_name": row["task_name"],
                "attack_variant": row["attack_variant"],
                "target": row["target"],
                "target_p11": row["target_p11"],
                "predicted_p11": float(probability[3]),
                **{
                    f"prob_{name}": float(probability[index])
                    for index, name in enumerate(JOINT_OUTCOME_CLASSES)
                },
            }
        )
    return predictions, {
        "arm": arm,
        "seed": seed,
        "training_tasks": len({row["task_name"] for row in train_rows}),
        "training_rows": len(train_rows),
        "confirmation_tasks": len({row["task_name"] for row in test_rows}),
        "confirmation_rows": len(test_rows),
        "state_width": int(state_train.shape[1]),
        "action_width": int(action_train.shape[1]),
        "pairwise_training_pairs": len(pairs),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "history": history,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--legacy-manifest", type=Path, required=True)
    parser.add_argument("--legacy-label-groups", type=Path, required=True)
    parser.add_argument("--paired-manifest", type=Path, required=True)
    parser.add_argument("--seed-result", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if protocol["status"] != "preregistered_before_results":
        raise ValueError("v34 protocol is not frozen")
    legacy = json.loads(args.legacy_manifest.read_text(encoding="utf-8"))
    paired = json.loads(args.paired_manifest.read_text(encoding="utf-8"))
    seed_results = [json.loads(path.read_text(encoding="utf-8")) for path in args.seed_result]
    paired_rows, data_audit = aggregate_paired_results(
        manifest_rows=paired["rows"],
        seed_results=seed_results,
        expected_seeds=protocol["execution"]["victim_seeds"],
    )
    confirmation = [row for row in paired_rows if row["attack_kind"] == "paired_factor"]
    training = targets_from_legacy_labels(
        legacy["rows"],
        _read_jsonl(args.legacy_label_groups),
        excluded_tasks=CONFIRMATION_TASKS,
    )
    if len(training) != 240 or len({row["task_name"] for row in training}) != 12:
        raise RuntimeError("v34 historical training split is not 240 rows / 12 tasks")
    if len(confirmation) != 32:
        raise RuntimeError("v34 confirmation split is not 32 attack rows")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write(args.output_dir / "data_audit.json", data_audit)
    _write(args.output_dir / "paired_targets.json", paired_rows)
    all_predictions = []
    runs = []
    for arm in ("structured_preexecution", "factorized_state_attack"):
        for seed in protocol["training"]["model_seeds"]:
            predictions, run = _train_one(
                arm=arm,
                train_rows=training,
                test_rows=confirmation,
                config=protocol["training"],
                seed=int(seed),
            )
            all_predictions.extend(predictions)
            runs.append(run)
    expected_fits = int(protocol["fixed_budget"]["model_fits"])
    if len(runs) != expected_fits:
        raise RuntimeError(f"incomplete v34 model budget: {len(runs)} != {expected_fits}")
    prediction_path = args.output_dir / "predictions.jsonl"
    prediction_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in all_predictions),
        encoding="utf-8",
    )
    _write(
        args.output_dir / "run_metrics.json",
        {
            "model_fits": len(runs),
            "runtime_failures": 0,
            "historical_training_rows": len(training),
            "historical_training_tasks": 12,
            "paired_confirmation_rows": len(confirmation),
            "paired_confirmation_tasks": 8,
            "victim_episodes": sum(item["summary"]["completed"] for item in seed_results),
            "real_external_endpoint_calls": 0,
            "runs": runs,
        },
    )


if __name__ == "__main__":
    main()
