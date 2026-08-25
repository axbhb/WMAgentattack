"""Train task-disjoint attack-conditioned four-cell residual rankers."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.feature_extraction import DictVectorizer
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wmagentattack.attack_conditioned_ranker import (
    AttackConditionedResidualRanker,
    align_attack_candidates,
    base_distribution_from_p11,
    pairwise_logistic_loss,
    task_balanced_weights,
    within_task_pairs,
)
from wmagentattack.joint_outcome_auxiliary import JOINT_OUTCOME_CLASSES


ARMS = (
    "structured_attack_residual",
    "world_attack_residual",
    "world_family_diagnostic",
)


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _append(path: Path, values: list[dict]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        for value in values:
            handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def _base_probabilities(rows: list[dict], prior: torch.Tensor, use_world: bool) -> torch.Tensor:
    if not use_world:
        return prior[None, :].repeat(len(rows), 1)
    p11 = torch.tensor([row["v5_p11"] for row in rows], dtype=torch.float32)
    return base_distribution_from_p11(p11, prior)


def _train_one(
    *,
    arm: str,
    train_rows: list[dict],
    test_rows: list[dict],
    config: dict,
    seed: int,
) -> tuple[list[dict], dict]:
    include_family = arm == "world_family_diagnostic"
    feature_key = "family_features" if include_family else "features"
    vectorizer = DictVectorizer(sparse=False, sort=True)
    train_x = vectorizer.fit_transform([row[feature_key] for row in train_rows]).astype(np.float32)
    test_x = vectorizer.transform([row[feature_key] for row in test_rows]).astype(np.float32)
    target = np.asarray([row["target"] for row in train_rows], dtype=np.float32)
    weights = task_balanced_weights(train_rows)
    pairs = within_task_pairs(train_rows, minimum_target_gap=float(config["minimum_pair_target_gap"]))
    prior = torch.tensor(np.average(target, axis=0, weights=weights), dtype=torch.float32)
    use_world = arm != "structured_attack_residual"
    train_base = _base_probabilities(train_rows, prior, use_world)
    test_base = _base_probabilities(test_rows, prior, use_world)

    _seed(seed)
    model = AttackConditionedResidualRanker(
        input_size=train_x.shape[1],
        hidden_size=int(config["hidden_size"]),
        dropout=float(config["dropout"]),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    x = torch.tensor(train_x)
    y = torch.tensor(target)
    w = torch.tensor(weights)
    pair_tensor = torch.tensor(pairs)
    history = []
    for epoch in range(int(config["epochs"])):
        _seed(seed * 1009 + epoch)
        model.train()
        logits = model(x, train_base)
        per_row = -(y * F.log_softmax(logits, dim=1)).sum(dim=1)
        joint_loss = (per_row * w).sum() / w.sum()
        p11 = torch.softmax(logits, dim=1)[:, 3]
        rank_loss = pairwise_logistic_loss(p11, pair_tensor)
        loss = joint_loss + float(config["pairwise_loss_weight"]) * rank_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["gradient_clip"]))
        optimizer.step()
        if epoch in (0, int(config["epochs"]) - 1):
            history.append(
                {
                    "epoch": epoch,
                    "loss": float(loss.detach()),
                    "joint_loss": float(joint_loss.detach()),
                    "rank_loss": float(rank_loss.detach()),
                }
            )
    model.eval()
    with torch.no_grad():
        probabilities = torch.softmax(model(torch.tensor(test_x), test_base), dim=1).numpy()
    predictions = []
    for row, probability in zip(test_rows, probabilities, strict=True):
        predictions.append(
            {
                "fold": int(row["fold"]),
                "seed": seed,
                "arm": arm,
                "row_id": row["row_id"],
                "task_name": row["task_name"],
                "attack_family": row["attack_family"],
                "target": row["target"],
                "target_p11": row["target_p11"],
                "v5_p11": row["v5_p11"],
                **{
                    f"prob_{name}": float(probability[index])
                    for index, name in enumerate(JOINT_OUTCOME_CLASSES)
                },
                "predicted_p11": float(probability[3]),
            }
        )
    return predictions, {
        "arm": arm,
        "seed": seed,
        "feature_width": int(train_x.shape[1]),
        "training_candidates": len(train_rows),
        "test_candidates": len(test_rows),
        "pairwise_training_pairs": len(pairs),
        "prior": prior.tolist(),
        "history": history,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--label-groups", type=Path, required=True)
    parser.add_argument("--v5-predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if protocol["status"] != "preregistered_before_results":
        raise ValueError("v33 protocol is not frozen")
    manifest_payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    rows, audit = align_attack_candidates(
        manifest_rows=manifest_payload["rows"],
        label_groups=_read_jsonl(args.label_groups),
        v5_predictions=_read_jsonl(args.v5_predictions),
    )
    if not audit["passed"]:
        raise RuntimeError(f"v33 alignment audit failed: {audit['checks']}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write(args.output_dir / "alignment_audit.json", audit)
    prediction_path = args.output_dir / "predictions.jsonl"
    prediction_path.write_text("", encoding="utf-8")
    runs = []
    for fold in range(5):
        train_rows = [row for row in rows if int(row["fold"]) != fold]
        test_rows = [row for row in rows if int(row["fold"]) == fold]
        if len(train_rows) != 320 or len(test_rows) != 80:
            raise ValueError(f"unexpected fold {fold}: {len(train_rows)}/{len(test_rows)}")
        for arm in ARMS:
            for seed in protocol["training"]["seeds"]:
                predictions, run = _train_one(
                    arm=arm,
                    train_rows=train_rows,
                    test_rows=test_rows,
                    config=protocol["training"],
                    seed=int(seed),
                )
                _append(prediction_path, predictions)
                runs.append({"fold": fold, **run})
    expected = int(protocol["fixed_budget"]["model_fits"])
    if len(runs) != expected:
        raise RuntimeError(f"incomplete model budget: {len(runs)} != {expected}")
    _write(
        args.output_dir / "run_metrics.json",
        {
            "model_fits": len(runs),
            "runtime_failures": 0,
            "attack_examples_generated": 0,
            "victim_llm_calls": 0,
            "sandbox_tool_calls": 0,
            "real_external_endpoint_calls": 0,
            "runs": runs,
        },
    )


if __name__ == "__main__":
    main()
