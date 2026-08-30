"""Train the frozen task-disjoint v35 comparison-reward policy."""

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

from wmagentattack.attack_conditioned_ranker import task_balanced_weights
from wmagentattack.comparison_reward_policy import (
    DEFAULT_REWARD_WEIGHTS,
    ComparisonRewardPolicy,
    align_preference_candidates,
    build_preference_pairs,
    soft_preference_loss,
)


ARMS = (
    "absolute_four_cell",
    "comparison_outcome_anchored",
    "family_comparison_diagnostic",
)


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _append(path: Path, rows: list[dict]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def _folds_from_v5(rows: list[dict]) -> dict[str, int]:
    folds: dict[str, set[int]] = {}
    for row in rows:
        if str(row.get("arm")) != "structured_joint_aux" or not bool(row.get("joint_trainable")):
            continue
        folds.setdefault(str(row["task_name"]), set()).add(int(row["fold"]))
    if len(folds) != 20 or any(len(values) != 1 for values in folds.values()):
        raise ValueError("v35 requires the frozen v5 five-fold task assignment")
    return {task: next(iter(values)) for task, values in folds.items()}


def _train_one(
    *,
    arm: str,
    train_rows: list[dict],
    test_rows: list[dict],
    config: dict,
    seed: int,
) -> tuple[list[dict], dict]:
    include_family = arm == "family_comparison_diagnostic"
    feature_key = "family_features" if include_family else "features"
    vectorizer = DictVectorizer(sparse=False, sort=True)
    train_x = vectorizer.fit_transform([row[feature_key] for row in train_rows]).astype(np.float32)
    test_x = vectorizer.transform([row[feature_key] for row in test_rows]).astype(np.float32)
    train_pairs, pair_audit = build_preference_pairs(
        train_rows,
        draws=int(config["posterior_draws"]),
        posterior_seed=int(config["posterior_seed"]),
        minimum_confidence_gap=float(config["minimum_confidence_gap"]),
    )
    if arm != "absolute_four_cell" and not len(train_pairs):
        raise RuntimeError("comparison arm has zero training pairs")

    _seed(seed)
    model = ComparisonRewardPolicy(
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
    target = torch.tensor(np.asarray([row["target"] for row in train_rows], dtype=np.float32))
    row_weights = torch.tensor(task_balanced_weights(train_rows))
    pairs = torch.tensor(train_pairs)
    reward_weights = torch.tensor(DEFAULT_REWARD_WEIGHTS, dtype=torch.float32)
    history = []
    for epoch in range(int(config["epochs"])):
        _seed(seed * 1009 + epoch)
        model.train()
        reward_score, outcome_logits = model(x)
        outcome_probability = torch.softmax(outcome_logits, dim=1)
        absolute_score = outcome_probability @ reward_weights
        per_row = -(target * F.log_softmax(outcome_logits, dim=1)).sum(dim=1)
        outcome_loss = (per_row * row_weights).sum() / row_weights.sum()
        if arm == "absolute_four_cell":
            comparison_loss = soft_preference_loss(absolute_score, pairs)
            loss = outcome_loss
        else:
            comparison_loss = soft_preference_loss(reward_score, pairs)
            loss = comparison_loss + float(config["outcome_anchor_weight"]) * outcome_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["gradient_clip"]))
        optimizer.step()
        if epoch in (0, int(config["epochs"]) - 1):
            history.append(
                {
                    "epoch": epoch,
                    "loss": float(loss.detach()),
                    "comparison_loss": float(comparison_loss.detach()),
                    "outcome_loss": float(outcome_loss.detach()),
                }
            )
    model.eval()
    with torch.no_grad():
        reward_score, outcome_logits = model(torch.tensor(test_x))
        probabilities = torch.softmax(outcome_logits, dim=1)
        if arm == "absolute_four_cell":
            selection_score = probabilities @ reward_weights
        else:
            selection_score = reward_score
    predictions = []
    for row, score, probability in zip(test_rows, selection_score.numpy(), probabilities.numpy(), strict=True):
        predictions.append(
            {
                "fold": int(row["fold"]),
                "seed": int(seed),
                "arm": arm,
                "row_id": row["row_id"],
                "task_name": row["task_name"],
                "attack_family": row["attack_family"],
                "counts": row["counts"],
                "target": row["target"],
                "target_p11": row["target_p11"],
                "target_utility": row["target_utility"],
                "target_reward": row["target_reward"],
                "predicted_score": float(score),
                "predicted_p11": float(probability[3]),
                "predicted_utility": float(probability[1] + probability[3]),
            }
        )
    return predictions, {
        "arm": arm,
        "seed": int(seed),
        "feature_width": int(train_x.shape[1]),
        "training_candidates": len(train_rows),
        "test_candidates": len(test_rows),
        "training_confident_pairs": len(train_pairs),
        "training_pair_audit": pair_audit,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "history": history,
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
        raise ValueError("v35 protocol is not frozen")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    v5_rows = _read_jsonl(args.v5_predictions)
    rows, alignment = align_preference_candidates(
        manifest_rows=manifest["rows"],
        label_groups=_read_jsonl(args.label_groups),
        fold_by_task=_folds_from_v5(v5_rows),
    )
    if not alignment["passed"]:
        raise RuntimeError(f"v35 alignment failed: {alignment['checks']}")
    all_pairs, support = build_preference_pairs(
        rows,
        draws=int(protocol["training"]["posterior_draws"]),
        posterior_seed=int(protocol["training"]["posterior_seed"]),
        minimum_confidence_gap=float(protocol["training"]["minimum_confidence_gap"]),
    )
    support_gate = protocol["data_support_gate"]
    support["passed"] = (
        support["confident_pairs"] >= int(support_gate["minimum_confident_pairs"])
        and support["tasks_with_at_least_twenty_pairs"]
        >= int(support_gate["minimum_tasks_with_twenty_confident_pairs"])
        and support["tasks_with_multiple_attack_families"]
        >= int(support_gate["minimum_tasks_with_multiple_attack_families"])
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write(args.output_dir / "alignment_audit.json", alignment)
    _write(args.output_dir / "preference_support_audit.json", support)
    prediction_path = args.output_dir / "predictions.jsonl"
    prediction_path.write_text("", encoding="utf-8")
    runs = []
    if support["passed"]:
        for fold in range(5):
            train_rows = [row for row in rows if int(row["fold"]) != fold]
            test_rows = [row for row in rows if int(row["fold"]) == fold]
            if len(train_rows) != 320 or len(test_rows) != 80:
                raise RuntimeError(f"unexpected fold sizes for fold {fold}")
            for arm in ARMS:
                for seed in protocol["training"]["model_seeds"]:
                    predictions, run = _train_one(
                        arm=arm,
                        train_rows=train_rows,
                        test_rows=test_rows,
                        config=protocol["training"],
                        seed=int(seed),
                    )
                    _append(prediction_path, predictions)
                    runs.append({"fold": fold, **run})
    expected = int(protocol["fixed_budget"]["model_fits"]) if support["passed"] else 0
    if len(runs) != expected:
        raise RuntimeError(f"incomplete v35 budget: {len(runs)} != {expected}")
    _write(
        args.output_dir / "run_metrics.json",
        {
            "data_support_passed": support["passed"],
            "model_fits": len(runs),
            "expected_model_fits": expected,
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
