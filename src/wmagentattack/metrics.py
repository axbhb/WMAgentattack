"""World-model evaluation metrics."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, brier_score_loss, f1_score, roc_auc_score


def evaluate_predictions(steps, predictions):
    skill_true = np.array([step.selected_skill for step in steps])
    risk_true = np.array([step.attack_success for step in steps], dtype=int)
    utility_true = np.array([step.task_success for step in steps], dtype=int)
    probabilities = predictions["next_skill_proba"]
    classes = predictions["skill_classes"]
    top_k = min(3, probabilities.shape[1])
    top_indices = np.argsort(probabilities, axis=1)[:, -top_k:]
    top3 = np.mean(
        [
            truth in classes[indices]
            for truth, indices in zip(skill_true, top_indices, strict=True)
        ]
    )

    def safe_auc(labels, scores):
        return float(roc_auc_score(labels, scores)) if len(set(labels)) > 1 else None

    risk_pred = predictions["risk_score"] >= 0.5
    return {
        "next_skill_accuracy": float(
            accuracy_score(skill_true, predictions["next_skill"])
        ),
        "next_skill_top3_accuracy": float(top3),
        "risk_auc": safe_auc(risk_true, predictions["risk_score"]),
        "risk_f1": float(f1_score(risk_true, risk_pred, zero_division=0)),
        "utility_auc": safe_auc(utility_true, predictions["utility_score"]),
        "calibration_brier_score": float(
            brier_score_loss(risk_true, predictions["risk_score"])
        ),
    }

