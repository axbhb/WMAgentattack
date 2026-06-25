"""Fast sklearn world model for the first end-to-end prototype."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from wmagentattack.schema import StepRecord


def step_to_text(step: StepRecord | dict, attack_action: str | None = None) -> str:
    value = step.model_dump() if isinstance(step, StepRecord) else step
    attack = attack_action if attack_action is not None else value.get("attack_action")
    return "\n".join(
        [
            f"goal: {value.get('user_goal', '')}",
            f"history: {value.get('agent_history', '')}",
            f"observation: {value.get('current_observation', '')}",
            f"previous_skills: {' '.join(value.get('previous_skills', []))}",
            f"candidates: {' '.join(value.get('candidate_skills', []))}",
            f"attack: {attack or 'NONE'}",
            f"target: {value.get('target_skill') or 'NONE'}",
        ]
    )


class SklearnWorldModel:
    def __init__(self, max_features: int = 12_000):
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2), max_features=max_features, min_df=2
        )
        self.skill_head = LogisticRegression(max_iter=500)
        self.risk_head = LogisticRegression(max_iter=500)
        self.utility_head = LogisticRegression(max_iter=500)

    def fit(self, steps: list[StepRecord]):
        texts = [step_to_text(step) for step in steps]
        features = self.vectorizer.fit_transform(texts)
        self.skill_head.fit(features, [step.selected_skill for step in steps])
        self.risk_head.fit(features, [step.attack_success for step in steps])
        self.utility_head.fit(features, [step.task_success for step in steps])
        return self

    def predict(self, steps: list[StepRecord | dict]):
        features = self.vectorizer.transform([step_to_text(step) for step in steps])
        return {
            "next_skill": self.skill_head.predict(features),
            "next_skill_proba": self.skill_head.predict_proba(features),
            "skill_classes": self.skill_head.classes_,
            "risk_score": self.risk_head.predict_proba(features)[:, 1],
            "utility_score": self.utility_head.predict_proba(features)[:, 1],
        }

    def score_actions(self, step: StepRecord | dict, actions: list[str]):
        features = self.vectorizer.transform(
            [step_to_text(step, attack_action=action) for action in actions]
        )
        return {
            "risk_score": self.risk_head.predict_proba(features)[:, 1],
            "utility_score": self.utility_head.predict_proba(features)[:, 1],
            "next_skill_proba": self.skill_head.predict_proba(features),
            "skill_classes": self.skill_head.classes_,
        }

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path):
        return joblib.load(path)

