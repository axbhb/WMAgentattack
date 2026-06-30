"""Offline DreamerV3-style world model adapter for AgentDojo trajectories.

This module intentionally keeps the existing sklearn prototype untouched.  It
adapts standardized :class:`StepRecord` trajectories into a sequence-learning
problem that can reuse SheepRL's DreamerV3 building blocks:

- MLPEncoder / MLPDecoder for vector observations,
- RecurrentModel + RSSM for latent dynamics and imagination,
- DreamerV3 initialization utilities.

The adapter is offline rather than Gym-environment based because AgentDojo
traces are already collected interaction trajectories.  It exposes the same
``predict`` and ``score_actions`` shape as ``SklearnWorldModel`` so downstream
selection code can compare both backbones.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any

import numpy as np

from wmagentattack.schema import StepRecord


TOKEN_RE = re.compile(r"[\w.-]+", re.UNICODE)


def step_to_dreamer_text(step: StepRecord | dict, attack_action: str | None = None) -> str:
    value = step.model_dump() if isinstance(step, StepRecord) else step
    attack = attack_action if attack_action is not None else value.get("attack_action")
    descriptions = value.get("candidate_skill_descriptions", {}) or {}
    skill_descriptions = " ".join(
        f"{skill}: {descriptions.get(skill, '')}" for skill in value.get("candidate_skills", [])
    )
    return "\n".join(
        [
            f"domain: {value.get('domain', '')}",
            f"task: {value.get('task_id', '')}",
            f"goal: {value.get('user_goal', '')}",
            f"trusted: {value.get('trusted_instruction', '')}",
            f"history: {value.get('agent_history', '')}",
            f"observation: {value.get('current_observation', '')}",
            f"untrusted: {value.get('untrusted_content') or ''}",
            f"previous_skills: {' '.join(value.get('previous_skills', []))}",
            f"candidates: {' '.join(value.get('candidate_skills', []))}",
            f"candidate_descriptions: {skill_descriptions}",
            f"attack: {attack or 'NONE'}",
            f"target: {value.get('target_skill') or 'NONE'}",
        ]
    )


def _hash_index(token: str, dim: int, seed: str) -> int:
    digest = hashlib.blake2b(f"{seed}|{token}".encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % dim


def hash_text_features(text: str, dim: int = 768) -> np.ndarray:
    """Stable dependency-light hashed bag-of-words features.

    We avoid scikit-learn here because the SheepRL conda environment may not be
    identical to the original ``wmagentattack`` environment.
    """

    features = np.zeros(dim, dtype=np.float32)
    tokens = [token.lower() for token in TOKEN_RE.findall(text)]
    for token in tokens:
        features[_hash_index(token, dim, "tok")] += 1.0
    for left, right in zip(tokens, tokens[1:], strict=False):
        features[_hash_index(f"{left}_{right}", dim, "bi")] += 0.5
    norm = float(np.linalg.norm(features))
    if norm > 0:
        features /= norm
    return features


@dataclass
class DreamerWorldModelConfig:
    obs_dim: int = 768
    encoder_layers: int = 2
    decoder_layers: int = 2
    dense_units: int = 256
    recurrent_state_size: int = 256
    stochastic_size: int = 16
    discrete_size: int = 16
    unimix: float = 0.01
    learning_rate: float = 3e-4
    batch_size: int = 16
    epochs: int = 5
    skill_loss_scale: float = 1.0
    risk_loss_scale: float = 1.0
    utility_loss_scale: float = 1.0
    final_utility_loss_scale: float = 1.0
    risk_pos_weight: float = 1.0
    utility_pos_weight: float = 1.0
    kl_scale: float = 0.01
    kl_dynamic_scale: float = 0.5
    kl_representation_scale: float = 0.1
    kl_free_nats: float = 1.0
    reconstruction_scale: float = 0.05
    seed: int = 7
    device: str = "auto"


def _require_torch_and_sheeprl():
    try:
        import torch
        from torch import nn
        import torch.nn.functional as F
        from sheeprl.algos.dreamer_v3.agent import MLPDecoder, MLPEncoder, RSSM, RecurrentModel
        from sheeprl.algos.dreamer_v3.utils import init_weights
    except Exception as exc:  # pragma: no cover - depends on optional remote env
        raise RuntimeError(
            "SheepRLDreamerWorldModel requires torch and sheeprl. "
            "Run it in the server conda environment `sheeprl_env`."
        ) from exc
    return torch, nn, F, MLPEncoder, MLPDecoder, RSSM, RecurrentModel, init_weights


def _build_vocab(steps: list[StepRecord]) -> list[str]:
    skills = set()
    for step in steps:
        skills.add(step.selected_skill)
        skills.update(step.candidate_skills)
        if step.target_skill:
            skills.add(step.target_skill)
    return sorted(skills)


def _group_steps(steps: list[StepRecord]) -> list[list[StepRecord]]:
    grouped: dict[str, list[StepRecord]] = {}
    for step in steps:
        grouped.setdefault(step.trajectory_id, []).append(step)
    return [sorted(items, key=lambda item: item.step_id) for items in grouped.values()]


def _binary_auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    labels = labels.astype(int)
    positives = int(labels.sum())
    negatives = int(len(labels) - positives)
    if positives == 0 or negatives == 0:
        return None
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    pos_rank_sum = ranks[labels == 1].sum()
    return float((pos_rank_sum - positives * (positives + 1) / 2) / (positives * negatives))


def evaluate_dreamer_predictions(steps: list[StepRecord], predictions: dict[str, Any]) -> dict[str, Any]:
    skill_true = np.array([step.selected_skill for step in steps])
    risk_true = np.array([step.attack_success for step in steps], dtype=int)
    utility_true = np.array([step.task_success for step in steps], dtype=int)
    probabilities = np.asarray(predictions["next_skill_proba"])
    classes = np.asarray(predictions["skill_classes"])
    top_k = min(3, probabilities.shape[1])
    top_indices = np.argsort(probabilities, axis=1)[:, -top_k:]
    top3 = np.mean([truth in classes[indices] for truth, indices in zip(skill_true, top_indices, strict=True)])
    risk_scores = np.asarray(predictions["risk_score"])
    utility_scores = np.asarray(predictions["utility_score"])
    risk_pred = risk_scores >= 0.5
    tp = float(((risk_pred == 1) & (risk_true == 1)).sum())
    fp = float(((risk_pred == 1) & (risk_true == 0)).sum())
    fn = float(((risk_pred == 0) & (risk_true == 1)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "next_skill_accuracy": float(np.mean(skill_true == predictions["next_skill"])),
        "next_skill_top3_accuracy": float(top3),
        "risk_auc": _binary_auc(risk_true, risk_scores),
        "risk_f1": float(f1),
        "utility_auc": _binary_auc(utility_true, utility_scores),
        "calibration_brier_score": float(np.mean((risk_true - risk_scores) ** 2)),
    }


class SheepRLDreamerWorldModel:
    """Offline AgentDojo adapter around SheepRL DreamerV3 RSSM components."""

    def __init__(self, config: DreamerWorldModelConfig | None = None, skill_classes: list[str] | None = None):
        self.config = config or DreamerWorldModelConfig()
        self.skill_classes = skill_classes or []
        self.skill_to_id = {skill: index for index, skill in enumerate(self.skill_classes)}
        self._module = None
        self.training_history: list[dict[str, float]] = []

    def _device_name(self) -> str:
        torch, *_ = _require_torch_and_sheeprl()
        if self.config.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.config.device

    def _make_module(self):
        torch, nn, F, MLPEncoder, MLPDecoder, RSSM, RecurrentModel, init_weights = _require_torch_and_sheeprl()
        cfg = self.config
        num_actions = len(self.skill_classes)

        class _OfflineDreamerModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.stochastic_size = cfg.stochastic_size
                self.discrete_size = cfg.discrete_size
                latent_size = cfg.recurrent_state_size + cfg.stochastic_size * cfg.discrete_size
                self.encoder = MLPEncoder(
                    keys=["obs"],
                    input_dims=[cfg.obs_dim],
                    mlp_layers=cfg.encoder_layers,
                    dense_units=cfg.dense_units,
                    symlog_inputs=False,
                )
                representation = nn.Sequential(
                    nn.Linear(cfg.recurrent_state_size + cfg.dense_units, cfg.dense_units),
                    nn.SiLU(),
                    nn.Linear(cfg.dense_units, cfg.stochastic_size * cfg.discrete_size),
                )
                transition = nn.Sequential(
                    nn.Linear(cfg.recurrent_state_size, cfg.dense_units),
                    nn.SiLU(),
                    nn.Linear(cfg.dense_units, cfg.stochastic_size * cfg.discrete_size),
                )
                recurrent = RecurrentModel(
                    input_size=cfg.stochastic_size * cfg.discrete_size + num_actions,
                    recurrent_state_size=cfg.recurrent_state_size,
                    dense_units=cfg.dense_units,
                )
                self.rssm = RSSM(
                    recurrent,
                    representation,
                    transition,
                    distribution_cfg={},
                    discrete=cfg.discrete_size,
                    unimix=cfg.unimix,
                )
                self.decoder = MLPDecoder(
                    keys=["obs"],
                    output_dims=[cfg.obs_dim],
                    latent_state_size=latent_size,
                    mlp_layers=cfg.decoder_layers,
                    dense_units=cfg.dense_units,
                )
                self.skill_head = nn.Linear(latent_size, num_actions)
                self.risk_head = nn.Linear(latent_size, 1)
                self.utility_head = nn.Linear(latent_size, 1)
                self.final_utility_head = nn.Linear(latent_size, 1)
                self.apply(init_weights)

            def forward(self, obs, action_ids):
                batch, steps, _ = obs.shape
                encoded = self.encoder({"obs": obs.reshape(batch * steps, -1)}).reshape(batch, steps, -1)
                prev_actions = torch.zeros(batch, steps, num_actions, device=obs.device)
                if steps > 1:
                    prev_ids = action_ids[:, :-1].clamp_min(0)
                    prev_actions[:, 1:, :].scatter_(2, prev_ids.unsqueeze(-1), 1.0)
                recurrent_state, posterior = self.rssm.get_initial_states((1, batch))
                outputs = []
                posterior_states = []
                recurrent_states = []
                posterior_logits = []
                prior_logits = []
                for index in range(steps):
                    is_first = torch.zeros(1, batch, 1, device=obs.device)
                    if index == 0:
                        is_first.fill_(1.0)
                    recurrent_state, posterior, prior, post_logits, prior_logits_t = self.rssm.dynamic(
                        posterior,
                        recurrent_state,
                        prev_actions[:, index, :].unsqueeze(0),
                        encoded[:, index, :].unsqueeze(0),
                        is_first,
                    )
                    posterior_flat = posterior.reshape(1, batch, -1)
                    feature = torch.cat((posterior_flat, recurrent_state), dim=-1).squeeze(0)
                    outputs.append(feature)
                    posterior_states.append(posterior_flat.squeeze(0))
                    recurrent_states.append(recurrent_state.squeeze(0))
                    posterior_logits.append(post_logits.squeeze(0))
                    prior_logits.append(prior_logits_t.squeeze(0))
                features = torch.stack(outputs, dim=1)
                return {
                    "features": features,
                    "posterior_states": torch.stack(posterior_states, dim=1),
                    "recurrent_states": torch.stack(recurrent_states, dim=1),
                    "skill_logits": self.skill_head(features),
                    "risk_logits": self.risk_head(features).squeeze(-1),
                    "utility_logits": self.utility_head(features).squeeze(-1),
                    "final_utility_logits": self.final_utility_head(features).squeeze(-1),
                    "reconstruction": self.decoder(features.reshape(batch * steps, -1))["obs"].reshape(batch, steps, -1),
                    "posterior_logits": torch.stack(posterior_logits, dim=1),
                    "prior_logits": torch.stack(prior_logits, dim=1),
                }

            def kl_loss(self, posterior_logits, prior_logits):
                post = posterior_logits.reshape(*posterior_logits.shape[:2], self.stochastic_size, self.discrete_size)
                prior = prior_logits.reshape(*prior_logits.shape[:2], self.stochastic_size, self.discrete_size)
                post_log = F.log_softmax(post, dim=-1)
                prior_log = F.log_softmax(prior, dim=-1)
                post_prob = post_log.exp()
                dynamic_loss = (
                    post_prob.detach() * (post_log.detach() - prior_log)
                ).sum(dim=(-1, -2))
                representation_loss = (
                    post_prob * (post_log - prior_log.detach())
                ).sum(dim=(-1, -2))
                free_nats = torch.full_like(dynamic_loss, cfg.kl_free_nats)
                dynamic_loss = cfg.kl_dynamic_scale * torch.maximum(dynamic_loss, free_nats)
                representation_loss = cfg.kl_representation_scale * torch.maximum(
                    representation_loss, free_nats
                )
                return dynamic_loss + representation_loss

        return _OfflineDreamerModule()

    def _ensure_module(self):
        if self._module is None:
            self._module = self._make_module().to(self._device_name())
        return self._module

    def _vectorize_step(self, step: StepRecord | dict, attack_action: str | None = None) -> np.ndarray:
        return hash_text_features(step_to_dreamer_text(step, attack_action), self.config.obs_dim)

    def _tensorize_sequences(self, sequences: list[list[StepRecord]]):
        torch, *_ = _require_torch_and_sheeprl()
        max_len = max(len(sequence) for sequence in sequences)
        obs = np.zeros((len(sequences), max_len, self.config.obs_dim), dtype=np.float32)
        actions = np.full((len(sequences), max_len), -1, dtype=np.int64)
        risk = np.zeros((len(sequences), max_len), dtype=np.float32)
        utility = np.zeros((len(sequences), max_len), dtype=np.float32)
        mask = np.zeros((len(sequences), max_len), dtype=np.float32)
        for row, sequence in enumerate(sequences):
            for col, step in enumerate(sequence):
                obs[row, col] = self._vectorize_step(step)
                actions[row, col] = self.skill_to_id[step.selected_skill]
                risk[row, col] = float(step.attack_success)
                utility[row, col] = float(step.task_success)
                mask[row, col] = 1.0
        return {
            "obs": torch.from_numpy(obs),
            "actions": torch.from_numpy(actions),
            "risk": torch.from_numpy(risk),
            "utility": torch.from_numpy(utility),
            "mask": torch.from_numpy(mask),
        }

    def fit(self, steps: list[StepRecord], *, epochs: int | None = None, batch_size: int | None = None):
        torch, _, F, *_ = _require_torch_and_sheeprl()
        if not steps:
            raise ValueError("Cannot train a Dreamer world model with zero steps.")
        self.skill_classes = _build_vocab(steps)
        self.skill_to_id = {skill: index for index, skill in enumerate(self.skill_classes)}
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)
        module = self._make_module().to(self._device_name())
        self._module = module
        device = next(module.parameters()).device
        sequences = _group_steps(steps)
        data = self._tensorize_sequences(sequences)
        optimizer = torch.optim.AdamW(module.parameters(), lr=self.config.learning_rate)
        epochs = epochs or self.config.epochs
        batch_size = batch_size or self.config.batch_size
        num_sequences = data["obs"].shape[0]
        self.training_history = []
        for epoch in range(1, epochs + 1):
            permutation = torch.randperm(num_sequences)
            totals = {
                "loss": 0.0,
                "skill": 0.0,
                "risk": 0.0,
                "utility": 0.0,
                "final_utility": 0.0,
                "reconstruction": 0.0,
                "kl": 0.0,
            }
            seen = 0.0
            for start in range(0, num_sequences, batch_size):
                indices = permutation[start : start + batch_size]
                obs = data["obs"][indices].to(device)
                actions = data["actions"][indices].to(device)
                risk = data["risk"][indices].to(device)
                utility = data["utility"][indices].to(device)
                mask = data["mask"][indices].to(device)
                out = module(obs, actions)
                flat_mask = mask.reshape(-1) > 0
                skill_loss = F.cross_entropy(
                    out["skill_logits"].reshape(-1, len(self.skill_classes))[flat_mask],
                    actions.reshape(-1)[flat_mask],
                )
                risk_pos_weight = torch.tensor(self.config.risk_pos_weight, device=device)
                utility_pos_weight = torch.tensor(self.config.utility_pos_weight, device=device)
                risk_loss = F.binary_cross_entropy_with_logits(
                    out["risk_logits"].reshape(-1)[flat_mask],
                    risk.reshape(-1)[flat_mask],
                    pos_weight=risk_pos_weight,
                )
                utility_loss = F.binary_cross_entropy_with_logits(
                    out["utility_logits"].reshape(-1)[flat_mask],
                    utility.reshape(-1)[flat_mask],
                    pos_weight=utility_pos_weight,
                )
                sequence_lengths = mask.sum(dim=1).long().clamp_min(1)
                final_indices = sequence_lengths - 1
                batch_indices = torch.arange(mask.shape[0], device=device)
                final_utility_logits = out["final_utility_logits"][batch_indices, final_indices]
                final_utility_targets = utility[batch_indices, final_indices]
                final_utility_loss = F.binary_cross_entropy_with_logits(
                    final_utility_logits,
                    final_utility_targets,
                    pos_weight=utility_pos_weight,
                )
                reconstruction_loss = ((out["reconstruction"] - obs) ** 2).mean(dim=-1)
                reconstruction_loss = (reconstruction_loss * mask).sum() / mask.sum().clamp_min(1.0)
                kl_loss = module.kl_loss(out["posterior_logits"], out["prior_logits"])
                kl_loss = (kl_loss * mask).sum() / mask.sum().clamp_min(1.0)
                loss = (
                    self.config.skill_loss_scale * skill_loss
                    + self.config.risk_loss_scale * risk_loss
                    + self.config.utility_loss_scale * utility_loss
                    + self.config.final_utility_loss_scale * final_utility_loss
                    + self.config.reconstruction_scale * reconstruction_loss
                    + self.config.kl_scale * kl_loss
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(module.parameters(), 100.0)
                optimizer.step()
                batch_weight = float(mask.sum().item())
                seen += batch_weight
                for key, value in [
                    ("loss", loss),
                    ("skill", skill_loss),
                    ("risk", risk_loss),
                    ("utility", utility_loss),
                    ("final_utility", final_utility_loss),
                    ("reconstruction", reconstruction_loss),
                    ("kl", kl_loss),
                ]:
                    totals[key] += float(value.detach().cpu()) * batch_weight
            epoch_stats = {"epoch": float(epoch), **{key: value / max(seen, 1.0) for key, value in totals.items()}}
            self.training_history.append(epoch_stats)
        return self

    def _predict_arrays(self, steps: list[StepRecord | dict], attack_actions: list[str | None] | None = None):
        torch, _, F, *_ = _require_torch_and_sheeprl()
        module = self._ensure_module()
        device = next(module.parameters()).device
        attack_actions = attack_actions or [None] * len(steps)
        obs = np.stack([self._vectorize_step(step, action) for step, action in zip(steps, attack_actions, strict=True)])
        obs_t = torch.from_numpy(obs.astype(np.float32)).unsqueeze(1).to(device)
        actions = torch.zeros((len(steps), 1), dtype=torch.long, device=device)
        module.eval()
        with torch.no_grad():
            out = module(obs_t, actions)
            skill_proba = F.softmax(out["skill_logits"][:, 0, :], dim=-1).detach().cpu().numpy()
            risk = torch.sigmoid(out["risk_logits"][:, 0]).detach().cpu().numpy()
            utility = torch.sigmoid(out["utility_logits"][:, 0]).detach().cpu().numpy()
        return skill_proba, risk, utility

    def predict(self, steps: list[StepRecord | dict]) -> dict[str, Any]:
        skill_proba, risk, utility = self._predict_arrays(steps)
        classes = np.asarray(self.skill_classes)
        return {
            "next_skill": classes[np.argmax(skill_proba, axis=1)],
            "next_skill_proba": skill_proba,
            "skill_classes": classes,
            "risk_score": risk,
            "utility_score": utility,
        }

    def score_actions(self, step: StepRecord | dict, actions: list[str]) -> dict[str, Any]:
        repeated_steps = [step for _ in actions]
        skill_proba, risk, utility = self._predict_arrays(repeated_steps, actions)
        return {
            "risk_score": risk,
            "utility_score": utility,
            "next_skill_proba": skill_proba,
            "skill_classes": np.asarray(self.skill_classes),
        }

    def rollout_score_step(self, step: StepRecord | dict, *, horizon: int = 3) -> dict[str, Any]:
        """Score a state with true RSSM latent imagination.

        This is the Dreamer-specific counterpart of the text-based rollout used
        by the sklearn baseline.  The current observation is encoded once, then
        future latent states are generated with ``rssm.imagination`` and decoded
        only through the prediction heads.
        """

        torch, _, F, *_ = _require_torch_and_sheeprl()
        module = self._ensure_module()
        device = next(module.parameters()).device
        target_skill = step.target_skill if isinstance(step, StepRecord) else step.get("target_skill")
        target_index = self.skill_to_id.get(target_skill) if target_skill else None

        obs = self._vectorize_step(step).astype(np.float32)
        obs_t = torch.from_numpy(obs).reshape(1, 1, -1).to(device)
        actions = torch.zeros((1, 1), dtype=torch.long, device=device)
        step_value = step.model_dump(mode="json") if isinstance(step, StepRecord) else step
        branch_skill_names = []
        if target_skill and target_skill in self.skill_to_id:
            branch_skill_names.append(target_skill)
        branch_skill_names.extend(
            skill
            for skill in step_value.get("candidate_skills", [])
            if skill in self.skill_to_id and skill not in branch_skill_names
        )
        if not branch_skill_names:
            branch_skill_names = ["finish"] if "finish" in self.skill_to_id else [self.skill_classes[0]]

        module.eval()
        with torch.no_grad():
            out = module(obs_t, actions)
            base_stochastic_state = out["posterior_states"][:, -1, :].unsqueeze(0)
            base_recurrent_state = out["recurrent_states"][:, -1, :].unsqueeze(0)

            branch_summaries = []
            for first_skill in branch_skill_names:
                stochastic_state = base_stochastic_state.clone()
                recurrent_state = base_recurrent_state.clone()
                imagined_skills: list[str] = []
                risk_scores: list[float] = []
                utility_scores: list[float] = []
                final_utility_scores: list[float] = []
                target_probabilities: list[float] = []
                action_skill_id = self.skill_to_id[first_skill]

                for rollout_index in range(max(horizon, 1)):
                    action = torch.zeros(1, 1, len(self.skill_classes), device=device)
                    action[:, :, action_skill_id] = 1.0
                    stochastic_state, recurrent_state = module.rssm.imagination(
                        stochastic_state, recurrent_state, action
                    )
                    stochastic_state = stochastic_state.reshape(1, 1, -1)
                    feature = torch.cat(
                        (stochastic_state.squeeze(0), recurrent_state.squeeze(0)), dim=-1
                    )
                    skill_prob = F.softmax(module.skill_head(feature), dim=-1)
                    risk_score = torch.sigmoid(module.risk_head(feature)).squeeze(-1)
                    utility_score = torch.sigmoid(module.utility_head(feature)).squeeze(-1)
                    final_utility_score = torch.sigmoid(module.final_utility_head(feature)).squeeze(-1)
                    predicted_next_id = int(torch.argmax(skill_prob, dim=-1).item())
                    predicted_next_skill = self.skill_classes[predicted_next_id]

                    imagined_skill = first_skill if rollout_index == 0 else predicted_next_skill
                    imagined_skills.append(imagined_skill)
                    risk_scores.append(float(risk_score.item()))
                    utility_scores.append(float(utility_score.item()))
                    final_utility_scores.append(float(final_utility_score.item()))
                    target_probabilities.append(
                        float(skill_prob[0, target_index].item()) if target_index is not None else 0.0
                    )
                    action_skill_id = predicted_next_id

                max_risk = max(risk_scores) if risk_scores else 0.0
                mean_risk = float(np.mean(risk_scores)) if risk_scores else 0.0
                mean_utility = float(np.mean(utility_scores)) if utility_scores else 0.0
                min_utility = min(utility_scores) if utility_scores else 0.0
                final_utility = final_utility_scores[-1] if final_utility_scores else mean_utility
                max_target_probability = max(target_probabilities) if target_probabilities else 0.0
                mean_target_probability = (
                    float(np.mean(target_probabilities)) if target_probabilities else 0.0
                )
                target_reached = float(target_skill in imagined_skills) if target_skill else 0.0
                selection_score = (
                    max_risk
                    + 0.3 * mean_risk
                    + 0.5 * mean_utility
                    + 0.3 * max_target_probability
                    + 0.2 * target_reached
                )
                branch_summaries.append(
                    {
                        "branch_first_skill": first_skill,
                        "risk_score": max_risk,
                        "utility_score": mean_utility,
                        "min_utility_score": min_utility,
                        "final_utility_score": final_utility,
                        "target_skill_probability": max_target_probability,
                        "selection_score": selection_score,
                        "rollout_mean_risk_score": mean_risk,
                        "rollout_mean_target_skill_probability": mean_target_probability,
                        "rollout_target_reached": target_reached,
                        "rollout_imagined_skills": imagined_skills,
                    }
                )

        best_branch = max(branch_summaries, key=lambda item: item["selection_score"])
        compact_branch_summaries = sorted(
            branch_summaries, key=lambda item: item["selection_score"], reverse=True
        )[:3]
        return {
            **best_branch,
            "rollout_backend": "sheeprl_rssm_branch_imagination",
            "rollout_branch_count": len(branch_summaries),
            "rollout_top_branch_summaries": compact_branch_summaries,
        }

    def save(self, path: str | Path):
        torch, *_ = _require_torch_and_sheeprl()
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        module = self._ensure_module()
        torch.save(module.state_dict(), path / "model.pt")
        metadata = {
            "config": asdict(self.config),
            "skill_classes": self.skill_classes,
            "training_history": self.training_history,
            "backend": "sheeprl_dreamer_v3_offline",
        }
        (path / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path):
        torch, *_ = _require_torch_and_sheeprl()
        path = Path(path)
        metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
        config = DreamerWorldModelConfig(**metadata["config"])
        model = cls(config=config, skill_classes=metadata["skill_classes"])
        model.training_history = metadata.get("training_history", [])
        module = model._ensure_module()
        state = torch.load(path / "model.pt", map_location=model._device_name())
        if "final_utility_head.weight" not in state and "utility_head.weight" in state:
            state["final_utility_head.weight"] = state["utility_head.weight"].clone()
            state["final_utility_head.bias"] = state["utility_head.bias"].clone()
        module.load_state_dict(state)
        module.eval()
        return model
