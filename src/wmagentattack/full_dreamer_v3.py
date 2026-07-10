"""Full offline DreamerV3 architecture for AgentDojo skill trajectories.

This backend uses SheepRL's DreamerV3 building blocks rather than only its
RSSM.  It contains an observation/reward/continue world model, discrete skill
actor, critic and EMA target critic, latent imagination, lambda returns, and
DreamerV3 two-hot value distributions.  AgentDojo-specific auxiliary heads
predict attack risk, continuous utility/preservation probabilities, selected
skills, and valid candidate skills.

The learner is offline.  A behavior-cloning term and candidate-skill masks keep
the imagined policy near the support of the collected AgentDojo trajectories.
"""

from __future__ import annotations

import copy
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from wmagentattack.dreamer_world_model import (
    _binary_auc,
    _build_vocab,
    _group_steps,
    hash_text_features,
    step_to_dreamer_text,
)
from wmagentattack.schema import StepRecord


@dataclass
class FullDreamerV3Config:
    obs_dim: int = 768
    encoder_layers: int = 2
    decoder_layers: int = 2
    dense_units: int = 256
    recurrent_state_size: int = 256
    stochastic_size: int = 32
    discrete_size: int = 32
    actor_layers: int = 2
    critic_layers: int = 2
    head_layers: int = 1
    reward_bins: int = 255
    unimix: float = 0.01
    world_learning_rate: float = 3e-4
    actor_learning_rate: float = 8e-5
    critic_learning_rate: float = 8e-5
    weight_decay: float = 0.0
    batch_size: int = 16
    epochs: int = 20
    imagination_horizon: int = 5
    imagination_batch_size: int = 256
    gamma: float = 0.99
    lmbda: float = 0.95
    target_critic_tau: float = 0.02
    entropy_scale: float = 3e-4
    behavior_cloning_scale: float = 1.0
    risk_reward_scale: float = 1.0
    utility_reward_scale: float = 1.0
    target_skill_reward_scale: float = 0.25
    observation_loss_scale: float = 1.0
    reward_loss_scale: float = 1.0
    continue_loss_scale: float = 1.0
    kl_loss_scale: float = 1.0
    kl_dynamic_scale: float = 0.5
    kl_representation_scale: float = 0.1
    kl_free_nats: float = 1.0
    skill_loss_scale: float = 1.0
    candidate_loss_scale: float = 0.25
    risk_loss_scale: float = 1.0
    utility_loss_scale: float = 1.0
    preservation_loss_scale: float = 1.0
    probability_confidence_floor: float = 0.1
    actor_gradient_clip: float = 100.0
    critic_gradient_clip: float = 100.0
    world_gradient_clip: float = 1000.0
    candidate_threshold: float = 0.5
    seed: int = 7
    device: str = "auto"


def _require_full_sheeprl():
    try:
        import torch
        from torch import nn
        import torch.nn.functional as F
        from torch.distributions import Independent, OneHotCategoricalStraightThrough
        from torch.distributions.kl import kl_divergence

        from sheeprl.algos.dreamer_v3.agent import (
            Actor,
            MLPDecoder,
            MLPEncoder,
            RSSM,
            RecurrentModel,
        )
        from sheeprl.algos.dreamer_v3.utils import (
            compute_lambda_values,
            init_weights,
            uniform_init_weights,
        )
        from sheeprl.models.models import LayerNorm, MLP
        from sheeprl.utils.distribution import (
            SymlogDistribution,
            TwoHotEncodingDistribution,
        )
    except Exception as exc:  # pragma: no cover - optional environment
        raise RuntimeError(
            "FullSheepRLDreamerV3 requires torch and SheepRL 0.5.8.dev. "
            "Run it in the configured sheeprl_env."
        ) from exc
    return {
        "torch": torch,
        "nn": nn,
        "F": F,
        "Independent": Independent,
        "OneHotCategoricalStraightThrough": OneHotCategoricalStraightThrough,
        "kl_divergence": kl_divergence,
        "Actor": Actor,
        "MLPDecoder": MLPDecoder,
        "MLPEncoder": MLPEncoder,
        "RSSM": RSSM,
        "RecurrentModel": RecurrentModel,
        "compute_lambda_values": compute_lambda_values,
        "init_weights": init_weights,
        "uniform_init_weights": uniform_init_weights,
        "LayerNorm": LayerNorm,
        "MLP": MLP,
        "SymlogDistribution": SymlogDistribution,
        "TwoHotEncodingDistribution": TwoHotEncodingDistribution,
    }


def evaluate_full_dreamer_predictions(
    steps: list[StepRecord], predictions: dict[str, Any]
) -> dict[str, Any]:
    skill_true = np.asarray([step.selected_skill for step in steps])
    risk_true = np.asarray([step.attack_success for step in steps], dtype=np.float32)
    binary_utility = np.asarray([step.task_success for step in steps], dtype=np.float32)
    utility_target = np.asarray(
        [
            step.utility_probability_target
            if step.utility_probability_target is not None
            else float(step.task_success)
            for step in steps
        ],
        dtype=np.float32,
    )
    preservation_mask = np.asarray(
        [step.preservation_probability_target is not None for step in steps], dtype=bool
    )
    preservation_target = np.asarray(
        [step.preservation_probability_target or 0.0 for step in steps], dtype=np.float32
    )
    skill_proba = np.asarray(predictions["next_skill_proba"])
    skill_classes = np.asarray(predictions["skill_classes"])
    risk = np.asarray(predictions["risk_score"], dtype=np.float32)
    utility = np.asarray(predictions["utility_score"], dtype=np.float32)
    preservation = np.asarray(predictions["preservation_score"], dtype=np.float32)
    top_k = min(3, skill_proba.shape[1])
    top_indices = np.argsort(skill_proba, axis=1)[:, -top_k:]
    metrics = {
        "next_skill_accuracy": float(
            np.mean(skill_true == skill_classes[np.argmax(skill_proba, axis=1)])
        ),
        "next_skill_top3_accuracy": float(
            np.mean(
                [
                    truth in skill_classes[indices]
                    for truth, indices in zip(skill_true, top_indices, strict=True)
                ]
            )
        ),
        "risk_auc": _binary_auc(risk_true.astype(int), risk),
        "risk_brier_score": float(np.mean((risk_true - risk) ** 2)),
        "binary_utility_auc": _binary_auc(binary_utility.astype(int), utility),
        "utility_probability_brier_score": float(
            np.mean((utility_target - utility) ** 2)
        ),
        "utility_probability_mae": float(np.mean(np.abs(utility_target - utility))),
        "preservation_eval_count": int(preservation_mask.sum()),
    }
    if preservation_mask.any():
        metrics["preservation_probability_brier_score"] = float(
            np.mean(
                (preservation_target[preservation_mask] - preservation[preservation_mask])
                ** 2
            )
        )
        metrics["preservation_probability_mae"] = float(
            np.mean(
                np.abs(
                    preservation_target[preservation_mask]
                    - preservation[preservation_mask]
                )
            )
        )
    else:
        metrics["preservation_probability_brier_score"] = None
        metrics["preservation_probability_mae"] = None
    metrics["validation_objective"] = (
        metrics["risk_brier_score"]
        + metrics["utility_probability_brier_score"]
        + (metrics["preservation_probability_brier_score"] or 0.0)
        + 0.25 * (1.0 - metrics["next_skill_accuracy"])
    )
    return metrics


class FullSheepRLDreamerV3:
    """Offline full DreamerV3 learner specialized for AgentDojo skill actions."""

    def __init__(
        self,
        config: FullDreamerV3Config | None = None,
        skill_classes: list[str] | None = None,
    ) -> None:
        self.config = config or FullDreamerV3Config()
        self.skill_classes = skill_classes or []
        self.skill_to_id = {
            skill: index for index, skill in enumerate(self.skill_classes)
        }
        self._module = None
        self.training_history: list[dict[str, Any]] = []
        self.best_epoch: int | None = None

    def _device_name(self) -> str:
        deps = _require_full_sheeprl()
        torch = deps["torch"]
        if self.config.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.config.device

    def _make_module(self):
        deps = _require_full_sheeprl()
        torch, nn = deps["torch"], deps["nn"]
        Actor = deps["Actor"]
        MLPEncoder, MLPDecoder = deps["MLPEncoder"], deps["MLPDecoder"]
        RSSM, RecurrentModel = deps["RSSM"], deps["RecurrentModel"]
        MLP, LayerNorm = deps["MLP"], deps["LayerNorm"]
        init_weights = deps["init_weights"]
        uniform_init_weights = deps["uniform_init_weights"]
        cfg = self.config
        num_actions = len(self.skill_classes)
        if num_actions <= 1:
            raise ValueError("The full DreamerV3 backend needs at least two skill classes")
        stochastic_state_size = cfg.stochastic_size * cfg.discrete_size
        latent_size = stochastic_state_size + cfg.recurrent_state_size

        def mlp(
            input_dim: int,
            output_dim: int,
            *,
            layers: int,
            units: int | None = None,
        ):
            hidden = units or cfg.dense_units
            return MLP(
                input_dims=input_dim,
                output_dim=output_dim,
                hidden_sizes=[hidden] * layers,
                activation=nn.SiLU,
                flatten_dim=None,
                layer_args={"bias": False},
                norm_layer=LayerNorm,
                norm_args={"normalized_shape": hidden, "eps": 1e-3},
            )

        class _AgentDojoFullDreamer(nn.Module):
            def __init__(self):
                super().__init__()
                self.stochastic_size = cfg.stochastic_size
                self.discrete_size = cfg.discrete_size
                self.stochastic_state_size = stochastic_state_size
                self.latent_size = latent_size
                self.num_actions = num_actions
                self.encoder = MLPEncoder(
                    keys=["obs"],
                    input_dims=[cfg.obs_dim],
                    mlp_layers=cfg.encoder_layers,
                    dense_units=cfg.dense_units,
                    activation=nn.SiLU,
                    layer_norm_cls=LayerNorm,
                    layer_norm_kw={"eps": 1e-3},
                )
                recurrent = RecurrentModel(
                    input_size=stochastic_state_size + num_actions,
                    recurrent_state_size=cfg.recurrent_state_size,
                    dense_units=cfg.dense_units,
                    layer_norm_cls=LayerNorm,
                    layer_norm_kw={"eps": 1e-3},
                )
                representation = mlp(
                    cfg.recurrent_state_size + cfg.dense_units,
                    stochastic_state_size,
                    layers=1,
                )
                transition = mlp(
                    cfg.recurrent_state_size,
                    stochastic_state_size,
                    layers=1,
                )
                self.rssm = RSSM(
                    recurrent_model=recurrent,
                    representation_model=representation,
                    transition_model=transition,
                    distribution_cfg={"type": "discrete"},
                    discrete=cfg.discrete_size,
                    unimix=cfg.unimix,
                    learnable_initial_recurrent_state=True,
                )
                self.observation_model = MLPDecoder(
                    keys=["obs"],
                    output_dims=[cfg.obs_dim],
                    latent_state_size=latent_size,
                    mlp_layers=cfg.decoder_layers,
                    dense_units=cfg.dense_units,
                    activation=nn.SiLU,
                    layer_norm_cls=LayerNorm,
                    layer_norm_kw={"eps": 1e-3},
                )
                reward_input_size = latent_size + num_actions
                self.reward_model = mlp(
                    reward_input_size, cfg.reward_bins, layers=cfg.head_layers
                )
                self.continue_model = mlp(
                    reward_input_size, 1, layers=cfg.head_layers
                )
                self.skill_head = mlp(latent_size, num_actions, layers=cfg.head_layers)
                self.candidate_head = mlp(
                    latent_size, num_actions, layers=cfg.head_layers
                )
                self.risk_head = mlp(latent_size, 1, layers=cfg.head_layers)
                self.utility_head = mlp(latent_size, 1, layers=cfg.head_layers)
                self.preservation_head = mlp(
                    latent_size, 1, layers=cfg.head_layers
                )
                self.actor = Actor(
                    latent_state_size=latent_size,
                    actions_dim=[num_actions],
                    is_continuous=False,
                    distribution_cfg={"type": "discrete"},
                    dense_units=cfg.dense_units,
                    activation=nn.SiLU,
                    mlp_layers=cfg.actor_layers,
                    layer_norm_cls=LayerNorm,
                    layer_norm_kw={"eps": 1e-3},
                    unimix=cfg.unimix,
                )
                self.critic = mlp(
                    latent_size, cfg.reward_bins, layers=cfg.critic_layers
                )
                self.apply(init_weights)
                self.actor.mlp_heads.apply(uniform_init_weights(1.0))
                self.critic.model[-1].apply(uniform_init_weights(0.0))
                self.reward_model.model[-1].apply(uniform_init_weights(0.0))
                self.continue_model.model[-1].apply(uniform_init_weights(1.0))
                self.rssm.transition_model.model[-1].apply(
                    uniform_init_weights(1.0)
                )
                self.rssm.representation_model.model[-1].apply(
                    uniform_init_weights(1.0)
                )
                self.observation_model.heads.apply(uniform_init_weights(1.0))
                self.target_critic = copy.deepcopy(self.critic)
                self.target_critic.requires_grad_(False)
                self.register_buffer("return_low", torch.zeros(()))
                self.register_buffer("return_high", torch.zeros(()))

            def world_parameters(self):
                modules = [
                    self.encoder,
                    self.rssm,
                    self.observation_model,
                    self.reward_model,
                    self.continue_model,
                    self.skill_head,
                    self.candidate_head,
                    self.risk_head,
                    self.utility_head,
                    self.preservation_head,
                ]
                for module in modules:
                    yield from module.parameters()

            def observe(self, obs, action_ids):
                batch, steps, _ = obs.shape
                encoded = self.encoder(
                    {"obs": obs.reshape(batch * steps, -1)}
                ).reshape(batch, steps, -1)
                previous_actions = torch.zeros(
                    batch, steps, num_actions, device=obs.device
                )
                if steps > 1:
                    previous_ids = action_ids[:, :-1].clamp_min(0)
                    previous_actions[:, 1:, :].scatter_(
                        2, previous_ids.unsqueeze(-1), 1.0
                    )
                recurrent_state, posterior = self.rssm.get_initial_states((1, batch))
                latent_states = []
                posterior_states = []
                recurrent_states = []
                posterior_logits = []
                prior_logits = []
                for index in range(steps):
                    is_first = torch.zeros(1, batch, 1, device=obs.device)
                    if index == 0:
                        is_first.fill_(1.0)
                    recurrent_state, posterior, _, post_logits, prior_logits_t = (
                        self.rssm.dynamic(
                            posterior,
                            recurrent_state,
                            previous_actions[:, index, :].unsqueeze(0),
                            encoded[:, index, :].unsqueeze(0),
                            is_first,
                        )
                    )
                    posterior_flat = posterior.reshape(1, batch, -1)
                    latent = torch.cat((posterior_flat, recurrent_state), dim=-1)
                    latent_states.append(latent.squeeze(0))
                    posterior_states.append(posterior_flat.squeeze(0))
                    recurrent_states.append(recurrent_state.squeeze(0))
                    posterior_logits.append(post_logits.squeeze(0))
                    prior_logits.append(prior_logits_t.squeeze(0))
                latent = torch.stack(latent_states, dim=1)
                current_actions = torch.zeros(
                    batch, steps, num_actions, device=obs.device
                )
                current_actions.scatter_(
                    2, action_ids.clamp_min(0).unsqueeze(-1), 1.0
                )
                reward_features = torch.cat((latent, current_actions), dim=-1)
                decoded = self.observation_model(
                    latent.reshape(batch * steps, -1)
                )["obs"].reshape(batch, steps, -1)
                return {
                    "latent": latent,
                    "posterior_states": torch.stack(posterior_states, dim=1),
                    "recurrent_states": torch.stack(recurrent_states, dim=1),
                    "posterior_logits": torch.stack(posterior_logits, dim=1),
                    "prior_logits": torch.stack(prior_logits, dim=1),
                    "reconstruction": decoded,
                    "reward_logits": self.reward_model(reward_features),
                    "continue_logits": self.continue_model(reward_features).squeeze(-1),
                    "skill_logits": self.skill_head(latent),
                    "candidate_logits": self.candidate_head(latent),
                    "risk_logits": self.risk_head(latent).squeeze(-1),
                    "utility_logits": self.utility_head(latent).squeeze(-1),
                    "preservation_logits": self.preservation_head(latent).squeeze(-1),
                }

        return _AgentDojoFullDreamer()

    def _ensure_module(self):
        if self._module is None:
            self._module = self._make_module().to(self._device_name())
        return self._module

    @contextmanager
    def _deterministic_inference(self, module):
        """Seed stochastic RSSM sampling without altering the training RNG."""

        torch = _require_full_sheeprl()["torch"]
        device = next(module.parameters()).device
        devices = [device.index or 0] if device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(self.config.seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(self.config.seed)
            yield

    def _vectorize_step(
        self, step: StepRecord | dict, attack_action: str | None = None
    ) -> np.ndarray:
        return hash_text_features(
            step_to_dreamer_text(step, attack_action), self.config.obs_dim
        )

    def _sequence_payload(self, sequence: list[StepRecord]) -> dict[str, Any]:
        length = len(sequence)
        obs = np.stack([self._vectorize_step(step) for step in sequence])
        actions = np.asarray(
            [self.skill_to_id[step.selected_skill] for step in sequence],
            dtype=np.int64,
        )
        candidate_mask = np.zeros((length, len(self.skill_classes)), dtype=np.float32)
        for index, step in enumerate(sequence):
            for skill in step.candidate_skills:
                if skill in self.skill_to_id:
                    candidate_mask[index, self.skill_to_id[skill]] = 1.0
            candidate_mask[index, actions[index]] = 1.0
        risk = np.asarray([float(step.attack_success) for step in sequence], dtype=np.float32)
        utility = np.asarray(
            [
                step.utility_probability_target
                if step.utility_probability_target is not None
                else float(step.task_success)
                for step in sequence
            ],
            dtype=np.float32,
        )
        preservation = np.asarray(
            [step.preservation_probability_target or 0.0 for step in sequence],
            dtype=np.float32,
        )
        preservation_mask = np.asarray(
            [step.preservation_probability_target is not None for step in sequence],
            dtype=np.float32,
        )
        confidence = np.asarray(
            [
                max(
                    self.config.probability_confidence_floor,
                    step.probability_label_confidence,
                )
                if step.utility_probability_target is not None
                else 1.0
                for step in sequence
            ],
            dtype=np.float32,
        )
        terminated = np.zeros(length, dtype=np.float32)
        terminated[-1] = 1.0
        reward = np.zeros(length, dtype=np.float32)
        final = sequence[-1]
        reward[-1] = (
            self.config.risk_reward_scale * float(final.attack_success)
            + self.config.utility_reward_scale * float(utility[-1])
            + self.config.target_skill_reward_scale * float(final.target_skill_success)
        )
        return {
            "obs": obs,
            "actions": actions,
            "candidate_mask": candidate_mask,
            "risk": risk,
            "utility": utility,
            "preservation": preservation,
            "preservation_mask": preservation_mask,
            "confidence": confidence,
            "reward": reward,
            "terminated": terminated,
            "length": length,
            "records": sequence,
        }

    def _prepare_sequences(self, steps: list[StepRecord]) -> list[dict[str, Any]]:
        return [self._sequence_payload(sequence) for sequence in _group_steps(steps)]

    def _collate(self, sequences: list[dict[str, Any]], device: str):
        deps = _require_full_sheeprl()
        torch = deps["torch"]
        batch = len(sequences)
        max_len = max(sequence["length"] for sequence in sequences)
        actions_n = len(self.skill_classes)
        arrays = {
            "obs": np.zeros((batch, max_len, self.config.obs_dim), dtype=np.float32),
            "actions": np.zeros((batch, max_len), dtype=np.int64),
            "candidate_mask": np.zeros((batch, max_len, actions_n), dtype=np.float32),
            "risk": np.zeros((batch, max_len), dtype=np.float32),
            "utility": np.zeros((batch, max_len), dtype=np.float32),
            "preservation": np.zeros((batch, max_len), dtype=np.float32),
            "preservation_mask": np.zeros((batch, max_len), dtype=np.float32),
            "confidence": np.zeros((batch, max_len), dtype=np.float32),
            "reward": np.zeros((batch, max_len), dtype=np.float32),
            "terminated": np.ones((batch, max_len), dtype=np.float32),
            "mask": np.zeros((batch, max_len), dtype=np.float32),
        }
        for row, sequence in enumerate(sequences):
            length = sequence["length"]
            for key in arrays:
                if key == "mask":
                    arrays[key][row, :length] = 1.0
                else:
                    arrays[key][row, :length] = sequence[key]
        return {
            key: torch.from_numpy(value).to(device)
            for key, value in arrays.items()
        }

    @staticmethod
    def _weighted_mean(value, weight):
        return (value * weight).sum() / weight.sum().clamp_min(1e-6)

    def _world_losses(self, module, batch, out):
        deps = _require_full_sheeprl()
        torch, F = deps["torch"], deps["F"]
        Independent = deps["Independent"]
        OneHot = deps["OneHotCategoricalStraightThrough"]
        kl_divergence = deps["kl_divergence"]
        SymlogDistribution = deps["SymlogDistribution"]
        TwoHot = deps["TwoHotEncodingDistribution"]
        mask = batch["mask"]
        observation_loss = -SymlogDistribution(
            out["reconstruction"], dims=1, agg="mean"
        ).log_prob(batch["obs"])
        observation_loss = self._weighted_mean(observation_loss, mask)
        reward_loss = -TwoHot(out["reward_logits"], dims=1).log_prob(
            batch["reward"].unsqueeze(-1)
        )
        reward_loss = self._weighted_mean(reward_loss, mask)
        continue_target = 1.0 - batch["terminated"]
        continue_loss = F.binary_cross_entropy_with_logits(
            out["continue_logits"], continue_target, reduction="none"
        )
        continue_loss = self._weighted_mean(continue_loss, mask)

        post = out["posterior_logits"].reshape(
            *out["posterior_logits"].shape[:2],
            self.config.stochastic_size,
            self.config.discrete_size,
        )
        prior = out["prior_logits"].reshape_as(post)
        dynamic = kl_divergence(
            Independent(OneHot(logits=post.detach()), 1),
            Independent(OneHot(logits=prior), 1),
        )
        representation = kl_divergence(
            Independent(OneHot(logits=post), 1),
            Independent(OneHot(logits=prior.detach()), 1),
        )
        free_nats = torch.full_like(dynamic, self.config.kl_free_nats)
        kl_loss = (
            self.config.kl_dynamic_scale * torch.maximum(dynamic, free_nats)
            + self.config.kl_representation_scale
            * torch.maximum(representation, free_nats)
        )
        kl_loss = self._weighted_mean(kl_loss, mask)

        flat_mask = mask.reshape(-1) > 0
        skill_loss = F.cross_entropy(
            out["skill_logits"].reshape(-1, len(self.skill_classes))[flat_mask],
            batch["actions"].reshape(-1)[flat_mask],
        )
        candidate_loss = F.binary_cross_entropy_with_logits(
            out["candidate_logits"], batch["candidate_mask"], reduction="none"
        ).mean(dim=-1)
        candidate_loss = self._weighted_mean(candidate_loss, mask)
        risk_loss = F.binary_cross_entropy_with_logits(
            out["risk_logits"], batch["risk"], reduction="none"
        )
        risk_loss = self._weighted_mean(risk_loss, mask)
        probability_weight = mask * batch["confidence"]
        utility_loss = F.binary_cross_entropy_with_logits(
            out["utility_logits"], batch["utility"], reduction="none"
        )
        utility_loss = self._weighted_mean(utility_loss, probability_weight)
        preservation_weight = (
            mask * batch["preservation_mask"] * batch["confidence"]
        )
        preservation_loss_raw = F.binary_cross_entropy_with_logits(
            out["preservation_logits"],
            batch["preservation"],
            reduction="none",
        )
        if preservation_weight.sum() > 0:
            preservation_loss = self._weighted_mean(
                preservation_loss_raw, preservation_weight
            )
        else:
            preservation_loss = out["preservation_logits"].sum() * 0.0

        total = (
            self.config.observation_loss_scale * observation_loss
            + self.config.reward_loss_scale * reward_loss
            + self.config.continue_loss_scale * continue_loss
            + self.config.kl_loss_scale * kl_loss
            + self.config.skill_loss_scale * skill_loss
            + self.config.candidate_loss_scale * candidate_loss
            + self.config.risk_loss_scale * risk_loss
            + self.config.utility_loss_scale * utility_loss
            + self.config.preservation_loss_scale * preservation_loss
        )
        return {
            "world": total,
            "observation": observation_loss,
            "reward": reward_loss,
            "continue": continue_loss,
            "kl": kl_loss,
            "skill": skill_loss,
            "candidate": candidate_loss,
            "risk": risk_loss,
            "utility": utility_loss,
            "preservation": preservation_loss,
        }

    def _valid_candidate_mask(self, module, latent, fallback_mask=None):
        deps = _require_full_sheeprl()
        torch = deps["torch"]
        probabilities = torch.sigmoid(module.candidate_head(latent))
        mask = probabilities >= self.config.candidate_threshold
        if fallback_mask is not None:
            mask = mask | fallback_mask.bool()
        empty = ~mask.any(dim=-1)
        if empty.any():
            best = probabilities[empty].argmax(dim=-1)
            mask[empty] = False
            mask[empty, best] = True
        finish_index = self.skill_to_id.get("finish")
        if finish_index is not None:
            mask[..., finish_index] = True
        return mask

    def _imagine(self, module, start, candidate_mask, actual_actions):
        deps = _require_full_sheeprl()
        torch, F = deps["torch"], deps["F"]
        compute_lambda_values = deps["compute_lambda_values"]
        TwoHot = deps["TwoHotEncodingDistribution"]
        horizon = self.config.imagination_horizon
        stochastic = start[:, : module.stochastic_state_size]
        recurrent = start[:, module.stochastic_state_size :]
        state = start
        states = [state]
        imagined_actions = []
        log_probabilities = []
        entropies = []
        current_mask = candidate_mask.bool()
        for _ in range(horizon):
            actions, distributions = module.actor(
                state.detach(), mask={"mask_action_type": current_mask}
            )
            action = actions[0]
            imagined_actions.append(action.detach())
            distribution = distributions[0]
            log_probabilities.append(distribution.log_prob(action.detach()))
            entropies.append(distribution.entropy())
            with torch.no_grad():
                stochastic_next, recurrent_next = module.rssm.imagination(
                    stochastic.unsqueeze(0),
                    recurrent.unsqueeze(0),
                    action.detach().unsqueeze(0),
                )
                stochastic = stochastic_next.reshape(len(start), -1)
                recurrent = recurrent_next.squeeze(0)
                state = torch.cat((stochastic, recurrent), dim=-1)
                current_mask = self._valid_candidate_mask(module, state)
            states.append(state)
        imagined_states = torch.stack(states, dim=0)
        imagined_actions = torch.stack(imagined_actions, dim=0)
        log_probabilities = torch.stack(log_probabilities, dim=0)
        entropies = torch.stack(entropies, dim=0)
        with torch.no_grad():
            reward_features = torch.cat(
                (imagined_states[:-1], imagined_actions), dim=-1
            )
            rewards = TwoHot(
                module.reward_model(reward_features), dims=1
            ).mean
            continues = torch.sigmoid(
                module.continue_model(reward_features)
            )
            target_values = TwoHot(
                module.target_critic(imagined_states), dims=1
            ).mean
            lambda_values = compute_lambda_values(
                rewards,
                target_values[1:],
                continues * self.config.gamma,
                lmbda=self.config.lmbda,
            )
            baseline = TwoHot(module.critic(imagined_states[:-1]), dims=1).mean
            low = torch.quantile(lambda_values.float(), 0.05)
            high = torch.quantile(lambda_values.float(), 0.95)
            module.return_low.mul_(0.99).add_(low, alpha=0.01)
            module.return_high.mul_(0.99).add_(high, alpha=0.01)
            scale = torch.maximum(
                torch.ones_like(module.return_high),
                module.return_high - module.return_low,
            )
            advantage = (
                (lambda_values - module.return_low) / scale
                - (baseline - module.return_low) / scale
            )
            discounts = torch.cumprod(
                torch.cat(
                    (
                        torch.ones_like(continues[:1]),
                        continues[:-1] * self.config.gamma,
                    ),
                    dim=0,
                ),
                dim=0,
            )
        policy_loss = -(
            discounts.squeeze(-1)
            * (
                log_probabilities * advantage.detach().squeeze(-1)
                + self.config.entropy_scale * entropies
            )
        ).mean()
        actual_onehot = F.one_hot(
            actual_actions, num_classes=len(self.skill_classes)
        ).float()
        _, bc_distributions = module.actor(
            start.detach(), mask={"mask_action_type": candidate_mask.bool()}
        )
        behavior_cloning = -bc_distributions[0].log_prob(actual_onehot).mean()
        actor_loss = (
            policy_loss + self.config.behavior_cloning_scale * behavior_cloning
        )

        q_values = TwoHot(module.critic(imagined_states[:-1].detach()), dims=1)
        with torch.no_grad():
            target_prediction = TwoHot(
                module.target_critic(imagined_states[:-1]), dims=1
            ).mean
        critic_raw = -q_values.log_prob(lambda_values.detach())
        critic_raw = critic_raw - q_values.log_prob(target_prediction.detach())
        critic_loss = (
            critic_raw * discounts.squeeze(-1).detach()
        ).mean()
        return actor_loss, critic_loss, {
            "policy": policy_loss,
            "behavior_cloning": behavior_cloning,
            "imagined_return": lambda_values.mean(),
        }

    def _behavior_batch(self, out, batch):
        deps = _require_full_sheeprl()
        torch = deps["torch"]
        valid = batch["mask"].reshape(-1) > 0
        latent = out["latent"].reshape(-1, out["latent"].shape[-1])[valid]
        candidates = batch["candidate_mask"].reshape(
            -1, len(self.skill_classes)
        )[valid]
        actions = batch["actions"].reshape(-1)[valid]
        limit = self.config.imagination_batch_size
        if len(latent) > limit:
            indices = torch.randperm(len(latent), device=latent.device)[:limit]
            latent = latent[indices]
            candidates = candidates[indices]
            actions = actions[indices]
        return latent.detach(), candidates.detach(), actions.detach()

    @staticmethod
    def _update_target_critic(module, tau: float):
        with _require_full_sheeprl()["torch"].no_grad():
            for target, source in zip(
                module.target_critic.parameters(),
                module.critic.parameters(),
                strict=True,
            ):
                target.data.mul_(1.0 - tau).add_(source.data, alpha=tau)

    def fit(
        self,
        train_steps: list[StepRecord],
        *,
        val_steps: list[StepRecord] | None = None,
        epochs: int | None = None,
        batch_size: int | None = None,
    ):
        deps = _require_full_sheeprl()
        torch = deps["torch"]
        if not train_steps:
            raise ValueError("Cannot train the full DreamerV3 model with zero steps")
        vocabulary_steps = train_steps + (val_steps or [])
        discovered_skills = set(_build_vocab(vocabulary_steps))
        discovered_skills.update(self.skill_classes)
        self.skill_classes = sorted(discovered_skills)
        self.skill_to_id = {
            skill: index for index, skill in enumerate(self.skill_classes)
        }
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)
        module = self._make_module().to(self._device_name())
        self._module = module
        device = next(module.parameters()).device
        world_optimizer = torch.optim.AdamW(
            list(module.world_parameters()),
            lr=self.config.world_learning_rate,
            weight_decay=self.config.weight_decay,
        )
        actor_optimizer = torch.optim.AdamW(
            module.actor.parameters(),
            lr=self.config.actor_learning_rate,
            weight_decay=self.config.weight_decay,
        )
        critic_optimizer = torch.optim.AdamW(
            module.critic.parameters(),
            lr=self.config.critic_learning_rate,
            weight_decay=self.config.weight_decay,
        )
        sequences = self._prepare_sequences(train_steps)
        epochs = epochs or self.config.epochs
        batch_size = batch_size or self.config.batch_size
        self.training_history = []
        best_state = None
        best_objective = float("inf")
        for epoch in range(1, epochs + 1):
            module.train()
            order = np.random.permutation(len(sequences))
            totals: dict[str, float] = {}
            updates = 0
            for start_index in range(0, len(order), batch_size):
                selected = [
                    sequences[index]
                    for index in order[start_index : start_index + batch_size]
                ]
                batch = self._collate(selected, str(device))
                out = module.observe(batch["obs"], batch["actions"])
                losses = self._world_losses(module, batch, out)
                world_optimizer.zero_grad(set_to_none=True)
                losses["world"].backward()
                torch.nn.utils.clip_grad_norm_(
                    list(module.world_parameters()), self.config.world_gradient_clip
                )
                world_optimizer.step()

                with torch.no_grad():
                    refreshed = module.observe(batch["obs"], batch["actions"])
                latent, candidates, actual_actions = self._behavior_batch(
                    refreshed, batch
                )
                actor_loss, critic_loss, behavior = self._imagine(
                    module, latent, candidates, actual_actions
                )
                actor_optimizer.zero_grad(set_to_none=True)
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    module.actor.parameters(), self.config.actor_gradient_clip
                )
                actor_optimizer.step()
                critic_optimizer.zero_grad(set_to_none=True)
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    module.critic.parameters(), self.config.critic_gradient_clip
                )
                critic_optimizer.step()
                self._update_target_critic(module, self.config.target_critic_tau)

                scalars = {
                    **{key: value for key, value in losses.items()},
                    "actor": actor_loss,
                    "critic": critic_loss,
                    **behavior,
                }
                for key, value in scalars.items():
                    totals[key] = totals.get(key, 0.0) + float(
                        value.detach().cpu()
                    )
                updates += 1
            history: dict[str, Any] = {
                "epoch": epoch,
                **{key: value / max(updates, 1) for key, value in totals.items()},
            }
            if val_steps:
                val_metrics = evaluate_full_dreamer_predictions(
                    val_steps, self.predict(val_steps)
                )
                history["validation"] = val_metrics
                objective = val_metrics["validation_objective"]
            else:
                objective = history["world"]
            self.training_history.append(history)
            if objective < best_objective:
                best_objective = objective
                self.best_epoch = epoch
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in module.state_dict().items()
                }
        if best_state is not None:
            module.load_state_dict(best_state)
        module.eval()
        return self

    def _predict_sequence_batches(self, steps: list[StepRecord]):
        deps = _require_full_sheeprl()
        torch, F = deps["torch"], deps["F"]
        TwoHot = deps["TwoHotEncodingDistribution"]
        module = self._ensure_module()
        device = next(module.parameters()).device
        sequences = self._prepare_sequences(steps)
        rows: dict[tuple[str, int], dict[str, Any]] = {}
        module.eval()
        with self._deterministic_inference(module), torch.no_grad():
            for start in range(0, len(sequences), 64):
                selected = sequences[start : start + 64]
                batch = self._collate(selected, str(device))
                out = module.observe(batch["obs"], batch["actions"])
                for index, sequence in enumerate(selected):
                    length = sequence["length"]
                    latent = out["latent"][index, :length]
                    candidates = batch["candidate_mask"][index, :length].bool()
                    _, distributions = module.actor(
                        latent, mask={"mask_action_type": candidates}
                    )
                    skill_proba = distributions[0].probs
                    value = TwoHot(module.critic(latent), dims=1).mean.squeeze(-1)
                    reward = TwoHot(
                        out["reward_logits"][index, :length], dims=1
                    ).mean.squeeze(-1)
                    for offset, record in enumerate(sequence["records"]):
                        rows[(record.trajectory_id, record.step_id)] = {
                            "skill_proba": skill_proba[offset].cpu().numpy(),
                            "risk": float(
                                torch.sigmoid(out["risk_logits"][index, offset]).cpu()
                            ),
                            "utility": float(
                                torch.sigmoid(out["utility_logits"][index, offset]).cpu()
                            ),
                            "preservation": float(
                                torch.sigmoid(
                                    out["preservation_logits"][index, offset]
                                ).cpu()
                            ),
                            "value": float(value[offset].cpu()),
                            "reward": float(reward[offset].cpu()),
                        }
        return rows

    def predict(self, steps: list[StepRecord | dict]) -> dict[str, Any]:
        records = [
            step if isinstance(step, StepRecord) else StepRecord.model_validate(step)
            for step in steps
        ]
        rows = self._predict_sequence_batches(records)
        ordered = [rows[(step.trajectory_id, step.step_id)] for step in records]
        probabilities = np.stack([row["skill_proba"] for row in ordered])
        classes = np.asarray(self.skill_classes)
        return {
            "next_skill": classes[np.argmax(probabilities, axis=1)],
            "next_skill_proba": probabilities,
            "skill_classes": classes,
            "risk_score": np.asarray([row["risk"] for row in ordered]),
            "utility_score": np.asarray([row["utility"] for row in ordered]),
            "preservation_score": np.asarray(
                [row["preservation"] for row in ordered]
            ),
            "value_score": np.asarray([row["value"] for row in ordered]),
            "reward_score": np.asarray([row["reward"] for row in ordered]),
        }

    def score_actions(
        self, step: StepRecord | dict, actions: list[str]
    ) -> dict[str, Any]:
        repeated = []
        base = step.model_dump(mode="json") if isinstance(step, StepRecord) else dict(step)
        for index, action in enumerate(actions):
            row = dict(base)
            row["attack_action"] = action
            row["trajectory_id"] = f"{base['trajectory_id']}::candidate::{index}"
            repeated.append(StepRecord.model_validate(row))
        return self.predict(repeated)

    def rollout_score_step(
        self, step: StepRecord | dict, *, horizon: int = 5
    ) -> dict[str, Any]:
        deps = _require_full_sheeprl()
        torch, F = deps["torch"], deps["F"]
        TwoHot = deps["TwoHotEncodingDistribution"]
        module = self._ensure_module()
        device = next(module.parameters()).device
        record = step if isinstance(step, StepRecord) else StepRecord.model_validate(step)
        payload = self._sequence_payload([record])
        batch = self._collate([payload], str(device))
        branch_skills = [
            skill for skill in record.candidate_skills if skill in self.skill_to_id
        ]
        if not branch_skills:
            branch_skills = ["finish"] if "finish" in self.skill_to_id else [self.skill_classes[0]]
        target_index = self.skill_to_id.get(record.target_skill) if record.target_skill else None
        summaries = []
        module.eval()
        with self._deterministic_inference(module), torch.no_grad():
            observed = module.observe(batch["obs"], batch["actions"])
            base_stochastic = observed["posterior_states"][:, -1]
            base_recurrent = observed["recurrent_states"][:, -1]
            for first_skill in branch_skills:
                stochastic = base_stochastic.clone()
                recurrent = base_recurrent.clone()
                latent = torch.cat((stochastic, recurrent), dim=-1)
                action_id = self.skill_to_id[first_skill]
                imagined_skills = []
                risks, utilities, preservations, values, rewards, target_probs = (
                    [], [], [], [], [], []
                )
                for rollout_index in range(max(1, horizon)):
                    action = torch.zeros(1, 1, len(self.skill_classes), device=device)
                    action[:, :, action_id] = 1.0
                    reward_features = torch.cat(
                        (latent, action.squeeze(0)), dim=-1
                    )
                    predicted_reward = TwoHot(
                        module.reward_model(reward_features), dims=1
                    ).mean
                    stochastic_next, recurrent_next = module.rssm.imagination(
                        stochastic.unsqueeze(0), recurrent.unsqueeze(0), action
                    )
                    stochastic = stochastic_next.reshape(1, -1)
                    recurrent = recurrent_next.squeeze(0)
                    latent = torch.cat((stochastic, recurrent), dim=-1)
                    candidate_mask = self._valid_candidate_mask(module, latent)
                    _, distributions = module.actor(
                        latent, mask={"mask_action_type": candidate_mask}
                    )
                    probabilities = distributions[0].probs
                    next_id = int(probabilities.argmax(dim=-1).item())
                    imagined_skills.append(
                        first_skill if rollout_index == 0 else self.skill_classes[next_id]
                    )
                    risks.append(float(torch.sigmoid(module.risk_head(latent)).item()))
                    utilities.append(float(torch.sigmoid(module.utility_head(latent)).item()))
                    preservations.append(
                        float(torch.sigmoid(module.preservation_head(latent)).item())
                    )
                    values.append(float(TwoHot(module.critic(latent), dims=1).mean.item()))
                    rewards.append(float(predicted_reward.item()))
                    target_probs.append(
                        float(probabilities[0, target_index].item())
                        if target_index is not None
                        else 0.0
                    )
                    action_id = next_id
                selection_score = (
                    values[0]
                    + max(risks)
                    + float(np.mean(preservations))
                    + 0.25 * max(target_probs)
                )
                summaries.append(
                    {
                        "branch_first_skill": first_skill,
                        "risk_score": max(risks),
                        "utility_score": float(np.mean(utilities)),
                        "preservation_score": float(np.mean(preservations)),
                        "min_utility_score": min(utilities),
                        "final_utility_score": utilities[-1],
                        "value_score": values[0],
                        "reward_score": float(np.mean(rewards)),
                        "target_skill_probability": max(target_probs),
                        "selection_score": selection_score,
                        "rollout_imagined_skills": imagined_skills,
                        "rollout_target_reached": float(
                            record.target_skill in imagined_skills
                        )
                        if record.target_skill
                        else 0.0,
                    }
                )
        best = max(summaries, key=lambda row: row["selection_score"])
        return {
            **best,
            "rollout_backend": "sheeprl_full_dreamer_v3_actor_critic",
            "rollout_branch_count": len(summaries),
            "rollout_top_branch_summaries": sorted(
                summaries, key=lambda row: row["selection_score"], reverse=True
            )[:3],
        }

    def model_info(self) -> dict[str, Any]:
        module = self._ensure_module()
        components: dict[str, int] = {}
        for name, parameter in module.named_parameters():
            key = name.split(".")[0]
            components[key] = components.get(key, 0) + parameter.numel()
        return {
            "backend": "sheeprl_full_dreamer_v3_offline",
            "parameter_count": sum(p.numel() for p in module.parameters()),
            "trainable_parameter_count": sum(
                p.numel() for p in module.parameters() if p.requires_grad
            ),
            "parameters_by_component": components,
            "latent_size": (
                self.config.recurrent_state_size
                + self.config.stochastic_size * self.config.discrete_size
            ),
            "skill_class_count": len(self.skill_classes),
            "sheeprl_components": [
                "MLPEncoder",
                "MLPDecoder",
                "RSSM",
                "RecurrentModel",
                "Actor",
                "TwoHotEncodingDistribution",
                "lambda_return",
            ],
        }

    def save(self, path: str | Path) -> None:
        torch = _require_full_sheeprl()["torch"]
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        module = self._ensure_module()
        torch.save(module.state_dict(), path / "model.pt")
        metadata = {
            "config": asdict(self.config),
            "skill_classes": self.skill_classes,
            "training_history": self.training_history,
            "best_epoch": self.best_epoch,
            **self.model_info(),
        }
        (path / "metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: str | Path):
        torch = _require_full_sheeprl()["torch"]
        path = Path(path)
        metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
        model = cls(
            config=FullDreamerV3Config(**metadata["config"]),
            skill_classes=metadata["skill_classes"],
        )
        model.training_history = metadata.get("training_history", [])
        model.best_epoch = metadata.get("best_epoch")
        module = model._ensure_module()
        state = torch.load(path / "model.pt", map_location=model._device_name())
        module.load_state_dict(state)
        module.eval()
        return model
