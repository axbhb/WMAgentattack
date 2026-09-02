"""Large hybrid semantic world model for the v45 full-data experiment.

The architecture deliberately keeps deterministic AgentDojo execution outside
the neural network.  A pretrained text model supplies field/candidate
embeddings, while three trainable large components model structured state,
victim action selection, and multi-step residual dynamics.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor, nn

from .joint_outcome_auxiliary import JOINT_OUTCOME_CLASSES


STATE_FIELDS = (
    "trusted_goal",
    "visible_observation",
    "legal_tools",
    "visible_prior_tool",
    "source_track",
)


def _compact_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def structured_state_texts(causal: Mapping[str, Any]) -> tuple[str, ...]:
    """Build five inference-visible text fields without outcome/task IDs."""
    required = {
        "trusted_goal", "visible_observation", "visible_prior_tool",
        "legal_tool_names", "tool_schemas", "source", "track",
    }
    missing = sorted(required - set(causal))
    if missing:
        raise ValueError(f"missing structured state fields: {missing}")
    schema_by_name: dict[str, Mapping[str, Any]] = {}
    for schema in causal["tool_schemas"]:
        if not isinstance(schema, Mapping):
            continue
        function = schema.get("function", {})
        if isinstance(function, Mapping) and function.get("name"):
            schema_by_name[str(function["name"])] = function
    tool_rows = []
    for name in sorted(map(str, causal["legal_tool_names"])):
        function = schema_by_name.get(name, {"name": name})
        tool_rows.append({
            "name": name,
            "description": str(function.get("description", "")),
            "parameters": function.get("parameters", {}),
        })
    return (
        "query: trusted user goal and obligations; " + str(causal["trusted_goal"]),
        "passage: visible environment observation; " + str(causal["visible_observation"]),
        "passage: currently legal tool interfaces; " + _compact_json(tool_rows),
        "passage: previously visible executed tool; " + str(causal["visible_prior_tool"]),
        "passage: sandbox source and trajectory track; "
        + str(causal["source"]) + "; " + str(causal["track"]),
    )


def candidate_text(candidate_id: str, candidate: Mapping[str, Any]) -> str:
    """Describe a legal candidate without runtime or outcome metadata."""
    function = candidate.get("function", {})
    if not isinstance(function, Mapping):
        function = {}
    return "passage: candidate tool action; " + _compact_json({
        "candidate": candidate_id,
        "name": function.get("name", candidate_id),
        "description": function.get("description", ""),
        "parameters": function.get("parameters", {}),
        "kind": candidate.get("kind", "tool"),
        "source": candidate.get("source", "unknown"),
    })


@dataclass(frozen=True)
class LargeWorldModelConfig:
    semantic_size: int = 768
    hidden_size: int = 768
    state_layers: int = 8
    action_layers: int = 6
    residual_layers: int = 6
    attention_heads: int = 12
    feedforward_size: int = 3072
    dropout: float = 0.1
    memory_tokens: int = 8
    outcome_size: int = 5

    def validate(self) -> None:
        if self.semantic_size <= 0 or self.hidden_size <= 0:
            raise ValueError("semantic and hidden sizes must be positive")
        if self.hidden_size % self.attention_heads:
            raise ValueError("hidden size must be divisible by attention heads")
        if min(self.state_layers, self.action_layers, self.residual_layers) < 1:
            raise ValueError("all large components require at least one layer")
        if self.memory_tokens < 1 or self.outcome_size < 1:
            raise ValueError("invalid memory/outcome size")


class LargeStructuredStateEncoder(nn.Module):
    """Fuse pretrained field embeddings with a deep typed state transformer."""

    def __init__(self, config: LargeWorldModelConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.input_projection = nn.Linear(config.semantic_size, config.hidden_size)
        self.field_embedding = nn.Embedding(len(STATE_FIELDS) + 1, config.hidden_size)
        self.summary_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.attention_heads,
            dim_feedforward=config.feedforward_size,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=config.state_layers, norm=nn.LayerNorm(config.hidden_size)
        )
        self.output_norm = nn.LayerNorm(config.hidden_size)

    def forward(self, fields: Tensor, field_mask: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if fields.ndim != 3 or fields.shape[1] != len(STATE_FIELDS):
            raise ValueError("state fields must have shape [batch, 5, semantic_size]")
        if field_mask.shape != fields.shape[:2]:
            raise ValueError("field mask shape mismatch")
        batch = fields.shape[0]
        ids = torch.arange(len(STATE_FIELDS), device=fields.device)[None, :]
        nodes = self.input_projection(fields) + self.field_embedding(ids)
        summary = self.summary_token.expand(batch, -1, -1) + self.field_embedding.weight[-1]
        tokens = torch.cat((summary, nodes), dim=1)
        mask = torch.cat((torch.ones(batch, 1, dtype=torch.bool, device=fields.device), field_mask), dim=1)
        encoded = self.transformer(tokens, src_key_padding_mask=~mask)
        state = self.output_norm(encoded[:, 0])
        return state, encoded[:, 1:], mask[:, 1:]


class LargeVictimActionDynamics(nn.Module):
    """Deep candidate-query cross-attention over structured state nodes."""

    def __init__(self, config: LargeWorldModelConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.candidate_projection = nn.Linear(config.semantic_size, config.hidden_size)
        layer = nn.TransformerDecoderLayer(
            d_model=config.hidden_size,
            nhead=config.attention_heads,
            dim_feedforward=config.feedforward_size,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            layer, num_layers=config.action_layers, norm=nn.LayerNorm(config.hidden_size)
        )
        self.action_score = nn.Sequential(
            nn.LayerNorm(config.hidden_size), nn.Linear(config.hidden_size, 1)
        )
        self.outcome_head = nn.Sequential(
            nn.LayerNorm(config.hidden_size),
            nn.Linear(config.hidden_size, config.hidden_size), nn.GELU(),
            nn.Linear(config.hidden_size, config.outcome_size),
        )
        self.joint_head = nn.Sequential(
            nn.LayerNorm(config.hidden_size),
            nn.Linear(config.hidden_size, config.hidden_size), nn.GELU(),
            nn.Linear(config.hidden_size, len(JOINT_OUTCOME_CLASSES)),
        )

    def encode_candidates(self, candidate_embeddings: Tensor) -> Tensor:
        if candidate_embeddings.ndim != 2:
            raise ValueError("candidate embeddings must be rank two")
        return self.candidate_projection(candidate_embeddings)

    def forward(
        self,
        state: Tensor,
        state_nodes: Tensor,
        state_mask: Tensor,
        candidate_embeddings: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        candidate = self.encode_candidates(candidate_embeddings)
        query = candidate[None, :, :].expand(state.shape[0], -1, -1)
        decoded = self.decoder(query, state_nodes, memory_key_padding_mask=~state_mask)
        logits = self.action_score(decoded).squeeze(-1)
        return logits, self.outcome_head(state), self.joint_head(state), candidate


class LargeTransformerResidualDynamics(nn.Module):
    """Zero-initialized Transformer memory dynamics for imagined rollouts."""

    def __init__(self, config: LargeWorldModelConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.memory = nn.Parameter(torch.zeros(1, config.memory_tokens, config.hidden_size))
        self.step_embedding = nn.Embedding(16, config.hidden_size)
        layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.attention_heads,
            dim_feedforward=config.feedforward_size,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.dynamics = nn.TransformerEncoder(
            layer, num_layers=config.residual_layers, norm=nn.LayerNorm(config.hidden_size)
        )
        self.transition_projection = nn.Linear(config.hidden_size, config.hidden_size)
        self.one_step_query = nn.Linear(config.hidden_size, config.hidden_size)
        self.rollout_query = nn.Linear(config.hidden_size, config.hidden_size)
        self.future_joint_head = nn.Linear(config.hidden_size, len(JOINT_OUTCOME_CLASSES))
        nn.init.zeros_(self.transition_projection.weight)
        nn.init.zeros_(self.transition_projection.bias)
        nn.init.zeros_(self.one_step_query.weight)
        nn.init.zeros_(self.one_step_query.bias)
        nn.init.zeros_(self.rollout_query.weight)
        nn.init.zeros_(self.rollout_query.bias)

    def one_step_delta_logits(self, state: Tensor, candidates: Tensor) -> Tensor:
        query = self.one_step_query(state)
        return torch.einsum("bh,ch->bc", query, candidates) / self.config.hidden_size**0.5

    def advance(self, hidden: Tensor, action: Tensor, step: int) -> Tensor:
        if step >= self.step_embedding.num_embeddings:
            raise ValueError("rollout step exceeds configured positional support")
        batch = hidden.shape[0]
        memory = self.memory.expand(batch, -1, -1)
        position = self.step_embedding.weight[step][None, None, :].expand(batch, 1, -1)
        tokens = torch.cat((hidden[:, None, :] + position, action[:, None, :], memory), dim=1)
        encoded = self.dynamics(tokens)
        return nn.functional.layer_norm(
            hidden + self.transition_projection(encoded[:, 0]),
            (self.config.hidden_size,),
        )

    def rollout_logits(self, hidden: Tensor, candidates: Tensor) -> Tensor:
        query = self.rollout_query(hidden)
        return torch.einsum("bh,ch->bc", query, candidates) / self.config.hidden_size**0.5

    def joint_logits(self, hidden: Tensor) -> Tensor:
        return self.future_joint_head(hidden)


class LargeHybridWorldModel(nn.Module):
    """Compose the three large learned components around exact sandbox execution."""

    def __init__(self, config: LargeWorldModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or LargeWorldModelConfig()
        self.config.validate()
        self.state_encoder = LargeStructuredStateEncoder(self.config)
        self.victim_dynamics = LargeVictimActionDynamics(self.config)
        self.residual_dynamics = LargeTransformerResidualDynamics(self.config)

    def teacher(
        self, fields: Tensor, field_mask: Tensor, candidates: Tensor
    ) -> dict[str, Tensor]:
        state, nodes, mask = self.state_encoder(fields, field_mask)
        action, outcome, joint, candidate_hidden = self.victim_dynamics(
            state, nodes, mask, candidates
        )
        return {
            "state": state,
            "state_nodes": nodes,
            "action_logits": action,
            "outcome_logits": outcome,
            "joint_logits": joint,
            "candidate_hidden": candidate_hidden,
        }

    def freeze_teacher(self) -> None:
        for module in (self.state_encoder, self.victim_dynamics):
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad_(False)

    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def architecture(self) -> dict[str, Any]:
        return {
            "config": asdict(self.config),
            "total_parameters": sum(p.numel() for p in self.parameters()),
            "trainable_parameters": self.trainable_parameter_count(),
            "exact_sandbox_inside_neural_model": False,
        }


def parameter_breakdown(model: LargeHybridWorldModel) -> dict[str, int]:
    return {
        "structured_state_encoder": sum(p.numel() for p in model.state_encoder.parameters()),
        "victim_action_dynamics": sum(p.numel() for p in model.victim_dynamics.parameters()),
        "multi_step_residual_dynamics": sum(p.numel() for p in model.residual_dynamics.parameters()),
        "total": sum(p.numel() for p in model.parameters()),
    }
