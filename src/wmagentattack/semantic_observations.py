"""Deterministic keys and structured features for semantic observations."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np

from wmagentattack.dreamer_world_model import step_to_dreamer_text
from wmagentattack.schema import StepRecord


def _step_value(step: StepRecord | dict[str, Any]) -> dict[str, Any]:
    return step.model_dump(mode="json") if isinstance(step, StepRecord) else dict(step)


def observation_structure_tokens(
    step: StepRecord | dict[str, Any], attack_action: str | None = None
) -> tuple[str, ...]:
    value = _step_value(step)
    group_id = str(value.get("multiseed_group_id") or "")
    parts = group_id.split("__") if group_id else []
    family = parts[-2] if len(parts) >= 2 and group_id.startswith("attack::") else "clean"
    injection = next(
        (part for part in parts if part.startswith("injection_task_")),
        "injection_task_NONE",
    )
    domain = str(value.get("domain") or "UNKNOWN")
    action = str(attack_action or value.get("attack_action") or "NONE")
    return (
        f"domain={domain}",
        f"family={family}",
        f"injection={injection}",
        f"domain_family={domain}|{family}",
        f"family_injection={family}|{injection}",
        f"attack_action={action}",
    )


def observation_cache_key(
    step: StepRecord | dict[str, Any], attack_action: str | None = None
) -> str:
    payload = {
        "text": step_to_dreamer_text(step, attack_action),
        "structure": observation_structure_tokens(step, attack_action),
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def hashed_structured_features(
    step: StepRecord | dict[str, Any],
    *,
    dim: int = 32,
    attack_action: str | None = None,
) -> np.ndarray:
    if dim <= 0:
        raise ValueError("Structured feature dimension must be positive")
    vector = np.zeros(dim, dtype=np.float32)
    for token in observation_structure_tokens(step, attack_action):
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:8], "little") % dim
        vector[index] += 1.0
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 0.0 else vector


def combine_semantic_and_structured(
    semantic: np.ndarray, structured: np.ndarray
) -> np.ndarray:
    return combine_feature_blocks(semantic, structured)


def combine_feature_blocks(*blocks: np.ndarray) -> np.ndarray:
    if not blocks:
        raise ValueError("At least one feature block is required")
    normalized = []
    for block in blocks:
        value = np.asarray(block, dtype=np.float32)
        if value.ndim != 1:
            raise ValueError("Feature blocks must be one-dimensional")
        block_norm = float(np.linalg.norm(value))
        normalized.append(value / block_norm if block_norm > 0.0 else value)
    combined = np.concatenate(normalized).astype(np.float32)
    norm = float(np.linalg.norm(combined))
    return combined / norm if norm > 0.0 else combined
