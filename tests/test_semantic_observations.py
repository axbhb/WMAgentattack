import numpy as np
import pytest

from wmagentattack.full_dreamer_v3 import (
    FullDreamerV3Config,
    FullSheepRLDreamerV3,
)
from wmagentattack.semantic_observations import (
    combine_feature_blocks,
    combine_semantic_and_structured,
    hashed_structured_features,
    observation_cache_key,
)


def _step(family: str = "tool_knowledge"):
    return {
        "domain": "workspace",
        "task_id": "user_task_1",
        "multiseed_group_id": (
            "attack::workspace__user_task_1__injection_task_0__"
            f"{family}__abc"
        ),
        "attack_action": "inject",
    }


def test_semantic_observation_key_includes_explicit_structure():
    assert observation_cache_key(_step("tool_knowledge")) != observation_cache_key(
        _step("dynamic_multistage")
    )
    structured = hashed_structured_features(_step(), dim=32)
    assert structured.shape == (32,)
    assert np.linalg.norm(structured) == pytest.approx(1.0)
    combined = combine_semantic_and_structured(np.ones(4), structured)
    assert combined.shape == (36,)
    assert np.linalg.norm(combined) == pytest.approx(1.0)
    fused = combine_feature_blocks(np.ones(4), np.ones(3), structured)
    assert fused.shape == (39,)
    assert np.linalg.norm(fused) == pytest.approx(1.0)


def test_full_dreamer_reads_precomputed_observations_and_fails_closed(tmp_path):
    step = _step()
    path = tmp_path / "features.npz"
    expected = np.asarray([[0.1, 0.2, 0.3]], dtype=np.float32)
    np.savez_compressed(
        path,
        keys=np.asarray([observation_cache_key(step)], dtype="<U64"),
        features=expected,
    )
    model = FullSheepRLDreamerV3(
        FullDreamerV3Config(
            obs_dim=3,
            observation_feature_mode="precomputed",
            observation_feature_path=str(path),
        ),
        skill_classes=["a", "b"],
    )
    assert np.allclose(model._vectorize_step(step), expected[0])
    with pytest.raises(KeyError):
        model._vectorize_step(_step("dynamic_multistage"))
