import torch

from wmagentattack.large_hybrid_world_model import (
    LargeHybridWorldModel,
    LargeWorldModelConfig,
    candidate_text,
    parameter_breakdown,
    structured_state_texts,
)


def tiny_config():
    return LargeWorldModelConfig(
        semantic_size=16,
        hidden_size=24,
        state_layers=2,
        action_layers=2,
        residual_layers=2,
        attention_heads=4,
        feedforward_size=48,
        dropout=0.0,
        memory_tokens=2,
        outcome_size=5,
    )


def test_text_fields_exclude_outcomes_and_task_ids():
    causal = {
        "trusted_goal": "Find the meeting",
        "visible_observation": "Calendar is visible",
        "visible_prior_tool": "calendar_read",
        "legal_tool_names": ["calendar_read"],
        "tool_schemas": [{"function": {"name": "calendar_read", "description": "Read calendar", "parameters": {}}}],
        "source": "agentdojo",
        "track": "attacked",
        "utility": True,
        "security": True,
        "task_id": "forbidden",
    }
    fields = structured_state_texts(causal)
    assert len(fields) == 5
    combined = " ".join(fields)
    assert "forbidden" not in combined
    assert "utility" not in combined
    assert "security" not in combined


def test_candidate_description_is_deterministic():
    candidate = {"function": {"name": "calendar_read", "parameters": {}}, "kind": "tool", "source": "agentdojo"}
    assert candidate_text("x", candidate) == candidate_text("x", candidate)


def test_large_hybrid_shapes_and_legal_mask_compatibility():
    model = LargeHybridWorldModel(tiny_config())
    fields = torch.randn(3, 5, 16)
    mask = torch.ones(3, 5, dtype=torch.bool)
    candidates = torch.randn(7, 16)
    output = model.teacher(fields, mask, candidates)
    assert output["action_logits"].shape == (3, 7)
    assert output["outcome_logits"].shape == (3, 5)
    assert output["joint_logits"].shape == (3, 4)
    assert output["state"].shape == (3, 24)


def test_residual_is_zero_initialized_at_h1_and_transition():
    model = LargeHybridWorldModel(tiny_config()).eval()
    state = torch.randn(2, 24)
    actions = torch.randn(2, 24)
    candidates = torch.randn(7, 24)
    assert torch.equal(model.residual_dynamics.one_step_delta_logits(state, candidates), torch.zeros(2, 7))
    advanced = model.residual_dynamics.advance(state, actions, 1)
    expected = torch.nn.functional.layer_norm(state, (24,))
    assert torch.allclose(advanced, expected, atol=1e-6)


def test_freeze_teacher_leaves_only_residual_trainable():
    model = LargeHybridWorldModel(tiny_config())
    before = parameter_breakdown(model)
    model.freeze_teacher()
    assert model.trainable_parameter_count() == before["multi_step_residual_dynamics"]
