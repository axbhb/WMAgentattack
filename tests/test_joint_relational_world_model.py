import importlib.util
from pathlib import Path

import torch

from wmagentattack.joint_relational_world_model import (
    JointRelationalSuccessorTransition,
    global_pointer_probabilities,
)


def test_shapes_and_zero_start_residual():
    model = JointRelationalSuccessorTransition(8, 7, 12, 6, 5, 4)
    state = torch.randn(2, 8)
    action = torch.randn(2, 7)
    records = torch.randn(4, 6)
    conflicts = torch.randn(3, 4)
    initial = model.initial_hidden(state)
    hidden, record_logits, delta, conflict_logits, execution = model(state, action, records, conflicts)
    assert torch.allclose(hidden, model.next_norm(initial), atol=1e-6)
    assert record_logits.shape == (2, 4)
    assert delta.shape == (2, 5)
    assert conflict_logits.shape == (2, 3)
    assert execution.shape == (2,)
    relation = model.relation_logits(hidden[:1], records, torch.randn(5, 5))
    assert relation.shape == (4, 5)


def test_global_pointer_is_joint_noisy_or():
    records = torch.tensor([10.0, -10.0])
    relations = torch.tensor([[10.0, -10.0], [10.0, 10.0]])
    probability = global_pointer_probabilities(records, relations)
    assert probability[0] > 0.99
    assert probability[1] < 0.001


def test_training_contract_binds_record_local_goal_edges():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("v30_train_test", root / "scripts/271_train_joint_relational_successor_v30.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    record = {
        "entity_type": "webpage", "link_status": "UNIQUE",
        "attributes": [{"name": "content", "kind": "SINGLE_VALUED"}],
        "newly_matched_goal_term_indices": [1],
    }
    signature = module.record_signature(record)
    row = {
        "model_input": {
            "normalized_action": {"tool_id": "slack::get_webpage"},
            "current_semantic_state": {"goal": {"fact_terms": ["a", "b"]}},
        },
        "model_target": {"relational_successor_delta": {
            "added_evidence_records": [record], "newly_matched_goal_term_indices": [1],
        }},
    }
    record_y, relation_y, pointer_y = module.row_targets(row, [signature], [0])
    assert record_y.tolist() == [1.0]
    assert relation_y.tolist() == [[0.0, 1.0]]
    assert pointer_y.tolist() == [0.0, 1.0]
