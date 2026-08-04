import json

import pytest

from wmagentattack.state_storage_v2 import (
    ContentAddressedStateStore,
    FeatureEnvelope,
    ModelTower,
    ScopedFeature,
    VisibilityScope,
    build_exact_state_transition,
)


def test_content_addressed_store_deduplicates_and_roundtrips(tmp_path):
    store = ContentAddressedStateStore(tmp_path)
    first = store.put({"reservation": {"status": "open"}})
    duplicate = store.put({"reservation": {"status": "open"}})
    changed = store.put({"reservation": {"status": "closed"}})
    assert first == duplicate
    assert first.fingerprint != changed.fingerprint
    assert len(list((tmp_path / "blobs").rglob("*.json"))) == 2
    assert store.get(first, requesting_tower=ModelTower.SIMULATOR) == {
        "reservation": {"status": "open"}
    }


def test_internal_state_cannot_be_dereferenced_by_model_towers(tmp_path):
    store = ContentAddressedStateStore(tmp_path)
    reference = store.put({"private": {"unobserved": "fact"}})
    for tower in (
        ModelTower.VICTIM_PROPOSAL,
        ModelTower.KNOWLEDGE_PROGRESS,
        ModelTower.COMPLETION_VALUE,
        ModelTower.PLANNER_VALUE,
    ):
        with pytest.raises(PermissionError):
            store.get(reference, requesting_tower=tower)


def test_feature_envelope_enforces_visibility_matrix():
    envelope = FeatureEnvelope(
        fields=(
            ScopedFeature(
                name="trusted_goal",
                visibility_scope=VisibilityScope.VICTIM_OBSERVED,
                value="find a hotel",
            ),
            ScopedFeature(
                name="planner_state_view",
                visibility_scope=VisibilityScope.PLANNER_PRIVILEGED,
                value={"reservation_count": 1},
            ),
            ScopedFeature(
                name="exact_state_ref",
                visibility_scope=VisibilityScope.SIMULATOR_INTERNAL,
                value="blobs/aa/hash.json",
            ),
        )
    )
    assert envelope.view(
        ModelTower.VICTIM_PROPOSAL, requested_fields=["trusted_goal"]
    ) == {"trusted_goal": "find a hotel"}
    with pytest.raises(PermissionError):
        envelope.view(
            ModelTower.VICTIM_PROPOSAL,
            requested_fields=["trusted_goal", "planner_state_view"],
        )
    assert "planner_state_view" in envelope.allowed_view(ModelTower.COMPLETION_VALUE)
    assert "exact_state_ref" not in envelope.allowed_view(ModelTower.COMPLETION_VALUE)
    with pytest.raises(PermissionError):
        envelope.view(
            ModelTower.COMPLETION_VALUE, requested_fields=["exact_state_ref"]
        )
    assert envelope.view(
        ModelTower.SIMULATOR, requested_fields=["exact_state_ref"]
    ) == {"exact_state_ref": "blobs/aa/hash.json"}


def test_transition_stores_references_delta_and_no_outcomes(tmp_path):
    store = ContentAddressedStateStore(tmp_path)
    record = build_exact_state_transition(
        store,
        episode_id="episode-1",
        call_index=0,
        initial_state={"reservation": []},
        state_before={"reservation": []},
        state_after={"reservation": [{"id": "R1"}]},
        exact_delta=(
            {"op": "add", "path": "/reservation/0", "value": {"id": "R1"}},
        ),
        execution_status="success",
        error_type=None,
    )
    assert record.initial_state_ref == record.state_before_ref
    assert record.state_after_ref != record.state_before_ref
    assert record.delta_operation_count == 1
    assert record.delta_roots == ("/reservation",)
    assert record.state_changed
    payload = record.model_dump(mode="json")
    assert payload["outcome_labels_present"] is False
    serialized = json.dumps(payload).lower()
    assert '"utility"' not in serialized
    assert '"security"' not in serialized
    assert '"canonical_state"' not in serialized
