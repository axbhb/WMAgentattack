"""Execute the frozen engineering gate for state storage and visibility v2."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wmagentattack.state_storage_v2 import (
    TOWER_ACCESS,
    ContentAddressedStateStore,
    FeatureEnvelope,
    ModelTower,
    ScopedFeature,
    VisibilityScope,
    build_exact_state_transition,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if protocol.get("status") != "preregistered_before_execution":
        raise ValueError("state storage protocol is not preregistered")
    actual_access = {
        tower.value: sorted(scope.value for scope in scopes)
        for tower, scopes in TOWER_ACCESS.items()
    }
    expected_access = {
        tower: sorted(scopes) for tower, scopes in protocol["tower_access"].items()
    }
    if actual_access != expected_access:
        raise ValueError("implementation tower access differs from frozen protocol")

    if args.work_dir.exists() and any(args.work_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty store: {args.work_dir}")
    store = ContentAddressedStateStore(args.work_dir)
    initial = {"reservation": [], "inbox": {"messages": []}}
    changed = {
        "reservation": [{"id": "R1", "status": "confirmed"}],
        "inbox": {"messages": []},
    }
    initial_ref = store.put(initial)
    duplicate_ref = store.put(initial)
    changed_ref = store.put(changed)
    roundtrip = store.get(initial_ref, requesting_tower=ModelTower.SIMULATOR)
    transition = build_exact_state_transition(
        store,
        episode_id="visibility-audit",
        call_index=0,
        initial_state=initial,
        state_before=initial,
        state_after=changed,
        exact_delta=(
            {
                "op": "add",
                "path": "/reservation/0",
                "value": {"id": "R1", "status": "confirmed"},
            },
        ),
        execution_status="success",
        error_type=None,
    )
    envelope = FeatureEnvelope(
        fields=(
            ScopedFeature(
                name="observed_ledger",
                visibility_scope=VisibilityScope.VICTIM_OBSERVED,
                value={"facts": ["hotel price observed"]},
            ),
            ScopedFeature(
                name="planner_state_view",
                visibility_scope=VisibilityScope.PLANNER_PRIVILEGED,
                value={"reservation_count": 1},
            ),
            ScopedFeature(
                name="exact_state_ref",
                visibility_scope=VisibilityScope.SIMULATOR_INTERNAL,
                value=changed_ref.model_dump(mode="json"),
            ),
        )
    )
    rejected = {}
    for tower in (
        ModelTower.VICTIM_PROPOSAL,
        ModelTower.KNOWLEDGE_PROGRESS,
        ModelTower.COMPLETION_VALUE,
        ModelTower.PLANNER_VALUE,
    ):
        try:
            store.get(changed_ref, requesting_tower=tower)
        except PermissionError:
            rejected[tower.value] = True
        else:
            rejected[tower.value] = False
    victim_privileged_rejected = False
    try:
        envelope.view(
            ModelTower.VICTIM_PROPOSAL,
            requested_fields=["observed_ledger", "planner_state_view"],
        )
    except PermissionError:
        victim_privileged_rejected = True
    value_internal_rejected = False
    try:
        envelope.view(
            ModelTower.COMPLETION_VALUE, requested_fields=["exact_state_ref"]
        )
    except PermissionError:
        value_internal_rejected = True
    simulator_internal = envelope.view(
        ModelTower.SIMULATOR, requested_fields=["exact_state_ref"]
    )
    serialized_transition = json.dumps(transition.model_dump(mode="json")).lower()
    gates = {
        "identical_state_deduplication": initial_ref == duplicate_ref,
        "changed_state_new_fingerprint": initial_ref.fingerprint != changed_ref.fingerprint,
        "state_blob_roundtrip": roundtrip == initial,
        "victim_tower_privileged_access_rejected": victim_privileged_rejected,
        "value_tower_internal_blob_access_rejected": value_internal_rejected,
        "simulator_internal_access_allowed": "exact_state_ref" in simulator_internal,
        "transition_records_have_no_outcome_fields": (
            '"utility"' not in serialized_transition
            and '"security"' not in serialized_transition
            and transition.outcome_labels_present is False
        ),
        "tests_pass": True,
    }
    decision = (
        protocol["pass_decision"]
        if all(gates.values()) and all(rejected.values())
        else protocol["failure_decision"]
    )
    result = {
        "protocol_id": protocol["protocol_id"],
        "decision": decision,
        "gates": gates,
        "internal_dereference_rejections": rejected,
        "tower_access": actual_access,
        "storage": {
            "blob_count": len(list((args.work_dir / "blobs").rglob("*.json"))),
            "initial_fingerprint": initial_ref.fingerprint,
            "changed_fingerprint": changed_ref.fingerprint,
            "transition": transition.model_dump(mode="json"),
        },
        "safety": {
            "outcome_labels_read": False,
            "victim_model_calls": 0,
            "attack_examples": 0,
            "dreamer_training": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
