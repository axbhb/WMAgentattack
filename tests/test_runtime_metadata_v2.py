from pathlib import Path

from wmagentattack.runtime_metadata_v2 import (
    EpisodeLocalMetadataNormalizer,
    load_runtime_metadata_registry,
)


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = load_runtime_metadata_registry(
    ROOT / "configs" / "0726_runtime_metadata_normalization_v2.json"
)


def test_registry_is_narrow_and_preserves_raw_artifacts():
    assert set(REGISTRY.rules) == {"create_calendar_event", "send_email"}
    assert all(rule.volatile_fields == ("timestamp",) for rule in REGISTRY.rules.values())
    assert all(rule.raw_exact_state_retained for rule in REGISTRY.rules.values())
    assert set(REGISTRY.hard_boundaries.values()) == {True}


def test_registered_wall_clock_is_episode_local_and_replay_stable():
    left = EpisodeLocalMetadataNormalizer(REGISTRY)
    right = EpisodeLocalMetadataNormalizer(REGISTRY)
    left.observe_transition(
        tool_name="send_email",
        call_index=4,
        runtime_output={"id_": "1", "timestamp": "2026-01-01T01:02:03.1"},
        exact_delta=[],
    )
    right.observe_transition(
        tool_name="send_email",
        call_index=4,
        runtime_output={"id_": "1", "timestamp": "2026-02-02T02:03:04.2"},
        exact_delta=[],
    )
    assert left.normalize({"id_": "1", "timestamp": "2026-01-01T01:02:03.1"}) == right.normalize(
        {"id_": "1", "timestamp": "2026-02-02T02:03:04.2"}
    )
    assert left.semantic_fingerprint(
        {"id_": "1", "timestamp": "2026-01-01T01:02:03.1"}
    ) == right.semantic_fingerprint(
        {"id_": "1", "timestamp": "2026-02-02T02:03:04.2"}
    )


def test_unregistered_tools_and_user_dates_are_not_normalized():
    normalizer = EpisodeLocalMetadataNormalizer(REGISTRY)
    assert normalizer.observe_transition(
        tool_name="get_day_calendar_events",
        call_index=0,
        runtime_output={"timestamp": "2026-01-01T01:02:03"},
        exact_delta=[],
    ) == ()
    value = {
        "timestamp": "2026-01-01T01:02:03",
        "start_time": "2026-08-01T10:00:00",
        "end_time": "2026-08-01T11:00:00",
    }
    assert normalizer.normalize(value) == value


def test_nested_state_delta_and_structured_attribute_share_one_token():
    normalizer = EpisodeLocalMetadataNormalizer(REGISTRY)
    timestamp = "2026-01-01T01:02:03.123456"
    delta = [
        {
            "op": "add",
            "path": "/inbox/emails/1",
            "value": {"id_": "1", "timestamp": timestamp},
        },
        {"op": "replace", "path": "/metadata/timestamp", "value": timestamp},
    ]
    tokens = normalizer.observe_transition(
        tool_name="create_calendar_event",
        call_index=2,
        runtime_output={"id_": "event-1"},
        exact_delta=delta,
    )
    assert tokens == ("VOLATILE::timestamp::CALL_002::ITEM_000",)
    normalized_delta = normalizer.normalize(delta)
    assert normalized_delta[0]["value"]["timestamp"] == tokens[0]
    assert normalized_delta[1]["value"] == tokens[0]
    record = {"name": "timestamp", "value": timestamp, "kind": "TIME_SCOPED"}
    assert normalizer.normalize(record)["value"] == tokens[0]
    assert normalizer.public_manifest()["binding_count"] == 1
