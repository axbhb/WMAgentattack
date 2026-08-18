from wmagentattack.persistence_conflict import _readback_matches, _resolve_arguments


def test_dynamic_argument_uses_prior_output():
    step = {
        "arguments": {"event_id": "placeholder", "new_start_time": "2027-01-16 11:00"},
        "dynamic_arguments": {"event_id": {"step_index": 0, "path": ["id_"]}},
    }
    assert _resolve_arguments(step, [{"id_": "42"}])["event_id"] == "42"


def test_suite_readback_validators():
    assert _readback_matches(
        [{"id": 7, "subject": "wm-v19-alpha"}],
        {"kind": "banking_subject", "id": 7, "value": "wm-v19-alpha"},
    )
    assert _readback_matches(
        [{"body": "wm-v19-alpha"}],
        {"kind": "slack_message", "body": "wm-v19-alpha", "present": True},
    )
    assert _readback_matches(
        [{"title": "x", "start_time": "2027-01-15T09:00:00"}],
        {"kind": "calendar_start", "title": "x", "value": "2027-01-15T09:00:00"},
    )
    assert _readback_matches(
        {"shared_with": {"a@example.com": "rw"}},
        {"kind": "file_permission", "email": "a@example.com", "value": "rw"},
    )
