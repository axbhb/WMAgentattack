from wmagentattack.toolsandbox_replica_recovery import (
    canonical_replica_payload,
    replicas_semantically_identical,
    runtime_uuid_mapping,
)


def _payload(before, after, output=None, error=None, status="success"):
    return canonical_replica_payload(
        before=before,
        after=after,
        output=output,
        error=error,
        status=status,
    )


def test_memory_addresses_are_representation_only():
    first, count_a = _payload(
        {}, {}, error={"type": "NoDataError", "message": "Expr at 0x7F01 found"}, status="error"
    )
    second, count_b = _payload(
        {}, {}, error={"type": "NoDataError", "message": "Expr at 0x9ABC found"}, status="error"
    )
    assert replicas_semantically_identical(first, second)
    assert count_a == count_b == {"memory_addresses": 1, "runtime_uuids": 0}


def test_only_new_runtime_uuids_are_canonicalized():
    existing = "11111111-1111-4111-8111-111111111111"
    new_a = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
    new_b = "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"
    before = {"contacts": [{"id": existing}]}
    first, count_a = _payload(before, {"contacts": [{"id": existing}], "messages": [{"id": new_a}]}, new_a)
    second, count_b = _payload(before, {"contacts": [{"id": existing}], "messages": [{"id": new_b}]}, new_b)
    assert replicas_semantically_identical(first, second)
    assert count_a == count_b == {"memory_addresses": 0, "runtime_uuids": 1}
    assert runtime_uuid_mapping(before, before, existing) == {}


def test_semantic_differences_remain_visible():
    first, _ = _payload({}, {"messages": [{"content": "hello"}]}, "ok")
    second, _ = _payload({}, {"messages": [{"content": "different"}]}, "ok")
    assert not replicas_semantically_identical(first, second)


def test_error_type_is_never_normalized():
    first, _ = _payload({}, {}, error={"type": "A", "message": "x at 0x111"}, status="error")
    second, _ = _payload({}, {}, error={"type": "B", "message": "x at 0x222"}, status="error")
    assert not replicas_semantically_identical(first, second)
