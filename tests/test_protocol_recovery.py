import pytest
from wmagentattack.protocol_recovery import explicit_call, valid_runtime_arguments


def validate(name, args):
    return name == "lookup" and set(args) == {"query"} and isinstance(args["query"], str)


@pytest.mark.parametrize("text", [
    '<function=lookup>{"query":"x"}</function>',
    '<function=name="lookup" parameters={"query":"x"}</function>',
    '<function name="lookup" parameters={"query":"x"}></function>',
    '<function=lookup({"query":"x"})</function>',
    '<function>lookup{"query":"x"}</function>',
    '{"name":"lookup","parameters":{"query":"x"}}',
    '```json\n{"function":"lookup","arguments":{"query":"x"}}\n```',
])
def test_explicit_serialization_only(text):
    assert explicit_call(text, {"lookup"}, validate) == {"function": "lookup", "args": {"query": "x"}}


@pytest.mark.parametrize("text", [
    'I will use lookup now.', 'The answer is 42.',
    'Example: <function=lookup>{"query":"x"}</function>',
    '<function=lookup>{"query":"x"}</function><function=lookup>{"query":"y"}</function>',
    '{"name":"lookup"}', '{"name":"lookup","parameters":{"query":1}}',
    '{"name":"unknown","parameters":{"query":"x"}}',
    '<function=lookup>__import__("os")</function>',
])
def test_no_prose_inference_invention_or_multiple_calls(text):
    assert explicit_call(text, {"lookup"}, validate) is None


def test_nested_json_keeps_values_not_string_coercion():
    text = '<function=name="lookup" parameters={"query":"x}y","items":[{"id":1}]}</function>'
    result = explicit_call(text, {"lookup"}, lambda n, a: True)
    assert result["args"] == {"query": "x}y", "items": [{"id": 1}]}


def test_runtime_validation_rejects_unknown_missing_or_coerced_args():
    from pydantic import BaseModel
    from types import SimpleNamespace
    class Args(BaseModel):
        count: int
    runtime = SimpleNamespace(functions={"lookup": SimpleNamespace(parameters=Args)})
    assert valid_runtime_arguments(runtime, "lookup", {"count": 1})
    assert not valid_runtime_arguments(runtime, "lookup", {"count": "1"})
    assert not valid_runtime_arguments(runtime, "lookup", {})
    assert not valid_runtime_arguments(runtime, "lookup", {"count": 1, "extra": 2})
