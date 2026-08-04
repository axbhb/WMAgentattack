from types import SimpleNamespace

import torch

from wmagentattack.qwen_agentdojo import TransformersQwenLLM


def test_large_yaml_tool_output_keeps_metadata_and_compacts_content():
    adapter = TransformersQwenLLM.__new__(TransformersQwenLLM)
    adapter.max_tool_output_chars = 1_000
    content = (
        "- filename: example.txt\n"
        "  id_: '1'\n"
        "  size: 999\n"
        f"  content: {'x' * 3_000}\n"
    )

    compacted = adapter._compact_tool_output(content)
    assert len(compacted) < len(content)
    assert "filename: example.txt" in compacted
    assert "size: 999" in compacted
    assert "content compacted for local context" in compacted


def test_parser_accepts_empty_argument_function_call():
    message = TransformersQwenLLM._parse_completion(
        "<function=list_files></function>"
    )
    assert message["tool_calls"][0].function == "list_files"
    assert message["tool_calls"][0].args == {}


def test_parser_rejects_non_object_arguments():
    message = TransformersQwenLLM._parse_completion(
        "<function=list_files>[]</function>"
    )
    assert message["tool_calls"] == []


def test_native_parser_accepts_qwen_tool_call():
    message = TransformersQwenLLM._parse_native_completion(
        'Checking now.\n<tool_call>\n{"name": "list_files", "arguments": {}}\n</tool_call>'
    )
    assert message["tool_calls"][0].function == "list_files"
    assert message["tool_calls"][0].args == {}
    assert "Checking now." in message["content"][0]["content"]


def test_native_parser_accepts_llama_bare_parameters_call():
    message = TransformersQwenLLM._parse_native_completion(
        '{"name": "get_flights", "parameters": {"origin": "SIN"}}',
        allowed_functions={"get_flights"},
    )
    assert message["tool_calls"][0].function == "get_flights"
    assert message["tool_calls"][0].args == {"origin": "SIN"}


def test_native_parser_accepts_llama_call_after_short_preamble():
    message = TransformersQwenLLM._parse_native_completion(
        'I will check.\n```json\n{"name": "list_files", "parameters": {}}\n```',
        allowed_functions={"list_files"},
    )
    assert message["tool_calls"][0].function == "list_files"
    assert message["tool_calls"][0].args == {}
    assert "I will check." in message["content"][0]["content"]


def test_native_parser_rejects_unknown_function():
    message = TransformersQwenLLM._parse_native_completion(
        '{"name": "not_a_real_tool", "parameters": {}}',
        allowed_functions={"list_files"},
    )
    assert message["tool_calls"] == []


def test_native_parser_keeps_only_first_valid_tool_call():
    message = TransformersQwenLLM._parse_native_completion(
        '{"name": "first", "parameters": {}}\n'
        '{"name": "second", "parameters": {}}',
        allowed_functions={"first", "second"},
    )
    assert len(message["tool_calls"]) == 1
    assert message["tool_calls"][0].function == "first"


def test_format_only_profile_adds_no_task_specific_hints():
    adapter = TransformersQwenLLM.__new__(TransformersQwenLLM)
    adapter.prompt_profile = "format_only"
    rules = adapter._function_tag_rules()
    assert "<function=name>" in rules
    assert "exactly one tool call" in rules
    assert "get_current_day" not in rules
    assert "search_files" not in rules


def test_constraint_checklist_is_generic_and_preserves_entity_binding():
    adapter = TransformersQwenLLM.__new__(TransformersQwenLLM)
    adapter.prompt_profile = "constraint_checklist"
    rules = adapter._function_tag_rules()
    assert "constraints satisfied by different candidates" in rules
    assert "final answer" in rules
    assert "exactly one call" in rules
    for task_specific_hint in (
        "get_current_day",
        "search_files",
        "calendar",
        "hotel",
        "restaurant",
    ):
        assert task_specific_hint not in rules


def test_repaired_parser_accepts_observed_inline_argument_tag():
    message = TransformersQwenLLM._parse_repaired_completion(
        '<function=get_all_restaurants_in_city({"city": "Paris"})</function>',
        allowed_functions={"get_all_restaurants_in_city"},
    )
    assert message["tool_calls"][0].function == "get_all_restaurants_in_city"
    assert message["tool_calls"][0].args == {"city": "Paris"}


def test_repaired_parser_accepts_observed_name_parameters_tag():
    message = TransformersQwenLLM._parse_repaired_completion(
        '<function name="get_all_hotels_in_city" '
        'parameters={"city": "Paris"}></function>',
        allowed_functions={"get_all_hotels_in_city"},
    )
    assert message["tool_calls"][0].function == "get_all_hotels_in_city"
    assert message["tool_calls"][0].args == {"city": "Paris"}


def test_repaired_parser_accepts_split_function_and_arguments():
    message = TransformersQwenLLM._parse_repaired_completion(
        '<function>get_all_restaurants_in_city</function> {"city": "Paris"}',
        allowed_functions={"get_all_restaurants_in_city"},
    )
    assert message["tool_calls"][0].args == {"city": "Paris"}


def test_repaired_parser_accepts_bare_function_json_synonym():
    message = TransformersQwenLLM._parse_repaired_completion(
        '{"function": "get_current_day", "parameters": {}}',
        allowed_functions={"get_current_day"},
    )
    assert message["tool_calls"][0].function == "get_current_day"


def test_repaired_parser_rejects_unknown_function():
    message = TransformersQwenLLM._parse_repaired_completion(
        '<function>unknown_tool</function> {"city": "Paris"}',
        allowed_functions={"get_all_restaurants_in_city"},
    )
    assert message["tool_calls"] == []


def test_repaired_parser_accepts_observed_name_attribute_variant():
    message = TransformersQwenLLM._parse_repaired_completion(
        '<function=name="get_all_hotels_in_city" '
        'parameters={"city": "Paris"}></function>',
        allowed_functions={"get_all_hotels_in_city"},
    )
    assert message["tool_calls"][0].function == "get_all_hotels_in_city"
    assert message["tool_calls"][0].args == {"city": "Paris"}


def test_repaired_parser_accepts_function_body_call_variant():
    message = TransformersQwenLLM._parse_repaired_completion(
        '<function>get_all_restaurants_in_city({"city": "Paris"})</function>',
        allowed_functions={"get_all_restaurants_in_city"},
    )
    assert message["tool_calls"][0].function == "get_all_restaurants_in_city"
    assert message["tool_calls"][0].args == {"city": "Paris"}


def test_retry_intent_requires_both_intention_and_tool_action():
    assert TransformersQwenLLM._should_retry_tool_intent(
        "I will fetch the available rental companies first."
    )
    assert not TransformersQwenLLM._should_retry_tool_intent(
        "The available rental company is Blue Car Rentals."
    )
    assert not TransformersQwenLLM._should_retry_tool_intent(
        "I will provide the final answer now."
    )


def test_query_retries_once_on_first_turn_explicit_tool_intent():
    class FakeTokenizer:
        eos_token_id = 0

        def apply_chat_template(self, messages, **kwargs):
            return {"input_ids": torch.tensor([[1, 2]])}

        def decode(self, token_ids, **kwargs):
            return (
                "I will fetch the available files first."
                if int(token_ids[0]) == 10
                else "<function=list_files>{}</function>"
            )

    class FakeModel:
        def __init__(self):
            self.calls = 0

        def generate(self, **kwargs):
            self.calls += 1
            token = 10 if self.calls == 1 else 11
            return torch.tensor([[1, 2, token]])

    adapter = TransformersQwenLLM.__new__(TransformersQwenLLM)
    adapter.protocol = "function_tags_repair_retry"
    adapter.tokenizer = FakeTokenizer()
    adapter.model = FakeModel()
    adapter.device = torch.device("cpu")
    adapter.max_input_tokens = 8192
    adapter.max_new_tokens = 32
    adapter.do_sample = False
    adapter.temperature = 0.7
    adapter.top_p = 0.95
    adapter._to_qwen_messages = lambda messages, runtime: [
        {"role": "system", "content": "tools"},
        {"role": "user", "content": "list files"},
    ]
    runtime = SimpleNamespace(functions={"list_files": object()})

    result = adapter.query(
        "list files",
        runtime,
        messages=(
            {"role": "system", "content": []},
            {"role": "user", "content": []},
        ),
    )

    assert adapter.model.calls == 2
    assert result[3][-1]["tool_calls"][0].function == "list_files"
