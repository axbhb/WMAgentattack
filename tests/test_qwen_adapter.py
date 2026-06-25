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
