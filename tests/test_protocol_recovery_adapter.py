from types import SimpleNamespace
from pydantic import BaseModel
import torch

from wmagentattack.protocol_recovery_adapter import ConservativeProtocolLLM, GenerationRecorder


class Args(BaseModel):
    query: str


def adapter(completions, mode="function_tags_repair_retry"):
    class Tokenizer:
        eos_token_id = 0
        def apply_chat_template(self, messages, **kwargs):
            return {"input_ids": torch.tensor([[1, 2]])}
        def decode(self, ids, **kwargs):
            return completions[int(ids[0]) - 10]
    class Model:
        calls = 0
        def generate(self, **kwargs):
            token = 10 + self.calls
            self.calls += 1
            return torch.tensor([[1, 2, token]])
    a = ConservativeProtocolLLM.__new__(ConservativeProtocolLLM)
    a.protocol, a.tokenizer = mode, Tokenizer()
    a.model = GenerationRecorder(Model(), a.tokenizer)
    a.recovered_calls = 0
    a.device, a.max_input_tokens, a.max_new_tokens = "cpu", 8192, 256
    a.do_sample, a.temperature, a.top_p = False, 0.7, 0.95
    a._to_qwen_messages = lambda messages, runtime: [{"role": "system", "content": "tools"}, {"role": "user", "content": "query"}]
    return a, SimpleNamespace(functions={"lookup": SimpleNamespace(parameters=Args)})


def test_syntax_recovers_valid_call_and_records_original_generation():
    a, runtime = adapter(['<function=name="lookup" parameters={"query":"x"}</function>'], "function_tags_repair")
    result = a.query("query", runtime, messages=[{"role": "user", "content": []}])
    assert result[3][-1]["tool_calls"][0].args == {"query": "x"}
    assert a.recovered_calls == 1 and len(a.model.events) == 1
    assert a.model.first_prompt_ids == [1, 2]


def test_retry_happens_at_most_once_even_if_correction_still_has_no_call():
    a, runtime = adapter(["I will search now.", "I will search now."])
    result = a.query("query", runtime, messages=[{"role": "user", "content": []}])
    assert not result[3][-1]["tool_calls"] and len(a.model.events) == 2


def test_no_retry_on_a_later_assistant_turn():
    a, runtime = adapter(["I will search now."])
    a.query("query", runtime, messages=[{"role": "user", "content": []}, {"role": "assistant", "content": [], "tool_calls": []}])
    assert len(a.model.events) == 1


def test_direct_answer_is_not_forced_to_call_a_tool():
    a, runtime = adapter(["The answer is 42."])
    result = a.query("query", runtime, messages=[{"role": "user", "content": []}])
    assert len(a.model.events) == 1 and not result[3][-1]["tool_calls"]


def test_new_recovery_rejects_wrong_types_without_inventing_values():
    a, runtime = adapter(['<function=name="lookup" parameters={"query":42}</function>'], "function_tags_repair")
    result = a.query("query", runtime, messages=[{"role": "user", "content": []}])
    assert not result[3][-1]["tool_calls"] and a.recovered_calls == 0
