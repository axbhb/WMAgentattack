"""Candidate adapter; the historical strict adapter stays byte-for-byte intact."""
from agentdojo.functions_runtime import FunctionCall
from agentdojo.types import ChatAssistantMessage, text_content_block_from_string
from .qwen_agentdojo import TransformersQwenLLM
from .protocol_recovery import explicit_call, valid_runtime_arguments


class GenerationRecorder:
    def __init__(self, model, tokenizer):
        self.model, self.tokenizer = model, tokenizer
        self.reset()

    def reset(self):
        self.events = []
        self.first_prompt_ids = None

    def generate(self, **kwargs):
        ids = kwargs["input_ids"]
        if self.first_prompt_ids is None:
            self.first_prompt_ids = ids[0].detach().cpu().tolist()
        output = self.model.generate(**kwargs)
        generated = output[0, ids.shape[1]:]
        self.events.append({"completion": self.tokenizer.decode(generated, skip_special_tokens=True).strip(),
                            "input_tokens": int(ids.shape[1]), "output_tokens": int(generated.shape[0])})
        return output


class ConservativeProtocolLLM(TransformersQwenLLM):
    def _parse_repaired_completion(self, completion, allowed_functions=None):
        strict = self._parse_completion(completion, allowed_functions)
        if strict["tool_calls"]:
            return strict
        recovered = explicit_call(completion, set(allowed_functions or []),
                                   lambda n, a: valid_runtime_arguments(self._active_runtime, n, a))
        if recovered is None:
            return strict
        self.recovered_calls += 1
        return ChatAssistantMessage(role="assistant", content=[text_content_block_from_string(completion.strip())],
                                    tool_calls=[FunctionCall(**recovered)])

    def query(self, query, runtime, env=None, messages=(), extra_args=None):
        self._active_runtime = runtime
        kwargs = dict(messages=messages, extra_args=extra_args or {})
        if env is not None:
            kwargs["env"] = env
        return super().query(query, runtime, **kwargs)
