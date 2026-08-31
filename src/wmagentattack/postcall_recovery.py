"""One optional post-tool correction; original adapter and parser unchanged."""
import re
from agentdojo.types import text_content_block_from_string
from .postcall_audit import text_of
from .protocol_recovery_adapter import ConservativeProtocolLLM


class PostcallRecoveryLLM(ConservativeProtocolLLM):
    def reset_postcall(self, arm, prompts):
        self.postcall_arm = arm
        self.postcall_prompts = prompts
        self.postcall_probe = None
        self.postcall_corrections = 0

    def query(self, query, runtime, env=None, messages=(), extra_args=None):
        result = super().query(query, runtime, env, messages, extra_args)
        response = result[3][-1]
        text = text_of(response)
        trigger = (any(m['role'] == 'tool' for m in messages)
                   and not response.get('tool_calls')
                   and (bool(re.search(r'<function\b', text, re.I)) or self._should_retry_tool_intent(text)))
        if trigger and self.postcall_probe is None:
            self.postcall_probe = {'prefix_event_count': len(self.model.events), 'completion': text}
        if not trigger or self.postcall_arm not in ('generic', 'grounded') or self.postcall_corrections:
            return result
        self.postcall_corrections = 1
        instruction = self.postcall_prompts[self.postcall_arm]
        self.postcall_probe['instruction'] = instruction
        correction_history = [*result[3], {'role': 'user', 'content': [text_content_block_from_string(instruction)]}]
        corrected = super().query(query, runtime, env, correction_history, extra_args)
        # Retain the diagnostic prompt/completion in recorder/probe, not as fabricated user task history.
        return result[0], result[1], result[2], [*messages, corrected[3][-1]], result[4]
