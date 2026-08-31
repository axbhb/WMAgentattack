# v38–v40: clean protocol recovery

## v38 conclusion: GO_PARSER_VIABILITY_V38

The single remote, read-only replay of the 60 archived v37 clean traces recovered two additional argument-validated calls in two episodes. It changed zero strict parsed calls, reproduced every archived strict parse, and left all five successful tool-free episodes unchanged. No model generation, tool execution, or new trajectory occurred. All frozen viability checks passed; 42 remote tests passed. Candidate implementation: `86cff9e`.

Counterevidence: the historical permissive parser recognized eight additional unvalidated calls. The conservative, schema-validated candidate recovered only two; syntax recovery alone may have little effect on task success. Recognition is not execution and does not establish utility. The grammar is frozen and will not be broadened in response to this audit.

Archive: `/share/guozhix/wmagentattack/0831/protocol_recovery_v38/formal_v1`, including the original preregistration, submission receipt and gate. No new content checksums are used.

## v39: preregistered, awaiting submission

Run 20 retained tasks × 3 new seeds × 3 arms = 180 clean episodes on one GPU with one Llama-3.1-70B-Instruct 4bit model load. Compare unchanged strict parsing, validated syntax recovery, and syntax recovery plus the unchanged one-shot first-turn intention correction. Preserve all original prompts and model settings. Share actual task/seed RNG allocation; verify first prompt tokens and first completions exactly match across arms.

The correction arm is a policy intervention with an extra generation, not a same-query-budget parser improvement. Archive hidden correction completions and report total generation calls and output tokens. Protect legitimate tool-free successes explicitly.

Require at least +10 percentage points clean utility, six improved tasks, task-level one-sided exact sign-flip p≤0.025, no suite degradation exceeding 10 points, no regressions on successful tool-free controls, at least 12 stable tasks and three suites with at least two stable tasks. Prefer syntax over syntax_retry if both pass. Otherwise stop: no post-result tuning, threshold relaxation, or reruns.

## v40: conditional, not yet authorized

Only after v39 GO, compare strict against the frozen selected arm on 20 tasks × 3 independent seeds × 2 arms = 120 clean episodes. Require the same effect and eligibility gates, p≤0.05, and at least ten stable tasks shared with the selected v39 arm. These are new seeds on familiar tasks, not unseen-task validation.

At most 300 new clean episodes across this cycle. No attack data, real external endpoints, model fitting, or world-model training is authorized. After the fixed cycle's conclusion, stop the monitor and summarize both positive evidence and counterevidence.
