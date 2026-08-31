# v38--v40: isolate serialization repair from bounded first-turn correction

Frozen after v37, before any parser replay or new outcomes. Three bounded
stages: read-only replay of 60 old traces; fresh 180-episode three-arm clean
comparison; only on GO, a 120-episode independent-seed confirmation. Maximum
300 new clean episodes, sequential single-GPU jobs, no attacks or fitting.

The prior findings are known: v37 has 25 zero-call failures, 13 containing
function-tag text, but also five tool-free successes. Therefore retain direct
answers, do not infer tools or arguments from prose, and preserve strict valid
calls. Recovery accepts one explicit full-message name/JSON serialization;
it validates new argument objects without converting their values. It does
not execute any tool during replay. The historical adapter file is unchanged.

All arms retain the same initial prompt and model. Syntax-only isolates parser
behavior. Syntax+retry adds one existing first-turn intent-triggered correction;
this is a policy intervention, not merely parsing, and has a reported extra
generation cost. It cannot be described as equal-query-budget improvement.
Record every completion and the first prompt token array, directly compare
paired arrays/text without content digests, and reset actual RNG per task/seed.
Use 20 task-level effects for inference, with a two-comparison correction in
v39 and an independent-seed test in v40. These are familiar tasks, not unseen
task confirmation. The full thresholds and fixed selection rule are in JSON.

Literature mapping: [BFCL's official evaluation description](https://gorilla.cs.berkeley.edu/blogs/8_berkeley_function_calling_leaderboard.html)
distinguishes syntax/parameter checks from execution-based correctness. This
motivates separating v38 parser viability from v39/v40 end-to-end clean utility;
an extra parse is not a successful task. [Hugging Face's Llama-3.1 integration](https://github.com/huggingface/blog/blob/main/llama31.md)
describes custom JSON function calling, supporting the possibility of an
adapter-format mismatch. It does not prove that native tool calling will
improve this project's tasks, so we do not change the prompt/template here.

Counterevidence to seek: harm to previously tool-free successes; executable
but semantically wrong arguments; gains confined to one suite; retry overhead;
failure to repeat gains on new seeds. Stop at any failed stage. No downstream
attack pilot or large world-model training is part of this fixed cycle.
