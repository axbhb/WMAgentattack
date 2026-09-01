# v41 onward: post-call reliability research

## Fixed scope and budget

Continue from v39 scientific NO_GO without modifying its parser, labels, gate, or withheld v40 seeds. V41 is a single descriptive audit of all 180 existing clean traces (including successes); zero tool/model execution. The next protocol, frozen only after this diagnostic, may use at most 240 fresh screening episodes and 120 conditional confirmation episodes. All real data processing and inference run remotely. No attacks, world-model fitting, or real endpoints. No content checksums.

## Primary literature mapped to concrete repository decisions

- [Reflexion](https://arxiv.org/abs/2303.11366): environment feedback and reflective memory can support agent improvement. Here, inspect actual tool error records before designing a feedback mechanism. Do not give the online agent final evaluator labels or claim replication of Reflexion's reported benchmark performance.
- [Large Language Models Cannot Self-Correct Reasoning Yet](https://arxiv.org/abs/2310.01798): intrinsic correction without external feedback can fail or degrade performance in the examined reasoning settings. Repository implication: include an extra-generation generic correction control; preserve successful trajectories and report additional generation/token cost. This is counterevidence to assuming every retry helps, not proof that all agent correction is impossible.
- [Key Condition Verification / ProCo](https://arxiv.org/abs/2405.14092): structured verification is a counterexample to blanket claims of ineffective self-correction on its studied tasks. Repository implication: separate an observable condition check from vague self-reflection. Its benefits do not establish efficacy on AgentDojo; use new evaluation seeds and task-level paired tests.

## v41 observable taxonomy

`postcall_audit.indicators` reads only messages and generation diagnostics, not utility/security. Outcomes are attached separately for reporting. Record explicit tool errors, unparsed terminal function markers after tool results, post-tool intention wording, consecutive identical calls, and input/output token-cap hits. These are overlapping indicators, not proven root causes. Preserve an unknown category and successful counterexamples.

Frozen protocol: `configs/0901_postcall_audit_v41_protocol.json`. Audit archive: `/share/guozhix/wmagentattack/0901/postcall_audit_v41/formal_v1`.

### V41 result: DESCRIPTIVE_AUDIT_COMPLETE_V41

All 180 traces were audited once remotely; 26 focused remote tests passed. Audit implementation `4172259`.

Among the 17 syntax_retry failures despite parsed calls, only two episodes (two tasks) had explicit tool errors. Four episodes had an unparsed terminal function marker after tool results; five had post-tool intention wording (these overlap). There were no input-token-cap hits or consecutive identical calls in any arm. One failing episode hit the output-token cap, which alone does not establish the cause. The remaining failures include unsupported stopping or wrong conclusions, not just serialization.

Counterevidence: among 38 successful syntax_retry episodes, two had tool errors, two had terminal function markers, and two had terminal intention wording. A local indicator cannot be treated as an oracle for task failure. In particular, actual tool execution can succeed while the final natural-language reply still contains misleading action intentions. No labels or parser grammar were changed.

### V42/V43 selected mechanism and preregistration

Select a single optional post-tool termination correction. After a tool result, trigger only when the next assistant reply has no parsed call and contains a function marker or the unchanged intention heuristic. Neither task ID nor final evaluation is available to the trigger. Preserve the existing first-turn correction as an experimental anchor, not an already accepted baseline replacement.

V42 compares four arms on all 20 tasks and seeds 701/709/719 (240 clean episodes): historical strict reference, anchor, anchor plus generic review, and anchor plus explicit execution-status/serialization feedback. The last two arms each have at most one additional post-tool correction, with identical max token budget. Permit a final answer instead of forcing a tool, and explicitly prohibit repeating completed side effects. First prompt/completion and the pre-correction generation prefix must agree across paired arms. If no trigger occurs, the entire trajectory and outcome must be unchanged versus anchor.

Each candidate must improve anchor by at least 10 percentage points on at least six tasks, with task-level exact one-sided p≤0.025, zero successful-anchor regressions, no suite drop exceeding 10 points, twelve stable tasks, and three suites with two stable tasks. Prefer the generic prompt if both pass. The grounded-vs-generic comparison is diagnostic; do not claim feedback-specific superiority solely from the overall improvement. Fixed protocol: `configs/0901_postcall_recovery_v42_v43_protocol.json`.

Only a passing V42 selection may enter V43 (120 episodes, new seeds 727/733/739, anchor versus selected candidate; p≤0.05 and at least ten stable-task overlap in addition to the effect/safety gates). No use of the withheld old V40 seeds. Total fresh budget is 360 clean episodes, two sequential one-GPU jobs at most. Stop after NO_GO/INVALID or confirmation conclusion; no post-result retry, threshold relaxation, attack generation, or model fitting.

V42 was submitted exactly once as **Slurm 7567** at 2026-08-31 16:17:32 UTC (2026-09-01 00:17:32 China time), implementation `4958f9b`; first snapshot PENDING. **74 remote tests passed** on experiment source `4265e82` (unchanged at submission), with 43 local synthetic tests passed. No V43 submission before a passing V42 selection. The exclusive remote submitter `scripts/293_submit_postcall_stage_once.py` refuses a duplicate/uncertain archive and records submission receipts; it does not rerun an uncertain job. Archive: `/share/guozhix/wmagentattack/0901/postcall_recovery_v42/formal_v1`.

## V42 result: NO_GO_POSTCALL_RECOVERY

Slurm 7567 completed all 240 clean episodes at 2026-08-31 22:23:38 UTC. The archive contains 240 result rows, 240 raw traces, and 240 diagnostics, with zero runtime failures and zero blocked network attempts. CUDA preflight passed on physical GPU 0. The final log contains no Traceback, OOM, or CUDA/runtime error. The stored gate was recomputed independently from the archived rows and artifacts, and the exact task-level sign-flip probabilities were independently enumerated; both checks matched.

The strict reference obtained 32/60 successes (53.33%) and 11 stable tasks. The experimental anchor obtained 38/60 (63.33%) and 13 stable tasks. Generic post-call review obtained 40/60 (66.67%), 14 stable tasks, and 11 post-call corrections. Grounded execution-status feedback obtained 39/60 (65.00%), 13 stable tasks, and 11 corrections.

Against the anchor, generic gained only 2/60 episodes (+3.33 percentage points), on two tasks, with exact task-level p=0.25. Grounded gained 1/60 (+1.67 points), on one task, with p=0.5. Both preserved every successful anchor episode, met stability and suite non-inferiority clauses, and had no negative paired tasks, but both failed the frozen effect-size, improved-task-count, and paired-significance clauses. Generic used 269 generations and 19,098 output tokens versus the anchor's 249 generations and 16,974 output tokens; grounded used 264 generations and 17,856 output tokens. Thus the small local gains came with extra inference cost and do not establish broad clean-reliability improvement.

Counterevidence is especially clear by domain: Travel remained 1/15 successful (6.67%) in anchor, generic, and grounded. Both gains were confined to Workspace: generic improved `workspace|user_task_16` and `workspace|user_task_18`; grounded improved only `workspace|user_task_16`. The explicit execution-status prompt was not better than generic review, but their direct contrast is diagnostic only and cannot establish feedback-specific inferiority.

The scientific interpretation is that bounded post-tool correction is safe under these seeds and can rescue isolated Workspace episodes, but post-tool termination is not the dominant remaining bottleneck. Failures after parsed calls, task reasoning, evidence interpretation, and Travel solvability remain unresolved. No candidate is promoted, V43 is **NOT_AUTHORIZED**, and this fixed cycle ends without attacks, world-model fitting, threshold changes, or post-result reruns.
