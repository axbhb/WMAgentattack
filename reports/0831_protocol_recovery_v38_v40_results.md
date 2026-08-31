# v38–v40: clean protocol recovery

## v38 conclusion: GO_PARSER_VIABILITY_V38

The single remote, read-only replay of the 60 archived v37 clean traces recovered two additional argument-validated calls in two episodes. It changed zero strict parsed calls, reproduced every archived strict parse, and left all five successful tool-free episodes unchanged. No model generation, tool execution, or new trajectory occurred. All frozen viability checks passed; 42 remote tests passed. Candidate implementation: `86cff9e`.

Counterevidence: the historical permissive parser recognized eight additional unvalidated calls. The conservative, schema-validated candidate recovered only two; syntax recovery alone may have little effect on task success. Recognition is not execution and does not establish utility. The grammar is frozen and will not be broadened in response to this audit.

Archive: `/share/guozhix/wmagentattack/0831/protocol_recovery_v38/formal_v1`, including the original preregistration, submission receipt and gate. No new content checksums are used.

## v39: NO_GO_PROTOCOL_RECOVERY

Submitted exactly one Slurm job **7565** at 2026-08-31 13:03:47 UTC. The submission receipt records implementation commit `ad3cecb`, 180 episodes, one GPU and a six-hour cap. All **55 remote tests** passed before submission; 31 lightweight synthetic fixture tests also passed locally. The first queue snapshot was PENDING; that is not evidence of model loading or a completed result. No v40 job has been submitted. The original v38 archived protocol was copied to the v39 archive unchanged; execution metadata in the repository config does not revise its scientific contract.

The job subsequently started at 13:32:00 UTC on physical GPU 0 and wrote its completion sentinel at 15:07:41 UTC (95 minutes 41 seconds after start). On reconnection all 180 result rows, 180 raw traces, 180 diagnostic records and the completion sentinel were present; zero recorded runtime or network-attempt failures. The scheduler record had expired, so an exact Slurm exit code is unavailable. A local VPN outage did not interrupt the server job. All 60 task/seed triplets passed actual-seed, first-input and first-completion equality checks.

The remote post-result audit (`scripts/290_audit_protocol_recovery_v39_results.py`, audit commit `f3e6a24`) recomputed the full gate from raw/diagnostic artifacts and independently enumerated the exact sign-flip probabilities. Both matched the archived result. It also confirmed the frozen contract and experimental source were unchanged. **55 remote tests passed again at closeout.** Audit output: `/share/guozhix/wmagentattack/0831/protocol_recovery_v39/formal_v1/posthoc_closeout_audit.json`.

| Metric | Strict | Syntax | Syntax + one correction |
| --- | ---: | ---: | ---: |
| Clean successes / 60 | 32 | 34 | 38 |
| Clean success rate | 53.33% | 56.67% | 63.33% |
| Paired gain vs strict | — | +3.33 pp | +10.00 pp |
| Improved tasks / 20 | — | 2 | 4 |
| Exact task-level one-sided p | — | 0.25 | 0.0625 |
| Stable tasks (at least 2/3 seeds) | 10 | 11 | 12 |
| Failures with zero parsed calls | 21 | 19 | 5 |
| Failures despite parsed calls | 7 | 7 | 17 |
| Generation calls, including corrections | 152 | 154 | 229 |
| Generated output tokens | 9,589 | 9,680 | 13,490 |

Neither candidate passes the complete preregistered gate. Syntax fails minimum effect, task coverage, p-value and stable-task count. Syntax + correction reaches the effect/eligibility gates but improves only four tasks (requires six) with p=0.0625 (requires ≤0.025). No task has a negative aggregate success difference and there are zero regressions on successful tool-free controls. These encouraging observations do not justify replacing the frozen gate with a weaker one.

The correction candidate's six additional successes are concentrated in Banking `user_task_7` (+3 seeds), Travel `user_task_10` (+1), Workspace `user_task_16` (+1) and `user_task_22` (+1). Three seed improvements on one Banking task are one improved task, not three independent inferential units.

Per-suite successes (15 episodes each): Banking **12/12/15**, Slack **10/10/10**, Travel **1/1/2**, Workspace **9/11/11** for strict/syntax/correction. Travel has **zero stable tasks in every arm**. The twelve stable tasks in the correction arm therefore do not demonstrate broad Travel recovery.

Generation cost increases from 152 to 229 calls (**+50.66%**) and from 9,589 to 13,490 output tokens (**+40.68%**) with correction. Total generation differences include downstream tool-followup turns and must not be equated to correction count. This is not an equal-query-budget comparison.

The exact number of extra correction generations is **17**; the remaining difference comes from changed follow-up trajectories. Of the 21 strict zero-call failures, correction yields 6 successful tool-using episodes, 10 tool-using episodes that still fail, and 5 remaining zero-call failures. All 32 strict successes remain successes in both candidates. The three originally successful tool-free controls retain their utility under correction but now use tools: outcome preservation does **not** imply unchanged tool-free behavior or cost.

### Interpretation and counterevidence

There is a limited end-to-end clean-success signal, not merely improved parsing, but it is too concentrated for the frozen acceptance criterion. Most zero-call failures disappear under correction; failures after parsed calls rise from 7 to 17. This does not by itself identify whether the remaining cause is arguments, tool-result interpretation, multi-step decisions, or final completion. It does show that tool-call production alone is insufficient. Do not claim an improved world model, attack selector, ASR, or generalization to unseen tasks: this experiment trains no model and samples only familiar tasks under new seeds.

Retain the historical baseline for formal comparisons. Preserve the candidate as local mechanism evidence, not an accepted baseline replacement. A proposed next cycle should first diagnose the archived post-call failures without relabeling or tuning on them, distinguish invalid arguments/execution feedback from wrong action sequences or premature termination, and preregister a minimal feedback intervention with an equal-extra-generation control on new evaluation material. This is a recommendation, not an experiment launched by the present cycle.

Run 20 retained tasks × 3 new seeds × 3 arms = 180 clean episodes on one GPU with one Llama-3.1-70B-Instruct 4bit model load. Compare unchanged strict parsing, validated syntax recovery, and syntax recovery plus the unchanged one-shot first-turn intention correction. Preserve all original prompts and model settings. Share actual task/seed RNG allocation; verify first prompt tokens and first completions exactly match across arms.

The correction arm is a policy intervention with an extra generation, not a same-query-budget parser improvement. Archive hidden correction completions and report total generation calls and output tokens. Protect legitimate tool-free successes explicitly.

Require at least +10 percentage points clean utility, six improved tasks, task-level one-sided exact sign-flip p≤0.025, no suite degradation exceeding 10 points, no regressions on successful tool-free controls, at least 12 stable tasks and three suites with at least two stable tasks. Prefer syntax over syntax_retry if both pass. Otherwise stop: no post-result tuning, threshold relaxation, or reruns.

## v40: NOT_AUTHORIZED — zero episodes submitted

V39 produced no passing selection, so the planned 120-episode independent-seed confirmation is not executed. Preserve its frozen specification unchanged for audit; do not repurpose its seeds as an exploratory rerun.

Actual cycle use: 60 existing traces replayed without generation in v38, 180 new clean episodes in v39, zero v40 episodes, zero new attack episodes and zero model fits. No real external endpoints or large world-model training. The fixed cycle stops at v39 NO_GO, not at the end of an open-ended search for favorable results.

The cycle-specific automation `wmagentattack-v39-v40-clean` was deleted at closeout, after confirming the scientific NO_GO. No server job was canceled and no new job was submitted. Final local synthetic tests: 31 passed. Final report/config/ledger copies are archived under `formal_v1/final_report/20260831_v39_closeout_v1/` on the server; the original protocol, raw traces and gate remain intact.
