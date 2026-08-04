# AgentDojo-v2 method-improvement round and fresh-task audit

Date: 2026-07-16

Decision: **NO-GO for Travel attack-data construction, Dreamer scaling, or replacement of the frozen E5 control.** `function_tags_repair_retry` remains a useful execution repair, but its two development-retained Travel tasks failed the frozen unseen-seed confirmation and the durable task intersection is empty. The value-model probes also remain negative: the strongest point estimate is the secondary top-anchor ranker at Top-1 joint `0.81`, but its delta over E5 is only `+0.03` and its interval crosses zero.

## Fixed research budget

This round used the following bounded sequence and retained negative results:

1. clean-conditioned preservation targets: direct attacked probability, probability uplift, and logit residual over five task-held-out folds;
2. a held-out diagnostic of whether predictive uncertainty was actually rank-calibrated;
3. two top-heavy ranking objectives plus the existing largest-gap E5 control, with no hyperparameter search;
4. a label-blind E5 truncation audit followed by one frozen field-aware representation over five folds;
5. a clean-only screen over every AgentDojo v1.2.2 task excluded from the existing 20-task benchmark, followed by a complete three-seed clean census;
6. no Dreamer training run unless a cheap probe cleared the fixed improvement gate.

The value-model probes did not clear their gate, so the budget correctly stopped before another expensive Dreamer run. A subsequent bounded clean execution-protocol loop did clear its separate gate; that result repairs data eligibility but does not reverse the Dreamer/value-model NO-GO.

## Method comparison

| Method | Claim status | Top-1 ASR | Top-1 BUP | Top-1 joint | Result |
|---|---|---:|---:|---:|---|
| Frozen Dreamer OOF | historical formal baseline | 0.23 | 0.49 | 0.72 | reference |
| E5 + structured largest-gap | frozen semantic control | 0.32 | 0.46 | 0.78 | reference |
| Logit residual + uncertainty penalty | preregistered residual primary | 0.20 | 0.40 | 0.60 | NO-GO |
| Direct attacked probability | residual secondary | 0.23 | 0.43 | 0.66 | NO-GO |
| Probability uplift | residual secondary | 0.16 | 0.46 | 0.62 | NO-GO |
| Logit residual, penalty removed | post-hoc diagnostic only | 0.31 | 0.42 | 0.73 | below E5 |
| Lambda-NDCG@3 | preregistered top-heavy primary | 0.34 | 0.46 | 0.80 | NO-GO |
| Top-anchor | secondary signal | 0.34 | 0.47 | 0.81 | unconfirmed |
| Field-aware dual E5 view | frozen representation probe | 0.32 | 0.46 | 0.78 | no Top-1 change |

For top-anchor versus E5, the joint delta is `+0.03`, bootstrap 95% interval `[-0.03, 0.10]`, and exact one-sided sign-flip `p=0.28125`. It changed five of twenty task selections: three improved and two worsened. Lambda-NDCG@3 changed only one task. These results are compatible with a small ranking signal but not with a confirmed improvement.

## Counterevidence and failure analysis

### 1. Predictive uncertainty is not decision uncertainty

Fold-1 validation selected an uncertainty penalty of `1.0`, but the five-fold held-out joint score fell from `0.73` without the penalty to `0.60` with it. The predicted uncertainty/error correlation was only `0.194`. The penalty therefore suppresses useful candidates without reliably identifying ranking mistakes. Future uncertainty must be calibrated on task-level selection regret or posterior probability of being best, not on pointwise regression variance.

Residual and E5 selections also had almost no exploitable complementarity: a held-out oracle choosing between them reached only `0.79` versus E5 at `0.78`. This rejects a more complicated router on the current data.

### 2. Top-heavy loss is directionally useful but data-limited

Ranking literature supports weighting pairs by the metric impact near the top, but the formal Lambda-NDCG@3 variant changed only one of twenty decisions. The secondary top-anchor variant changed five, yet its confidence interval and exact test do not support replacement. The remaining obstacle is not merely objective mismatch: repeated five-seed outcomes often do not identify a unique best candidate. A posterior Monte Carlo audit found that the posterior-mean winner had mean probability-best only `0.353`; only `11/20` tasks agreed with the probability-best winner, and `8/20` had probability-best below `0.30`.

### 3. E5 truncation is real but not the current Top-1 bottleneck

All 2,000 attack first-step texts exceed E5's 512-token limit. The attack marker is truncated in `90.2%` of examples and the target marker in `100%`; Travel loses the untrusted marker in `40%`. The critical field view reduces the maximum to 457 tokens and retains every audited field. Nevertheless, the field-aware model selected exactly the same candidate on all twenty tasks and remained at joint `0.78`. The loader should still be repaired for correctness and future distribution shift, but this representation change alone does not justify larger training.

### 4. The 20-task OOF set is now a development benchmark

Although every individual run respected task-held-out folds, many sequential method choices have been informed by aggregate outcomes on the same twenty tasks. Those tasks should no longer support a new confirmatory claim. They remain useful for development and rejection, while a new task partition must be frozen before attack outcomes are generated.

## Fresh-task clean screen

AgentDojo v1.2.2 contains 97 tasks across banking, slack, travel, and workspace. Excluding the existing twenty leaves 77 fresh tasks. The first Llama-3.1-70B clean seed completed all 77 without runtime failures but solved only 18:

| Domain | Fresh tasks | Seed-101 clean successes |
|---|---:|---:|
| banking | 11 | 6 |
| slack | 16 | 7 |
| travel | 15 | 1 |
| workspace | 35 | 4 |
| total | 77 | 18 |

This is direct evidence that observed preservation combines attack damage with victim/task solvability, and that the mixture differs sharply by domain. Conditioning the next dataset on one lucky seed would introduce selection bias, especially for Travel. Therefore seeds 103 and 107 were expanded to all 77 fresh tasks before either outcome was read. No attack episode was generated during this screen.

The complete census retained only 14 tasks:

| Domain | Fresh tasks | 0/3 | 1/3 | 2/3 | 3/3 | Retained >=2/3 |
|---|---:|---:|---:|---:|---:|---:|
| banking | 11 | 2 | 6 | 2 | 1 | 3 |
| slack | 16 | 4 | 5 | 4 | 3 | 7 |
| travel | 15 | 13 | 2 | 0 | 0 | 0 |
| workspace | 35 | 24 | 7 | 2 | 2 | 4 |
| total | 77 | 43 | 20 | 8 | 6 | 14 |

The retained IDs are banking `10, 11, 14`; slack `2, 4, 6, 12, 13, 17, 18`; and workspace `8, 12, 25, 31`. Travel has no task meeting the fixed threshold. A four-domain confirmatory attack set therefore cannot be constructed under the current victim protocol without silently changing the eligibility rule.

## Agent execution failure audit and native-tool repair

The 231 trace audit found 177 utility failures, of which 116 never reached a tool. This failure mode is concentrated in the two weakest domains:

| Domain | Utility failures | Failures with zero tool calls |
|---|---:|---:|
| banking | 20 | 6 |
| slack | 26 | 5 |
| travel | 43 | 36 |
| workspace | 88 | 69 |

For a representative Travel failure, the model explicitly wrote that it would fetch rental companies but emitted no custom function tag. Inspection of the local Llama-3.1 tokenizer confirmed a protocol mismatch: its native tool template requests bare `{"name": ..., "parameters": ...}` JSON, while the adapter's native parser recognized only Qwen-style `<tool_call>` blocks. The adapter now supports both forms, validates names against the active runtime, and keeps one call per assistant message as required by the Llama template.

The first native smoke exposed the multiple-call template constraint and failed closed; the retained failure artifact is `smoke/result_job4657_multiple_call_failure.json`. After the one-call repair, the same task completed with eight valid tool calls and no runtime error, although utility remained false.

The fixed paired clean-only evaluation completed all 45 episodes without execution errors. Native removed every zero-tool failure (`36 -> 0`) but improved utility only from `2/45` to `3/45`: candidate wins `3`, losses `2`, ties `40`, paired delta `+0.0222`, and exact two-sided sign-test `p=1.0`. It recovered one stable task (`travel::user_task_13`, 3/3), while losing the isolated base successes on tasks 7 and 15. Native therefore fails the minimum two-task and statistical gates. This is strong counterevidence that fixing tool serialization alone fixes task reasoning.

The next bounded experiment kept the original function-tags representation and added only an output-format instruction: one function tag per turn, no prose around the tag, and no merely verbalized tool intent. It deliberately excluded the full robust profile's date, calendar, and file-strategy hints. Across the same 45 pairs, zero-tool failures fell from `36` to `14`, but utility fell from `2/45` to `1/45`; candidate wins were `1`, losses `2`, ties `42`, paired delta `-0.0222`, and exact `p=1.0`. No task was retained. Global format prompting therefore fails: it improves syntax but perturbs useful reasoning.

All 36 base zero-tool failures occurred on the first assistant turn. At least 24 explicitly described an intended tool action, and many contained an unambiguous but malformed call: bare Llama JSON, `<function>name</function>{...}`, a `name=`/`parameters=` tag, or arguments placed inside the function tag. The next lower-impact intervention is a tolerant, runtime-validated parser that accepts only these unambiguous forms while leaving the base prompt and sampled completion unchanged. A later model retry is justified only for residual prose-only intent failures.

The preregistered offline precheck found that `20/36` zero-tool failures contain a deterministically repairable name-plus-JSON call, exceeding the fixed minimum of eight. The live parser additionally validates the name against the active AgentDojo runtime, keeps only the first valid call, and never infers a tool from prose. Jobs `4673/4674` completed all 45 pairs without errors. Utility improved from `2/45` to `4/45`, with wins `2`, losses `0`, ties `43`, paired delta `+0.0444`, and exact two-sided sign-test `p=0.5`. Zero-tool failures fell from `36` to `20`, but only one task became stable (`travel::user_task_18`, at least 2/3). This is directional evidence for tolerant parsing, not a clean-eligibility result: it misses the fixed `+0.05`, two-task, and statistical gates.

The final bounded execution-layer candidate adds two residual unambiguous syntaxes and permits one serialization-only regeneration when the first response still has no valid call and contains both explicit tool intent and an action verb. Residual failures were inspected to define this rule, so its eligibility audit is transparently a compute gate rather than a blind confirmatory test. The audit found `5` parser-v2 syntax cases and `14` retry-eligible cases among the `20` parser-v1 zero-tool failures, passing the frozen minimum of eight. Jobs `4680/4681` then completed all 45 clean pairs without Traceback, OOM, CUDA, or result-completeness errors.

Against archived strict function tags, clean utility rose from `2/45` to `8/45`: wins `6`, losses `0`, ties `39`, paired delta `+0.1333`, and exact two-sided sign-test `p=0.03125`. Zero-tool failures fell from `36` to `1`, and two tasks reached the fixed 2/3 retention threshold: `travel::user_task_7` and `travel::user_task_18`. All prespecified development gates are met. Against parser-v1, utility rose from `4/45` to `8/45`, with wins `4`, losses `0`, delta `+0.0889`, but exact `p=0.125`. Mechanism inspection maps two incremental wins to the new deterministic syntax forms (`seed101/task13`, `seed103/task9`) and two to the one-shot retry (`seed101/task7`, `seed107/task16`). Episode RNG is reset from `run_seed` and row ID before every task, so the historical 4-chunk strict archive and 2-chunk candidates remain seed-aligned.

The positive result has important counterevidence. It is sequential development evidence, not an independent confirmation: the same residual failures shaped the candidate. Overall utility is still only `8/45`; nine tasks remain 0/3, four are 1/3, two are 2/3, and none is 3/3. Moreover, `36/37` remaining utility failures now execute at least one tool. The intervention has largely removed serialization as the dominant failure mode, exposing planning and task reasoning as the next bottleneck. The protocol is accepted and frozen, but no additional prompt/parser experiment is submitted in this loop.

## Independent unseen-seed confirmation

The frozen protocol was rerun without code, prompt, model, or hyperparameter changes on unseen run seeds `109, 113, 127` (jobs `4687/4688`). All 45 episodes and six chunks completed without runtime failures, Traceback, OOM, or CUDA errors. The confirmation panel produced only `6/45` clean successes, distributed `4, 1, 1` across the three seeds. Ten tasks were 0/3, four were 1/3, one was 2/3, and none was 3/3.

The only confirmation-retained task was `travel::user_task_1` at 2/3, but it was 0/3 in development. Conversely, the development-retained tasks did not replicate: `user_task_18` fell from 2/3 to 0/3 and `user_task_7` fell from 2/3 to 1/3. The durable development-confirmation intersection is therefore empty, failing both prespecified task gates. Even pooling all six seeds as a non-gating diagnostic yields no task at 4/6; the best is task 7 at 3/6. This rules out attack-data construction from the current Travel panel.

Execution is no longer the dominant problem. Zero-tool failures remained low at `5/39` confirmation failures, while `34/39` failures executed one or more tools. The bottleneck has shifted to stochastic planning, multi-step state tracking, and exact task completion. More parser or format tuning on these tasks would reuse the same evidence and is not justified.

## Next stage after failed clean confirmation

1. Do not generate Travel attack data and do not launch Dreamer training from this panel. Preserve the empty durable intersection as the frozen result.
2. Keep parser-v2/retry as an engineering execution repair, but stop prompt/parser tuning on these 15 development tasks and six observed seeds.
3. Expand the clean-solvability boundary before attacks: evaluate a stronger agent scaffold or a separately frozen open-weight victim/model pool, and/or add unseen AgentDojo-compatible tasks. The intervention must target planning competence rather than serialization.
4. Freeze the new victim/task protocol before outcomes, repeat development and unseen-seed confirmation, and require at least two tasks in the durable intersection. Do not weaken the 2/3 plus cross-panel rule.
5. Only after that gate passes should attack trajectories be generated. Keep the E5 largest-gap control and top-anchor as the two predeclared selectors; large Dreamer training remains gated on fresh paired attack outcomes.

## Literature mapping

- Uplift consistency motivates modeling joint outcomes rather than optimizing a proxy, while top-k uplift counterevidence warns that objectives can fail to generalize to the actual selection budget: <https://arxiv.org/abs/2011.00041>, <https://arxiv.org/abs/2002.05897>.
- Counterfactual regression motivates separating task-level clean solvability from attacked outcomes, without implying that this benchmark establishes a causal effect: <https://arxiv.org/abs/1606.03976>.
- Beta-binomial modeling supports retaining repeated-trial uncertainty rather than collapsing five runs to a hard label: <https://arxiv.org/abs/1003.1325>.
- LambdaRank/LambdaLoss motivate top-weighted pair losses, while listwise work covers ties and top-heavy metrics: <https://www.microsoft.com/en-us/research/publication/on-the-optimality-of-lambdarank/>, <https://research.google/pubs/the-lambdaloss-framework-for-ranking-metric-optimization/>, <https://arxiv.org/abs/2001.01828>.
- Ranking calibration work supports distinguishing score uncertainty from calibrated decision confidence: <https://arxiv.org/abs/2101.04356>.

## Reproducibility map

- Residual protocol and archive: `configs/0716_residual_preservation_protocol.json`; `/share/guozhix/wmagentattack/0716/residual_preservation/residual_preservation_20260716_v2_numeric_tie`.
- Top-heavy protocol and archive: `configs/0716_top_heavy_ranking_protocol.json`; `/share/guozhix/wmagentattack/0716/top_heavy_ranking/top_heavy_ranking_20260716_v1`.
- Field-aware protocol and archive: `configs/0716_field_aware_e5_protocol.json`; `/share/guozhix/wmagentattack/0716/field_aware_e5/field_aware_e5_20260716_v1`.
- Fresh-screen protocol and archive: `configs/0716_fresh_task_screen_protocol.json`; `/share/guozhix/wmagentattack/0716/fresh_task_screen/fresh_task_screen_20260716_v1`.
- Native-tool repair protocol and archive: `configs/0716_native_tool_protocol_repair.json`; `/share/guozhix/wmagentattack/0716/native_tool_protocol/native_tool_protocol_20260716_v1`.
- Format-only function-tags protocol: `configs/0716_function_tag_format_repair.json`; planned archive `/share/guozhix/wmagentattack/0716/function_tag_format/function_tag_format_20260716_v1`.
- Tolerant function-tag parser protocol and archive: `configs/0716_function_tag_parser_repair.json`; `/share/guozhix/wmagentattack/0716/function_tag_parser/function_tag_parser_20260716_v1`.
- Final parser-v2/retry protocol, compact results, and archive: `configs/0716_function_tag_parser_retry.json`; `reports/0716_function_tag_parser_retry_results.json`; `/share/guozhix/wmagentattack/0716/function_tag_parser_retry/function_tag_parser_retry_20260716_v1`.
- Unseen-seed confirmation protocol, compact results, and archive: `configs/0716_function_tag_parser_retry_confirmation.json`; `reports/0716_function_tag_parser_retry_confirmation_results.json`; `/share/guozhix/wmagentattack/0716/function_tag_parser_retry_confirmation/function_tag_parser_retry_confirmation_20260716_v1`.
- Implementations: `scripts/91_probe_v2_residual_preservation.py`, `scripts/93_diagnose_v2_residual_preservation.py`, `scripts/94_probe_v2_top_heavy_ranking.py`, `scripts/96_audit_v2_e5_truncation.py`, `scripts/97_probe_v2_field_aware_e5.py`, `scripts/98_build_fresh_clean_task_manifest.py`, `scripts/99_run_selected_clean_agentdojo.py`, `scripts/102_summarize_fresh_clean_multiseed.py`, `scripts/103_audit_fresh_clean_failures.py`, `scripts/104_filter_fresh_clean_manifest.py`, `scripts/105_compare_clean_tool_protocols.py`, `scripts/106_audit_function_tag_repairability.py`, `scripts/107_audit_intent_retry_eligibility.py`, `scripts/108_evaluate_parser_retry_confirmation.py`, and `src/wmagentattack/qwen_agentdojo.py`.

All execution remains inside the synthetic AgentDojo sandbox; the fresh screen runs clean tasks only and has no external side effects.

## 0724 exact-state and observed clean replay follow-up

The next fixed-budget clean-only loop implemented exact Pydantic environment
snapshots/deltas and audited all 97 registered AgentDojo v1.2.2 user tasks.
All 339 expert calls executed, all tasks passed final utility, and every state
snapshot round-tripped as canonical JSON. However, 239 calls were read-only,
36 tasks were entirely read-only, and 21 tasks appeared utility-complete before
the final expert call when the fixed final answer was supplied. State delta is
therefore valid simulator state but not a dense progress label.

The adapter was then tested on the frozen 90-episode Llama-3.1-70B Travel clean
panel. The first preregistered replay retained a NO-GO because one terminal
assistant proposal had no following tool result and was incorrectly counted as
executed. A minimal, separately archived causal-pairing repair now records 456
proposals as 455 executed transitions plus one terminal unexecuted proposal.
All 90 final utilities recompute exactly and all pairing gates pass.

The repaired panel provides strong counterevidence to state-only value
modeling: only 13/455 calls changed state, while five task/final-state groups
contained both successes and failures. Strict target-only expert-slot coverage
was higher for success than failure in all eight mixed-outcome tasks, motivating
a goal-conditioned evidence ledger and separate progress/value heads. This is
an architecture result only. The clean-data gate remains blocked, so no attack
data or large Dreamer run is authorized. Full details are in
`reports/0724_clean_state_instrumentation_and_replay_results.md`.

## 0728–0729 factorized evaluator, balanced panel, and architecture result

The original custom-panel result remains `CUSTOM_PANEL_DATA_SUFFICIENCY_NO_GO`;
its labels were not rewritten. A factorized evaluator-v2 was instead developed on
the old 24-task adjudication set. It preserved all five strict durable successes
and all nine genuine failure tasks while recovering the ten checker/schema
artifact tasks, with zero task-level regression mismatches.

A new template-disjoint custom panel-v2 then balanced Banking, Slack, Travel, and
Workspace across L1/L2/L3. Its frozen 144-episode budget consists of 48 one-shot
greedy tasks and 16 preselected tasks with six true stochastic samples each. All
episodes completed without runtime failures. Independent dynamics and evidence
gates passed, but conditional reporting failed because training had only three
negative independent tasks against the required four. Attack data and Dreamer
therefore remained blocked, while a small clean-only three-backbone probe became
eligible.

Slurm 4786 exposed a pretraining implementation bug: twelve raw proposals carried
parser-extra argument keys outside the declared schema vocabulary. It failed
before any model fit and is retained as an engineering failure, not a scientific
result. The repair uses post-validation typed argument fields, adds a fail-closed
preflight invariant, preserves every scientific setting, and independently
rebuilds the 144-episode dataset byte-for-byte. Slurm 4787 then completed all nine
frozen runs with 31 tests and 34 archive checksums verified.

The binding architecture result is
`CUSTOM_PANEL_V2_ARCHITECTURE_NO_INCREMENT`. Observable execution improved
confirmation action NLL from 3.9883 to 3.7870 and action accuracy from 0.2263 to
0.3171, but worsened evidence NLL from 1.3423 to 1.4628, violating the frozen
cross-head guard. Ledger-v2 improved evidence NLL relative to observable execution
from 1.4628 to 1.4333, but worsened dynamics and still did not beat Semantic
Markov's 1.3423. Every increment gate failed, so both accepted heads remain
Semantic Markov.

The counterevidence is structured rather than uniformly null. Observable dynamics
gains are concentrated in Slack and disappear or reverse in Banking, Travel, and
L3. Ledger-v2 evidence gains appear in L1/L2 but reverse in L3, consistent with the
current hashed cumulative ledger lacking an explicit candidate-by-constraint,
join, coverage, and uniqueness representation. Training nearly saturates with
only 196 training prefixes, while confirmation remains weak; only eight evidence
rows are `CONTRADICTED`. The next valid proposal would need more independent tasks
and a relational evidence state, not a larger MLP or post-hoc threshold change.
No additional experiment is submitted in this fixed-budget loop. Completion,
attacks, H2, and Dreamer remain blocked. Full details are in
`reports/0729_custom_panel_v2_architecture_ablation_results_v2.md`.
