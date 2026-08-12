# Multi-source replication and tau3 tail-horizon results

## Final decisions

- ToolSandbox + InjecAgent auxiliary multi-seed expansion: `AUXILIARY_MULTI_SEED_EXPANSION_COMPLETE_AFTER_MEASUREMENT_RECOVERY`.
- tau3 tail-horizon pilot: `TAIL_HORIZON_PILOT_NO_GO__DO_NOT_CONFIRM_OR_SCALE`.

All scheduled jobs in this fixed-budget loop have terminated. No related job remains in the Slurm queue. The multi-source release is complete for auxiliary probability and robustness analysis, but it does not add independent tasks or authorize Dreamer training. tau3 remains excluded from scale-up under the current assistant-transition formulation.

## Fixed budget and execution

- Shared victim: `meta-llama/Meta-Llama-3.1-70B-Instruct`, frozen snapshot, 4-bit NF4.
- Additional sampling seeds: `311`, `313`.
- ToolSandbox: 190 LLM decisions, one GPU job `6659`.
- InjecAgent: 4,216 LLM decisions in four serial shards, Slurm array `6660`, maximum one GPU at a time.
- ToolSandbox measurement recovery: CPU-only jobs `6683` (source-path preflight failure, zero replay) and `6685` (completed).
- Final recovery summary: `6686`, completed after `6685` and `6660`.
- Total new LLM decisions: 4,406. Hyperparameter searches: 0. Real external endpoint calls: 0.

## Multi-source release gate

| Source | Rows | Text responses | Tool calls | Tool-call rate | Runtime failures |
|---|---:|---:|---:|---:|---:|
| ToolSandbox | 190 | 97 | 93 | 0.489474 | 0 |
| InjecAgent | 4,216 | 2,067 | 2,149 | 0.509725 | 0 |
| Total | 4,406 | 2,164 | 2,242 | 0.508852 | 0 |

All four InjecAgent chunks contain 1,054 rows, pass their frozen audits, and preserve all 2,108 clean/poison pairs. All outputs are non-empty, use one frozen LLM contract, and record zero real endpoint calls.

## ToolSandbox measurement recovery

The original ToolSandbox audit correctly remains a raw NO-GO: 4 of 93 exact executions had byte-level replica differences. A separately preregistered, CPU-only audit replayed all 93 decisions twice without model calls and reproduced exactly the same four raw mismatches.

- Two `remove_contact` pairs differed only in Python object memory addresses embedded in `NoDataError` text.
- Two `send_message_with_phone_number` pairs differed only in replay-generated UUIDs that appeared in the tool output and post-state.
- The allowlist normalized 4 memory-address occurrences and 4 runtime UUID occurrences.
- Status, exception type, and `state_changed` agreed for every pair.
- Canonical semantic mismatches after normalization: 0/93.
- LLM calls added: 0; records regenerated: 0; original outputs overwritten: false.

This is a measurement recovery, not a retrospective rewrite of the raw gate. Both the original failed audit and recovered semantic audit are retained.

## InjecAgent paired counterevidence

Across 2,108 task-seed pairs, attacker-tool selection was:

- poisoned-only: 435;
- clean-only: 7;
- both: 3;
- neither: 1,663.

The paired poisoned-minus-clean attacker-tool selection rate is `+0.203036`. Conversely, expected user-tool selection changes by `-0.198292` from clean to poisoned. These are descriptive behavior-selection diagnostics, not verified attack success and not a release gate.

## tau3 tail-horizon counterevidence

The 24-episode tail pilot completed with zero runtime failures and reduced forced stops from 8 to 4, but failed six frozen sufficiency clauses:

- agent tool-decision rate `0.301038 < 0.35`;
- state-changing assistant transitions `1 < 4`;
- tasks with an assistant state change `1 < 2`;
- domains with an assistant state change `1 < 2`;
- paired state-change gain `0 < 3`;
- supported transition targets `3 < 4`.

Therefore additional horizon alone does not repair the assistant-side transition signal. A 96-episode confirmation and tau3 scale-up are not authorized. Any future tau3 work should preregister a joint user-agent or dual-control dynamics formulation rather than extending the horizon again.

## Claim boundary and next authorized work

ToolSandbox and InjecAgent can be retained as auxiliary sources for stochastic-response estimation under the shared LLM contract. Their extra seeds do not increase independent task count, and the current results do not overturn the earlier scale NO-GO. tau3 cannot join the scaled dataset under the current transition target.

The next scientifically authorized step is a task-disjoint data-sufficiency and Semantic Markov experiment using the released AgentDojo, ToolSandbox, and InjecAgent surfaces, with source-aware weighting and no tau3 assistant-transition scale-up. Large Dreamer training and attack generation remain unauthorized until that gate passes.

## Reproducibility

- Multi-source archive: `/share/guozhix/wmagentattack/0811/multisource_replication/auxiliary_v1`
- Final gate SHA256: `26807040d9a5bf3fc039e58afb810986efbe9f26ca6b3f37e763dc0f6b3082d3`
- Released records SHA256: `396e3ea67e831ac4ac5ee117422be722f5189cb498df4cb102af7f5b5a1ad14d`
- ToolSandbox recovery-audit SHA256: `901086a860cb8741de19a1637f75718f246c11d6639c2303832588eddb7ddbf0`
- tau3 tail gate SHA256: `219d4bee003b844989fc51fc2b92cbe3d7eb972582473c3ee840a8ec8ad1a0c5`
