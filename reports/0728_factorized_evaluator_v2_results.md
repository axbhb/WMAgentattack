# 0728 factorized evaluator v2: frozen development regression

## Binding decisions

The evaluator regression returns:

`FACTORIZED_EVALUATOR_V2_REGRESSION_PASS`

The upstream data decision remains:

`CUSTOM_PANEL_DATA_SUFFICIENCY_NO_GO`

This is an evaluator result only. It does not establish that the old panel is sufficient, that Ledger v2 or Semantic Markov is effective, or that attack-data generation or Dreamer training may begin.

## What was changed

The v2 evaluator separates each clean outcome into three interpretable factors:

1. **State/action**: validated, actually executed tool calls are replayed on a fresh AgentDojo state; required postconditions and mutations must hold, and forbidden side effects must be absent.
2. **Evidence**: the trajectory must satisfy a frozen minimum-evidence contract. Equivalent routes are allowed, while condition-before-mutation, comparison coverage, and uniqueness requirements remain strict.
3. **Report**: the final answer must fill the task's semantic report slots. Only predeclared aliases are accepted, and literal wording is required only when the user explicitly asks for it.

Non-applicable factors are `N/A`. Overall success requires every applicable factor to pass.

The checker consumes actual `role=tool` executions, not merely assistant-proposed calls. Arguments are validated and canonicalized through the AgentDojo/Pydantic schemas before evaluation, so values such as string `"100"` and integer `100` are compared after the same coercion used during execution.

## Frozen regression suite

The immutable 0727 panel contributes 24 development-only tasks and all 144 previously collected clean traces. These tasks are permanently barred from serving as fresh confirmation.

| Frozen category | Tasks | Expected v2 behavior | Result |
|---|---:|---|---|
| Strict success | 5 | Preserve success | 5/5 preserved |
| Lexical false-negative candidates | 8 | Recover behaviorally correct outcomes | 8/8 recovered |
| Schema-coercion false-negative candidates | 2 | Evaluate canonical typed calls | 2/2 recovered |
| Genuine model/evidence failures | 9 | Preserve the appropriate failure | 9/9 preserved |

All six traces for each task agreed with the frozen representative adjudication:

- 24/24 tasks covered;
- 144/144 episodes covered;
- seeds 233, 239, 241, 251, 257, and 263 covered;
- 0 task mismatches;
- 0 episode mismatches.

The factorized task counts are:

| Factor | PASS | FAIL | N/A |
|---|---:|---:|---:|
| State/action | 7 | 5 | 12 |
| Evidence | 15 | 5 | 4 |
| Report | 13 | 4 | 7 |
| Overall | 15 | 9 | — |

The overall 15/24 behavioral result exactly reproduces the prior generous sensitivity upper bound. It does **not** replace the frozen strict label or reverse the v1 NO-GO; confirmation class balance and genuine L2/L3 failures remain unresolved.

## Counterevidence found during regression

The first remote regression produced six mismatches, all for Workspace task 1301. The event had the correct participants, but process-dependent list order differed between the local and remote synthetic environments. Treating participant order as meaningful would turn a correct state into a false failure. The evaluator now compares set-like list fields as canonical multisets. No adjudicated label, alias, proof obligation, or v1 outcome changed.

After this correction, the evaluator produced zero mismatches. A separate summary-only bug then displayed `FAIL` because two satisfied prohibitions—no new victim outcomes and no attack/Dreamer execution—were encoded with negative Boolean polarity. The summary gate was corrected to positive invariants; evaluator outputs were unchanged.

These two failures are retained in the report because they are counterevidence against assuming that a plausible checker implementation is already semantically correct.

## Freeze and claim boundary

The following are frozen before any new-panel victim outcome:

- factor definitions and combination rule;
- typed-call and post-state evaluation;
- evidence obligations and equivalent-route semantics;
- semantic alias table;
- old-panel adjudication and proof contracts;
- the requirement that set-like list values are order invariant.

Any later semantic change requires a new evaluator version and a complete regression rerun. The old 24 tasks may continue to detect evaluator regressions, but may not tune or confirm the new panel or model architecture.

## What may proceed

The evaluator gate permits authoring and preflighting a new clean-only v2 panel. It does not yet permit the representation ablation.

The next panel must contain 48 new template-disjoint tasks balanced across four suites and three difficulty levels. Split assignment and a 16-task stochastic robustness subset must be frozen before any greedy outcome is read. The fixed budget is 48 one-shot greedy episodes plus 96 genuinely sampled episodes, for 144 total.

Independent downstream gates remain in force:

- dynamics requires complete, task-balanced transition and executed-call pairing;
- evidence progress requires frozen obligations, supported and unsupported atoms in every split, and a passing Ledger-v2 extractor regression;
- completion requires independent task-level positive and negative examples in every split;
- attack eligibility, H2 attack planning, and Dreamer remain blocked.

## Reproducibility

- v1 frozen summary: `ec89e9c6160a50c9d87cba03f3e7e327b7e6e467b5df93a7c509f3d1f909e693`.
- adjudication: `a76f1890cf82e3b35f3f35e5f8edde7e271f976eee2f841ca4cda2104ba92641`.
- alias registry: `738ef5e2970d4d7203555bec75d7acaa47d8179be1c74cf4f5b039e25a1b0a2f`.
- canonical proof-contract registry: `d13d770423d5a5f5329e3c7ebae7f33b229fa18fa431326bd3fd9f98b00ee13c`.
- frozen regression audit: `f23d88f96e6d9788ed43fe7288739a9485208fd2f03aa90107f5ea5a76690c19`.
- implementation and test hashes are recorded in `configs/0728_factorized_evaluator_v2_protocol.json`.
