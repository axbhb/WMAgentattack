# tau3 interaction-faithful repair results

## Frozen decision

`INTERACTION_DATA_NO_GO__DO_NOT_SCALE_OR_RUN_METHOD_TEST`

The 96-episode pilot completed without a runtime failure, but restoring the
official user--agent--environment turn-taking did not recover a usable
assistant-side transition surface. The preregistered data gate failed, so the
Semantic Markov/Structured v3/full-history method comparison, large tau3
collection, attack generation, and Dreamer/planner training remain forbidden.

## Exact results

| Metric | Result | Frozen gate | Decision |
|---|---:|---:|---|
| Complete episodes | 96 | 96 | PASS |
| Natural user messages | 510 | >=96 | PASS |
| Adjacent assistant transitions | 254 | >=100 | PASS |
| Episodes with >=2 assistant transitions | 50 | >=36 | PASS |
| Tasks with an assistant transition | 37 | >=30 | PASS |
| Unique assistant tools | 18 | >=10 | PASS |
| Agent tool-decision rate | 0.3547 | 0.35--0.90 | PASS |
| Dominant action fraction | 0.3142 | <=0.65 | PASS |
| Forced-budget-stop episodes | 78 | <=24 | **FAIL** |
| State-changing assistant transitions | 1 | >=15 | **FAIL** |
| State-unchanged assistant transitions | 253 | >=30 | PASS |
| Tasks with an assistant state change | 1 | >=8 | **FAIL** |
| Domains with an assistant state change | 1 | >=2 | **FAIL** |
| Paired changed-transition gain over parent | -4 | >=10 | **FAIL** |
| Supported transition targets | 3 | >=4 | **FAIL** |

All six chunks passed their 16/16 audit. Exact live execution and both fresh
replay replicas agreed for every complete tool sequence. There were zero
runtime failures, communication errors, nondeterministic tool events, private
scenario exposures, task-split overlaps, semantic-state leakage findings, or
real external endpoint calls.

## Counterevidence and diagnosis

This candidate repaired interaction fidelity, dialogue volume, action balance,
and transition count, but those repairs were not enough:

- 47 episodes hit the agent's eight-generation cap and 31 hit the user's cap;
  only 18 ended without either forced cap.
- The domain split was agent/user forced stops of 19/5 for airline, 22/1 for
  retail, and 6/25 for telecom.
- The agent executed 254 tools but produced 79 tool errors. Only one assistant
  event changed state, from `update_reservation_passengers`.
- The paired parent had five changed assistant transitions, so the purported
  repair reduced rather than increased the mutation count (`1 - 5 = -4`).
- The user executed 24 tools and nine changed user-side state. Those are valid
  exogenous dynamics, especially in telecom, but the frozen target contract
  explicitly excluded user actions from assistant transition labels. They are
  retained as diagnostic evidence and are not used to retroactively pass the
  gate.
- `execution_error`, `goal_overlap_gained`, and `novel_observation` had both
  classes in training and confirmation. `state_changed` did not, while
  `output_nonempty` was constant-positive, leaving only three supported
  targets.

The strongest immediate explanation is finite-horizon truncation, not lack of
tool use: 78/96 conversations reached a hard role cap. The contrary evidence
is also material: the 31.1% assistant tool-error rate means a longer horizon
could merely extend invalid loops. The next test therefore changes only the
horizon and includes an error-rate non-regression clause.

## Next preregistered mechanism

The next authorized work is a 24-episode, same-task/same-seed paired pilot that
doubles each role's call horizon from 8 to 16 and the matching orchestrator
horizon from 32 to 64. Model weights, prompts, decoding, tool schemas, exact
execution, transition extraction, and assistant-only mutation target remain
unchanged. Selection uses only frozen manifest metadata and hashes, never
state-change labels or current completions.

A pilot GO authorizes only the full 96-episode horizon confirmation. It does
not authorize the predictive-method comparison or large-scale collection.

## Reproducibility

- Generation array: `6457`
- Frozen summary: `6532`
- Code commit: `f53bc5f7fad76ade08f6e7675ea8005178908127`
- Archive: `/share/guozhix/wmagentattack/0806/tau3_interaction_faithful_repair/pilot_v1`
- Frozen protocol SHA256: `6cae3ea5583dba82a62713163e89d845791184c5d405943e31d5448bfac8e641`
- Manifest SHA256: `d635d46ada7f189e91d3737216877f78197082f9bc7ce623539114fdda0596d5`
- Dataset SHA256: `c1d3aaca520d82da4d1cf69247f47ce25d77150840b3b9ac9c1893241a79c631`
- Dataset audit SHA256: `a45fe84370507904df814d241b2ae390de65dc9e7dfadeafbd732018738c961a`
- Gate SHA256: `cf6ce03a9b5399ef6ab856153b997297934c052efb9271c70e126de549d70afd`
- Archived report SHA256: `997a88f1085d5d802c662b4ea82e18577bf4cc884bd0f7a08ec552f4c41b25a8`
- Archive checksum-file SHA256: `c5ae7eaf74e4f2f5beb4d8f50387772efb4571973e15ad56296619fd93c376da`
