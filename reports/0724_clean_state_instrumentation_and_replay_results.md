# 0724 clean state instrumentation and observed replay results

## Decision

The fixed research loop ends at:

`OBSERVED_CLEAN_EXECUTED_CALL_PAIRING_READY_CLEAN_GATE_BLOCKED`

This is a positive engineering result and a negative data-readiness result.
Exact AgentDojo state snapshots, state deltas, tool status, and causal pairing
between proposed and executed calls are now reproducible. The existing Travel
panel still has only `14/90` clean successes and an empty durable
development-confirmation task intersection, so attack-data construction and
large Dreamer training remain prohibited.

## Why this stage was needed

The 0723 task-macro ablation retained an architecture signal but did not rule
out short cuts. Event identity improved next-skill NLL over first-order Markov
controls, yet length/static controls remained competitive in free generation,
semantic first-order history reproduced nearly all higher-order prefix gain,
and attack-semantic features slightly hurt victim dynamics. The next admissible
question was therefore whether AgentDojo exposes exact, label-blind state and
progress signals that can support a genuinely closed-loop model.

## Experiment A: all-task clean state interface audit

Protocol: `configs/0724_clean_state_interface_audit_protocol.json`

Archive:
`/share/guozhix/wmagentattack/0724/clean_state_interface_audit/fixed_v1`

The audit replayed the built-in ground-truth calls for every registered clean
user task in AgentDojo v1.2.2. It loaded no LLM, generated no attacks, contacted
no endpoint, and emitted no training examples.

### Frozen gates

All preregistered gates passed:

- `97/97` tasks passed final official utility.
- `339/339` calls executed without runtime failure.
- All expert traces matched their target-only goal slots.
- Every before/after state was finite JSON and round-tripped exactly.
- All archive checksums verified; stderr was empty; preflight was `8 passed`.

The canonical state adapter is therefore ready, but three requested fields are
not available from AgentDojo and remain null rather than guessed:

- fractional task progress;
- irreversibility annotations;
- current-state dynamic tool preconditions.

### Counterevidence

- `239/339` calls (`70.5%`) were read-only.
- `36/97` tasks had no state-mutating call at all.
- Official utility was already true before the final expert call for 21 tasks,
  usually because the audit supplied the fixed ground-truth final answer at
  each prefix. Official utility is consequently an episode-end checker, not a
  valid dense progress label.
- Exact scalar argument-to-state linking was unique for `315/663` values
  (`47.5%`), ambiguous for `161/663` (`24.3%`), and absent for `187/663`
  (`28.2%`). Ambiguity must remain explicit.

These results reject the assumption `state delta == task progress`.

## Experiment B: frozen observed clean replay v1

Protocol: `configs/0724_observed_clean_state_replay_protocol.json`

Archive:
`/share/guozhix/wmagentattack/0724/observed_clean_state_replay/fixed_v1`

The pilot replayed the existing 90 Llama-3.1-70B clean Travel traces from the
development seeds `101/103/107` and confirmation seeds `109/113/127`. It made
zero new victim-model calls.

The frozen v1 result was a valid NO-GO:

- `90/90` traces replayed and `90/90` final utilities matched the archive.
- One trace had `16` assistant tool proposals but only `15` tool-result
  messages, failing the exact pairing gate.
- The failure was `development / seed107 / user_task_19`: its final assistant
  message proposed `get_all_hotels_in_city`, but the rollout ended before the
  runtime executed it.

The v1 archive and decision were retained unchanged. This is evidence that a
victim proposal is not automatically an environment transition.

## Experiment C: minimal executed-call pairing repair

Protocol: `configs/0724_executed_call_pairing_repair_protocol.json`

Archive:
`/share/guozhix/wmagentattack/0724/observed_clean_state_pairing_repair/fixed_v1`

The repair was specified after diagnosing the sole v1 failure and is therefore
an engineering development result, not an independent confirmation. Its fixed
rule is causal and deterministic:

1. A simulator transition exists only when a following tool message contains
   exactly the same function and arguments as the pending proposal.
2. A proposal left at end-of-trace is recorded as terminal and unexecuted.
3. Mid-trajectory missing results, orphan tool messages, and signature
   mismatches fail closed.
4. No call may be dropped or inferred merely to force alignment.

All repair gates passed:

- `90/90` episodes and all 15 tasks across six seeds were present.
- `456` proposals became `455` executed transitions plus one terminal
  unexecuted proposal.
- There were zero mid-trajectory missing results, orphan tool messages,
  signature mismatches, replay failures, utility mismatches, or outcome labels
  inside transition records.
- Preflight was `14 passed`; the final full suite was `201 passed` with five
  previously known Transformer warnings.
- The 17-file archive was complete, stderr was empty, and all checksums
  verified.

## Scientific findings from the repaired 90-episode panel

### 1. Environment state alone is insufficient for Travel utility

- Only `13/455` executed calls (`2.86%`) changed environment state;
  `442/455` (`97.14%`) were read-only.
- Only `13/90` episodes changed state at all.
- Five task/final-state groups contained both successes and failures. These
  groups covered 29 episodes and 8 of the 14 total successes.

For read-only search tasks, the canonical environment remains unchanged while
the agent accumulates facts in tool observations and then composes a final
answer. A state-only world model necessarily collapses many successful and
failed trajectories to the same state.

State change was more common in successful episodes (`6/14`) than failures
(`7/76`). Treating episodes as independent gives an odds ratio of `7.39` and a
two-sided Fisher value of `0.00449`, but that calculation is not inferentially
valid because seeds are clustered within only 15 tasks. Within the eight tasks
that contain both outcomes, the state-change difference was positive for three,
negative for one, and tied for four. The apparent episode-level association is
substantially task-confounded.

### 2. Goal-evidence coverage is the strongest clean diagnostic found here

Strict exact matching against target-only expert call slots gave:

- success mean coverage: `0.7217`;
- failure mean coverage: `0.4405`;
- episode-level diagnostic AUC: `0.8289`;
- within all eight mixed-outcome tasks, success coverage exceeded failure
  coverage (`8/8` directional signs).

This signal must not be used directly as a model input. It depends on the
expert plan, can under-credit equivalent plans, and is available only as an
offline target. It does, however, justify learning a goal-conditioned evidence
or progress head from causally observed history.

### 3. Exact path identity is too sparse to establish sufficiency

Across 90 episodes there were 84 distinct exact task-conditioned call paths.
No exact path group contained mixed utility, but this is not evidence that path
identity solves utility: there are almost no replicated paths on which to test
that claim. The model must generalize over tool purpose, arguments, returned
evidence, and goal coverage rather than memorize whole paths.

### 4. Other retained diagnostics

- Runtime tool errors: `15/455` (`3.30%`).
- Episodes with no executed tool call: `6/90` (`6.67%`).
- Argument links: `1159/1427` unique (`81.2%`), `148/1427` ambiguous
  (`10.4%`), and `120/1427` unmatched (`8.4%`).
- Exact state-delta roots occurred only in synthetic inbox, calendar, and
  reservation state. Read-only observations therefore carry most Travel
  information.

## Revised model architecture

The evidence supports a five-part closed-loop architecture:

1. **Victim proposal model** predicts tool, arguments, stop, and proposal
   uncertainty from trusted goal, event history, evidence memory, and the
   suite-wide candidate manifest. Attack semantics stay out of this tower.
2. **Execution pairing layer** distinguishes proposed, executed, errored,
   terminal-unexecuted, and censored calls. Only exact executed pairs enter the
   simulator.
3. **Exact AgentDojo simulator** applies the function to the canonical state
   and returns state delta, status, and the causally observed tool result.
4. **Goal-conditioned evidence ledger** summarizes facts retrieved by
   read-only tools, argument/entity links, errors, novelty, and which parts of
   the trusted goal appear supported. This memory changes even when the
   environment state does not.
5. **Separate progress/value heads** predict target-only progress supervision
   and episode-end utility. Outcome gradients must not leak into the victim
   dynamics encoder by default. A later security/configuration tower may use
   attack semantics only after the clean gate passes.

Terminal unexecuted proposals are right-censored victim-policy events, not
`finish` actions and not simulator transitions.

## Next admissible fixed-budget study

Before collecting more attack data, run a clean-only evidence-ledger ablation
on the frozen 90 traces, then confirm on a separately frozen stronger
victim/task panel:

- task/length/static control;
- event tool-and-argument history;
- canonical state only;
- event plus state;
- event plus state plus goal-conditioned evidence ledger;
- output-length and within-task shuffled-evidence controls.

Use task-grouped evaluation and task-macro metrics. Predict two separate
targets: target-only expert-slot coverage and episode-end utility. Treat the
90-trace study as an architecture probe because it contains only 15 independent
tasks. Regardless of its outcome, formal attack-data eligibility still requires
a new clean development/confirmation panel with at least two durable tasks in
the prespecified cross-panel intersection.

## Reproducibility map

- State adapter: `src/wmagentattack/clean_state_instrumentation.py`
- Proposal/execution pairing: `src/wmagentattack/trace_execution_pairing.py`
- Interface audit: `scripts/119_audit_agentdojo_clean_state_interfaces.py`
- Observed replay: `scripts/120_replay_observed_clean_state.py`
- Interface audit JSON SHA256:
  `d55508cd6d4bf8676d00bddc3ee900b7021cb2ed6b0a424927eccf65edf77b43`
- Frozen replay v1 JSON SHA256:
  `298ea1177781665ef5b59c0232fc48f59a7ff67a2f326802add4e3ac9c1fc510`
- Pairing-repair JSON SHA256:
  `c9d32917b2c9b5f3a2a609dd1cf2c0bc6a509633b4f2560941997331c957a73d`

All work in this loop stayed inside the synthetic AgentDojo clean sandbox.

