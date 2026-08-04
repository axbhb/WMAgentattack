# 0723 Task-Macro Victim-Dynamics Ablation: Result and Next Revision

## Frozen decision

`SHORTCUT_NOT_RULED_OUT_CLEAN_GATE_BLOCKED`

The full semantic event model retains a real next-event likelihood signal, but
the fixed-budget mechanism audit does **not** show that it has learned sufficiently
general higher-order victim dynamics or a reliable trajectory-value representation.
The experiment therefore rejects the current full architecture as a basis for
attack planning, selective deployment, or Dreamer training.

The result is not “the event model is useless.”  It is more specific:

- event identity improves teacher-forced task-macro NLL over a length/static
  condition and both candidate Markov baselines;
- attack-semantic fields do not improve victim dynamics and slightly hurt it;
- the full model is not the best free-running generator;
- a semantic first-order Markov prefix reproduces almost all of the dynamic joint
  head's apparent value gain;
- validation task directions and seed stability are not adequate.

## Execution and integrity

- Slurm job: `4716`.
- Archive:
  `/share/guozhix/wmagentattack/0723/task_macro_dynamics_ablation/fixed_budget_v1`.
- Runtime: 2026-07-22 01:27:13–01:30:16 China Standard Time, about 3 minutes.
- Fixed budget: 3 variants × 3 seeds × 12 maximum epochs; no grid search.
- Device: one Slurm-allocated RTX 6000 Ada job.
- Remote full regression: 189 passed, 5 known Transformer performance warnings.
- New targeted regression: 8 passed, including a one-epoch end-to-end CPU smoke.
- Nine model outputs present; `COMPLETE` present; `FAILED` absent.
- No Traceback, OOM, CUDA error, or runtime failure.
- All 56 archived checksums verified.
- Frozen ontology fingerprint matched:
  `62171756d5051d395da68cc20a1b9ac2118ea8404d08eba8835d4d5bfb08ef4d`.
- No task overlap and no forbidden outcome fields in the event ontology.

## 1. Task-macro victim next-event prediction

Lower NLL is better.  Gains are baseline NLL minus full-model NLL.

| Split | Full semantic event | Plain candidate Markov | Gain | Semantic candidate Markov | Gain |
|---|---:|---:|---:|---:|---:|
| Validation | 1.2923 | 1.3668 | +0.0745 | 1.7107 | +0.4184 |
| Test | 1.9538 | 2.7141 | +0.7602 | 2.8253 | +0.8715 |

This is genuine evidence that event identity contains predictive information.
The `length_semantic` condition has NLL 1.5030 on validation and 2.0434 on test,
so the full model improves it by 0.2107 and 0.0896 respectively.  The frozen
`event_identity_beyond_length` gate passes.

However, the attack-semantics ablation reverses the intended interpretation:

| Split | Full semantic event | Event history without attack semantics | Full-model gain |
|---|---:|---:|---:|
| Validation | 1.2923 | **1.2579** | -0.0344 |
| Test | 1.9538 | **1.9483** | -0.0055 |

Masking attack family, role, trigger, payload position, knowledge level, endpoint
policy, and depth does not hurt victim prediction.  It slightly improves it.  The
`attack_semantics_increment` gate therefore fails.  These fields should not be
mixed into the victim-dynamics encoder in the next model.

## 2. Free-running trajectories contradict a planning claim

Lower normalized edit distance is better.

| Split | Full | Plain Markov | Semantic Markov | No attack semantics | Length/static condition |
|---|---:|---:|---:|---:|---:|
| Validation | 0.3573 | 0.4291 | 0.3642 | 0.3495 | **0.3390** |
| Test | 0.4491 | 0.5265 | 0.4361 | 0.4301 | **0.4032** |

The full model improves the plain Markov baseline, but its validation advantage
over semantic Markov is only 0.0068, below the frozen 0.02 requirement.  On test,
semantic Markov is better by 0.0130.  Both simpler neural ablations also produce
better greedy paths than the full model, and the length/static condition is best.

Therefore, better teacher-forced probability assignment has not translated into
better open-loop greedy generation.  The full model cannot yet be treated as a
trajectory planner.

## 3. Task-level heterogeneity is severe

The validation task-macro gain over plain Markov is only 0.0745, while the
micro-averaged gain is 0.2830.  The earlier micro result was therefore amplified
by long tasks—especially Slack—receiving many more event votes.

Against the strongest Markov baseline for each task:

- Validation: only Slack improves materially; Banking is nearly tied, and Travel
  and Workspace are materially worse.  Result: 1/4 improved and only 2/4 not worse.
- Test: all four tasks improve materially.

The opposite validation/test patterns are not evidence of robust generalization.
With only four independent tasks in each split, the experiment cannot distinguish
task composition from a stable population effect.  The frozen per-task-direction
gate fails.

## 4. Dynamic joint-value gains are mostly low-order

The observed prefix improves the full dynamic joint NLL over the static semantic
head by 0.2452 on validation and 0.1912 on test.  Taken alone, that appears strong.
The negative controls change the conclusion.

| Control minus observed NLL | Validation | Test |
|---|---:|---:|
| Static semantic | +0.2452 | +0.1912 |
| Shuffled event multiset | +0.0310 | +0.0021 |
| Length only | +0.0581 | +0.0288 |
| Random legal prefix | +0.0524 | +0.0183 |
| Plain Markov prefix | +0.0512 | **-0.0068** |
| Semantic Markov prefix | **+0.0034** | **+0.0062** |

Positive values favor the observed prefix.  The observed prefix beats static,
length, random, and plain Markov controls on validation, but it is essentially
indistinguishable from semantic Markov.  At task level, semantic Markov is better
on three of four validation tasks and one test task; its only meaningful loss is
Slack.

The shuffled-order gain also falls from 0.0310 on validation to 0.0021 on test.
Consequently, the dynamic head is learning useful aggregate trajectory cues, but
the current evidence cannot separate higher-order event dynamics from semantic
first-order transition statistics.  The frozen prefix-content gate fails.

## 5. Seed stability remains insufficient

| Diagnostic | Validation | Test | Frozen requirement |
|---|---:|---:|---:|
| All-seed full-sequence agreement | 0.3835 | 0.5049 | ≥0.40 |
| Mean pairwise joint total variation | 0.1191 | 0.0958 | ≤0.10 |
| Minimum pairwise joint-success rank Spearman | 0.8351 | 0.8719 | ≥0.80 |
| Maximum observed joint total variation | 0.3843 | 0.3861 | descriptive |

Ranking direction is reasonably stable, but validation sequence agreement and
mean joint-distribution disagreement fail.  Aggregate NLL/edit standard deviations
pass, which shows why aggregate seed standard deviation alone is not a sufficient
stability test.

## Gate outcome

| Frozen gate | Result |
|---|---|
| Representation integrity | PASS |
| Task-macro predictive and free performance vs both Markovs | FAIL |
| Event identity beyond length | PASS |
| Attack-semantics increment | FAIL |
| Prefix content beyond all low-order controls | FAIL |
| Prefix order beyond shuffled multiset | PASS, driven by validation |
| Aggregate seed stability | PASS |
| Outcome/path/ranking seed stability | FAIL |
| Per-task direction | FAIL |
| Independent clean eligibility | FAIL |

The only scientifically defensible decision is
`SHORTCUT_NOT_RULED_OUT_CLEAN_GATE_BLOCKED`.

## Minimum architecture revision implied by the controls

The next model should not add more capacity to the same mixed encoder.  It should
separate three causal roles.

### A. Victim dynamics tower

Predict the next victim tool, normalized argument slots, execution status, and
termination from:

- trusted goal representation;
- canonical current AgentDojo state;
- prior normalized tool events;
- current legal candidate manifest.

Attack-configuration semantics are excluded from this tower.  The current
`event_no_attack_semantics` condition becomes the neural dynamics baseline.

### B. Clean utility/progress tower

Predict task progress and eventual clean utility from exact state differences,
unmet goal slots, checker progress, irreversible effects, and termination.  This
tower must not infer progress from attack vocabulary or sequence length.

### C. Security/configuration tower

Attack family, role, trigger, payload position, knowledge, and endpoint policy
remain useful here, together with interactions between untrusted content, victim
events, target tools, and state effects.  They are not used to steer generic
victim dynamics.

The utility and security towers may share a low-level event encoder only after
state/progress features exist.  Initially, outcome gradients should be stopped or
strongly down-weighted at the dynamics encoder so noisy joint labels cannot make
configuration vocabulary dominate event learning.

## Required data revision before another value-model experiment

The frozen v2 audit confirms that four critical fields do not exist:

- `argument_entity_links`;
- `canonical_state_delta`;
- `task_progress_delta`;
- `irreversible_effect`.

They must not be synthesized from final success labels.  The next admissible
research cycle is therefore a clean-only AgentDojo instrumentation and independent
task-pool expansion:

1. snapshot canonical sandbox state before and after every exact tool execution;
2. compute typed state deltas from simulator state, not raw language observations;
3. expose checker-derived completed/unmet goal slots for diagnosis while retaining
   the official final utility checker as the eligibility decision;
4. record argument-to-entity links, tool execution status, premature termination,
   and known irreversible effects;
5. add genuinely different tasks and environment initial states, not paraphrases
   or more repeated configurations;
6. use task-held-out calibration and test panels with at least double-digit
   independent tasks;
7. only then evaluate clean closed-loop exact-simulator rollout and clean-only
   DAgger-style state aggregation.

The fixed `max_probability < 0.35` rule remains rejected.  Task-bootstrap ensemble
disagreement and risk–coverage calibration are deferred until enough independent
calibration tasks exist.

## Safety and scope boundary

No attack data were generated.  No H2 attack planning, real endpoint, selective
deployment claim, or Dreamer training was started.  The failed clean gate remains
authoritative.

## Repository and artifact map

- Frozen protocol: `configs/0723_task_macro_dynamics_ablation_protocol.json`
- Frozen ontology: `src/wmagentattack/event_ontology.py`
- Ontology audit: `scripts/116_audit_frozen_event_ontology.py`
- Task-macro experiment: `scripts/117_train_task_macro_event_ablation.py`
- Frozen summarizer: `scripts/118_summarize_task_macro_dynamics_ablation.py`
- Machine-readable result:
  `reports/0723_task_macro_dynamics_ablation_results.json`
- This analysis: `reports/0723_task_macro_dynamics_ablation_analysis.md`
- Remote frozen archive:
  `/share/guozhix/wmagentattack/0723/task_macro_dynamics_ablation/fixed_budget_v1`
