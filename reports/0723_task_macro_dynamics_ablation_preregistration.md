# 0723 task-macro victim-dynamics mechanism audit

## Frozen starting point

The controlling result remains `ARCHITECTURE_SIGNAL_ONLY_CLEAN_GATE_BLOCKED`.
The 0722 semantic candidate-constrained event model improved micro-averaged
next-skill likelihood and free sequence edit distance, but validation and test
each contain only four independent base tasks.  The current archive therefore
cannot establish task-general victim dynamics, long-horizon planning, calibrated
rejection, or causal clean-stable attack effects.

This cycle is a development-only mechanism audit.  It uses the existing
AgentDojo-v2 synthetic sandbox data and cannot authorize new attack collection,
H2 attack planning, selective deployment, or Dreamer training.

## Questions fixed before execution

1. Does the full model still beat legal-candidate Markov baselines when every
   task receives equal weight instead of every event?
2. Does event identity help beyond static semantics and sequence length?
3. Do attack-semantic fields help beyond domain, clean prior, and event history?
4. Does the observed event prefix improve the joint outcome distribution beyond
   static semantics, a shuffled event multiset, length only, random legal events,
   a Markov prefix, and a semantic-Markov prefix?
5. Are free paths, joint probabilities, and trajectory score rankings stable
   across the three frozen random initializations?

No conventional task-level significance claim is possible here: with four
independent tasks, even four favorable task directions have a one-sided sign-test
probability of `1/16 = 0.0625`.

## Frozen representation and data audit

- Event ontology: `wmagentattack.event.v2.3`.
- Ontology fingerprint:
  `62171756d5051d395da68cc20a1b9ac2118ea8404d08eba8835d4d5bfb08ef4d`.
- Candidate policy: `wmagentattack.candidates.current-state.v1`.
- Teacher forcing may use the exact current-state candidate set.
- Free rollout may use only the initial current-state candidate set; it may not
  inspect future true candidate sets.
- Outcome labels, probability targets, raw observations, and raw tool outputs
  are excluded from event-model inputs.
- The current v2 archive does not contain exact entity links, canonical state
  deltas, task-progress deltas, or irreversible-effect annotations.  These fields
  remain explicitly unavailable rather than being inferred from final labels.

The frozen split contains 12 train tasks, 4 validation tasks, and 4 test tasks,
with 103 trajectories per task.  Trajectory counts are balanced, but event counts
are not: long Slack tasks contribute far more micro-averaged events than short
Travel tasks.  The new loss is an unbiased equal-task objective under uniform
trajectory sampling, and checkpoint selection uses validation task-macro metrics.

## Fixed component map

| Requested control | Repository realization |
|---|---|
| Candidate mask only | Candidate-uniform legal-set baseline |
| First-order dynamics | Domain/previous-event hierarchical candidate Markov |
| Markov with the same semantics | Field-factorized semantic candidate Markov |
| Static semantics | `length_semantic` next-skill condition and static joint head |
| Event history without attack semantics | `event_no_attack_semantics` |
| Full model | `semantic_event` |
| Shuffled prefix | Deterministic, length- and multiset-matched shuffle |
| Length-only prefix | BOS at every observed event position |
| Random prefix | Deterministic initial-candidate samples at matched length |
| Markov prefix | Candidate Markov generation at matched length |
| Semantic-Markov prefix | Factorized semantic Markov generation at matched length |
| Tool-name-only history | Current normalized selected skill/tool identity |
| Argument ablation | Argument slots are only an auxiliary target; no argument-input claim |
| State-delta ablation | Not run because canonical deltas are absent from v2 |
| Outcome-head ablation | Static semantic head versus dynamic prefix residual |
| Random-init ensemble | Seeds 7, 17, and 29 with path/TV/ranking stability diagnostics |

## Fixed budget and decision boundary

- Three variants, three seeds, 12 epochs each.
- Hidden size 64, two layers, four heads, batch size 64, learning rate `3e-4`.
- One Slurm job and no hyperparameter grid.
- Test is never used for checkpoint selection.
- Primary metrics are task-macro next-skill NLL, free normalized edit distance,
  dynamic joint count NLL, and per-task direction.  Micro metrics are diagnostic.
- A full mechanism signal requires advantage over both Markov baselines, event
  identity beyond length, an attack-semantics increment, observed-prefix advantage
  over every negative control including shuffled order, seed stability, and a
  favorable majority of held-out task directions.
- Every possible decision remains `CLEAN_GATE_BLOCKED`.

Exact thresholds and all integrity conditions are frozen in
`configs/0723_task_macro_dynamics_ablation_protocol.json`; they must not be
changed after reading results.

## Method basis and deferred stages

DAgger motivates collecting labels under learner-induced states, but it is
deferred to a later clean-only cycle because the current round first isolates
whether event content is useful at all
([Ross et al., 2011](https://proceedings.mlr.press/v15/ross11a.html)).
Pointer-style candidate scoring is the planned solution for a genuinely dynamic
tool catalog, but the current frozen representation retains the tied compositional
catalog so ontology and training changes are not bundled
([Vinyals et al., 2015](https://arxiv.org/abs/1506.03134)).
Three random initializations are reported as a development diagnostic; formal
task-bootstrap ensembles and selective risk require many more independent tasks
([Lakshminarayanan et al., 2017](https://papers.neurips.cc/paper_files/paper/2017/hash/9ef2ed4b7fd2c810847ffa5fa85bce38-Abstract.html),
[Angelopoulos et al., 2024](https://arxiv.org/abs/2208.02814)).

If this mechanism audit passes, the next admissible stage is clean-only expansion
to double-digit independent calibration and test task pools.  If it fails, the
negative control that matches the full model determines the minimum architecture
revision; no attack data are collected in either case.

## Repository map

- Event ontology: `src/wmagentattack/event_ontology.py`
- Ontology audit: `scripts/116_audit_frozen_event_ontology.py`
- Task-macro training and controls: `scripts/117_train_task_macro_event_ablation.py`
- Frozen summarizer: `scripts/118_summarize_task_macro_dynamics_ablation.py`
- Protocol: `configs/0723_task_macro_dynamics_ablation_protocol.json`
- Slurm wrapper: `scripts/server/run_0723_task_macro_dynamics_ablation.sbatch`
- Tests: `tests/test_event_ontology.py`,
  `tests/test_task_macro_event_ablation.py`, and
  `tests/test_task_macro_dynamics_summary.py`
- Planned archive:
  `/share/guozhix/wmagentattack/0723/task_macro_dynamics_ablation/fixed_budget_v1`

Local targeted regression currently passes 8 tests.  The remote configured
environment remains the authoritative full-regression and execution environment.
