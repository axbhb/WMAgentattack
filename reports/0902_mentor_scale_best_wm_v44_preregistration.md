# v44 mentor-scale confirmation: preregistration

## Question

Before the mentor discussion, run a larger confirmation budget for the strongest
retained small WMagentattack architecture. The experiment asks whether the
validated v5/v21/v22 mechanisms replicate across five model seeds without the
capacity overfit observed in v32.

## Frozen method

The candidate is a hybrid rather than a monolithic Dreamer model:

1. Structured Semantic State v3;
2. one-step Structured Markov teacher with the normalized four-cell
   task-success × attack-success auxiliary;
3. zero-initialized recurrent residual action dynamics for H1--H5;
4. the v21 shared semantic effect head, without execution-conditional experts.

The multi-step component is optional under a frozen fallback: if it does not
beat the repeated one-step teacher while preserving H1, the final selected
action model remains the one-step teacher. This prevents a weak recurrent
component from being hidden inside an aggregate score.

## Data and budget

- Existing AgentDojo sandbox data only: 2,060 trajectories, 6,763 event rows,
  4,703 adjacent transitions, 20 tasks and four suites.
- Existing v21 effect surface: 121 rows, evaluated separately because it has a
  different causal target.
- Frozen task-disjoint folds; five training seeds: 7, 17, 29, 43 and 61.
- 25 baseline teachers, 25 four-cell teachers, 25 residual models, 15 effect
  baselines and 15 v21 effect candidates: 105 fits total.
- No new LLM trajectories, attack generation, external endpoints, Dreamer,
  task removal, result-conditioned rerun, or content checksums.

## Pre-result scheduling amendment

The first GPU submission, Slurm 7654, remained pending with a next-day start
estimate and was cancelled before allocation or execution. Because all models
are small and the scale comes from data coverage and 105 repeated fits, the
unchanged experiment is executed with eight CPU threads in `formal_v2_cpu`.
No task, seed, model, loss, metric, threshold or fit count changed, and the
superseded GPU submission record remains archived in `formal_v1`.

## Primary metrics

- one-step action NLL and accuracy;
- four-cell cross-entropy and Brier score versus the training-fold prior;
- observable-outcome BCE;
- H1/H2/H3/H5 free-rollout action NLL, accuracy and legality;
- effect BCE, positive-label NLL/recall, pair assignment and rollout BCE;
- task-level and seed-level replication for every claimed gain.

The exact thresholds and fallback rule are frozen in
`configs/0902_mentor_scale_best_wm_v44_protocol.json` before any v44 result.
