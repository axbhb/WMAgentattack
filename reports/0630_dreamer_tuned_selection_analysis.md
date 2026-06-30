# 0630 Dreamer Tuned Selection Analysis

## Fixed research budget

This analysis used a fixed budget before summarizing:

1. Run tuned Dreamer validation and test selection grids with the same K/seed
   protocol as the previous Dreamer grid.
2. Compare tuned Dreamer against initial Dreamer and sklearn rollout.
3. Search for counterevidence where tuned Dreamer did not improve.
4. Cross-check step-level metrics against candidate-pool diagnostics.
5. Map each finding back to repo files that should be changed next.

## Tuned model metrics

Compared with the initial Dreamer adapter, risk and utility prediction improved:

| Model | Risk AUC | Utility AUC | Brier | Next-skill top-3 |
|---|---:|---:|---:|---:|
| Initial Dreamer | 0.8173 | 0.8093 | 0.1760 | 0.6754 |
| Tuned Dreamer | 0.8519 | 0.8352 | 0.1532 | 0.6716 |

This confirms that risk/utility loss scaling helped the supervised heads.

## Selection-grid comparison

World-model-top selection, averaged over seeds 7/13/21:

| Split | Model | K | ASR | BUP | ASR+BUP |
|---|---|---:|---:|---:|---:|
| val | Initial Dreamer | 16 | 0.2500 | 0.1875 | 0.4375 |
| val | Tuned Dreamer | 16 | 0.1667 | 0.2292 | 0.3958 |
| val | Initial Dreamer | 24 | 0.2083 | 0.1250 | 0.3333 |
| val | Tuned Dreamer | 24 | 0.2083 | 0.1667 | 0.3750 |
| val | Initial Dreamer | 32 | 0.1875 | 0.1563 | 0.3438 |
| val | Tuned Dreamer | 32 | 0.1875 | 0.1771 | 0.3646 |
| test | Initial Dreamer | 16 | 0.3333 | 0.1875 | 0.5208 |
| test | Tuned Dreamer | 16 | 0.3750 | 0.2500 | 0.6250 |
| test | Initial Dreamer | 24 | 0.3333 | 0.1944 | 0.5278 |
| test | Tuned Dreamer | 24 | 0.3611 | 0.2361 | 0.5972 |
| test | Initial Dreamer | 32 | 0.2917 | 0.2396 | 0.5313 |
| test | Tuned Dreamer | 32 | 0.3021 | 0.2604 | 0.5625 |

The tuned Dreamer improves on test for all K values, but validation contains
counterevidence: K=16 loses ASR enough that ASR+BUP drops.

## Comparison against sklearn rollout

On test, tuned Dreamer is still behind sklearn rollout in BUP:

| K | Tuned Dreamer ASR | Tuned Dreamer BUP | sklearn ASR | sklearn BUP |
|---:|---:|---:|---:|---:|
| 16 | 0.3750 | 0.2500 | 0.4375 | 0.4375 |
| 24 | 0.3611 | 0.2361 | 0.3333 | 0.4583 |
| 32 | 0.3021 | 0.2604 | 0.3125 | 0.3750 |

The strongest tuned-Dreamer result is K=24 for ASR, but its utility
preservation remains substantially weaker than sklearn.

## Candidate-pool diagnostics

The candidate-pool diagnostics explain the mismatch between improved head AUC
and weaker BUP:

| Split | Model | Risk candidate AUC | Utility candidate AUC | Selection AUC utility |
|---|---|---:|---:|---:|
| val | Initial | 0.9351 | 0.6169 | 0.6551 |
| val | Tuned | 0.9109 | 0.7092 | 0.7280 |
| test | Initial | 0.8804 | 0.7406 | 0.7608 |
| test | Tuned | 0.8816 | 0.7850 | 0.7934 |

Tuned Dreamer improves utility ranking, but the selected top-K remains
concentrated in slack/banking candidates and imagined futures collapse mostly
to `finish -> record_read -> record_read`. This suggests the current adapter
uses the RSSM, but the action-conditioned latent imagination is not yet
sufficiently task-specific.

## Counterevidence

- Validation K=16: tuned Dreamer improves BUP by +0.0417 but ASR drops by
  -0.0833, so ASR+BUP decreases.
- Tuned Dreamer improves test over initial Dreamer, but not over sklearn BUP at
  any K.
- Candidate-level risk AUC on validation decreases from 0.9351 to 0.9109 after
  tuning, even though step-level risk AUC improves.
- `risk_f1` remains 0.0 because raw risk probabilities are still calibrated
  below a fixed 0.5 threshold; ranking works better than thresholding.

## Mapping findings back to repo files

| Finding | Repo location to change next |
|---|---|
| Utility improves but BUP remains weak | `src/wmagentattack/dreamer_world_model.py::fit` needs pair-level or final-trajectory utility supervision, not only step labels. |
| Imagined skill paths collapse to `finish/record_read` | `src/wmagentattack/dreamer_world_model.py::rollout_score_step` needs candidate-action branching instead of greedy self-rollout. |
| Dreamer adapter is not full DreamerV3 | `src/wmagentattack/dreamer_world_model.py` should adopt DreamerV3-style KL balancing/free nats and reward/continue heads. |
| Selection weights alone are insufficient | `scripts/17_tune_selection_weights.py` is useful for cheap sweeps, but improvements must move into model training. |
| Current observations are hashed bags of text | `step_to_dreamer_text` and `hash_text_features` should be replaced or augmented with learned text embeddings. |
| Evaluation should stay leakage-safe | `scripts/11_select_world_model_agentdojo_pairs.py` should continue using `--standardized-steps` and split-specific `--allowed-trajectories`. |

## Recommended next implementation plan

1. Add candidate-action branching in RSSM imagination: for each candidate target
   skill/action, imagine one to three steps and score that branch.
2. Add final-trajectory labels to training: supervise whether a prefix belongs
   to a trajectory with final attack success and final utility success.
3. Replace or augment hashed text features with frozen sentence embeddings or
   small local encoder embeddings.
4. Add DreamerV3-style KL balancing/free-nats to the offline adapter instead of
   the current single KL term.
5. Run a small ablation table: current tuned, branch-imagination, final-label
   supervision, embedding features, and combined.

