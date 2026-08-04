# 0712 Within-task contrast, shrinkage validation, and grouped-task transfer

## Executive conclusion

This round establishes a useful but bounded result:

1. Supplying the actual injection text to Dreamer materially improves within-task attack ordering. On 24 tasks, 96 injection pairs, and 480 fresh Llama-3.1-70B outcomes, `injection_world_ridge` beat `clean_world_ridge` by +0.1557 task-mean pairwise accuracy (95% task-bootstrap CI [0.0545, 0.2686]).
2. The unshrunk model failed the preregistered confirmation gate because its Brier score degraded by +0.0141, above the allowed +0.01. The planned 160-outcome confirmation was therefore not run.
3. A fixed 0.5 clean/injection probability blend corrected the calibration failure on the development data and directionally replicated on 80 new pair-heldout outcomes: pairwise difference +0.25 (95% CI [0, 0.5]) and Brier difference -0.0200 (95% CI [-0.0364, -0.0036]). Only four of eight tasks were informative, so this was not a strong replication.
4. The legacy train/val/test data split was found to leak every test user task into training. A new `(suite, user_task_id)`-grouped split was implemented and verified: 68/14/15 train/val/test tasks with zero overlap.
5. Three Dreamer seeds retrained on the grouped split show that injection conditioning transfers reliably to the **risk head**, but harms the **utility head**. A dual-view model—risk from injection-conditioned imagination, utility from clean-prefix imagination—is the strongest current architecture. Its validation improvement is statistically positive, while the held-out test estimate remains underpowered and seed-sensitive.

The correct status is therefore **mechanism PARTIAL-GO, general claim NO-GO**. Injection-aware risk modeling works; utility/value modeling must be decoupled, and fresh repeated outcomes on truly grouped held-out tasks are still required.

## Experiment A: 480-outcome within-task contrast

### Frozen design

- 24 tasks: six per AgentDojo suite.
- Four injection pairs per task: 96 pairs.
- Five independent sampling replicates per pair: 480 outcomes.
- Base seeds: 51, 57, 63, 69, 75; chunk substream seed `base + 1000 * chunk`.
- Selection was label-blind and excluded validation/test pair identities.
- Evaluation used five-repeat, six-fold, suite-balanced leave-user-task-out OOF predictions.
- Primary metric: mean attack/utility within-task pairwise accuracy.
- Uncertainty: 20,000 suite-stratified task bootstrap samples.

All 20 replay chunks passed seed, row-count, uniqueness, and frozen-selection audits.

### Outcome distribution

| Quantity | Result |
|---|---:|
| Attempts | 480 |
| Observed ASR | 0.2125 |
| Observed BUP | 0.2604 |
| Pairs with variable attack outcomes | 45 / 96 |
| Pairs with variable utility outcomes | 37 / 96 |
| Tasks with injection-dependent attack rates | 19 / 24 |
| Tasks with injection-dependent utility rates | 14 / 24 |

### Model comparison

| Model | Primary pairwise | Attack pairwise | Utility pairwise | Mean soft Brier |
|---|---:|---:|---:|---:|
| Context ridge | 0.5000 | 0.5000 | 0.5000 | 0.06023 |
| Text + context, no Dreamer | 0.7019 | 0.7722 | 0.6316 | 0.06551 |
| Clean-world ridge | 0.6456 | 0.6772 | 0.6140 | **0.04069** |
| **Injection-world ridge** | **0.7472** | **0.8101** | **0.6842** | 0.05474 |
| Hierarchical beta-binomial | 0.6965 | 0.7089 | 0.6842 | 0.07836 |
| Joint text multinomial | 0.7043 | 0.7595 | 0.6491 | 0.04783 |
| Raw clean world | 0.5616 | 0.6582 | 0.4649 | 0.10549 |
| Raw injection world | 0.6819 | 0.7848 | 0.5789 | 0.11430 |

Key task-bootstrap comparisons:

| Comparison | Pairwise difference (95% CI) | Brier difference (95% CI) |
|---|---:|---:|
| Injection ridge - clean ridge | **+0.1557 [0.0545, 0.2686]** | +0.0141 [-0.0035, 0.0337] |
| Injection ridge - text/context | +0.0712 [-0.0697, 0.2000] | -0.0108 [-0.0383, 0.0160] |
| Hierarchical - injection ridge | -0.0659 [-0.1303, -0.0101] | +0.0236 [0.0095, 0.0411] |
| Joint text - text/context | -0.0061 [-0.1016, 0.0754] | **-0.0177 [-0.0363, -0.0018]** |

Interpretation:

- Injection-conditioned world features add a clear ranking signal over the clean-world model.
- The incremental ranking advantage over the text/context diagnostic is positive but uncertain, so the current data do not isolate all gains as uniquely world-model-derived.
- The hierarchical beta-binomial model is worse on both ranking and Brier; all fits pushed concentration to the upper bound, indicating that this parameterization did not exploit useful overdispersion.
- The joint text model improves calibration relative to text-only/context but does not improve ranking.

### Preregistered gate

The gate required all three conditions:

- Pairwise improvement at least +0.03: **passed**, +0.1557.
- Pairwise 95% CI lower bound at least -0.02: **passed**, +0.0545.
- Brier degradation no more than +0.01: **failed**, +0.0141.

Decision: **NO-GO** for the original 160-outcome confirmation. This saved the GPU budget and triggered calibration diagnosis instead.

## Experiment B: fixed shrinkage on new pair-heldout outcomes

The development OOF curve showed that a fixed clean/injection blend could retain ordering while correcting calibration:

| Injection weight | Primary pairwise | Mean Brier |
|---:|---:|---:|
| 0.000 | 0.6456 | 0.04069 |
| 0.125 | 0.6955 | 0.03923 |
| 0.250 | 0.7043 | **0.03869** |
| **0.500** | **0.7608** | 0.04036 |
| 0.750 | 0.7623 | 0.04571 |
| 1.000 | 0.7472 | 0.05474 |

Alpha was frozen at 0.5 before collecting validation outcomes. The validation set contained eight calibrator-task-disjoint test tasks, two new injection pairs per task, and five fresh repeats per pair (80 outcomes). These pairs were held out from world-model training, but the user tasks were not; this limitation was written into the protocol before collection.

| Model | Primary pairwise | Mean Brier |
|---|---:|---:|
| Clean-world ridge | 0.875 | 0.07051 |
| Injection-world ridge | 0.875 | **0.04589** |
| Text/context multinomial | **1.000** | 0.08916 |
| Joint text multinomial | 0.875 | 0.06397 |
| **Fixed alpha=0.5 shrinkage** | **1.000** | 0.05055 |

Shrinkage minus clean:

- Task-mean pairwise difference: +0.25, 95% CI [0, 0.5].
- Brier difference: -0.01996, 95% CI [-0.03636, -0.00356].
- Informative ranking tasks: 4 / 8.

Decision: **directional replication passed; strong replication failed** because the ranking CI lower bound was exactly zero and only half the tasks supplied non-tied comparisons.

## Data audit: legacy task leakage and grouped repair

The old split was trajectory-level. All 73 legacy test user tasks also appeared in train, so prior val/test results measured new trajectories or injection pairs within seen tasks—not unseen-task transfer.

`scripts/09_split_real_agentdojo_dataset.py` now defaults to a grouped split and retains the legacy trajectory mode only as an explicit option. A 5,000-candidate balance search produced:

| Split | User tasks | Trajectories | Steps | Attack rate | Utility rate |
|---|---:|---:|---:|---:|---:|
| Train | 68 | 732 | 2,237 | 0.1134 | 0.1995 |
| Validation | 14 | 152 | 492 | 0.1382 | 0.2171 |
| Test | 15 | 162 | 565 | 0.1111 | 0.2099 |

Task overlap is zero for train-val, train-test, and val-test. Continuous probability labels were regenerated using train-only global/context evidence for val/test; clean evidence covered 97/97 tasks.

## Experiment C: three-seed grouped-task Dreamer transfer

Three models were retrained with unchanged architecture/hyperparameters and seeds 7/13/21. Best epochs were 23/13/22. Metadata records data paths, SHA-256 hashes, task counts, and zero train-validation task overlap.

### Unseen-task step-level metrics

Across the three test seeds:

- Risk AUC: 0.9416–0.9620.
- Risk Brier: 0.0565–0.0829.
- Continuous utility Brier: 0.02694–0.03658.
- Next-skill accuracy: 0.3292–0.3558.
- Next-skill top-3 accuracy: 0.5841–0.6478.

Thus the world model itself transfers to unseen user tasks reasonably well at the step level.

### Candidate-level view comparison

The dual-view rule uses injection-conditioned risk and clean-prefix utility.

| Split / view | Primary pairwise | Attack pairwise | Utility pairwise | Mean Brier |
|---|---:|---:|---:|---:|
| Val clean | 0.5211 | 0.6765 | 0.3657 | 0.12840 |
| Val injection | 0.5577 | 0.7721 | 0.3433 | 0.13645 |
| **Val dual** | **0.5689** | **0.7721** | 0.3657 | 0.13290 |
| Val symmetric alpha=0.5 | 0.5130 | 0.7574 | 0.2687 | 0.13115 |
| Test clean | 0.5000 | 0.5000 | 0.5000 | 0.13190 |
| Test injection | 0.5054 | 0.7442 | 0.2667 | 0.14341 |
| **Test dual** | **0.6221** | **0.7442** | 0.5000 | 0.14052 |
| Test symmetric alpha=0.5 | 0.4805 | 0.7209 | 0.2400 | 0.13683 |

Dual minus clean:

| Split | Task-mean pairwise difference (95% CI) | Brier difference (95% CI) | Informative tasks |
|---|---:|---:|---:|
| Validation | **+0.0984 [0.0256, 0.1938]** | +0.00561 [-0.00285, 0.01307] | 8 / 14 |
| Test | +0.0663 [-0.1875, 0.1949] | +0.00982 [0.00233, 0.01677] | 4 / 15 |

Per-seed test primary scores for dual versus clean were:

- Seed 7: 0.6979 vs 0.5758.
- Seed 13: 0.4958 vs 0.5133.
- Seed 21: 0.5116 vs 0.3836.

The ensemble and two of three seeds favor dual-view, but the test task-bootstrap interval crosses zero. The symmetric shrinkage discovered on the seen-task development set does not transfer; the structural head separation does.

## Current method interpretation

The strongest supported architecture is:

1. Encode the trusted task state with the clean-prefix world-model path.
2. Inject untrusted text only into a parallel risk-imagination path.
3. Predict attack risk from the injection-conditioned latent rollout.
4. Predict task utility/preservation from the clean latent rollout until an injection-aware utility adapter is shown to improve held-out utility.
5. Use a downstream calibrated selector to combine attack and utility, with task-grouped OOF fitting.

This is better aligned with AgentDojo causality than feeding injection text indiscriminately to every value head. The injection is directly informative for attacker behavior, while the current utility head exhibits negative transfer under that intervention.

## Limitations and counterevidence

- The first 480-outcome experiment used tasks seen by the fixed world model; it is a stochastic within-task contrast study, not a task-generalization result.
- The 80-outcome shrinkage validation held out pairs and calibrator tasks, but not world-model user tasks.
- Grouped test candidate labels are one Bernoulli outcome per pair. Only four test tasks supplied informative pairwise comparisons, causing wide uncertainty.
- The grouped test analysis is explicitly exploratory because candidate labels were inspected while developing the dual-view interpretation.
- Dual-view is not seed-uniform: seed 13 is slightly worse than clean on test.
- Injection-conditioned utility is consistently worse than clean utility. This is the principal remaining modeling failure.
- Text/context alone can rank some small held-out sets very well, so future claims must continue to include a text-only counterbaseline.

## Recommended next experiment

Freeze a fresh-replay grouped-task confirmation before collecting outcomes:

- Select two grouped-test tasks per suite and four injection pairs per task using score spans/disagreement only.
- Run five fresh Llama-3.1-70B repeats: 8 tasks × 4 pairs × 5 = 160 outcomes.
- Compare clean, injection, symmetric shrinkage, and dual-view with task-stratified bootstrap.
- Primary claim: dual-view improves attack/utility pair ordering without more than +0.01 Brier degradation.
- Train an injection-to-utility residual adapter only if this confirmation shows utility variation that the clean head cannot capture; keep the clean utility path as a fixed fallback.

This is the minimum experiment needed for a defensible unseen-task claim. Scaling models or collecting a much larger dataset should wait until it clears.

## Reproducibility map

Local code:

- `scripts/09_split_real_agentdojo_dataset.py`: grouped user-task split.
- `scripts/11_select_world_model_agentdojo_pairs.py`: injection-conditioned rollout scoring.
- `scripts/37_merge_within_task_contrast_replays.py`: multi-seed replay merge.
- `scripts/38_evaluate_hierarchical_contrast_models.py`: grouped OOF models and task bootstrap.
- `scripts/44_evaluate_pair_heldout_shrinkage.py`: fixed alpha=0.5 validation.
- `scripts/45_evaluate_grouped_task_transfer.py`: clean/injection/dual unseen-task analysis.
- `scripts/server/run_grouped_sheeprl_full_dreamer_v3_70b.sbatch`: three-seed grouped retraining.
- `configs/0712_within_task_confirmation_gate.json`: original gate.
- `configs/0712_pair_heldout_shrinkage_protocol.json`: pair-heldout shrinkage protocol.
- `configs/0712_grouped_task_dreamer_protocol.json`: grouped retraining protocol.

Remote artifacts:

- `/share/guozhix/wmagentattack/0712/within_task_contrast`
- `/share/guozhix/wmagentattack/0712/grouped_user_task_split_raw`
- `/share/guozhix/wmagentattack/0712/grouped_user_task_split_continuous_probability`
- `/share/guozhix/wmagentattack/0712/grouped_user_task_dreamer`

Verification: local full suite passed **91/91** tests; all final Slurm jobs completed and the queue is empty.
