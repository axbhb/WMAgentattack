# Semantic residual victim-event model: fixed-budget result

## Bottom line

The second factorized model is a **feasible replacement for the first diagnostic Event Transformer**, but it is **not yet a validated attack planner or a reason to start Dreamer training**.

All pre-registered architecture gates passed. The model removed the false OOV failure, beat a stronger candidate-aware Markov baseline in teacher-forced NLL and free-running sequence quality, and improved repeated joint utility/security likelihood after conditioning on the event prefix. The independent clean-solvability gate remains false, however, and the post-hoc counterevidence exposes substantial task-count, long-horizon, uncertainty-calibration, and seed-stability limitations.

Frozen decision: `ARCHITECTURE_SIGNAL_ONLY_CLEAN_GATE_BLOCKED`.

## What changed

The first Event Transformer treated each exact attack string as a categorical context and built its skill vocabulary only from skills selected in training. It was evaluated only with true prefixes. This round changed the factorization in four ways:

1. **Training-candidate vocabulary.** The skill vocabulary is the union of `candidate_skills` visible in the training split, not the union of observed labels. `restaurant_generate`, which caused 17 false OOV targets in the first round, is therefore represented without reading validation or test labels.
2. **Compositional skills.** A skill such as `restaurant_generate` is encoded from the name tokens `restaurant` and `generate`, plus a small learned skill residual.
3. **Semantic attack context.** Separate embeddings represent attack family, role, trigger stage, payload position, knowledge level, endpoint policy, and required-depth bucket. Unseen values map field-by-field to `<UNK>` instead of collapsing the whole configuration.
4. **Anchored value distribution.** A semantic static four-cell Dirichlet-multinomial value model is learned first; the causal event prefix contributes a residual to its logits. AgentDojo tool execution, state transitions, and checkers remain exact and are not reconstructed by the neural model.

Teacher-forced predictions are restricted to the candidate set produced by the observed exact state. Free-running predictions use only the first-step candidate set, so they do not inspect future true states. The round uses maximum likelihood and reports the teacher/free gap directly; it does not use scheduled sampling.

## Frozen experiment and integrity

- Existing synthetic AgentDojo-v2 data only: 1,236 train, 412 validation, and 412 test trajectories.
- Three fixed seeds: 7, 17, and 29; 12 epochs; hidden size 64; no hyperparameter grid.
- Slurm job: `4715`; runtime 57 seconds on an RTX 6000 Ada; no runtime failure, Traceback, OOM, or CUDA error.
- Remote regression before submission: 211 tests passed. Four warnings were the known PyTorch Transformer `norm_first` nested-tensor performance notice.
- All 22 original archive checks, 6 pre-run code checks, 17 data checks, and 4 post-hoc checks verified.
- Pre-registered protocol SHA256: `e3c6df9637e400046a54e61603cf839b0d6b374e283f13b5467774e3fab73060`.
- Summary SHA256: `c7feade47dbfa9e1d931fdda91bd56f3e3dfff237256c55e25878454b2df9b91`.
- Counterevidence SHA256: `8a62359bba38bc435ed49885534ea105633e5f1a65bba5bd54ebfd66ab8062b8`.

Slurm accounting storage is disabled on this cluster, so completion is established by the job's atomic `COMPLETE` marker, absence of `FAILED`, `runtime_failures=0`, complete outputs for all three seeds, clean error scan, and verified checksums.

## Main results

### Victim next-skill prediction

| Split | Semantic model NLL | Candidate Markov NLL | NLL delta | Model accuracy | Markov accuracy |
|---|---:|---:|---:|---:|---:|
| Validation | 1.4209 ± 0.0647 | 1.7033 | -0.2824 | 48.64% ± 3.71 pp | 47.76% |
| Test | 2.0365 ± 0.0100 | 2.5933 | -0.5568 | 32.86% ± 0.41 pp | 31.84% |

The NLL improvement is larger than the top-1 improvement. This is useful: the new model assigns better calibrated mass to the victim's stochastic choices rather than merely changing the argmax.

### Free-running skill trajectories

| Split | Model exact | Markov exact | Exact delta | Model edit ↓ | Markov edit ↓ | Edit delta |
|---|---:|---:|---:|---:|---:|---:|
| Validation | 43.61% | 28.64% | +14.97 pp | 0.3698 | 0.4291 | -0.0593 |
| Test | 36.81% | 20.39% | +16.42 pp | 0.4359 | 0.5265 | -0.0906 |

The task-cluster bootstrap 95% interval for the paired edit delta is `[-0.0948, -0.0223]` on validation and `[-0.1671, -0.0242]` on test. Lower is better, so both intervals favor the model. The exact-match interval is `[0.0000, 0.2994]` on validation and `[0.0016, 0.4086]` on test.

These intervals must be read cautiously: validation and test each contain only **four independent task groups**, one per domain. The 412 trajectories are mostly repeated configurations/seeds, not 412 independent tasks.

### Joint utility/security distribution

| Split | Constant NLL | Semantic static NLL | Dynamic teacher NLL | Dynamic free NLL |
|---|---:|---:|---:|---:|
| Validation | 4.3864 | 2.7242 | 2.5163 | 2.6231 |
| Test | 4.0913 | 2.4229 | 2.2495 | 2.2136 |

The static semantic anchor accounts for most of the gain over the pooled constant. The event-prefix residual still lowers NLL in both teacher and free-running evaluations, which supports the intended decomposition: configuration-level value first, trajectory evidence second.

## Counterevidence and unresolved problems

The architecture gate passed, but the following findings prevent a stronger claim:

1. **Only four independent tasks per evaluation split.** Task-cluster intervals are wide, and every domain-specific result is based on one task. The current evidence tests repeated configurations much more than task generalization.
2. **Long-horizon exact success remains zero.** For both 5–8 and 9–16 step groups, exact free-running trajectory accuracy is 0%. Validation edit distance even worsens by 0.0139 for the 2–4 step group. The aggregate gain is driven heavily by one-step trajectories.
3. **The fixed support rule is too pessimistic and not calibrated.** A single seed would conservatively truncate 59.22% of validation and 52.75% of test trajectories. All three seeds agree to continue on only 22.57% and 35.44%, respectively. All 80 unseen-semantic trajectories per split are automatically rejected.
4. **Support does not reliably identify model advantage.** In validation, the all-seed-supported subset exactly ties Markov; the improvement comes from trajectories the support rule would reject. A fixed maximum-softmax threshold is therefore not a validated uncertainty criterion.
5. **Sequence stability is limited.** All three seeds produce the same complete sequence on only 46.84% of validation and 54.13% of test trajectories. Maximum pairwise total variation in the joint outcome prediction reaches 0.384 and 0.433.
6. **Some domains and lengths do not improve.** Travel exactly ties Markov. Slack exact success is 0% on validation and 0.32% on test. Longer sequences improve edit distance modestly but never complete exactly.
7. **Historical test top-1 accuracy is lower.** Relative to the first Event Transformer, validation teacher accuracy rises by 4.98 pp while test accuracy falls by 4.48 pp. This is not a direct confirmatory comparison because the old model scored a selected-only vocabulary with `<UNK>` targets and no candidate mask, but the regression is retained as counterevidence.
8. **The ablation is bundled.** Candidate masking, compositional skill encoding, semantic fields, and the residual value head changed together. This round establishes that the package is promising, not which component caused the gain.
9. **The independent clean gate is still NO-GO.** The earlier unseen-seed confirmation produced zero durable development/confirmation tasks. This remains the controlling data-validity failure.

## Interpretation

The viable method is now:

> static attacker/configuration semantics → candidate-constrained victim event model → exact AgentDojo transition/checkers → joint outcome distribution, with event-prefix residual value updates.

This is better aligned with the actual problem than a monolithic Dreamer reconstruction. In the present dataset the attacker configuration is fixed before rollout, so the outer decision is closer to a contextual configuration selector than a full sequential attacker MDP. Dreamer becomes justified only if future experiments define genuine attacker actions at multiple time steps.

The compositional candidate catalog is consistent with work that represents tools as reusable tokens, while causal event likelihood follows the trajectory-sequence modeling line ([ToolkenGPT](https://arxiv.org/abs/2305.11554), [Trajectory Transformer](https://arxiv.org/abs/2106.02039), [ViLPAct](https://arxiv.org/abs/2210.05556)). The distributional joint head is closer to distributional trajectory value modeling than point-label regression ([SwitchTT](https://arxiv.org/abs/2203.07413)). Teacher/free evaluation is kept explicit because scheduled sampling has a known objective-consistency critique ([Scheduled Sampling](https://arxiv.org/abs/1506.03099), [How (not) to Train your Generative Model](https://arxiv.org/abs/1511.05101)). Conservative rollout stopping is directionally motivated by offline model-based work, but this experiment shows that its threshold must be calibrated rather than assumed ([MOPO](https://arxiv.org/abs/2005.13239), [MOPP](https://arxiv.org/abs/2105.07351), [TATU](https://arxiv.org/abs/2304.04660)).

## Decision and next research gate

Retain the semantic residual event model as the new diagnostic baseline. Reject the fixed `max_probability < 0.35` support rule. Do not generate new attack data, run H2 attack planning, or start Dreamer training yet.

The next cycle should be pre-registered only after addressing the data gate:

1. Run a clean-only victim/task-pool expansion until independent unseen seeds recover durable tasks; this remains the first blocker.
2. Increase **unique task** coverage, not merely seeds or configurations. Use task-held-out validation/test with enough tasks per domain to estimate domain effects.
3. On the larger task split, isolate candidate masking, compositional skill encoding, semantic context, and dynamic value residual in separate ablations.
4. Replace the fixed confidence cutoff with held-out selective-risk calibration, preferably task-grouped conformal or ensemble-disagreement calibration. Report coverage versus sequence error; do not tune on test.
5. Add exact-simulator H2 evaluation only after the clean gate passes and the one-step/free-running model retains its gains. Stop or penalize rollouts when calibrated support is exceeded.
6. Treat attack selection as contextual configuration ranking while the attacker has only one pre-rollout decision. Reconsider Dreamer only after defining and collecting true multi-step attacker actions.

## Repository and archive map

- Frozen protocol: `configs/0722_semantic_residual_event_protocol.json`
- Model: `src/wmagentattack/semantic_residual_event_model.py`
- Training/evaluation: `scripts/113_train_semantic_residual_event_model.py`
- Frozen summary: `scripts/114_summarize_semantic_residual_round.py`
- Counterevidence audit: `scripts/115_analyze_semantic_residual_counterevidence.py`
- Structured result: `reports/0722_semantic_residual_event_results.json`
- Remote archive: `/share/guozhix/wmagentattack/0722/semantic_residual_event/fixed_budget_v1`

No new attack data and no large Dreamer training were launched in this cycle.
