# Factorized belief world model v4 pilot protocol

Status: `PREREGISTERED_AND_FROZEN_BEFORE_TRAINING`

This pilot changes the world-model representation and dynamics while holding the AgentDojo data, task-disjoint folds, training seeds, epoch budget, candidate catalog, observable targets, and legal-action mask fixed.

The current baseline compresses the causal state into one fixed hashed vector and applies an MLP. The v4 candidate keeps that structured vector as a residual branch, but additionally represents seven visible causal fields as typed nodes, performs relational self-attention across them, and applies a selected-action-conditioned recurrent transition. The learned components are factorized into victim next-action and observable execution-outcome heads. Legal actions remain an exact mask outside the network.

The multi-horizon arm is trained on the true adjacent action chain for horizons 1--5. During confirmation rollout it sees the observed action at the current step, then advances using its own expected action embedding without future observations. This directly tests whether the learned latent supports short imagination rather than only one-step classification.

The fixed comparison has three arms: the existing Structured Markov v3 MLP, the typed recurrent model trained at one step, and the same model with multi-horizon supervision. There are five frozen task-disjoint folds and three seeds, for 45 neural fits. No LLM calls, new tool executions, attacks, Dreamer runs, utility/value heads, or planner are allowed.

The candidate is retained only if it clears every gate in `configs/0817_factorized_belief_wm_v4_pilot.json`. The one-step typed ablation, task-paired intervals, individual horizons, Brier scores, and predictive entropy remain diagnostic counterevidence and cannot be used to alter the frozen gate after results are visible.

Runtime note recorded before any formal model fit: the first GPU submission (7053) remained pending with zero output and an approximately 18-hour scheduler delay. It was cancelled before start and replaced by one CPU submission after the label-blind CPU smoke passed. Every paired arm uses the same CPU runtime; no scientific field, threshold, seed, epoch, or data split changed.

Method motivation is mapped to recent primary work on code-grounded world models, language-agent world models, belief-state modeling, and temporal abstraction. The concrete repository hypothesis is narrower than those papers: typed causal fields and action-conditioned multi-step latent dynamics should reduce task-disjoint next-action NLL and free-rollout error on the existing audited traces.

## Formal result

Decision: `NO_GO_FNS_BWM_V4_DOES_NOT_CLEAR_FROZEN_GATE`.

Slurm 7054 completed all 45 preregistered fits in 44 minutes 38 seconds on the remote server. It produced 157,842 prediction rows with zero runtime failures. Six directed tests passed, the two independent typed-state builds were byte-identical, all 1--5 step surfaces were complete, all predictions respected the exact legal-action mask, and the archived checksum manifest verifies.

The full v4 model did not improve the primary one-step task-disjoint objective. Structured Markov v3 plus MLP achieved task-macro NLL 1.788638, while typed one-step v4 achieved 3.558275 and multi-horizon v4 achieved 3.673056. Relative to the baseline, multi-horizon v4 changed NLL by -1.884418 and accuracy by -0.046903; all 20 tasks had worse NLL. The 95% paired task bootstrap interval for NLL gain was [-2.708654, -1.238252], so this is strong counterevidence rather than an ambiguous miss.

The training behavior identifies the failure as task-disjoint overfitting. Typed one-step v4 ended with lower training action loss than the baseline (0.4915 versus 0.7754) while its confirmation NLL nearly doubled. Raw goal, observation, and tool-schema nodes therefore supplied enough task-specific lexical capacity to memorize training tasks but did not produce a transferable causal state.

There is one useful positive result: multi-horizon supervision improved free rollout over the otherwise identical typed one-step model by +0.332351 NLL on average over horizons 2--5, passing all three frozen multi-step clauses. The effect was negative at horizon 2 (-0.166267), modest at horizon 3 (+0.128930), and strong at horizons 4 (+0.619890) and 5 (+0.746852). Observable-outcome BCE also improved by +0.028141, but execution-error BCE degraded by -0.013638 and failed noninferiority.

## Scientific interpretation and retained architecture

The typed raw-text encoder is rejected and must not replace Structured Markov v3. The multi-horizon recurrent objective is retained only as a mechanism hypothesis, not as an accepted model. No attack generation, utility/value head, planner, Dreamer run, or post-result threshold change is authorized by this result.

The next rational candidate is a conservative residual dynamics model: freeze or strongly anchor the existing Structured Markov one-step representation and head, attach a zero-initialized action-conditioned residual only for horizons 2--5, and enforce one-step KL/noninferiority so long-horizon learning cannot destroy the validated short-horizon policy. Typed evidence should be slot/entity normalized before learning rather than represented as raw hashed goal and schema text. Execution error should remain a separate calibrated rare-event head. This candidate requires a new preregistered loop; it was not run in v4.
