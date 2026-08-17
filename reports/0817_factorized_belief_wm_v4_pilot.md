# Factorized belief world model v4 pilot protocol

Status: `PREREGISTERED_AND_FROZEN_BEFORE_TRAINING`

This pilot changes the world-model representation and dynamics while holding the AgentDojo data, task-disjoint folds, training seeds, epoch budget, candidate catalog, observable targets, and legal-action mask fixed.

The current baseline compresses the causal state into one fixed hashed vector and applies an MLP. The v4 candidate keeps that structured vector as a residual branch, but additionally represents seven visible causal fields as typed nodes, performs relational self-attention across them, and applies a selected-action-conditioned recurrent transition. The learned components are factorized into victim next-action and observable execution-outcome heads. Legal actions remain an exact mask outside the network.

The multi-horizon arm is trained on the true adjacent action chain for horizons 1--5. During confirmation rollout it sees the observed action at the current step, then advances using its own expected action embedding without future observations. This directly tests whether the learned latent supports short imagination rather than only one-step classification.

The fixed comparison has three arms: the existing Structured Markov v3 MLP, the typed recurrent model trained at one step, and the same model with multi-horizon supervision. There are five frozen task-disjoint folds and three seeds, for 45 neural fits. No LLM calls, new tool executions, attacks, Dreamer runs, utility/value heads, or planner are allowed.

The candidate is retained only if it clears every gate in `configs/0817_factorized_belief_wm_v4_pilot.json`. The one-step typed ablation, task-paired intervals, individual horizons, Brier scores, and predictive entropy remain diagnostic counterevidence and cannot be used to alter the frozen gate after results are visible.

Method motivation is mapped to recent primary work on code-grounded world models, language-agent world models, belief-state modeling, and temporal abstraction. The concrete repository hypothesis is narrower than those papers: typed causal fields and action-conditioned multi-step latent dynamics should reduce task-disjoint next-action NLL and free-rollout error on the existing audited traces.
