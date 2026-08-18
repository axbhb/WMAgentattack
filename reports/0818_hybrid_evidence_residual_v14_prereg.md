# Hybrid exact–evidence world model v14 preregistration

v12 showed that a true action-event graph contains strong task-disjoint multi-step action signal. v13 showed that predicting the entire 263-dimensional graph is worse than a training-fold feature prior and loses the oracle gain. v14 changes the target rather than increasing model capacity.

The frozen, label-blind partition treats action, argument, history, legal-action, and goal–argument relation features as exact protocol state available after sampling an action and applying the AgentDojo sandbox transition. Receipt structure, goal–receipt relations, argument–receipt relations, and entity-schema changes form the stochastic evidence residual.

The first gate only verifies deterministic partitioning, exact row reconstruction, source hashes, and absence of outcome labels. If it passes, two equal-capacity oracle arms will mask the original 263-dimensional graph to either exact or evidence features. Each arm must independently preserve one-step performance and recover a substantial, broad, seed-replicated fraction of the v12 full-graph multi-step gain. The result selects a deterministic renderer, a learned evidence residual, both, or an interaction-focused late-fusion redesign. No predictor is trained before this attribution gate.
