# v32 medium-scale GPU diagnostic preregistration

## Question and claim boundary

This fixed-budget experiment asks whether additional capacity helps the strongest retained local mechanisms without degrading task-disjoint generalization. It is a scaling diagnostic, not formal large-scale training.

The candidate contains a 768-dimensional Structured Markov v3/four-cell teacher, zero-start recurrent H1--H5 residual dynamics, and a separate 768-dimensional v21 shared-effect transition probe. The combined modular footprint is expected to be 7,166,317 trainable parameters. Failed v30/v31 relation heads, open-vocabulary claims, Dreamer, utility/value heads, attacks, and planning remain disabled.

## Frozen data and controls

- Action dynamics: 6,763 events from 2,060 AgentDojo trajectories, 20 tasks, five task-disjoint folds.
- Effect dynamics: the frozen 121-row v21 hard-label view, three task-disjoint folds.
- Seeds: 7, 17, and 29.
- Controls: immutable v22 small-model action predictions and immutable v21 selected shared-effect metrics.
- One RTX 6000 Ada, float32, deterministic algorithms, 39 total fits, no hyperparameter search or post-result rerun.

## Frozen gate

The candidate must retain H1 NLL/accuracy, improve H2--H5 NLL by at least 0.01 with task and seed replication, produce only legal actions, and keep effect positive NLL, recall, and rollout BCE within frozen non-inferiority margins in at least two folds and two seeds. Exact paired-key coverage, CUDA execution, parameter bounds, complete fit budget, and zero runtime failures are mandatory.

A GO only retains a medium closed-vocabulary/action-dynamics checkpoint. It does not override the v31 binding failure and cannot authorize attacks, Dreamer, a planner, or formal large-scale training.
