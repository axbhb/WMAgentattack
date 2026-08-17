# Structured Markov four-cell auxiliary v5

Status: `PREFLIGHT_PASSED_AND_FROZEN_BEFORE_TRAINING`.

The Structured Markov v3 representation, next-action head, observable-outcome head, task-disjoint folds, seeds, and optimization budget remain unchanged. The only candidate mechanism is a four-logit trajectory-outcome head supervised by the repeated-seed distribution over task failure/success and attack failure/success.

The four-cell labels are reconstructed deterministically from the frozen 400 attack configurations with five seeds each. They are never added to causal state inputs. The joint loss is normalized hierarchically so that tasks, attack configurations, seed trajectories, and steps within a trajectory do not become pseudo-replicates. Clean trajectories remain in the dynamics losses but are excluded from the attack joint-outcome loss.

The candidate is retained only if it both improves the original task-disjoint next-action objective and predicts the four-cell distribution better than the training-fold prior under the frozen gate.

Two independent builds were byte-identical. The frozen view contains 6,763 events, including 6,564 attacked events with four-cell soft targets. All seven label-integrity checks passed and no joint label occurs inside `causal_model_input`.

## Formal result

Decision: `GO_RETAIN_STRUCTURED_JOINT_AUXILIARY`.

Slurm 7055 completed all 30 paired fits with zero runtime failures. All 11 frozen clauses passed. The original Structured Markov baseline had task-macro next-action NLL 1.788638 and accuracy 0.424642. Adding only the trajectory-normalized four-cell auxiliary changed these to NLL 1.768879 and accuracy 0.446973, corresponding to gains of +0.019759 and +0.022331. Action NLL improved on 70% of tasks and its paired task bootstrap interval was [0.000504, 0.039655].

The joint head also learned information beyond the training-fold prior: task/group-normalized joint cross-entropy improved by +0.069338 and joint Brier score by +0.013059, each positive on 60% of tasks. Their paired bootstrap lower bounds were +0.017510 and +0.003693. Observable-outcome BCE changed by -0.003036, which stayed inside the preregistered noninferiority margin of 0.005.

This result supports retaining the four-cell auxiliary head and loss on the Structured Markov baseline. It does not justify putting success labels into the causal state, filtering to only successful attacks, or starting a planner automatically. The four probabilities remain prediction targets; `p11` can later become a constrained planning objective after the next architecture and clean-preservation gates.
