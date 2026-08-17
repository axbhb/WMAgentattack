# Structured Markov four-cell auxiliary v5

Status: `PREFLIGHT_PASSED_AND_FROZEN_BEFORE_TRAINING`.

The Structured Markov v3 representation, next-action head, observable-outcome head, task-disjoint folds, seeds, and optimization budget remain unchanged. The only candidate mechanism is a four-logit trajectory-outcome head supervised by the repeated-seed distribution over task failure/success and attack failure/success.

The four-cell labels are reconstructed deterministically from the frozen 400 attack configurations with five seeds each. They are never added to causal state inputs. The joint loss is normalized hierarchically so that tasks, attack configurations, seed trajectories, and steps within a trajectory do not become pseudo-replicates. Clean trajectories remain in the dynamics losses but are excluded from the attack joint-outcome loss.

The candidate is retained only if it both improves the original task-disjoint next-action objective and predicts the four-cell distribution better than the training-fold prior under the frozen gate.

Two independent builds were byte-identical. The frozen view contains 6,763 events, including 6,564 attacked events with four-cell soft targets. All seven label-integrity checks passed and no joint label occurs inside `causal_model_input`.
