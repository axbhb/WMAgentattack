# Structured joint zero-residual v6

Status: `GO_RETAIN_STRUCTURED_JOINT_RESIDUAL_V6`.

This candidate starts from the retained Structured Markov v3 plus four-cell auxiliary teacher. The teacher is frozen before residual training. A zero-initialized recurrent residual receives only the structured context and action embeddings; raw goal, observation, and schema text nodes are absent. One-step predictions are anchored to the teacher with KL and a noninferiority gate. Horizons 2--5 use teacher-forced training and free-running expected-action feedback at confirmation. A latent consistency term aligns imagined hidden states with future frozen structured contexts, while a small future four-cell loss preserves the updated outcome objective.

## Formal result

Slurm job 7056 completed without runtime failures. All 10 preregistered clauses passed, the frozen inputs and implementation verified, and the archive checksums pass.

On the task-disjoint one-step test, original Structured Markov v3 obtained task-macro NLL 1.788638 and accuracy 0.424642. The retained v5 joint-outcome teacher obtained NLL 1.768879 and accuracy 0.446973. The v6 residual obtained NLL 1.771762 and accuracy 0.446900: gains of 0.016876 NLL and 0.022258 accuracy over original Structured Markov, while remaining within the frozen noninferiority margins relative to v5. The residual is therefore not a better one-step predictor than v5 itself; its value is the added multi-step mechanism without materially destroying v5's one-step capability.

For free rollout, v6 beat the rejected typed-text v4 at horizons 2--5 by task-macro NLL 1.471274 on average. The gain was positive for 20/20 held-out tasks and all three seeds. Absolute v6 NLL was 1.685191, 1.762698, 1.621061, and 1.497548 at horizons 2, 3, 4, and 5, respectively, versus 3.437179, 3.240663, 2.796326, and 2.654184 for typed v4. These horizon-wise absolute values use progressively smaller eligible trajectory subsets and must not be interpreted as NLL improving automatically with horizon.

The future four-cell head achieved CE 1.262605 versus the training-task prior CE 1.284054, a gain of 0.021449. The task-level bootstrap interval crosses zero and the exact sign test is not significant, so this is gate-clearing directional evidence, not definitive cross-task proof.

## Interpretation and limitation

The retained architecture is now **Structured Markov v3 + trajectory-normalized four-cell auxiliary + zero-initialized recurrent residual**. It rejects raw typed goal/schema/observation nodes, anchors one-step predictions with KL, and applies multi-horizon, latent-consistency, and future-outcome supervision only through a conservative residual.

This is the first candidate in this sequence that is better than original Structured Markov on the frozen one-step metrics and also supplies a successful multi-step rollout mechanism. It is retained as the leading world-model candidate, not yet declared the final model: the current multi-step control is typed v4 rather than an independently rolled-out Structured-only control, and the outcome-head uncertainty remains broad. A fresh task-disjoint confirmation with a Structured-only rollout ablation is required before claiming general superiority.

Archive: `/share/guozhix/wmagentattack/0817/structured_residual_v6/formal_v1`.
