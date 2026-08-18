# Late-fusion action distillation v15 results

## Conclusion

The frozen result is `NO_GO_LATE_FUSION_DISTILLED_WORLD_MODEL_V15`. Distillation successfully makes the student imitate the full-graph oracle better than an equal-capacity control, but that imitation does not improve task-disjoint free-running action dynamics over v6. The neural future-latent direction is stopped.

## Fixed experiment

- Slurm 7110 completed with exit code 0 in 10 minutes 14 seconds, CPU only.
- Budget: 15 Structured Markov teachers, 15 full-graph oracles, 15 equal-capacity controls, and 15 distilled students.
- Five task-disjoint folds, seeds 7/17/29, 41,433 paired rows per arm.
- Parameter counts match; all predictions are legal; four-cell outcomes copy frozen v6 exactly; stderr is empty; hashes verify.

## Exact effects

- Horizon-1 NLL gain over v6: **+0.002763**.
- Horizon-1 accuracy gain over v6: **+0.003567**.
- Horizon-2–5 NLL gain over equal-capacity control: **+0.011642**.
- Oracle action cross-entropy gain over control: **+0.006788**, positive in all three seeds.
- Horizon-2–5 NLL gain over v6: **-0.011201**, negative in all three seeds.
- Positive held-out task fraction: **0.40**.
- Full-graph oracle gain over v6: **+0.036264**.
- Four-cell future-joint CE difference from v6: **0.0**.

The candidate passes the imitation, control, one-step, integrity, and outcome-isolation clauses. It fails absolute multi-step improvement, oracle-retention, task breadth, and seed replication.

## Interpretation

The student learned what the distillation loss asked for, so the failure is not simply insufficient optimization. The full oracle uses true future evidence at every step; matching its action distribution under teacher-forced training creates a target that the student's own free-running state cannot support. This is another form of exposure mismatch. Increasing latent size, KL weight, or recurrent depth would repeat the same assumption and is not authorized.

The next direction is retrieval-augmented dynamics. Training-task transition prototypes can ground a forecast in real observed successors and expose distance/coverage uncertainty, avoiding unconstrained future-state generation. A conservative residual over v6 will only be accepted when retrieval has close task-disjoint support. This is consistent with recent retrieval-augmented world-model and prototype-retrieval work, and with evidence that long-horizon generative simulation degrades rapidly.
