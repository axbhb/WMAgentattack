# Factorized semantic transition world model v11

This loop changes the prediction target rather than adding another latent encoder. For every non-terminal adjacent transition it derives four label-blind semantic factors: visible observation change, evidence acquisition type, goal-evidence progress, and execution status. Labels use only the source event's observed execution outcome and the next victim-visible observation. Final task/attack outcomes and the next action are excluded from factor construction.

Stage F1 must produce two byte-identical builds over all 4,703 adjacent transitions, cover all 20 tasks, retain unique source events, expose at least two classes per factor, avoid a class above 98%, and pass leakage/fingerprint tests. Failure blocks neural training.

If F1 passes, Stage F2 retains the v6 Structured Markov teacher. The candidate predicts all four factor distributions and conditions its zero-initialized action/dynamics residual on those predictions. A matched control receives equal capacity without factor supervision. True factors may be used only as a confirmation diagnostic to determine whether failure comes from factor predictability or factor usefulness; oracle metrics cannot authorize the model. The complete gate requires transferable factor prediction, one-step noninferiority, multi-step gains over both v6 and the capacity control, task/seed breadth, future-joint noninferiority, and legal actions.

## Stage F1 result

The frozen label gate is `GO_FACTORIZED_LABEL_GATE`. Two independent builds are byte-identical and contain exactly 4,703 unique adjacent source events across all 20 tasks. All fingerprint, leakage, task coverage, multiclass, and 98% dominance clauses pass. Observed classes are 4/4/4/3 for observation, evidence, progress, and execution respectively. The label dataset SHA256 is `74925561691739f984f0f0d6ed317e7d3c9cba1339bbf052bad3254c748c9297`; Stage F2 is authorized without changing its model gate.
