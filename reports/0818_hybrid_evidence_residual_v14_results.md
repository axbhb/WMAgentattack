# Hybrid exact–evidence world model v14 results

## Conclusion

The frozen decision is `NO_GO_INDIVIDUAL_PARTITIONS_INTERACTION_REQUIRED_V14`. The full graph reliably improves multi-step action prediction, but neither preregistered partition independently clears every gate. The evidence partition misses the 70% retained-gain threshold narrowly; the threshold is not changed after observing results.

## Data gate

All 6,763 events across 20 tasks were partitioned deterministically and reconstructed exactly. The 263 graph features split into 174 exact action/protocol features and 89 stochastic receipt/evidence features. The partitions are disjoint, label-blind, and byte-identical across two builds. Exact features account for 39.46% of active feature instances.

## Oracle attribution

- Slurm job 7109 completed with exit code 0 in 9 minutes 45 seconds using CPU only.
- Budget: 15 teacher fits plus 45 equal-capacity oracle fits across five task-disjoint folds and three seeds.
- Each arm produced 41,433 paired rows; all predictions were legal; stderr was empty; frozen hashes passed.

Task-macro horizon-2–5 NLL gains over v6:

- Full modular graph: **+0.036264**, 18/20 tasks and 3/3 seeds positive.
- Exact protocol only: **+0.018390**, 17/20 tasks and 3/3 seeds positive.
- Stochastic evidence only: **+0.025278**, 17/20 tasks and 3/3 seeds positive.

The evidence arm retains **69.705%** of the full gain versus the frozen 70% requirement. The exact arm also misses the absolute +0.020 requirement. Both preserve one-step noninferiority and have broad task/seed support, so this is strong mechanism evidence but not a pass.

## Interpretation and next direction

The benefit is not localized to either partition. Full performance requires an interaction between known action/tool protocol structure and receipt/evidence changes. This explains why v13 dense graph reconstruction failed: it optimized every sparse feature independently rather than the action-relevant interaction.

v15 therefore uses late fusion. Exact protocol structure remains a deterministic path. A compact evidence residual is trained only for its contribution to future action dynamics and is distilled from the frozen full-graph oracle's action distributions or latent transitions. It does not reconstruct the 89 evidence features and does not share a hidden outcome head. The frozen v6 four-cell outcome branch remains unchanged.
