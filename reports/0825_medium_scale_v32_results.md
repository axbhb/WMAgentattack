# v32 medium-scale GPU diagnostic results

## Decision

`NO_GO_MEDIUM_SCALE_CAPACITY_V32` (9/17 frozen clauses passed).

Slurm 7322 completed the complete preregistered budget on one RTX 6000 Ada: 15 action teachers, 15 recurrent action residuals, and 9 shared-effect models, for 39/39 successful fits. Runtime was 504 seconds, peak allocated GPU memory was 2.25 GB, six tests passed, and there were no OOM, CUDA, runtime, or external-endpoint failures. No result-driven rerun was made.

The actual modular footprint was 7,931,245 parameters: 1,503,752 in the Structured/four-cell teacher, 4,341,510 in recurrent residual dynamics, and 2,085,983 in the shared-effect transition module. This remained inside the frozen 3M--10M medium-scale range.

The preregistration's nominal estimate was 7,166,317 parameters because it used the configured hash widths as the full input widths. The realized Structured and effect vectors also contain fixed non-hash fields, producing the larger audited count. No architecture, threshold, or budget was changed; both parameter-bound clauses passed.

## Main comparison with the retained small models

| Task-disjoint metric | Frozen small reference | Medium v32 | Change/gain |
|---|---:|---:|---:|
| H1 task-macro NLL gain | -- | -- | **-0.5275** |
| H1 task-macro accuracy gain | -- | -- | -0.00320 |
| H2--H5 task-macro NLL gain | -- | -- | **-0.5153** |
| H2--H5 positive-task fraction | -- | 0.20 | required >= 0.55 |
| H2--H5 positive seeds | -- | 0/3 | required >= 2/3 |
| hard-effect positive NLL | 0.18572 | **0.24257** | -0.05685 |
| hard-effect positive recall | 0.96202 | 0.96111 | -0.00091 |
| effect rollout BCE | 0.02269 | **0.03283** | -0.01015 |
| effect pair assignment | 0.99306 | 0.99653 | +0.00347 |
| unseen-effect recall | 0.0000 | 0.0000 | unchanged |

All predicted actions were legal and all paired comparison keys were present. The failures are scientific, not infrastructural.

The horizon diagnostic tells the same story. Average NLL changed from 1.8520 to 2.4292 at H1, 1.8498 to 2.2951 at H2, 1.9270 to 2.5924 at H3, 1.8414 to 2.6768 at H5, and 2.0386 to 3.3833 at diagnostic H10. Some micro accuracies increased, especially at H5, while NLL became substantially worse. The larger model is therefore making more confident errors rather than learning a better predictive distribution.

## Capacity-overfit evidence

The clearest evidence is the effect branch. Its average training loss fell from 2.0365 to 0.000315, yet held-out positive NLL increased from 0.1857 to 0.2426 and rollout BCE increased from 0.02269 to 0.03283. Only one of three task folds and zero of three seeds satisfied the paired NLL/rollout non-inferiority condition.

The degradation was not confined to one fold. Effect positive NLL changed:

- fold 0: 0.2872 -> 0.4124;
- fold 1: 0.0698 -> 0.0729;
- fold 2: 0.2001 -> 0.2425.

The action branch likewise failed every multi-step seed and improved only 20% of held-out tasks. This cross-head, cross-horizon failure is counterevidence to the idea that the current system merely needed GPU-scale capacity.

## Scientific conclusion

The retained small v21/v22 models remain the strongest validated models. The medium checkpoints are archived as counterevidence but are not selected for downstream use.

The bottleneck is representation and support, not parameter count:

1. fixed closed-vocabulary outputs still have zero recall for labels absent from training;
2. v31 already showed that typed roles plus E5 do not bind repeated entity occurrences to goal predicates;
3. only 121 hard-effect rows supervise the effect module, which lets a 2.09M-parameter head nearly memorize training while worsening task-disjoint prediction;
4. scaling recurrent dynamics cannot repair missing entity identity, value provenance, or predicate binding.

The next admissible study is therefore the previously identified obligation-graph representation gate on existing data. It should be representation-only first, using anonymous entity-occurrence slots, parsed predicates, operation/comparator nodes, value provenance, and execution bridges. No larger model should be trained until that gate beats the same small controls.

## Archive

- Archive: `/share/guozhix/wmagentattack/0825/medium_scale_v32/formal_v1`
- Slurm: `7322`
- Scientific commit: `112b1eb7a1cbcf7130a9d01dc9d9cfd17b278d35`
- Gate SHA256: `df486240bd85cf091d7c8538c09322b986048984a15b580963028fedcc87879d`
- Metrics SHA256: `9b6928b89f3cc7a391c06500b3db77f2cd2565f92d304fe4085680deb1c72c53`
- Action predictions SHA256: `d11d7b3154eed874b0ce44c040da7276363fafab7246fb8e3e8af7304bb1915f`
- Diagnostics SHA256: `d14c49fd00f3d16186cbc33f60a9267675d21598b57ec7b843ee47ea8f53885e`
