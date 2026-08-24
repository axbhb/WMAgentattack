# v30 joint relational successor results

## Decision

`NO_GO_JOINT_RELATIONAL_SUCCESSOR_V30` (26/30 frozen clauses passed).

Slurm 7319 completed the exact preregistered budget: 30/30 CPU model fits, 30/30 metric rows, zero runtime failures, and five passing tests. Runtime was 446 seconds. Frozen implementation and data hashes matched before execution, stderr contained no Traceback/OOM/CUDA/runtime error, and no result-driven rerun was made.

The architecture is not retained as the primary open-vocabulary world model and does not authorize the 96-episode clean smoke, attacks, large-scale data generation, Dreamer, a planner, or large world-model training.

## Main result

The joint model substantially improved canonical task-disjoint transfer over v28, but failed the relation mechanism it was designed to validate.

| Task-disjoint metric | v28 | v30 | Change |
|---|---:|---:|---:|
| unseen positive recall | 0.7460 | **0.8889** | +0.1429 |
| unseen positive NLL | 2.4090 | **0.1607** | -2.2483 |
| unseen precision | 0.4691 | **0.6667** | +0.1976 |
| unseen false-positive rate | 0.0094 | **0.0000** | -0.0094 |
| bound-record F1 | 0.8308 | **0.9598** | +0.1290 |
| bound-record exact-set accuracy | 0.7328 | **0.9563** | +0.2235 |
| focused unseen recall | 0.3571 | **1.0000** | +0.6429 |
| goal-pointer F1 | **0.1717** | 0.1568 | -0.0149 |
| one-step BCE | 0.04112 | **0.03620** | -0.00492 |
| rollout BCE | 0.02789 | **0.02686** | -0.00103 |

The intended record--goal edge predictor reached only 0.0633 F1, 0.0402 precision, and 0.1481 recall. Goal-pointer F1 was 0.1568, below both the frozen 0.45 floor and v28's 0.1717. The four failed clauses were relation F1, relation recall, pointer F1, and pointer gain over v28.

## Counterevidence and causal diagnosis

The failure is not explained by missing record candidates. Candidate coverage was exactly 1.0, record recall was 0.9891, and record exact-set accuracy was 0.9563. The model therefore usually knew which evidence record appeared but could not reliably identify which current goal term that record satisfied.

The positive relation labels were highly unbalanced across the three held-out task folds: 16, 1, and 57 positive edges. Fold 0 contained 16 positive edges, yet all three seeds obtained zero relation recall and zero pointer F1 while canonical unseen recall was 1.0. This is direct counterevidence that the canonical gain came from the intended relation pathway; the record/delta renderer could recover the effects while the relation head failed.

Training loss also contradicts a simple optimization-failure account. In a representative fit, relation loss fell from 0.7769 to 0.1441 and pointer loss from 1.2940 to 0.1376. Strong training fit combined with weak task-disjoint edge F1 indicates relation memorization or shortcut fitting rather than transferable semantic binding.

The current goal representation explains this pattern. Goal terms are fixed hashes of normalized textual terms. Hashing is deterministic and leakage-safe, but it does not preserve semantic proximity. With only 24 linked records in the v29 dataset, the tri-linear hidden x record x goal scorer cannot infer that a new held-out goal phrase denotes the same entity/attribute/operation relation as a training phrase.

The open diagnostics are mixed rather than uniformly negative. Tool-family and source-heldout unseen recall both reached about 0.612, and pointer F1 reached 0.492 and 0.672 respectively, showing that the v29 static candidate inventory repaired v28's coverage failure. However, relation F1 remained only 0.244/0.298 because precision was low, and source-heldout NLL was 3.595. This supports retaining the candidate/data mechanism but not the learned relation model.

## Scientific conclusion

v30 validates two local mechanisms:

1. action-conditioned whole-record prediction plus complete static candidates improves open-vocabulary canonical transfer;
2. zero-start residual dynamics preserves and slightly improves the retained one-step and rollout surfaces.

It does not validate joint record--goal relation reasoning. The result is therefore a mechanism-only advance, not a superior world model.

The next admissible direction is representation/data repair before another model fit:

1. replace raw goal-term hashing with normalized entity, attribute, operation, relation-type, and evidence-status features; optionally add a frozen semantic alignment branch behind the typed bottleneck;
2. construct an edge-identifiability gate with balanced positive relation support for every task-disjoint fold and explicit hard negatives that share entity or attribute but not both;
3. require relation transfer itself, not only rendered canonical effects, to beat a record-only ablation;
4. keep the v30 record branch and v21/v22 residual controls frozen, and do not enlarge the model until the data gate passes.

This follows the relational/object-centric motivation from Graph Networks and SOLD while retaining the counterevidence emphasized by Vafa et al.: successful task outputs or low training loss alone do not establish a correct world-model representation.

## Archive

- Archive: `/share/guozhix/wmagentattack/0824/joint_relational_successor_v30/formal_v1`
- Slurm: `7319`
- Scientific commit: `c41367ac2a2e6a8571d5cbbfa9a2ce9d1377e9ab`
- Metrics SHA256: `1658a13ecc7e6aff5a78f1516cf51962b3b38a56f4e74a8301e1199e730d493b`
- Gate SHA256: `8f8654324a7711deb179bbaa6eceebd8406e9ca7afb3ccb714aca791687d6153`
