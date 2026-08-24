# v28 bound successor-record world model results

## Decision

`NO_GO_BOUND_SUCCESSOR_RECORDS_V28` (19/26 frozen clauses passed).

Slurm 7313 completed the exact 30/30 CPU fits and 30/30 metric rows in 358 seconds with zero runtime failures. Seven focused tests passed, all frozen hashes verified, and stderr was empty. No result-driven rerun was made.

## What improved

The bound-record mechanism produced a large task-disjoint gain over v26:

| Metric | v26 independent atoms | v28 bound records |
|---|---:|---:|
| unseen positive recall | 0.5053 | **0.7460** |
| unseen precision | 0.3574 | **0.4691** |
| unseen false-positive rate | 0.0260 | **0.0094** |
| matched-count=3 recall | 0.3810 | **1.0000** |
| focused entity/attribute recall | 0.3571 | 0.3571 |
| one-step BCE | 0.04285 | **0.04112** |
| rollout BCE | 0.02777 | 0.02789 |

Whole-record prediction itself was strong: record F1 0.8308, recall 0.7991, precision 0.8734, and exact-set accuracy 0.7328. The model had 100% exact-record candidate coverage in all three task folds. This supports the central mechanism claim: preserving entity-link-attribute bindings is better than v26's independent atom representation.

The deterministic pointer-cardinality renderer also repaired the high-count label. All three seeds in fold 2 reached 1.0 recall for `matched_count=3`; no independent count head was used.

## Why the full gate failed

The gain was not broad enough. Fold 0 remained exactly tied with v26 at 0.7143 unseen recall in all three seeds. Fold 2 rose to 0.7778 in all seeds, but its focused webpage entity/attribute effects remained at zero. Therefore only three of six affected fold/seed cells improved, below the frozen four-cell stability requirement. The aggregate improvement came from the count-3 repair rather than general relational transfer.

The model also learned count cardinality more readily than pointer identity. Task-disjoint goal-pointer F1 was only 0.1717. Fold 2 pointer F1 reached 0.4505--0.5271, but sparse folds 0 and 1 were near zero. The current target exposes bound records and a separate set of goal indices, but does not expose which record attribute or value-equality relation supports which goal term. The pointer problem is therefore underidentified after value information is removed.

Open diagnostics provide stronger counterevidence. Exact-record candidate coverage was only 0.4389 for tool-family-heldout and 0.5690 for source-heldout. Their unseen recalls were 0.0205 and 0.0056. A candidate-conditioned model cannot predict records absent from its train-plus-support candidate set, so these failures are primarily a data-support limitation rather than evidence for increasing model capacity.

Unseen NLL was 2.409, above the 1.25 ceiling, with seed-level values ranging from 0.840 to 4.572. This indicates unstable confidence even when thresholded recall is high. The external fixed-v21 reproduction also missed its deliberately exact threshold: the largest difference was 0.00347 in pair accuracy, while other differences were at most 0.00182. This is retained as CPU/thread-level reproduction counterevidence; the frozen threshold is not relaxed.

## Retained architecture and direction change

Retain fixed v21 for seen effects, v22 recurrent action supervision, and the v27 bound-successor data contract. Retain v28 whole-record scoring and deterministic pointer-cardinality only as validated mechanism components; do not promote v28 to the primary open-vocabulary branch.

Do not fit another larger encoder on these 131 rows. The next authorized step is a zero-model-fit relational data-sufficiency gate:

1. reconstruct privacy-safe `record_attribute -> goal_term_index` evidence links or equality/provenance edges, without exposing future outcomes or raw hidden simulator state;
2. add outcome-blind sibling support for the missing webpage records;
3. require exact-record candidate coverage for task-, tool-family-, and source-heldout diagnostics;
4. verify deterministic reconstruction, task disjointness, leakage, relation binding, and fold coverage before another model fit.

The literature-motivated object/slot principle was useful only when mapped to explicit semantic records. Dyn-O, Structured World Belief, HOWM, and SlotFormer motivate entity-preserving state units and interaction-aware dynamics; they do not justify adding a large object-centric network when the candidate and relation data are absent.

## Archive

- Archive: `/share/guozhix/wmagentattack/0824/bound_successor_records_v28/formal_v1`
- Slurm: `7313`
- Metrics SHA256: `aa4f6ba2facb5eebd21d3b8947c06e13cc326383aed0c9a18b54fb5e72267515`
- Gate SHA256: `4463d5f4d24af285f12611f8980a84831319d48e0b1769145f319b501eba09e0`
- Archive checksum SHA256: `08b08381fc655298d498e905760f025a8ba905b3f552e250c23c6dfb2f73dfc8`
