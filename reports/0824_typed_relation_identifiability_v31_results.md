# v31 typed relation identifiability results

## Decision

`NO_GO_TYPED_RELATION_REPRESENTATION_V31` (14/19 frozen clauses passed).

The single authorized scientific retry, Slurm 7321, completed both deterministic CPU builds in 36 seconds. The two datasets and audits were byte-identical, seven tests passed, all frozen hashes matched, and there were zero model fits, GPU requests, victim-LLM calls, sandbox tool calls, real endpoints, or runtime failures. Slurm 7320 is retained only as an invalid pre-output infrastructure attempt: it crashed before producing a dataset or metric because a namespaced tool ID was looked up in the frozen bare-name registry. The label-blind parser repair changed no task, label, seed, threshold, or score.

No clean support pilot, small model comparison, attack generation, data scale-up, large world-model training, planner, or Dreamer run is authorized.

## Exact result

| Frozen metric | Result | Gate |
|---|---:|---:|
| confirmation rows | 121 | 121 |
| unique positive record--goal pairs | 67 | 74 preregistered multiplicity |
| positive structural coverage | 0.7313 | >= 0.45 |
| positive typed-unit coverage | 0.8358 | >= 0.45 |
| hard negatives per positive | 4 | >= 4 |
| combined pair accuracy | **0.4627** | >= 0.65 |
| frozen-E5 semantic pair accuracy | 0.5075 | diagnostic |
| goal-blind record-only accuracy | **0.5000** | combined gain >= 0.10 |
| combined pair margin | +0.0200 | diagnostic |

Per-fold combined accuracy was 0.1667 on fold 0 (9 positive pairs, 36 comparisons), 1.0000 on fold 1 (only 1 positive pair, 4 comparisons), and 0.5000 on fold 2 (57 positive pairs, 228 comparisons). Fold 1's perfect number has no evidential weight for a transfer claim because its support is one pair.

The failed clauses were positive-edge reconstruction count, combined pair accuracy, gain over the goal-blind control, all-fold pair accuracy, and minimum positive relation support per fold.

## Counterevidence and diagnosis

Coverage is not the bottleneck. Structural relation types covered 73.13% of positive pairs and typed units covered 83.58%, yet the combined scorer ranked positives below verified same-record or same-goal negatives more often than the 50% goal-blind control. Frozen E5 alone was also effectively at chance. This is direct counterevidence to the hypothesis that a pretrained semantic encoder plus typed roles is sufficient for task-disjoint record--goal binding.

The 74-to-67 mismatch reveals a second representation defect rather than a reason to relax the gate. The preregistered 74 counted raw relation multiplicity from v29. After values were removed, repeated records with the same schema could collapse to the same `(record signature, goal index)` pair; fold 0 fell from 16 raw links to 9 unique pairs. The representation therefore knows that a hotel-like record has fields such as `name` and `price`, but not which anonymous hotel occurrence supplies the required evidence. Entity occurrence identity and value provenance are still missing.

Fold 0 is the strongest negative result: combined accuracy was 0.1667 and semantic accuracy 0.1111. Fold 2 remained near chance. A positive average margin coexisting with sub-chance accuracy indicates that a few large correct margins mask many incorrectly ordered pairs; it does not establish reliable binding.

## Retained architecture and next direction

The retained world-model core remains the previously validated Structured Markov v3 plus four-cell outcome auxiliary and zero-start recurrent residual mechanisms. v29's complete static candidate inventory and record-local target construction remain useful data mechanisms. The v31 typed/E5 relation representation is rejected and must not replace them.

The next admissible representation study should be preregistered before any new model fit or trajectory generation. It should construct an explicit obligation graph with:

1. anonymous per-row entity-occurrence slots, so repeated same-schema records remain distinct without leaking raw identity;
2. parsed goal predicates with entity, attribute, operation/comparator, and typed value/provenance slots;
3. action/tool-schema execution bridges and observable evidence-state updates;
4. verified predicate-compatible hard negatives, scored by whether the complete predicate binding is satisfied rather than by text similarity alone.

Only a new representation-only identifiability gate that beats the same goal-blind control in every supported held-out fold can authorize another small model comparison.

## Archive

- Archive: `/share/guozhix/wmagentattack/0824/typed_relation_identifiability_v31/formal_v1`
- Invalid pre-output attempt: Slurm `7320`
- Scientific retry: Slurm `7321`
- Scientific commit: `de8e06554f23a3759cdc33fa600947c968fd2de9`
- Dataset SHA256: `2662b92a2a477c3cd93aecc269f23a31d2b520f29da16f1cd0b9019f6de42f19`
- Audit SHA256: `e6a8a2f5a480281d7616b58551d0657952d5f4db5c58c13b0026df020b19e8a4`
- Gate SHA256: `2ddc59eeb013b41d285c16d50e39ebaf45d02440e36ec71ce997bd64ddad76cf`
