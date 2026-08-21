# Hard-label confirmation v21: results

Date: 2026-08-21  
Formal job: Slurm 7204  
Archive: `/share/guozhix/wmagentattack/0821/hard_label_confirmation_v21/formal_v1`

## Decision

The frozen gate passed with:

`GO_REPLACE_WITH_INTERVENTION_NO_EXECUTION_EXPERTS_V21`

The retained transition probe is therefore:

**Structured Semantic State v3 + normalized action encoder + zero-initialized
latent residual dynamics + separate execution head + one shared semantic
effect head + pair and sequence supervision.**

The success/error-conditional effect experts from full v20 are removed. They
improved the easy task-disjoint panel but overfit the observed tool families.
The pair loss remains because the no-pair variant failed the same held-out
tool-family gate as full v20.

## Data and integrity

- The v21 hard view contains 121 rows and 94 effect tokens.
- Exactly 121 mechanically action-implied `source=<tool>` labels were removed.
- Two independent builds are byte-identical.
- All 60 preregistered CPU fits completed.
- Eighteen tests passed; runtime failures and external endpoint calls are zero.
- Confirmation data was not used for training or model selection.

## Primary task-disjoint comparison

| Metric | Structured residual v6 | Retained no-expert v21 | Change |
|---|---:|---:|---:|
| Hard task-macro BCE ↓ | 0.06802 | **0.04426** | -34.9% |
| Positive-label task-macro NLL ↓ | 0.34016 | **0.18572** | -45.4% |
| Positive-label recall ↑ | 0.90136 | **0.96202** | +6.07 pp |
| v19 rollout hard BCE ↓ | 0.05698 | **0.02269** | -60.2% |
| Parameters ↓ | 51,104 | **32,575** | -36.3% |

The selected model beat v6 on positive NLL and rollout BCE in all three task
folds and all three seeds. It also outperformed full v20 and the no-pair
variant on every primary aggregate metric.

## Held-out diagnostics

On aggregate tool-family holdout, positive NLL improved from 2.0650 to 1.9466,
while recall changed from 0.6987 to 0.6827 and remained inside the frozen 0.05
non-inferiority margin. On source holdout, positive NLL improved from 2.2102
to 1.8763 and recall improved from 0.6591 to 0.6819.

Full v20 and the no-pair model both failed tool-family positive-NLL
non-inferiority. This is evidence against the hypothesis that more conditional
experts or the pair objective alone explain the v20 gain. The simplest shared
effect decoder transfers best on this panel.

## Counterevidence

The GO is limited to a closed, already observed label vocabulary.

1. Every arm has exactly zero recall on positive labels absent from its
   training partition. There are 16 such positive occurrences across the
   unique task folds, 213 in tool-family diagnostics, and 332 in source
   diagnostics.
2. On the held-out query/read family, the selected model's recall is 0.7439
   versus 0.8230 for v6. Pair assignment is 0.3846 versus 0.6923. Aggregate
   NLL improvement therefore hides a meaningful action-sensitivity regression
   in this family.
3. The fixed 94-way classifier learns one independent output weight per label;
   a class with no training positives is pushed toward zero by negative BCE
   and has no mechanism for semantic zero-shot transfer.
4. Tool-family and source diagnostics intentionally share tasks and some
   states. They diagnose action/source generalization but do not replace the
   task-disjoint primary inference.

The result is consistent with work showing that average predictive quality can
hide weak action controllability, motivating explicit counterfactual metrics
(CoCo: https://arxiv.org/abs/2608.04653). It also maps to classic and recent
zero-shot multi-label results: semantic label embeddings permit transfer to
classes not directly supervised, unlike a fixed independent classifier
(https://proceedings.mlr.press/v38/li15d.html and
https://proceedings.mlr.press/v235/liu24bq.html).

## Next research direction

Do not scale the current 94-way output head. The next model should keep the
selected no-expert latent residual but replace its fixed effect head with a
**candidate-conditioned compositional effect-token scorer**:

- parse each target into category, entity type, attribute name/kind, link,
  conflict, and count slots;
- encode those slots with one shared label encoder;
- score any candidate effect token against the predicted transition latent;
- train with matched positive/negative token pairs;
- evaluate unseen-positive recall as a primary gate, including query/read
  action-response consistency.

This is the smallest mechanism that directly addresses the observed zero
unseen-label recall. Attack generation, utility/value heads, planning, and
large-scale data expansion remain unauthorized until that open-vocabulary
gate passes.

## Hashes

- Hard-view dataset: `5ce08331f89f7e10da7512c98a26f44a7748bc8e3e755a38cf5c69edda6a4323`
- Metrics: `365def3161d381d3aab5634a4316026e822e54023a8df332af412a3dd6745fcb`
- Gate: `0eeabe1f17cdb81eeedf82ca30a60a9918dd0cd55d5803b5349bef01103930de`
- Frozen experiment commit: `d60a3aa40d69ab12a70c706ff6a57d4fae017704`
