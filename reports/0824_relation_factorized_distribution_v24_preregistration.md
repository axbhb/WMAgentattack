# Relation-factorized semantic distribution v24 preregistration

## Frozen question

The v23 frozen E5 prototype sharply improved unseen-label NLL but missed the task-disjoint recall floor and did not materially beat the independent-ID recall control. This cycle changes one mechanism: a single full-label embedding is replaced by relation-factorized semantic descriptions, similarity-distribution supervision, and train-only support-set selection.

The candidate does not retrieve whole successor transitions. Retrieval v16 already showed that semantically similar states are not causally equivalent. It also does not anonymize domain concepts: relational JEPA v7 showed that removing hotel, flight, rating, price, and related concepts damages Travel transfer.

## Data sufficiency counterevidence

The frozen 121-row dataset has only 16 task-disjoint unseen positive occurrences per seed. Their training-fold atom support is:

- exact entity: 0%;
- exact field: 18.75%;
- exact kind/category-kind: 75%;
- entity lexical-part overlap: 43.75%.

Therefore this experiment tests semantic relation transfer, not retrieval of observed causal exemplars. A failure closes architecture-only open-vocabulary search on this dataset and redirects the next loop to explicit label-disjoint branching support collection.

## Candidate

Each effect label is encoded by five frozen E5 channels: full description, category, entity, field, and kind/value. Each normalized action is encoded by full action, tool, and argument channels. Deterministic weighted aggregation feeds the retained zero-initialized semantic transition model.

Training adds a similarity-distribution loss: positives on fitted labels induce a soft target distribution over semantically related labels. At evaluation, fitted-label probabilities may diffuse through a fixed relation kernel into unseen candidates. The support weight and decision threshold are selected only from hashed label-heldout targets within the outer training fold.

## Fixed controls and budget

- Fresh arms: fixed v21, relation-E5 raw v24, relation-support-set v24.
- Immutable controls: v23 raw/calibrated E5, v22 independent candidate IDs, v22 long-action gate, and the frozen data-design gate.
- Splits and seeds are unchanged.
- Budget: 15 fixed-v21 fits, 15 inner relation fits, 15 outer relation fits; 45 total fits and 45 metric rows.
- No post-result rerun or threshold change is allowed.

## Gate

The support-set candidate must:

- find a feasible train-only support rule in every split/seed unit; a conservative runtime fallback is an automatic gate failure;
- reach task-disjoint unseen recall >=0.55 and improve by >=0.02 over v23 raw;
- keep unseen positive NLL within +0.1 of v23 raw;
- reach tool/source unseen recall floors;
- keep task unseen false-positive rate <=0.05 and precision >=0.20;
- keep mean predicted unseen set size <=2 times the true set size plus 0.5;
- preserve seen recall, one-step BCE, rollout BCE, and query/read recall;
- improve recall over relation-E5 raw by >=0.01 without worsening NLL by more than 0.1;
- satisfy cache, parameter, reproducibility, budget, and runtime-integrity checks.

Only a complete GO can authorize the frozen 96-episode three-source data smoke. No attack generation, medium/large data generation, planner, Dreamer, or large world-model training is authorized in this cycle.
