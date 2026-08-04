# 0723 task-macro dynamics ablation

## Decision

`SHORTCUT_NOT_RULED_OUT_CLEAN_GATE_BLOCKED`

This is a development-only mechanism audit on four validation and four test tasks. It cannot establish formal task generalization and cannot override the failed clean gate.

## Primary task-macro comparisons

| Split | Full NLL | Best Markov NLL | NLL gain | Full free edit | Best Markov free edit | Edit gain |
|---|---:|---:|---:|---:|---:|---:|
| validation | 1.2923 | 1.3668 | +0.0745 | 0.3573 | 0.3642 | +0.0068 |
| test | 1.9538 | 2.7141 | +0.7602 | 0.4491 | 0.4361 | -0.0130 |

## Frozen gates

- representation_integrity: `TRUE`
- task_macro_predictive_and_free_vs_markov: `FALSE`
- event_identity_beyond_length: `TRUE`
- attack_semantics_increment: `FALSE`
- prefix_content_beyond_static_length_random_markov: `FALSE`
- prefix_order_beyond_shuffled_multiset: `TRUE`
- seed_stability: `TRUE`
- outcome_and_ranking_seed_stability: `FALSE`
- per_task_direction: `FALSE`
- clean_eligibility_gate: `FALSE`

## Prefix-value counterfactual gains

### validation

- observed vs static_semantic: NLL gain +0.2452
- observed vs shuffled: NLL gain +0.0310
- observed vs length_only: NLL gain +0.0581
- observed vs random_length_matched: NLL gain +0.0524
- observed vs markov_length_matched: NLL gain +0.0512
- observed vs semantic_markov_length_matched: NLL gain +0.0034

### test

- observed vs static_semantic: NLL gain +0.1912
- observed vs shuffled: NLL gain +0.0021
- observed vs length_only: NLL gain +0.0288
- observed vs random_length_matched: NLL gain +0.0183
- observed vs markov_length_matched: NLL gain -0.0068
- observed vs semantic_markov_length_matched: NLL gain +0.0062

## Boundary and next action

No attack data, H2 attack planning, selective-deployment claim, or Dreamer training is authorized. The next admissible stage is clean-only expansion of independent tasks/victims, while retaining these frozen task-level metrics and negative controls.
