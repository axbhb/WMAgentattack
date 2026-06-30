# 0701 Utility Ranking and Calibration Round

## Research budget executed

This round optimized the utility/value stack rather than only sweeping
selectors:

1. Add a validation-trained logistic calibration script for cached utility
   scores.
2. Test calibrated utility scores on the existing value-head Dreamer candidate
   cache.
3. Add a same-task pairwise utility ranking loss to the SheepRL Dreamer adapter.
4. Train a new ranking-regularized value-head Dreamer on the server.
5. Evaluate step-level metrics, candidate-level calibration, Pareto selection,
   and fixed validation-threshold transfer.
6. Search for counterevidence against each improvement.
7. Archive all generated artifacts under `/share/guozhix/wmagentattack/0630`.

## Implemented repo changes

| Change | Repo location |
|---|---|
| Pairwise final-utility ranking loss | `src/wmagentattack/dreamer_world_model.py` |
| CLI flags for ranking loss scale and margin | `scripts/15_train_dreamer_world_model.py` |
| Logistic utility calibration for candidate caches | `scripts/19_calibrate_utility_scores.py` |
| Server ranking experiment job | `scripts/server/run_sheeprl_dreamer_value_rank_e5.sbatch` |

Archive:

```text
/share/guozhix/wmagentattack/0630/dreamer_sheeprl/utility_ranking_round
```

## Method

For each trajectory sequence, the model now stores:

```text
group_id = domain | task_id
final_utility = final task_success
```

Within each training batch, if two sequences share the same `group_id` and have
different final utility labels, the model applies a margin ranking loss:

```text
final_utility_logit(utility-preserving trajectory)
  >
final_utility_logit(utility-breaking trajectory)
```

The job used:

```text
--utility-ranking-loss-scale 1.0
--utility-ranking-margin 0.2
```

Calibration uses validation candidates only and appends:

- `calibrated_utility_score`
- `calibrated_final_utility_score`
- `calibrated_min_utility_score`

## Step-level metrics

| Model | Risk AUC | Utility AUC | Brier | Next-skill top-3 |
|---|---:|---:|---:|---:|
| Value-head Dreamer | 0.8482 | 0.8655 | 0.1672 | 0.6362 |
| Ranking value-head Dreamer | 0.8396 | 0.8687 | 0.1696 | 0.6642 |
| sklearn | 0.8722 | 0.9795 | 0.1116 | 0.8694 |

Interpretation: ranking loss slightly improved step-level utility AUC and
next-skill top-3, but hurt risk AUC and Brier. It is not a uniform win at the
step-prediction level.

## Candidate-level calibration

Validation-trained calibration changed probability quality, not ranking AUC.

| Model/key | Target Brier before | Target Brier after | Target AUC |
|---|---:|---:|---:|
| value `final_utility_score` | 0.0857 | 0.0720 | 0.7799 |
| ranking `utility_score` | 0.0703 | 0.0700 | 0.8742 |
| ranking `final_utility_score` | 0.0808 | 0.0718 | 0.8209 |
| ranking `min_utility_score` | 0.0711 | 0.0720 | 0.8506 |

Counterevidence: calibration improved Brier for final utility, but did not
change AUC, so it cannot fix bad candidate ordering by itself.

## Selection results

World-model-top, averaged over seeds 7/13/21.

| Split | Model / selector | K | Utility key | Threshold | ASR | BUP | ASR+BUP |
|---|---|---:|---|---:|---:|---:|---:|
| test | value-head Pareto best observed | 16 | final utility q0.80 | q0.80 | 0.3125 | 0.3125 | 0.6250 |
| test | value-head val-selected | 24 | min utility q0.80 | q0.80 | 0.2083 | 0.3333 | 0.5417 |
| test | ranking weighted baseline | 16 | selection score | none | 0.4375 | 0.4375 | 0.8750 |
| test | ranking Pareto best observed | 16 | final utility q0.70 | q0.70 | 0.3750 | 0.5000 | 0.8750 |
| test | ranking val-selected by ASR+BUP | 16 | utility q0.90 | q0.90 | 0.2500 | 0.3125 | 0.5625 |
| test | ranking val-selected by BUP | 24 | min utility q0.90 | q0.90 | 0.2500 | 0.3333 | 0.5833 |
| test | sklearn rollout | 16 | n/a | n/a | 0.4375 | 0.4375 | 0.8750 |
| test | sklearn rollout | 24 | n/a | n/a | 0.3333 | 0.4583 | 0.7917 |

Fixed validation-threshold transfer:

| Test selector | K | Utility key | Fixed threshold | ASR | BUP | ASR+BUP |
|---|---:|---|---:|---:|---:|---:|
| ranking Pareto | 16 | final utility | 0.228896 | 0.3125 | 0.5625 | 0.8750 |
| ranking Pareto | 24 | final utility | 0.228896 | 0.2500 | 0.4583 | 0.7083 |
| ranking Pareto | 16 | utility | 0.189925 | 0.3125 | 0.4375 | 0.7500 |
| ranking weighted baseline | 16 | selection score | none | 0.4375 | 0.4375 | 0.8750 |

## Findings

1. Pairwise utility ranking is the strongest improvement so far for utility
   ordering. Candidate-level test AUC for `utility_score` reached 0.8742,
   compared with 0.7934 for the previous value-head candidate cache.

2. The ranking model produced the best Dreamer-family BUP seen so far:
   fixed-threshold final-utility Pareto at K=16 reached BUP 0.5625 and
   ASR+BUP 0.8750.

3. The stronger result is not yet a clean val-selected headline. The strict
   val-selected ranking configurations transfer to test at ASR+BUP 0.5625 or
   0.5833, below the diagnostic fixed-threshold best.

4. Ranking also improves the weighted baseline substantially: ranking weighted
   K=16 reaches ASR 0.4375 / BUP 0.4375, matching sklearn K=16 ASR+BUP.

5. sklearn remains the broader counterexample at K=24: sklearn BUP 0.4583 and
   ASR+BUP 0.7917 still beat the strict val-selected Dreamer ranking result.

## Conclusion

This is a meaningful optimization step. The pairwise ranking loss changes the
utility critic in the right direction, unlike calibration alone. It gives the
first Dreamer-family result that matches sklearn at K=16 by ASR+BUP, and it can
exceed sklearn K=16 BUP under a fixed final-utility threshold. However, the
selection protocol is still not stable enough: validation-selected quantile
settings do not consistently pick the strongest test threshold.

## Next recommended experiment

1. Train a small grid over ranking margin/scale:
   - scale `{0.5, 1.0, 2.0}`
   - margin `{0.1, 0.2, 0.3}`
2. Select the final-utility fixed threshold on validation explicitly, then
   freeze the numeric threshold for test.
3. Report only this frozen-threshold test result as the headline.
