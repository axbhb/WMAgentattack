# 0701 Pareto Utility Selection Round

## Research budget executed

This round continued the utility/value line with a fixed budget:

1. Search arXiv for relevant constrained / Pareto / safe world-model methods.
2. Implement an inference-free Pareto / epsilon-constraint selector over cached
   world-model candidates.
3. Run the selector on value-head Dreamer val/test caches.
4. Run the same selector on KL-branch Dreamer caches as counterevidence.
5. Strictly check val-selected settings on test.
6. Re-run test with fixed thresholds copied from validation to avoid relying on
   split-specific quantile values.
7. Archive all outputs under `/share/guozhix/wmagentattack/0630`.

## arXiv method scan

| Source | Relevant idea | Mapping to this repo |
|---|---|---|
| [SafeDreamer: Safe Reinforcement Learning with World Models](https://arxiv.org/abs/2307.07176) | Add safety/cost constraints into Dreamer-style world-model planning with Lagrangian methods. | Treat utility loss as a cost/constraint, not just a positive reward term. |
| [Constrained Policy Optimization](https://arxiv.org/abs/1705.10528) | Formulate reward maximization under explicit behavioral/safety constraints. | Select attack-risk candidates only inside a predicted utility-feasible region. |
| Pareto-front MORL / epsilon-constraint methods | Decompose Pareto-front discovery into constrained single-objective problems. | Convert attack-vs-utility tradeoff into `maximize risk_target_score subject to utility >= threshold`. |
| Constrained MORL | Optimize multi-objective policies while satisfying explicit constraints / Pareto conditions. | Use utility quantiles as feasible-set constraints and report the ASR/BUP Pareto tradeoff. |

## Implemented repo changes

| Change | Repo location |
|---|---|
| Pareto / epsilon-constraint selection script | `scripts/18_pareto_utility_selection.py` |
| Cached value-head Pareto sweep | remote artifact `dreamer_value_{val,test}_pareto_utility_selection.json` |
| Cached KL-branch Pareto sweep | remote artifact `dreamer_kl_branch_{val,test}_pareto_utility_selection.json` |
| Validation-threshold fixed test check | remote artifact `dreamer_value_test_pareto_valfixed_selection.json` |

Archive:

```text
/share/guozhix/wmagentattack/0630/dreamer_sheeprl/pareto_utility_round
```

## Method

For each candidate, define:

```text
risk_target_score =
    risk_score
  + 0.3 * rollout_mean_risk_score
  + 0.3 * target_skill_probability
  + 0.2 * rollout_target_reached
```

Then select:

```text
maximize risk_target_score
subject to predicted_utility >= threshold
```

The utility signal can be:

- `utility_score`
- `final_utility_score`
- `min_utility_score`

Thresholds were swept over utility-score quantiles `{0.50, 0.60, 0.70, 0.80,
0.90}`. For a stricter cross-check, selected validation thresholds were also
applied to test as fixed numeric thresholds.

## Results

World-model-top, averaged over seeds 7/13/21.

| Split | Backend / selector | K | Utility key | Threshold | ASR | BUP | ASR+BUP |
|---|---|---:|---|---:|---:|---:|---:|
| val | value-head Pareto | 24 | min utility | q0.80 | 0.2083 | 0.2917 | 0.5000 |
| test | value-head Pareto, val-selected | 24 | min utility | q0.80 | 0.2083 | 0.3333 | 0.5417 |
| test | value-head Pareto, best observed | 16 | final utility | q0.80 | 0.3125 | 0.3125 | 0.6250 |
| test | value-head weighted baseline | 32 | selection score | none | 0.2812 | 0.3125 | 0.5938 |
| val | KL Pareto best | 24 | utility | q0.90 | 0.2083 | 0.2083 | 0.4167 |
| test | KL Pareto, val-selected | 24 | utility | q0.90 | 0.4167 | 0.2083 | 0.6250 |
| test | KL weighted baseline | 24 | selection score | none | 0.3750 | 0.2917 | 0.6667 |
| test | sklearn rollout | 24 | n/a | n/a | 0.3333 | 0.4583 | 0.7917 |
| test | sklearn rollout | 32 | n/a | n/a | 0.3125 | 0.3750 | 0.6875 |

Fixed-threshold cross-check using validation-derived numeric thresholds:

| Test selector | K | Utility key | Fixed threshold | ASR | BUP | ASR+BUP |
|---|---:|---|---:|---:|---:|---:|
| value-head Pareto | 24 | min utility | 0.135542 | 0.2083 | 0.3333 | 0.5417 |
| value-head Pareto | 32 | utility | 0.135542 | 0.2812 | 0.3125 | 0.5938 |
| value-head Pareto | 32 | final utility | 0.135542 | 0.2812 | 0.3125 | 0.5938 |

## Findings

1. The value-head model supports useful utility-constrained selection. The
   validation-selected Pareto setting transfers to test with BUP 0.3333 at K=24,
   improving over the value-head weighted K=24 BUP 0.2500 and KL-val-selected
   BUP 0.2083.

2. The Pareto selector improves BUP but sacrifices ASR. The val-selected
   value-head Pareto setting reaches BUP 0.3333 but ASR only 0.2083.

3. The best observed value-head Pareto point reaches ASR+BUP 0.6250, slightly
   higher than value-head weighted K=32 at 0.5938, but this is test-selected
   and should be treated as diagnostic rather than the headline result.

4. The KL branch is a counterexample: applying Pareto constraints to KL branch
   does not improve the final tradeoff. Its val-selected Pareto test BUP is
   only 0.2083, worse than the KL weighted baseline BUP 0.2917.

5. sklearn remains the strongest BUP counterexample. The current Dreamer-family
   utility/value stack still does not match sklearn rollout utility
   preservation.

## Conclusion

The arXiv scan supports the direction: attack selection should be framed as a
constrained optimization / Pareto problem. The experiment shows that this only
helps when the world model has a usable utility value signal. The value-head
Dreamer improves utility-constrained selection, but the utility critic is still
too weak to dominate sklearn.

## Next recommended experiment

The next useful step is not another selector sweep. It should improve the
utility critic itself:

1. Add pairwise/ranking utility loss within the same user task:

```text
utility_preserving branch > utility_breaking branch
```

2. Calibrate `utility_score`, `final_utility_score`, and `min_utility_score` on
   validation using a small logistic/isotonic calibration layer.

3. Then rerun this exact Pareto selector, selecting hyperparameters on val and
   reporting test once.
