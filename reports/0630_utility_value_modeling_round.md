# 0630 Utility/Value Modeling Round

## Research budget executed

This round used a fixed budget before summarizing:

1. Cross-check the prior evidence: Dreamer variants improved attack discovery
   but consistently lagged sklearn on BUP / utility preservation.
2. Search for a method-level explanation against DreamerV3 and SheepRL:
   stronger utility should be treated as value/return modeling, not only a
   scalar weight in the selector.
3. Add a cheap constrained utility scorer and test it on the existing KL-branch
   candidate cache as counterevidence.
4. Integrate a model-side final-utility value head into the SheepRL Dreamer
   adapter.
5. Train a new value-head Dreamer on the server and run val/test candidate
   cache sweeps.
6. Compare against KL branch and the previous sklearn counterexample.
7. Archive the results under `/share/guozhix/wmagentattack/0630`.

## Implemented repo changes

| Change | Repo location |
|---|---|
| Selection objectives: weighted vs utility-constrained shortfall penalty | `scripts/11_select_world_model_agentdojo_pairs.py` |
| Selectable utility signal: mean, final, or min utility score | `scripts/11_select_world_model_agentdojo_pairs.py`, `scripts/13_run_selection_grid.py` |
| Offline sweep presets for constrained, terminal-constrained, and min-constrained utility | `scripts/17_tune_selection_weights.py` |
| Final-utility value head trained on terminal trajectory utility | `src/wmagentattack/dreamer_world_model.py` |
| Value-head training CLI flag | `scripts/15_train_dreamer_world_model.py` |
| Server training/evaluation/sweep job | `scripts/server/run_sheeprl_dreamer_value_e5.sbatch` |

## Method design

The prior selector used a linear score:

```text
risk + mean_risk + utility + target probability + target reached
```

This is too soft for AgentDojo because an attack that breaks the benign task is
not a good attack. The new utility/value design has two layers:

1. A selector-side constrained objective:

```text
score = risk_target_score - penalty * max(0, utility_threshold - utility_score)
```

2. A model-side final utility value head:

```text
final_utility_head(latent_state_at_terminal_step) -> P(final task succeeds)
```

During RSSM branch imagination the model now exposes:

- `utility_score`: mean predicted utility over imagined steps.
- `min_utility_score`: worst predicted utility over imagined steps.
- `final_utility_score`: predicted terminal utility/value at the end of the
  imagined rollout.

The selector can use any of these via `--utility-score-key`.

## Server run

- Job: `4152 / wma-dreamer-value-e5`
- Model: `/share/guozhix/WMagentattack/artifacts/agentdojo_full_llama31_8b_dreamer_world_model_value_e5`
- Archive: `/share/guozhix/wmagentattack/0630/dreamer_sheeprl/utility_value_round`
- Candidate caches:
  - `dreamer_value_val_candidate_cache.json`
  - `dreamer_value_test_candidate_cache.json`
- Sweeps:
  - `dreamer_value_val_utility_value_sweep.json`
  - `dreamer_value_test_utility_value_sweep.json`

## Step-level metrics

| Model | Risk AUC | Utility AUC | Brier | Next-skill top-3 |
|---|---:|---:|---:|---:|
| Tuned Dreamer | 0.8519 | 0.8352 | 0.1532 | 0.6716 |
| KL-branch Dreamer | 0.8336 | 0.8570 | 0.1533 | 0.6549 |
| Value-head Dreamer | 0.8482 | 0.8655 | 0.1672 | 0.6362 |
| sklearn | 0.8722 | 0.9795 | 0.1116 | 0.8694 |

Interpretation: the final-utility value head improved utility AUC over the
KL-branch Dreamer, but it still trails sklearn by a wide margin and slightly
worsens Brier/top-3 next-skill metrics.

## Selection results

World-model-top, averaged over seeds 7/13/21.

| Split | Backend / scorer | K | ASR | BUP | ASR+BUP |
|---|---|---:|---:|---:|---:|
| val | KL branch best shown | 32 | 0.1875 | 0.2188 | 0.4063 |
| val | Value-head, target-light utility | 16 | 0.1875 | 0.3125 | 0.5000 |
| val | Value-head, terminal constrained | 24 | 0.1667 | 0.2917 | 0.4583 |
| test | KL branch | 16 | 0.5000 | 0.2500 | 0.7500 |
| test | KL branch | 24 | 0.3750 | 0.2917 | 0.6667 |
| test | KL branch | 32 | 0.3125 | 0.2188 | 0.5313 |
| test | Value-head, default/terminal/min constrained | 32 | 0.2812 | 0.3125 | 0.5938 |
| test | Value-head, constrained | 24 | 0.2917 | 0.2917 | 0.5833 |
| test | sklearn rollout | 16 | 0.4375 | 0.4375 | 0.8750 |
| test | sklearn rollout | 24 | 0.3333 | 0.4583 | 0.7917 |
| test | sklearn rollout | 32 | 0.3125 | 0.3750 | 0.6875 |

## Findings

1. Utility/value modeling helped BUP. On test K=32, BUP improved from KL
   branch 0.2188 to value-head 0.3125.

2. It did not dominate by ASR+BUP. KL branch K=16 still has much higher attack
   yield: ASR+BUP 0.7500 versus the best value-head test result 0.5938.

3. Validation selection did not transfer cleanly. The val-best value-head
   setting by ASR+BUP was `target_light_utility, K=16` with val ASR+BUP 0.5000,
   but test ASR+BUP fell to 0.3750.

4. Terminal/min constrained scoring did not select a different top set in the
   strongest test K=32 case; it changed the diagnostic utility signal but tied
   the observed result. This is a useful counterexample: just adding the new
   score fields is not enough unless the value head has enough resolution to
   reorder candidates.

5. sklearn remains the main counterevidence. Even after value-head training,
   sklearn rollout has substantially higher BUP on test K=16/24/32.

## Conclusion

The stronger utility/value modeling direction is valid but incomplete. The
final-utility value head gives a real BUP gain, especially at K=32, but the
Dreamer adapter still loses either ASR or BUP relative to the sklearn rollout.
The next improvement should not be another scalar sweep. It should make the
world model learn a sharper utility boundary and use a validation-selected
Pareto policy.

## Recommended next changes

| Goal | Concrete repo change |
|---|---|
| Sharpen final utility value | Train pairwise/ranking loss between utility-preserving and utility-breaking branches within the same user task. |
| Avoid scalar overfit | Add Pareto selection: maximize risk among candidates above a calibrated utility quantile, selected only on val. |
| Improve value signal resolution | Calibrate `final_utility_score` with isotonic/logistic calibration on validation states. |
| Improve representation | Replace hashed bag-of-words with frozen embedding features cached per step. |
| Make Dreamer more faithful | Add reward/continue/value objectives over imagined rollouts rather than only supervised heads. |

## External cross-check

DreamerV3 motivates learning compact latent dynamics together with reward,
continue, and value-style prediction under imagined rollouts. SheepRL's
DreamerV3 implementation also separates world-model losses from actor/critic
value losses. This supports the repo conclusion: for AgentDojo, utility should
be a first-class terminal value/constraint, not only a positive term in the
selection score.
