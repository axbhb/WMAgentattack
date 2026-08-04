# AgentDojo-v2 nested data-scale ablation and prospective selection

Date: 2026-07-15

## Conclusion

Expanding the training set from 25% to 100% produced a real but selective
improvement.  It improved risk ranking, first-step risk calibration, skill
prediction, and the primary low-budget prospective decision.  It did **not**
improve every probability head: utility and preservation transfer remained
non-monotonic, and the 25% model retained the best aggregate test loss.

The primary deployment-oriented result is the validation-selected,
first-step, calibrated ensemble with one attack configuration selected per
held-out user task:

- 25% data: ASR+BUP = 0.55.
- 50% data: ASR+BUP = 0.45.
- 100% data: ASR+BUP = 0.85.
- 100% minus 25%: +0.30; paired task bootstrap 95% interval [0.05, 0.65].
- The 100% result is above the random expectation of 0.60, but the one-sided
  task-balanced randomization result is p = 0.0857.  It is promising, not yet
  conventionally significant.

Thus, the answer to "did the expanded dataset help?" is **yes for the risk
model and low-budget attack selection, but no as a blanket statement about the
whole multi-head objective**.

## Fixed protocol

- Data subsets are nested and label-blind.
- Only attack-configuration count changes; all 12 clean training groups are
  retained at every scale.
- Every selected attack configuration retains all five victim-model seeds.
- All scales use the same 12 training user tasks, the same four validation
  tasks, the same four test tasks, and the same three model seeds (7, 13, 21).
- Architecture, loss weights, epoch budget, and validation checkpoint rule are
  unchanged.
- Decision recipes are selected on validation across the three model seeds and
  then frozen on test.
- The primary decision point is the first trajectory step, before the agent's
  subsequent actions and observations can enter history.
- Random comparisons use 200,000 task-balanced draws with the same number of
  selected configurations per held-out task.

| Scale | Attack configs | Attack episodes | Training steps |
|---|---:|---:|---:|
| 25% | 60 | 300 | 979 |
| 50% | 120 | 600 | 1,892 |
| 100% | 240 | 1,200 | 3,776 |

The 25% group set is a strict subset of the 50% group set, and both are subsets
of the full training set.  Each scale covers all 12 training tasks and all six
training attack families.

## Predictive test metrics before calibration

Values are means over the three checkpoint seeds.

| Scale | Group objective ↓ | Risk Brier ↓ | Utility Brier ↓ | Preservation Brier ↓ | Risk AUC ↑ | Skill accuracy ↑ |
|---|---:|---:|---:|---:|---:|---:|
| 25% | **0.3062** | 0.0258 | **0.0534** | **0.0515** | 0.8128 | 0.2983 |
| 50% | 0.3578 | 0.0452 | 0.0794 | 0.0634 | 0.8743 | 0.3212 |
| 100% | 0.3241 | **0.0202** | 0.0712 | 0.0644 | **0.8980** | **0.3267** |

The scale trend separates ranking from probability calibration:

- Risk AUC rises 0.8128 → 0.8743 → 0.8980.  The 100%-minus-25% AUC increase is
  +0.0853 and is positive for all three checkpoint seeds.
- Skill accuracy rises by +0.0284 on average from 25% to 100%, although one of
  the three checkpoint seeds regresses.
- Raw risk Brier is best at 100%, but the 50% run is badly mis-scaled despite
  having better AUC than 25%.
- Utility Brier worsens by +0.0178 from 25% to 100%, and this regression occurs
  in all three checkpoint seeds.
- Consequently, the aggregate test objective is not monotonic and remains
  best at 25%.

Validation metrics favored the 100% model on every main head, whereas test
utility and preservation did not transfer.  A likely explanation is the
intentional family shift: validation contains `contextual_holdout`, while test
contains `structured_holdout`.  This is an inference from the split design,
not proof of the causal mechanism.

## Frozen final-step probability calibration

Each scale independently selects identity or affine-logit regularization on
validation, then freezes it for test.

| Scale | Selected calibration | Test risk Brier ↓ | Test risk MAE ↓ | Test group objective ↓ |
|---|---|---:|---:|---:|
| 25% | reg1e-3 | 0.02305 | 0.09833 | **0.30337** |
| 50% | reg1e-4 | 0.03038 | 0.09353 | 0.34290 |
| 100% | reg0 | **0.01315** | **0.06091** | 0.31701 |

Calibration repairs much of the 50% risk-scale failure and makes the 100%
risk head clearly best, but it cannot fix utility or preservation because the
wrapper intentionally changes only risk.

## Strict prospective first-step selection

The table uses each scale's validation-selected first-step calibration and
validation-selected recipe.  ASR and BUP are empirical rates over the five
victim-seed repeats of each selected configuration.  `p(random ≥ observed)` is
the one-sided task-balanced randomization result.

| Scale | Budget/task | ASR | BUP | ASR+BUP | p(random ≥ observed) |
|---|---:|---:|---:|---:|---:|
| 25% | 1 | 0.050 | 0.500 | 0.550 | 0.6203 |
| 50% | 1 | 0.000 | 0.450 | 0.450 | 0.8697 |
| 100% | 1 | 0.250 | 0.600 | **0.850** | **0.0857** |
| 25% | 2 | 0.125 | 0.475 | 0.600 | 0.5210 |
| 50% | 2 | 0.125 | 0.525 | 0.650 | 0.3366 |
| 100% | 2 | 0.225 | 0.550 | **0.775** | **0.0688** |
| 25% | 4 | 0.150 | 0.4625 | 0.6125 | 0.4327 |
| 50% | 4 | 0.100 | 0.4875 | 0.5875 | 0.5729 |
| 100% | 4 | 0.150 | 0.500 | **0.650** | 0.2494 |

First-step validation risk Brier improves monotonically with scale:

| Scale | Raw first-step Brier ↓ | Calibrated first-step Brier ↓ |
|---|---:|---:|
| 25% | 0.03942 | 0.03154 |
| 50% | 0.03808 | 0.02832 |
| 100% | **0.03211** | **0.02728** |

For the primary one-per-task budget, the full-minus-25% ASR+BUP contrast is
+0.30.  Three held-out tasks improve and one ties.  The paired task bootstrap
interval is [0.05, 0.65], but only four test tasks exist, so this interval must
not be treated as definitive population-level evidence.  At budget two, the
contrast is +0.175 with interval [0.025, 0.40].  At budget four, the contrast is
only +0.0375 with an interval crossing zero.

The individual-checkpoint mean at budget one increases 0.5833 → 0.7667 →
0.8000, while the 50% ensemble drops to 0.45.  This is evidence that the
intermediate model's averaging/calibration is unstable, not evidence of a
smooth scaling law.

## Counterevidence: final-step selection is not prospective

The first downstream run aggregated the final step of every trajectory.  It
produced apparently strong results (for example, full-model budget-one
ASR+BUP = 1.20, randomization p = 0.000125), but the final-step history already
contains the agent's preceding interaction.  It is therefore a retrospective
prediction upper bound, not a valid pre-execution attack selector.

The corrected first-step result is much smaller: calibrated budget-one
ASR+BUP = 0.85 with p = 0.0857.  Only this corrected result should be used for
method claims.

## What remains unresolved

1. Test contains only four user tasks.  None of the prospective comparisons
   against random reaches p < 0.05, and three budgets were inspected.
2. More data improves the risk head but not the utility/value heads.  The
   current shared-latent multi-head objective does not guarantee equal transfer
   under a new attack family.
3. The first-step encoder is pre-response, but it is still constructed from a
   realized trace.  A deployable planner must construct the same state directly
   from an unexecuted manifest candidate, including payload, injection vector,
   optimizer, knowledge level, and target-tool depth.
4. The current result ranks complete attack configurations.  It is not yet a
   full Dreamer imagination loop that proposes a new attack/skill sequence and
   predicts its consequences before any AgentDojo execution.

## Recommended next experiment

Keep the 100% dataset for the risk/planner path, but do not simply add more
repeats to the same configurations.  The next informative test is grouped
five-fold out-of-fold evaluation over all 20 user tasks:

1. Pre-register one configuration per task as the primary budget and freeze
   the current four recipes.
2. Compare nested 25% and 100% training data in every fold with the same three
   checkpoint seeds.
3. Fit all calibration on fold-train data and select recipes only on fold-val.
4. Pool one out-of-fold prospective decision for every task, increasing the
   evaluation unit from four to 20 tasks without collecting new 70B traces.
5. In parallel, train a separate group-aware utility head or frozen utility
   calibrator; judge it by structured-holdout Brier and BUP, not by risk AUC.

Only after the 20-task out-of-fold result confirms the low-budget gain should
new 70B collection focus on additional user tasks and genuinely new attack
families.  Task/family coverage is now more valuable than additional seeds for
the existing 400 configurations.

## Artifacts

Remote archive:

`/share/guozhix/wmagentattack/0715/data_scale_ablation/formal_20260715_nested_v1`

Important files:

- `final_summary.json`: complete aggregate, per-seed deltas, and paired task
  bootstrap contrasts.
- `datasets/summary.json`: nested-subset and completeness audit.
- `downstream/pct25/result.json`, `downstream/pct50/result.json`: frozen
  first-step selection results.
- `downstream/*/randomization.json`: 200,000-draw randomization tests.
- `calibration/*/selection.json` and `final_summary.json`: strict validation
  selection and frozen test calibration.

Repository mapping:

- `scripts/70_evaluate_v2_downstream_selection.py`: retrospective/prospective
  selector, decision-time calibration, validation selection, and frozen test.
- `scripts/71_build_v2_size_ablation.py`: nested label-blind data subsets.
- `scripts/72_randomization_test_v2_selection.py`: task-balanced null tests.
- `scripts/73_summarize_v2_size_ablation.py`: final cross-scale aggregation.
- `scripts/server/run_agentdojo_v2_size_ablation_*.sbatch`: fixed training and
  calibration jobs.
- `scripts/server/run_agentdojo_v2_prospective_selection.sbatch`: first-step
  decision job.
- `tests/test_v2_scale_and_selection.py`: ordering, per-task budget, and metric
  regression tests.

Local verification: eight relevant tests passed.
