# AgentDojo v2 frozen monotonic risk calibration

Date: 2026-07-15

## Question

Can multi-seed posterior attack probabilities improve risk probability quality
without retraining or perturbing the DreamerV3 backbone, utility/preservation
heads, skill policy, or actor/critic?

## Method

The experiment reused the three validation-selected binary-risk checkpoints
from:

`/share/guozhix/wmagentattack/0715/dreamer_v2_final/formal_20260715_softprob_v1/binary_risk`

For each model seed, the checkpoint produced frozen train predictions. Only two
calibration parameters were fitted:

`calibrated_probability = sigmoid(softplus(a) * raw_logit + b)`

The positive scale makes the transform strictly monotonic, so binary risk
ordering and AUC are preserved. Training minimized configuration-level Brier
loss after averaging the five final trajectory probabilities in each attack
configuration. No Dreamer parameter received a gradient.

The fixed validation candidates were identity plus affine fits with identity
regularization 0, 1e-4, 1e-3, 1e-2, and 1e-1. The candidate with the lowest
mean group-aware validation objective over checkpoint seeds 7, 13, and 21 was
selected. Test data was not read before this choice.

- Validation job: `4476`.
- Frozen test job: `4479`.
- Selection SHA-256:
  `b2e8b0e00ae5ef0e641ea87355a31e7db6a5f694f1f6b154d86a303da1cc61e4`.
- Final summary SHA-256:
  `3708813b9d469f359573b3b556e223e8e860b31cee32c0c6717a856393ff032c`.
- Archive:
  `/share/guozhix/wmagentattack/0715/frozen_risk_calibration/formal_20260715_frozen_cal_v1`.

## Validation selection

| Candidate | Group objective | Group risk Brier | Risk AUC | Skill accuracy |
|---|---:|---:|---:|---:|
| Identity | 0.431830 +/- 0.020186 | 0.016313 | 0.912810 | 0.370033 |
| Reg 0 | **0.426363 +/- 0.019018** | **0.010846** | 0.912810 | 0.370033 |
| Reg 1e-4 | 0.426412 +/- 0.019048 | 0.010894 | 0.912810 | 0.370033 |
| Reg 1e-3 | 0.426838 +/- 0.019303 | 0.011321 | 0.912810 | 0.370033 |
| Reg 1e-2 | 0.429463 +/- 0.020267 | 0.013946 | 0.912810 | 0.370033 |
| Reg 1e-1 | 0.431612 +/- 0.020209 | 0.016094 | 0.912810 | 0.370033 |

Validation selected the unregularized affine fit (`reg0`). Its paired objective
deltas were -0.008057, -0.001390, and -0.006954 for seeds 7, 13, and 21. Thus
validation improved in all three checkpoints, with mean delta -0.005467 +/-
0.002918.

## Frozen test result

| Metric | Raw frozen checkpoint | Calibrated | Delta |
|---|---:|---:|---:|
| Group-aware objective | 0.324058 | **0.317010** | -0.007049 |
| Group risk Brier | 0.020194 | **0.013146** | -0.007049 |
| Group risk MAE | 0.097814 | **0.060907** | -0.036907 |
| All-step risk Brier | 0.068367 | **0.044549** | -0.023817 |
| Risk AUC | 0.898044 | 0.898044 | 0.000000 |
| Group utility Brier | 0.071187 | 0.071187 | 0.000000 |
| Group preservation Brier | 0.064355 | 0.064355 | 0.000000 |
| Next-skill accuracy | 0.326714 | 0.326714 | 0.000000 |

Group risk Brier improved by approximately 34.9%, group risk MAE by 37.7%,
and the full objective by 2.2%. The per-seed objective changes were:

| Seed | Raw test objective | Calibrated | Delta |
|---:|---:|---:|---:|
| 7 | 0.303872 | 0.294760 | -0.009112 |
| 13 | 0.303188 | 0.299745 | -0.003443 |
| 21 | 0.365114 | 0.356523 | -0.008591 |

The test improvement occurred in all three checkpoints. Mean paired delta was
-0.007049 with population standard deviation 0.002559.

All invariant checks passed exactly for every seed:

- Risk AUC delta: 0.0.
- Utility and preservation metric deltas: 0.0.
- Skill accuracy and top-3 accuracy deltas: 0.0.
- Dreamer/RSSM, reward, value, actor, critic, and policy parameters: unchanged.

## Fitted parameters

| Seed | Positive scale | Bias | Train group Brier before | After |
|---:|---:|---:|---:|---:|
| 7 | 0.494024 | -0.374152 | 0.007916 | 0.004655 |
| 13 | 0.640570 | -0.021013 | 0.009409 | 0.006834 |
| 21 | 0.425915 | -0.489083 | 0.007396 | 0.002653 |

All scales are below one, indicating that the raw risk logits were too sharp.
The negative biases in seeds 7 and 21 additionally shift risk probabilities
downward. This interpretation is limited to probability calibration; the
unchanged AUC confirms that ordering was neither improved nor damaged.

## Decision

**GO as a frozen inference-time risk calibrator.** This is the first v2
probability-label modification that improves validation and test in all three
seeds while preserving every non-risk result exactly.

It is not evidence that the world model learned better skills or attack
strategies. It improves the numerical reliability of risk probabilities only.
The previous binary Dreamer checkpoint remains the learned model, and the
calibrator is a two-parameter wrapper around its risk output.

## Next experiment

Integrate the selected per-checkpoint calibrator into downstream attack/skill
selection and repeat the frozen validation-selected decision evaluation:

1. Preserve an uncalibrated control using the same candidates and checkpoints.
2. Refit any probability threshold or scalarized Pareto coefficient on
   validation only.
3. Freeze the decision rule before test ASR/BUP evaluation.
4. Include a rank-only selector control. Because calibration is monotonic, the
   rank-only result should be exactly unchanged; any downstream gain must come
   from better probability-scale-sensitive decisions.

This tests whether improved probability quality has practical value for the
original world-model-guided selection goal.
