# Pretrained semantic hybrid v23: preregistration

Date: 2026-08-24

This cycle changes only the failed open-vocabulary effect representation. The
retained v21 direct head remains responsible for labels observed in an outer
training fold. A frozen `intfloat/e5-base-v2` prototype scorer is used only for
labels with zero positive support in that fold. The v22 recurrent action model
and its immutable H1-H5 metrics are not refit.

E5 consumes canonical effect-token descriptions and current normalized action
descriptions only. It never consumes task/source identifiers, outcome labels,
utility/security labels, future state, or confirmation labels. A deterministic
64-dimensional SVD is fit without supervision on the effect/action embeddings.

Calibration is label-disjoint and train-only: 20% of labels are selected by a
fixed SHA256 rule, masked from an inner semantic fit, and used to choose one
temperature and unseen-label bias from a frozen grid. The outer semantic model
is then refit on all labels observed in the outer training fold. Matched hard
negatives are the nearest fitted E5 prototypes, not confirmation examples.

The budget is exactly 45 fits: 15 fixed-v21 reproductions, 15 inner semantic
fits, and 15 outer semantic fits. Acceptance requires improvement over the
strong independent-ID counterexample on both unseen recall and NLL, while
preserving v21 seen, one-step, query/read, and v19 rollout surfaces. If and only
if every clause passes, the immutable v22 long-action clauses are recomposed
with the new effect-rollout clause and the already frozen 96-episode three-source
smoke is authorized. No medium/large generation is authorized in this cycle.

Primary evidence mapped to this mechanism:

- https://arxiv.org/abs/2602.05842: align textual world models in a pretrained
  embedding space rather than demanding token-level reconstruction.
- https://arxiv.org/abs/2608.04653: retain counterfactual/action consistency and
  do not accept predictive shortcuts.
- https://proceedings.mlr.press/v267/mehta25a.html: use a task-agnostic
  pretrained representation for zero-shot prediction.
- https://proceedings.mlr.press/v222/ji24c.html: explicitly control generalized
  zero-shot calibration and overfitting to seen labels.
