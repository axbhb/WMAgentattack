# 0713 Outer-crossfit Dreamer and text-anchored correction

## Decision

The current world-assisted candidate selector remains **NO-GO**.  Removing
upstream task leakage eliminates the apparent validation advantage of the
Dreamer features.  The deployable method under the new safety gate is therefore
`text_pointwise`.

The text-anchored correction is a safer research candidate: it lets the
outer-crossfit model change only attack ordering while preserving text utility
ordering and text probabilities.  It avoids the large utility regression seen
in the previous headwise method, but it produces zero attack-order gain on the
grouped validation tasks and therefore does not pass the predeclared gain gate.

## Fixed research budget and completion

- Outer folds: 4 suite-balanced task folds over 68 grouped-train user tasks.
- Seeds per fold: 7, 13, and 21.
- Dreamer checkpoints: 12; all 12 completed 30 epochs.
- OOF candidate pairs: 664 per seed/view, covering all 68 tasks exactly once as
  an outer-held task.
- Scoring views: clean-prefix rollout and injection-conditioned rollout.
- Llama-70B replay outcomes added: 0.
- Slurm job: `4386`, completed without OOM, traceback, or failed array member.
- Fold-manifest SHA-256:
  `da58322b18913aebcfef7558255c74ff81d4117a4b5fdd5172900e8aaa540cf4`.

The original finalizer checked an empty `DONE` sentinel with `-s`, so it could
never observe completion.  It was corrected to `-f`, covered by a regression
test, and the merge/evaluation then completed automatically.

## Training-health cross-check

All checkpoints have 5,952,588 parameters and reached the full epoch budget.
Across the 12 outer-held evaluations:

| Held-step metric | Mean | Std | Min | Max |
|---|---:|---:|---:|---:|
| Next-skill accuracy | 0.3523 | 0.0526 | 0.2755 | 0.4150 |
| Next-skill top-3 accuracy | 0.6201 | 0.0419 | 0.5310 | 0.6876 |
| Risk AUC | 0.8803 | 0.0597 | 0.7627 | 0.9528 |
| Risk Brier | 0.1123 | 0.0308 | 0.0660 | 0.1618 |
| Binary utility AUC | 0.6743 | 0.0587 | 0.5634 | 0.7380 |
| Continuous utility Brier | 0.0431 | 0.0110 | 0.0284 | 0.0651 |
| Preservation Brier | 0.0716 | 0.0149 | 0.0415 | 0.0895 |

These values rule out a simple failed-training explanation.  They are
step-level held-task metrics, however, and do not establish candidate-level
ranking quality.

## Leakage-safe grouped validation

The strongest text baseline remains `text_pointwise` with primary pairwise
accuracy 0.7104 (attack 0.8088, utility 0.6119).

| Training feature provenance | Selected world-assisted candidate | Primary | Attack | Utility | Gate |
|---|---|---:|---:|---:|---|
| Dreamer trained on all 68 train tasks | `world_pairwise_c0p03` | 0.6130 | 0.8603 | 0.3657 | NO-GO |
| True task-out-of-model outer-crossfit | `text_borda_alpha_0p25` | 0.5872 | 0.8088 | 0.3657 | NO-GO |

The world-only grouped-OOF probability Brier also deteriorates after leakage is
removed:

| Head | Full-train Dreamer | Outer-crossfit Dreamer | Change |
|---|---:|---:|---:|
| Attack | 0.0499 | 0.0831 | +0.0332 |
| Utility | 0.0208 | 0.0382 | +0.0173 |

Thus the earlier high attack ordering (0.8603) was partly supported by upstream
task exposure.  It should not be treated as evidence of cross-task world-model
generalization.

## Text-anchored mechanism correction

The prior world-assisted candidates replaced utility ordering with clean
Dreamer utility.  This was harmful on the seven ordinary remaining tasks:
utility pairwise accuracy was 0.26 versus 0.52 for text.

The corrected method is:

1. attack rank: validation-frozen outer-crossfit
   `text_borda_alpha_0p25`;
2. utility rank: `text_pointwise`;
3. attack and utility probabilities: `text_pointwise`.

On grouped validation, the corrected method is exactly tied with text:

- primary: 0.7104 versus 0.7104;
- attack: 0.8088 versus 0.8088;
- utility: 0.6119 versus 0.6119;
- attack mean task difference: 0.0000, 95% CI [0.0000, 0.0000].

It passes non-inferiority and exact-anchor checks but fails the required attack
gain of at least 0.03.  The implemented gate therefore falls back to text.

## Fresh-outcome cross-check (post-hoc only)

All 15 test-task outcomes had already been inspected before this correction;
these numbers cannot be presented as a new confirmation result.

| Cohort | Unsafe outer candidate | Text | Text-anchored correction | Correction minus text (task bootstrap) |
|---|---:|---:|---:|---:|
| Prior enriched 8 tasks | 0.7511 | 0.6645 | 0.6853 | +0.0238, CI [0.0000, 0.0556] |
| New remaining 7 tasks | 0.5467 | 0.6767 | 0.6767 | 0.0000, CI [0.0000, 0.0000] |
| All 15 tasks | 0.6301 | 0.6616 | 0.6755 | +0.0128, CI [0.0000, 0.0357] |

The correction removes the ordinary-task utility failure and retains a small
descriptive attack gain in the enriched stress cohort.  Because the method was
formed after those outcomes were known and validation shows no gain, this is a
candidate for future confirmation, not a positive result.

## Findings mapped to repository artifacts

| Finding | Implementation/evidence |
|---|---|
| Every training task is scored by a model that excluded it | `scripts/56_make_grouped_outer_crossfit_folds.py`, `configs/0713_grouped_outer_crossfit_protocol.json` |
| Six seed/view OOF archives contain the same 664 pairs | `scripts/57_merge_grouped_outer_crossfit_scores.py`, `reports/0713_outer_crossfit_merge_manifest.json` |
| Finalizer handles empty sentinels correctly | `scripts/server/finalize_grouped_outer_crossfit_if_complete.sh`, `tests/test_grouped_outer_crossfit_protocol.py` |
| Leakage-safe world-assisted validation remains NO-GO | `reports/0713_outer_crossfit_grouped_validation.json` |
| The result is not caused by checkpoint collapse | `scripts/60_summarize_outer_crossfit_checkpoints.py`, `reports/0713_outer_crossfit_checkpoint_health.json` |
| Utility replacement causes the largest ordinary-task regression | `scripts/58_diagnose_outer_crossfit_on_fresh_outcomes.py`, `reports/0713_outer_crossfit_fresh_posthoc.json` |
| Text anchoring prevents that regression and enforces fallback | `scripts/59_evaluate_text_anchored_outer_crossfit.py`, `reports/0713_text_anchored_outer_crossfit.json` |

## Next research step

Do not spend another Llama-70B replay budget on the current direct world-ranker.
The next model change should target candidate-level transfer explicitly:

1. keep text utility/probability heads fixed as the safety anchor;
2. train only a residual attack-order correction with task-out-of-model
   features;
3. supervise the residual with within-task candidate contrasts and repeated
   attack rates rather than one historical binary outcome;
4. use nested task-grouped cross-validation and require attack gain >=0.03
   before freezing a new external-task protocol;
5. confirm the frozen text-anchored candidate on genuinely unseen user tasks or
   another benchmark split before claiming improvement.

The present evidence supports the text baseline and a safer architecture, but
not yet the original claim that Dreamer imagination identifies a generally
better attack/skill candidate.
