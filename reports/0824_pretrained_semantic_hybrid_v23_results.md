# Pretrained Semantic Hybrid v23: fixed-budget result

## Conclusion

The frozen v23 cycle concludes:

- Open-vocabulary model: NO_GO_PRETRAINED_SEMANTIC_HYBRID_V23
- 96-episode data smoke: NO_GO_96_EPISODE_DATA_SMOKE_V23
- Gate: 13/16 clauses passed
- Authorization: no data generation, attack generation, or large world-model training

This is a scientific NO-GO rather than an infrastructure failure. The model completed all 45 preregistered fits and 45 metric rows with zero runtime failures. The final gate writer had a lowercase Python Boolean serialization error after metrics were already written; a label-blind gate-only repair reran no model and changed no seed, split, prediction, metric, or threshold.

## Frozen comparison

The experiment preserved the fixed v21 head for labels observed in each outer training fold and used frozen intfloat/e5-base-v2 prototypes only for truly unseen effect labels. It compared:

1. fixed_v21;
2. hybrid_e5_raw_v23;
3. hybrid_e5_calibrated_v23;
4. the immutable v22 independent candidate-ID control as counterevidence.

The semantic cache was deterministic, label-blind, finite, unit-normalized, and consumed no task/source IDs, outcomes, utility/security labels, future labels, or external endpoints.

## Main task-disjoint result

| Metric | Fixed v21 | Independent v22 | E5 raw v23 | E5 calibrated v23 | Gate |
|---|---:|---:|---:|---:|---|
| unseen positive recall | 0.0000 | 0.5079 | 0.5291 | 0.5291 | fail: require >=0.55 and >=+0.04 over v22 |
| unseen positive NLL | 7.2074 | 2.3938 | 0.9965 | 1.2561 | pass vs v22; calibration worsens raw by 0.2596 |
| seen positive recall | 0.9749 | 0.9717 | 0.9749 | 0.9749 | pass |
| one-step task-macro BCE | 0.0443 | 0.1336 | 0.0423 | 0.0407 | pass |
| rollout BCE | 0.0227 | 0.1168 | 0.0293 | 0.0278 | pass within +0.01 margin |
| query/read positive recall | 0.9739 | 0.9760 | 0.9757 | 0.9757 | pass |
| pair assignment accuracy | 0.9931 | 0.9896 | 0.9896 | 0.9896 | retained as counterevidence |

The E5 branch therefore learned substantially better probabilities for unseen labels, reducing task-disjoint unseen NLL by 1.1377 versus the independent-ID control. However, the recall gain was only 0.0212, below the frozen +0.04 requirement, and absolute recall remained 0.0209 below the 0.55 floor.

## Diagnostic splits and counterevidence

- Tool-family-held-out unseen recall reached 0.2798 and passed its 0.25 floor, but remained far below the independent-ID control at 0.5220.
- Source-held-out unseen recall reached 0.2008 and passed its 0.15 floor, but remained far below the independent-ID control at 0.4871.
- Train-only label-held-out calibration did not change recall and degraded task-disjoint unseen NLL from 0.9965 to 1.2561. It also degraded tool-held-out NLL from 1.1352 to 1.4458 and source-held-out NLL from 1.5163 to 1.6325.
- The recomposed v22 long-horizon gate passed: every retained recurrent-action clause stayed true and the new v23 effect-rollout non-inferiority clause passed.
- The frozen three-source data-generation design remains schema-ready, but execution is not authorized because the open-vocabulary model gate failed.

These results argue against treating general text-embedding proximity alone as sufficient evidence for causal effect transfer. The model ranks unseen labels more plausibly and is much less overconfidently wrong, yet still misses too many true labels and does not beat the non-semantic ID control by the preregistered margin.

## Interpretation and retained architecture

Keep:

- the v21 Structured Markov transition surface for observed effects;
- the v22 zero-initialized recurrent residual for multi-step action dynamics;
- the v23 frozen E5 prototype branch as a useful uncertainty/ranking diagnostic, not as the accepted open-vocabulary replacement.

Reject:

- the v23 train-only scalar temperature/bias calibration;
- scaling data merely because NLL improved;
- claims that pretrained semantic similarity has solved unseen causal effect prediction.

The next architecture should change the failed semantic mechanism rather than tune this result after the fact. The strongest next hypothesis is a support-aware relational effect model: canonical entity/field/relation slots plus retrieved label-disjoint demonstrations, a learned abstention/open-set head, and counterfactual action-consistency supervision. This directly targets the observed recall bottleneck and separates semantic plausibility from causal support. If explicit label-disjoint support cannot be collected, the next cycle should test calibrated conformal/set prediction with abstention rather than force a single thresholded label decision.

## Reproducibility

- Scientific Slurm job: 7291, elapsed 00:19:14
- Gate-only repair job: 7294
- Model fits / metric rows / runtime failures: 45 / 45 / 0
- Tests: 4 passed before training; 4 passed after gate-only repair
- Archive: /share/guozhix/wmagentattack/0824/pretrained_semantic_hybrid_v23/formal_v1
- Metrics SHA256: 8c7af9f2b3145c949d31053dd5f01757f7de5dcc8b66bf949774003ee4faa9ec
- Gate SHA256: e01906c89a6294e0eb7575f3f9787613533082eef08dd149bae7fc33bc4ee790
- Archive checksum manifest SHA256: 87c51756649de95f86a4ef94a4b09d2d69888bf88c87a9ca99811172e7ec9844

No 96-episode smoke, attack trajectory, medium/large dataset, or large world-model training was started.
