# Relation-factorized semantic distribution v24 result

## Conclusion

The frozen decision is NO_GO_RELATION_FACTORIZED_DISTRIBUTION_V24 and NO_GO_96_EPISODE_DATA_SMOKE_V24. The gate passed 15 of 20 clauses. No data generation, attack generation, medium/large scaling, planner, Dreamer, or large world-model training is authorized.

The formal job completed all 45 fits and 45 metric rows with zero runtime failures. Six tests and every pre-run hash passed.

## Main task-disjoint comparison

| Metric | Fixed v21 | E5 raw v23 | Relation raw v24 | Relation support v24 | Result |
|---|---:|---:|---:|---:|---|
| unseen positive recall | 0.0000 | 0.5291 | 0.2063 | 0.2619 | fail: require >=0.55 and >=+0.02 over v23 |
| unseen positive NLL | 7.2074 | 0.9965 | 0.9688 | 0.9874 | pass |
| unseen false-positive rate | n/a | n/a | 0.0659 | 0.0762 | fail: require <=0.05 |
| unseen precision | n/a | n/a | 0.1274 | 0.1349 | fail: require >=0.20 |
| mean predicted unseen set | n/a | n/a | 0.4378 | 0.4928 | pass: limit 0.7630 |
| seen positive recall | 0.9749 | 0.9749 | 0.9749 | 0.9749 | pass |
| one-step task-macro BCE | 0.0443 | 0.0423 | 0.0518 | 0.0515 | pass within +0.01 |
| rollout BCE | 0.0227 | 0.0293 | 0.0346 | 0.0341 | fail: 0.00142 beyond the noninferiority boundary |
| pair assignment accuracy | 0.9931 | 0.9896 | 0.9896 | 0.9896 | counterevidence |

Support diffusion improved the new relation model's own recall by 0.0556 and kept NLL within 0.0186 of relation raw. All train-only support rules were feasible. This establishes that the mechanism is active, but it does not establish transferable causal support: the final recall remains 0.2672 below v23.

## Diagnostic panels

- Tool-family-held-out unseen recall was 0.2892 and passed the 0.28 floor.
- Source-held-out unseen recall was 0.2454 and passed the 0.20 floor.
- Query/read positive recall and seen-label recall were preserved.
- The data-design protocol remains ready, but the combined model gate is not.

## Counterevidence and failure mechanism

The training-fold audit already showed that task-disjoint unseen effects have zero exact entity support and only 18.75% exact field support. The relation kernel therefore relies mainly on generic category/kind similarity.

The formal result matches that warning:

- Fold 0 has false-positive rates between 0.1538 and 0.2198 and precision between 0 and 0.0408.
- Fold 2 is more plausible, but recall varies from 0.1111 to 0.4444 across seeds.
- Lower NLL coexists with poor thresholded recall and excess false positives, so probability smoothing is not the missing causal mechanism.
- Support selection is feasible on inner held labels but does not transfer its error control to outer task-disjoint labels.

The result cross-checks the earlier v16 retrieval failure: semantic proximity is not causal equivalence. It also clarifies the v7 relational-slot failure: retaining domain words is necessary, but relation decomposition alone is insufficient when the training folds never observe the held-out entity effects.

## Retained architecture and direction change

Retain:

- fixed v21 Structured Markov transitions for observed effects;
- v22 zero-initialized recurrent action dynamics;
- v23 full-description E5 predictions as an uncertainty and ranking diagnostic.

Reject:

- v24 relation-factorized aggregation as a replacement for v23;
- similarity-distribution supervision as evidence of causal transfer;
- further threshold, temperature, kernel, or latent-capacity tuning on the same 121 rows.

The next research program must change the data rather than add another encoder. It should collect clean-only, label-disjoint matched branches in the AgentDojo sandbox: keep a canonical prefix fixed, intervene on legal action/entity/argument combinations, and record multiple successor effects. A coverage gate must first verify that every held-out effect has explicit entity, field, and operation support in training tasks. Only then should a support-conditioned relational world model be trained.

## Reproducibility

- Formal Slurm job: 7304
- Runtime: 972 seconds
- Fits / metric rows / failures: 45 / 45 / 0
- Tests: 6 passed
- Archive: /share/guozhix/wmagentattack/0824/relation_factorized_distribution_v24/formal_v1
- Metrics SHA256: 18ecea492f450db9ad040a053804d2b3d5a91a0e54ca2818bfd14cfcfe2adc13
- Gate SHA256: a664a9bf309859b7e9f832ed403a79507b776f4e3a92d748419d8501b95119d7
- Checksum manifest SHA256: 59dde39a76a9e61a12fc22766ce40d3c0480a56184b73f9216dc39f349d4c0bc

No 96-episode smoke or larger experiment was started.
