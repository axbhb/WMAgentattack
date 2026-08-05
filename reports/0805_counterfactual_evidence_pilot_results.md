# Clean counterfactual evidence execution pilot results

Date: 2026-08-05

Final decision: `GO_COLLECTOR__NO_GO_TRAINING__TARGETED_DATA_ROUND_REQUIRED`

## Scientific conclusion

The clean AgentDojo counterfactual execution harness is now valid and reproducible, but the resulting evidence labels are not sufficient to train a relational evidence world model. Freeze and retain the dataset. Do not run a model probe, attack generation, Dreamer, or large training job from this data version.

The result separates two questions that had previously been conflated:

1. **Can exact clean prefix states be reconstructed and legal alternatives executed reproducibly?** Yes.
2. **Does the current balanced 24-query sample contain enough coverage and rare evidence transitions for learning?** No; all four preregistered readiness clauses fail.

## Frozen execution

- Remote Slurm job: `6347`
- Code commit: `7a5c04fb7cd1e70470196762e96179e34f37b049`
- Archive: `/share/guozhix/wmagentattack/0805/counterfactual_evidence_execution_pilot/fixed_v3`
- Manifest SHA256: `a7e99a9c821757d200c6e943f0a22c764a22564a956897b68e384c7a25229569`
- Protocol SHA256: `fb0484072fe1eb65dcf3ed5b6cbcaf3190c2d53ce236febdec0e253d42732613`
- Dataset SHA256: `019afd9466bdd6f9457ef7a378bea13cb12361010bc7340325294686b01fdeef`
- Audit SHA256: `9cc209ed803552a81ffe3a8066295412e01288ef3d3ed9b85638efafda94d52b`
- Tests: 31 passed
- Archive verification: 20 of 20 recorded checksums passed; no Traceback, OOM, CUDA, or runtime-error signature
- Calls: 48 counterfactual + 36 exact prefix replays = 84 synthetic sandbox tool executions
- LLM calls, attack examples, model runs, and Dreamer runs: all zero

## Collector gate

Every frozen collector clause passed:

- 24 of 24 canonical outcomes and 48 of 48 fresh-state executions
- 24 of 24 byte-identical replica pairs
- all 12 suite-by-difficulty cells, with two rows per cell
- exactly 12 read-only and 12 mutating candidates
- zero `STOP` tool executions, prefix mismatches, infrastructure failures, invalid argument payloads, or semantic-state leakage findings
- complete adapter coverage for all 20 candidate tools and all 12 observed-replay tools, 24 unique tools in total

An independent reconstruction from the dataset reproduced these counts rather than relying only on the collector's audit.

## Training-readiness gate

| Clause | Frozen threshold | Result | Decision |
|---|---:|---:|---|
| Observed bound relation coverage | at least 25% | 457 / 6,282 = 7.2748% | fail |
| Counterfactual execution errors | at least 5 | 0 | fail |
| Conflict-positive outcomes | at least 5 | 0 | fail |
| Ambiguity-positive outcomes | at least 5 | 1 | fail |

Constraint progress comprised 66 `ALREADY_SUPPORTED`, 38 `NEWLY_SUPPORTED`, and 150 `UNCHANGED_UNSUPPORTED` rows. Twelve outcomes changed simulator state, matching the 12 mutating candidates exactly. There are still 529 executable action queries without outcome labels.

The coverage clause requires at least 1,571 observed relations, so the current dataset is short by 1,114 relation labels. At the pilot's mean 10.58 constraints per executed action, this is roughly 106 additional action queries, although the exact number depends on the selected states' constraint counts.

## Counterevidence and interpretation

The balanced selection and clean donor argument policy produced valid, mostly ordinary calls: all 24 counterfactual actions executed successfully. That is useful evidence that the harness works, but it is counterevidence against treating this dataset as sufficient for utility/value learning. It contains state-change signal but essentially no failure, contradiction, or entity-resolution boundary signal.

The failure therefore remains a data-identification problem, not a reason to tune Dreamer or enlarge the model. A learner trained now could predict common successful transitions while remaining untested on precisely the error, conflict, and ambiguity events needed for useful evidence/value modeling.

## Repair audit

- Job `6344` (`fixed_v1`) stopped at 23 outcomes because `get_current_day` was the sole missing label-blind adapter. Its archive remains unchanged.
- Job `6345` (`fixed_v2`) completed all calls, but one `send_email` pair differed because AgentDojo uses host `datetime.now()` microseconds. Its archive remains unchanged.
- Job `6347` (`fixed_v3`) froze the synthetic logical clock to the sandbox calendar day. Exactly one canonical row changed from v2 to v3—the preregistered `send_email` row—and all 24 pairs then became identical. The other 23 canonical rows were unchanged, supporting that the repair was isolated.

Two of three allowed nonsemantic repair attempts were consumed. No third repair is needed because the collector reached a scientific conclusion.

## Authorized next step

Preregister one targeted **clean-only** counterfactual data round in the AgentDojo sandbox. Keep the current task-disjoint grouping, manifest lineage, continuous semantic labels, deterministic logical clock, paired replicas, and existing readiness thresholds. Select additional rows using only pre-outcome structural strata:

1. raise relation coverage by at least 1,114 labels, with exact counts frozen before execution;
2. oversample schema-valid but referentially unsupported actions to expose execution-error boundaries without reading hidden outcomes;
3. oversample revisits to the same visible entity/attribute through independent tools to test conflict formation;
4. oversample tools whose visible records have zero or multiple entity-key matches to test ambiguity;
5. retain same-query fresh-state replicas and a uniform balanced control subset so targeted sampling itself can be audited.

No model training is authorized until a separately frozen targeted round passes the readiness gate. The thresholds in this completed experiment must not be relaxed after seeing the result.
