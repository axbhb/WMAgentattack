# Custom panel-v2 three-backbone architecture ablation: fixed-v2 result

## Binding conclusion

`CUSTOM_PANEL_V2_ARCHITECTURE_NO_INCREMENT`

Retain `semantic_markov` for both the victim-dynamics and evidence-progress
heads. This result does not authorize completion/reporting training, attack-data
generation, H2 planning, or Dreamer training.

## Execution integrity and the fixed-v1 erratum

Slurm 4786 (`fixed_v1`) failed before the first model fit while constructing the
argument-target tensor. Twelve prefixes contained raw parser-extra keys
(`properties`, `city`, or `min_price`) that AgentDojo/Pydantic ignored during
tool validation, while the target vocabulary contained only declared schema
fields. The failed archive remains unchanged and contains no scientific result.

The repair derives argument-key targets from post-validation `fields_set` and
adds a fail-closed vocabulary audit. No trajectory, task, split, architecture,
hyperparameter, seed, or acceptance threshold changed. The repaired dataset was
rebuilt independently and was byte-identical.

Slurm 4787 (`fixed_v2`) then completed all nine frozen runs. It made zero victim
LLM calls, generated zero attacks, and trained no completion/reporting or Dreamer
component. All 34 archived checksums verify, all 31 preflight tests pass, all
metrics are finite, and there are no task overlaps between the 24 training, 12
calibration, and 12 confirmation tasks.

Key frozen hashes:

- Protocol: `f4f3cb618ce2a3f229b0f116f0f53933d6fb3e442d1980f1d77125ea2e0df3ce`
- Dataset: `c01a0c29a8e2ce99e6f3dd81c82f1711131868ebe81da62a0298fe7e42745746`
- Dataset audit: `4a9ad30d9311e0693558707bcbecbec73ddd1d248b3389ec1a1258074f424374`
- Final summary: `65077d34d63a8de2beed81da2919da395099172b5ba1b77fa67f50066f9100a1`

## Frozen data and model budget

- 144 clean episodes from 48 independent custom AgentDojo tasks.
- 467 causal decision prefixes: 196 training, 123 calibration, 148 confirmation.
- 411 evidence-obligation rows: 146 training, 121 calibration, 144 confirmation.
- Evidence statuses: 137 `UNOBSERVED`, 162 `SUPPORTED`, 104 `AMBIGUOUS`, and
  only 8 `CONTRADICTED`.
- Three strictly nested representations and seeds `7/17/29`: nine runs total.
- Feature sizes at hash dimension 128 are 513 (`semantic_markov`), 901
  (`observable_execution`), and 1160 (`observable_execution_ledger_v2`), with a
  shared hidden size of 96.
- 7,902 predictions are present: 4,203 dynamics and 3,699 evidence predictions.

## Confirmation metrics

Mean over the three frozen training seeds:

| Representation | Action NLL ↓ | Action accuracy ↑ | Evidence NLL ↓ | Supported Brier ↓ |
|---|---:|---:|---:|---:|
| Semantic Markov | 3.988277 | 0.226281 | 1.342302 | 0.202611 |
| + observable execution | 3.787021 | 0.317075 | 1.462839 | 0.249721 |
| + Structured Ledger-v2 | 3.829244 | 0.313001 | 1.433285 | 0.217780 |

Training performance was nearly saturated while held-out performance remained
poor. For example, action accuracy was 0.961/0.965/0.969 on training but only
0.226/0.317/0.313 on confirmation. Evidence accuracy was
0.992/0.994/1.000 on training but 0.668/0.651/0.699 on confirmation. This is a
large task-generalization gap, not an optimization failure.

## Preregistered gate results

All four increments fail the complete frozen gate:

1. **Observable execution for dynamics — FAIL.** It improves action NLL on
   calibration by 0.1173 and confirmation by 0.2013, improves all three seeds,
   and improves 7/12 confirmation tasks. However, confirmation evidence NLL
   regresses by 0.1205, beyond the allowed 0.02 cross-head regression. The paired
   action-NLL bootstrap interval is `[-0.0572, 0.4962]`.
2. **Observable execution for evidence — FAIL.** Confirmation evidence NLL
   worsens by 0.1205 and Supported Brier worsens by 0.0471; all three seeds have
   negative NLL gain. Although 8/10 evidence-bearing tasks improve, two large
   regressions dominate the task-macro mean.
3. **Ledger-v2 for dynamics — FAIL.** Relative to observable execution,
   confirmation action NLL worsens by 0.0422, action accuracy falls by 0.0041,
   only 1/3 seeds improves, and only 4/12 tasks improve. Its paired bootstrap
   interval is `[-0.1496, 0.0755]`.
4. **Ledger-v2 for evidence — FAIL.** Relative to observable execution, evidence
   NLL improves by 0.0363 on calibration and 0.0296 on confirmation, with 2/3
   positive seeds and 6/10 positive confirmation tasks. It nevertheless fails
   because calibration Brier regresses by 0.0026 and confirmation dynamics NLL
   regresses by 0.0422. The evidence-NLL bootstrap interval is wide,
   `[-0.2389, 0.3163]`. Ledger-v2 also remains worse than Semantic Markov on
   confirmation evidence NLL (1.4333 versus 1.3423).

## Preserved counterevidence

The non-gating stratified diagnostic shows that the mean effects are not broadly
stable:

- Observable versus Semantic Markov action-NLL gain on confirmation is
  `-0.1269` Banking, `+0.8384` Slack, `-0.1129` Travel, and `+0.2064`
  Workspace. By difficulty it is `+0.3130` L1, `+0.3776` L2, and `-0.0868` L3.
- Ledger-v2 versus observable evidence-NLL gain on confirmation is `+0.3171`
  L1, `+0.2056` L2, and `-0.2903` L3. Thus the ledger's small average evidence
  gain does not extend to the cross-source/multi-constraint tier it was intended
  to help.
- Only 8 of 411 evidence rows are `CONTRADICTED`, making robust four-way
  evidence-state learning especially weak for failure recovery.
- The confirmation evidence analysis covers 10 evidence-bearing tasks rather
  than all 12; the fixed gate was applied exactly as preregistered, but power is
  limited.

## Interpretation

The experiment provides a directional but non-accepting signal that immediate
observable execution state helps predict the next victim action. It does not
show that one shared observable representation improves both heads, and the
effect does not transfer consistently across suites or L3 tasks.

The current Ledger-v2 is cumulative and typed, but its probe representation is
still a hashed bag of records. The L3 reversal is consistent with loss of
candidate-by-constraint relations, source joins, coverage, and uniqueness
certificates. This is an inference from the stratified result, not a proven
causal explanation. The severe train/held-out gap and the 196-prefix training set
also leave capacity and sample-size explanations open.

## Binding next boundary

This fixed-budget loop is complete and no additional run is submitted. The
scientifically valid next proposal, if opened later, is not a larger MLP or a
threshold rerun. It should preregister a genuinely relational evidence state
(candidate × constraint table, source links, coverage/uniqueness certificate),
increase independent task and contradiction coverage, and evaluate head-specific
encoders without changing the current confirmation result. Until a new clean
gate passes, Semantic Markov remains the baseline and completion, attacks, H2,
and Dreamer remain blocked.

Remote archive:
`/share/guozhix/wmagentattack/0729/custom_panel_v2_architecture_ablation/fixed_v2`.
