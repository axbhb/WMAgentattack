# Custom clean panel-v2 architecture ablation preregistration v2

Binding protocol: `0729_custom_panel_v2_architecture_ablation_fixed_v2`.

## Engineering erratum before training

Slurm 4786 (`fixed_v1`) stopped while constructing the argument-target tensor,
before any model fit. It therefore produced no architecture result. The failed
archive is retained unchanged. Twelve prefixes used raw parser-extra keys
(`properties`, `city`, or `min_price`) that AgentDojo/Pydantic ignores during
tool validation, while the target vocabulary correctly contained only declared
tool-schema fields.

The v2 dataset now derives argument-key targets from post-validation
`fields_set`. If a proposal fails validation, it retains only attempted keys that
are declared by the target tool schema. This matches the project's typed-call
contract. The 144 source trajectories, 48 tasks, split, 467 prefixes, 411 evidence
rows, three model variants, optimizer, seeds, and acceptance thresholds are
unchanged. The repaired dataset was independently rebuilt byte-for-byte, and its
preflight contains an explicit fail-closed out-of-vocabulary target gate.

## Frozen experiment

This round uses the already frozen 144 clean trajectories only. It makes zero new
victim-model calls and trains no completion/reporting, attack, H2, or Dreamer
component.

The comparison is head-specific and strictly nested:

1. `semantic_markov`: trusted goal, policy track, previous canonical action,
   legal candidates, and prefix index.
2. `observable_execution`: the first arm plus the victim-visible tool observation,
   executed/error receipt, and causal state-change summary.
3. `observable_execution_ledger_v2`: the second arm plus the cumulative typed,
   label-blind evidence ledger.

The dynamics primary target is the next legal tool or `STOP`; typed argument keys,
stopping, and error recovery are secondary diagnostics. The evidence primary
target is each frozen proof obligation's `UNOBSERVED`, `SUPPORTED`, `AMBIGUOUS`,
or `CONTRADICTED` state. Final reports, factorized terminal labels, legacy utility,
expert calls, and future events are barred from all inputs.

Training uses only the 24 training tasks. The 12 calibration and 12 confirmation
tasks are task-disjoint and receive no gradient. There is one fixed hyperparameter
setting and three seeds (`7/17/29`), for nine small runs total.

An incremental representation must improve its primary NLL by at least `0.02` on
both calibration and confirmation, improve in at least two of three training seeds
and at least six of twelve confirmation tasks, and keep the other head's NLL within
`0.02`. Paired-task bootstrap intervals are mandatory counterevidence but not a
hard gate because this intentionally small probe has only twelve confirmation
tasks.

No result from this round can authorize attack-data generation, H2 attack planning,
or Dreamer training. The failed conditional-reporting gate remains binding.
