# Custom clean panel-v2 architecture ablation preregistration

Binding protocol: `0729_custom_panel_v2_architecture_ablation_fixed_v1`.

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

The dynamics target is the next legal tool or `STOP`; argument keys, stopping, and
error recovery are secondary diagnostics. The evidence target is each frozen proof
obligation's `UNOBSERVED`, `SUPPORTED`, `AMBIGUOUS`, or `CONTRADICTED` state.
Final reports, factorized terminal labels, legacy utility, expert calls, and future
events are barred from all inputs.

Training uses only the 24 training tasks. The 12 calibration and 12 confirmation
tasks are task-disjoint and receive no gradient. There is one fixed hyperparameter
setting and three seeds (`7/17/29`), for nine small runs total.

An incremental representation must improve its primary NLL by at least `0.02` on
both calibration and confirmation, improve in at least two of three training seeds
and at least six of twelve confirmation tasks, and keep the other head's NLL within
`0.02`. Paired-task bootstrap intervals are reported as counterevidence but are not
a hard gate because this intentionally small probe has only twelve confirmation
tasks.

No result from this round can authorize attack-data generation, H2 attack planning,
or Dreamer training. The failed conditional-reporting gate remains binding.
