# Counterfactual evidence execution pilot: deterministic clock repair v3

Date: 2026-08-05

Status: preregistered after the second collector crash and before any deterministic-clock execution.

## Frozen failure evidence

Remote Slurm job `6345` completed all 48 counterfactual calls and 36 prefix-replay calls with zero infrastructure failures, prefix mismatches, missing adapters, schema failures, or leakage. Exactly one of 24 replica pairs differed: `workspace::send_email`. AgentDojo implements that mutation with the host process's `datetime.datetime.now()`, so sequential fresh-state replicas receive different microsecond timestamps. The other 23 pairs were byte-identical. The `fixed_v2` archive remains immutable, and its readiness metrics are diagnostics rather than a scientific gate decision because the collector failed.

## Sole permitted repair

Freeze the process wall clock during every sandbox tool call to `2024-05-15T12:00:00`, derived from the default Workspace sandbox calendar day `2024-05-15`. This keeps an observable, semantically meaningful time-scoped value while removing host timing noise. The clock context is restored immediately after each call. Add a unit test covering exact `now`, `utcnow`, `today`, date parsing, and restoration.

No task, state, candidate, argument, seed, manifest row, adapter, outcome threshold, readiness gate, or execution budget changes. The run must retain manifest SHA256 `a7e99a9c821757d200c6e943f0a22c764a22564a956897b68e384c7a25229569`, exactly 24 queries, two replicas, 48 counterfactual calls, 36 prefix-replay calls, and 84 total clean AgentDojo sandbox calls. The new immutable archive is `fixed_v3`.

This consumes the second of three preregistered nonsemantic repair attempts. A collector pass followed by readiness failure freezes the scientific NO-GO and forbids further reruns, model probes, attack generation, and Dreamer training.
