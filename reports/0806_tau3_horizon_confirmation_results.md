# tau3 full-horizon confirmation results

## Frozen decision

`HORIZON_CONFIRMATION_NO_GO__DO_NOT_RUN_METHOD_TEST_OR_SCALE`

The 96-episode same-contract confirmation failed four frozen clauses. The
negative result is binding: the predictive-method comparison and all scale-up
remain unauthorized. No threshold was relaxed and no outcome was rerun.

## Gate results

| Metric | Result | Frozen gate | Decision |
|---|---:|---:|---|
| Complete episodes | 96 | 96 | PASS |
| Forced-budget-stop episodes | 26 | <=24 | **FAIL** |
| Relative forced-stop reduction | 66.67% | >=50% | PASS |
| Adjacent assistant transitions | 367 | >=100 | PASS |
| Agent tool-decision rate | 33.45% | >=35% | **FAIL** |
| State-changing assistant transitions | 21 | >=15 | PASS |
| State-unchanged assistant transitions | 346 | >=30 | PASS |
| Tasks with an assistant state change | 11 | >=8 | PASS |
| Domains with an assistant state change | 2 | >=2 | PASS |
| Paired changed-transition gain | +20 | >=10 | PASS |
| Supported transition targets | 4 | >=4 | PASS |
| Assistant tool-error-rate change | -6.31 pp | <=+5 pp | PASS |
| Out-of-pilot changed transitions | 11 | >=5 | PASS |
| Out-of-pilot changed tasks | 5 | >=4 | PASS |
| Out-of-pilot changed domains | 2 | >=2 | PASS |
| Parent-prefix mismatches | 0 | 0 | PASS |
| Raw pilot-overlap episodes reproduced | 0/24 | 24/24 | **FAIL** |

The exact failed clauses were `forced_budget_stop_episodes`,
`minimum_agent_tool_decision_rate`, `pilot_overlap_reproducibility`, and
`integrity::all_pilot_overlap_episodes_reproduced`. Every other frozen gate and
integrity clause passed.

## Integrity and execution evidence

- Jobs: generation array `6565`; guarded `afterok:6565` summary `6568`.
- All six array tasks emitted their completion sentinel, wrote 16/16 episodes,
  and produced passing chunk audits. The dependent summary emitted its
  completion sentinel and had an empty stderr file.
- Slurm accounting storage is disabled and the completed jobs have expired
  from `scontrol`, so an archival `sacct` state is unavailable. Successful
  `afterok` execution, all seven completion sentinels, complete outputs, and
  passing audits provide terminal-success evidence without inventing an
  unavailable accounting record.
- All 25 entries in the archive checksum file verified. There were zero
  runtime failures, communication-error terminations, nondeterministic exact
  replays, private-scenario exposures, or real endpoint calls. Fatal-log scans
  found no Traceback, CUDA error, out-of-memory error, RuntimeError, or
  segmentation fault.

## Preserved counterevidence

The frozen raw-overlap equality check compared the full serialized episode and
therefore failed. A post-gate field audit found that all 24/24 overlapping
episodes had identical agent decisions and identical user generations; the
only differing top-level field was `trajectory`, and the observed element
difference was the runtime `trajectory[*].timestamp`. This diagnosis does not
recompute or rescue the gate. The raw mismatch and the binding NO-GO remain in
the archive.

Action coverage was heterogeneous. Agent tool-decision rates were 42.78% for
airline, 56.47% for retail, and 2.14% for telecom. Telecom recorded eight
assistant tool events and 44 user tool events. Those user-side device actions
remain exogenous and were not relabeled as assistant outcomes.

The useful signal is also retained: airline produced 12 changed assistant
transitions versus one in its paired parent; retail produced nine versus zero;
telecom produced zero versus zero. The confirmation therefore recovered a
strong two-domain mutation signal but did not satisfy the complete data-form
contract.

## Next smallest label-blind mechanism

Only a bounded tail-horizon pilot is preregistered next. It changes each role's
generation cap from 16 to 20 and the coherent orchestrator horizon from 64 to
80, while preserving the model, prompts, decoding, tools, role permissions,
parser, transition extraction, assistant-only targets, and two-replica exact
execution. Its 24-episode panel must be chosen deterministically from frozen
manifest metadata before outcomes. No implementation or Slurm submission is
authorized by this report.

Future overlap audits prospectively canonicalize only runtime trajectory
timestamps while requiring every behavioral and state-bearing field to remain
exact. Raw records and raw hashes remain mandatory, and the present NO-GO is
explicitly immutable.

## Reproducibility

- Archive: `/share/guozhix/wmagentattack/0806/tau3_horizon_extension/confirmation_v1`
- Frozen protocol SHA256: `6eef4d484095deba86ff6777a314bd05d931467bcb21542a4a8678df5a5a66cc`
- Manifest SHA256: `bea0961bcd1208af3df41057bf27826a94ceef0e77c28f6f3cc691472c034ea8`
- Dataset SHA256: `6d570ca8fc3ab7d556fa3faf4e470c6f326fdbcf59d6f7e686929f7856134498`
- Dataset audit SHA256: `3b4ede0d31d38e7970511ecf94ecc3554ebd7eba9f145bd0774c4fbc4c03193c`
- Gate SHA256: `d4936d31edcee855a0be42ae75f9e673616c85ff4a344a87d94cbfc1452d8521`
- Archived report SHA256: `7fb18082f715a9281d47812b26ba883d389c21e31cfe9287424ddec6acfd08ee`
- Archive checksum-file SHA256: `d33d03d92caad5bc29805aba858aa98848f738a680d6d927b2a401b8a904a916`
