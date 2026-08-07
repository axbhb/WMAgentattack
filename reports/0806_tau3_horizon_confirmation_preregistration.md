# tau3 full-horizon confirmation preregistration

The 24-episode horizon-only pilot passed every frozen clause. This
confirmation does not tune the mechanism: it keeps the same Llama-3.1-70B
4-bit snapshot, role prompts, decoding, tools, parser, assistant-only targets,
two fresh replay replicas, 16-call role caps, and 64-step orchestrator.

The confirmation uses all 96 rows from the interaction-v1 parent manifest: 48
tasks, both frozen seeds, and all task-disjoint split labels. No ranking or
outcome-based filtering is permitted. Twenty-four episodes overlap the pilot;
the other 72 episodes cover 36 tasks that were not selected for the pilot.

Every episode must reproduce its interaction-v1 parent prefix exactly before
any additional horizon suffix. The 24 pilot-overlap episodes must also
reproduce their complete pilot trajectories. These checks isolate the horizon
mechanism from hidden prompt, model, parser, environment, or execution drift.

The full-data thresholds retain the original interaction sufficiency gate:
96 complete episodes, at most 24 forced stops, at least 15 changed assistant
transitions across at least eight tasks and two domains, paired gain of at
least ten, four supported targets, and the original action/coverage clauses.
The paired forced-stop reduction must also remain at least 50%, and assistant
tool-error rate may not regress by more than five percentage points.

To prevent the pilot overlap from carrying the result, the 72 out-of-pilot
episodes must be exact and include at least five changed transitions across at
least four tasks and two domains. This holdout clause was frozen before any
confirmation completion was generated.

A confirmation GO authorizes only the already-specified task-disjoint
frequency/TF-IDF/Semantic Markov/Structured v3/full-history/observed-v4
comparison. It does not authorize large collection, attacks, Dreamer, a
planner, or any real endpoint. Large collection requires a later method GO.

## Execution status

The 96-episode manifest was frozen after two byte-identical label-blind builds,
and the remote implementation passed 17 focused tests. Three guarded submission
attempts on 2026-08-06 were not accepted by Slurm: the first timed out without
creating a queue entry, marker, log, or output, and the second and third returned
`Resource temporarily unavailable`. Post-attempt checks found no matching job
or lingering `sbatch` process. No job ID has been recorded and no experiment
was duplicated. At 2026-08-06T14:01Z, after the global queue dropped to 17
visible jobs and the frozen six-way array passed `sbatch --test-only`, Slurm
accepted the unique confirmation array as job `6565`. The guarded dependent
summary was not accepted because the array consumed the remaining job-record
capacity. The array marker is frozen; subsequent heartbeats must never resubmit
it. At 2026-08-06T14:11Z, a summary dry-run passed and Slurm accepted the
unique dependent summary as job `6568` with `afterok:6565`. Both submission
markers are now frozen; subsequent heartbeats may only monitor these jobs.

## Completed outcome

Both jobs completed and the frozen gate returned
`HORIZON_CONFIRMATION_NO_GO__DO_NOT_RUN_METHOD_TEST_OR_SCALE`. The confirmation
completed all 96 episodes, but recorded 26 forced stops against a maximum of
24, an agent tool-decision rate of 33.45% against a minimum of 35%, and 0/24
raw pilot-overlap reproductions. The overlap mismatch was subsequently traced
to runtime trajectory timestamps only; this post-gate diagnosis does not alter
the binding NO-GO. The full result and preserved counterevidence are recorded
in `reports/0806_tau3_horizon_confirmation_results.md`.

No predictive-method comparison or scale-up was run. Only the next smallest
label-blind bounded-tail mechanism is preregistered, with no implementation or
new Slurm submission.
