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
was duplicated; the next heartbeat may retry only after repeating those guards.
