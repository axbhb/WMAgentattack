# tau3 bounded-tail horizon pilot preregistration

The 96-episode 16/16-role, 64-step confirmation is a binding NO-GO. It
recovered 21 state-changing assistant transitions across 11 tasks and two
domains, with paired gain +20 and four supported targets, but failed the frozen
forced-stop ceiling by two episodes and the agent tool-decision floor by 1.55
percentage points. No method test or scale-up is authorized.

## Single mechanism

The next candidate adds a fixed tail of four generation calls to each role:
agent and user caps move from 16 to 20, and the coherent orchestrator horizon
moves from 64 to 80. This is the only behavioral change. The Llama-3.1-70B
4-bit snapshot, prompts, decoding, seeds, tool schemas, role permissions,
function-tag parser, private-information boundary, transition extraction,
assistant-only targets, and two fresh exact-replay replicas remain fixed.

The intervention is intentionally smaller than another horizon doubling. The
confirmation's 26 residual forced stops missed the full-data ceiling by only
two episodes, and the forced-stop subset had a higher agent tool-decision rate
than naturally terminated episodes. A short tail can therefore test both
remaining behavioral deficiencies without introducing a prompt, parser, role,
or target change.

## Label-blind panel

The pilot contains 12 tasks and both frozen seeds, for 24 episodes. Its pool is
the 36 tasks marked out-of-pilot in the already-frozen confirmation manifest;
that membership was fixed before confirmation outcomes. Four tasks per domain
must be selected by deterministic hash ranking while preserving structural
strata as closely as integer counts permit.

Selection may use only domain, task key, structural stratum, experimental
split, seed, episode ID, and frozen pilot-overlap membership. Completion text,
forced-stop outcomes, tool statuses, mutations, action labels, utilities, and
final outcomes are forbidden. The manifest must be built twice byte-identically
before any interactive outcome is generated.

Every candidate is paired to its exact 16/16/64 task and seed. Role calls and
tool-event prefixes must remain exact until the parent naturally stops or a
role reaches its cap; divergence is permitted only after a parent cap is
reached.

## Frozen gate

All clauses are required. In particular:

- 24/24 episodes, valid two-replica exact execution, and zero runtime,
  communication, privacy, or endpoint failures;
- at most six forced stops and at least 50% paired stop reduction;
- at least 25 adjacent assistant transitions, an agent tool-decision rate in
  [35%, 90%], and dominant-action fraction at most 65%;
- at least four changed assistant transitions across at least two tasks and
  two domains, with paired gain at least three;
- at least eight unchanged assistant transitions, four supported targets, and
  both classes in the frozen training and confirmation splits;
- assistant tool-error-rate increase at most five percentage points; and
- exact parent-prefix equivalence and label-blind panel selection.

User-side telecom device tools remain exogenous. They cannot be executed under
the assistant identity or relabeled as assistant outcomes to satisfy the gate.

## Prospective timestamp audit rule

The completed confirmation's raw pilot-overlap check failed because serialized
trajectory timestamps differed, even though all 24 agent-decision and user-
generation records matched. That failure remains binding and is not recomputed.

For future overlap checks only, the comparator removes exactly
`trajectory[*].timestamp` from both records and requires every other behavioral
and state-bearing field to be identical. Raw records, raw hashes, and raw
mismatch diagnostics remain mandatory. This prospective measurement rule
cannot rescue the prior NO-GO.

## Authorization boundary

This document preregisters only the candidate. It does not authorize
implementation or a Slurm submission. A later pilot GO could authorize only a
separately frozen same-contract 96-episode confirmation. Only a later full-data
GO could authorize the already-specified task-disjoint
frequency/TF-IDF/Semantic Markov/Structured v3/full-history/observed-v4 method
comparison, and only a subsequent method GO could authorize larger collection.
Attacks, real endpoints, Dreamer, and planner training remain forbidden.

## Frozen preregistration record

- Protocol: `configs/0807_tau3_tail_horizon_protocol.json`
- Protocol SHA256: `2b6a4ec560c8e4c756c32ee9738f3dd31a7a6d907cbc66ef3da4b17eed4a7a28`
- Status: preregistered only; implementation, manifest, jobs, and result are all
  unset.
