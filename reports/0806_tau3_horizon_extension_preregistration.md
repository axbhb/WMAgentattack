# tau3 bounded-horizon extension pilot preregistration

The interaction-faithful v1 pilot remains a binding NO-GO. It completed 96
episodes and 254 assistant transitions, but 78 episodes hit a hard per-role
generation cap, only one assistant transition changed state, and the paired
mutation gain over the preceding adapter was -4.

The next pilot changes one mechanism only: each role may make 16 rather than
8 generation calls, with the matching orchestrator horizon increased from 32
to 64 steps. The Llama-3.1-70B 4-bit snapshot, prompts, temperatures, seeds,
tool schemas, private-information boundary, function-tag parser, exact replay,
state representation, and assistant-only transition targets remain unchanged.

The panel contains 12 tasks and both frozen seeds, for 24 episodes. Four tasks
per domain are selected by deterministic hash ranking while preserving the
frozen structural strata as closely as integer counts allow. Selection may use
only manifest metadata. Previous completions, forced-stop flags, errors,
mutations, utility labels, and final outcomes are forbidden selection inputs.

Every candidate episode is paired to the exact interaction-v1 task and seed.
The first eight role calls must reproduce the parent prefix until the parent
naturally stopped or reached its cap. This checks that the candidate is a true
horizon intervention rather than an unrecorded prompt or inference change.

The complete gate is frozen in
`configs/0806_tau3_horizon_extension_protocol.json`. In particular, all 24
episodes and exact replays must be valid; forced stops must fall to at most six
and by at least 50% relative to the paired parent; the assistant side must
produce at least four changed transitions across at least two tasks and two
domains, with a paired gain of at least three; at least four transition targets
must have both classes; and the assistant tool-error rate may not increase by
more than five percentage points.

The principal counterhypothesis is that a longer horizon only prolongs invalid
lookup loops. That possibility is why error non-regression and state-change
clauses are both required. User-side mutations remain exogenous diagnostics
and cannot be relabeled as assistant outcomes to pass this gate.

A pilot GO authorizes only a full 96-episode horizon confirmation. It does not
authorize method training, large-scale collection, attacks, Dreamer, a planner,
or any real endpoint.
