# tau3 interaction-faithful data repair preregistration

The previous 96-episode multi-step pilot repaired the zero-transition problem
but failed its state-change gate: only 5 of 163 assistant tool transitions
changed state.  No predictive model was trained.

The next candidate changes one mechanism: it restores tau3's official
half-duplex user--agent--environment interaction.  The same frozen
Llama-3.1-70B-Instruct 4-bit model instance alternates between the agent and
user-simulator roles.  Role prompts and decoding settings are separate, but
the weights and tokenizer are identical.  The agent receives policy, legal
agent tools, natural user messages, and its own tool receipts; the task's
private `UserScenario` is available only to the user role.  Task-authorized
user tools are handled by the in-memory environment.

The experiment reuses the exact same 48 tasks, task splits, agent seeds, and
source commit as v1.  Every complete interleaved user/agent tool sequence is
executed live and replayed twice from fresh state.  Both agent and user DB
hashes are included in the deterministic state fingerprint.  Only assistant
tool calls become world-model transitions; user tool calls are retained as
exogenous causal context.

Each role may make at most eight logical model calls per episode.  At most
four additional calls per episode may repair an unambiguous tool-intent
serialization, for a fixed global ceiling of 1,920 physical model calls.
User tools are filtered to the task-authorized schema before execution, and
budget-forced terminations are recorded rather than hidden.  More than 24
budget-truncated episodes or any communication-error termination fails the
data gate.

The frozen data gate retains v1's absolute coverage and balance thresholds,
requires at least 15 changed assistant transitions, and adds task/domain
coverage, private-scenario separation, official user-turn coverage, a paired
gain of at least 10 changed transitions over v1, and support for at least four
transition labels.  Paired gains cannot substitute for any absolute gate.

Only a complete data GO authorizes the already-specified task-disjoint
Semantic Markov v4 comparison.  Attacks, Dreamer, planning, real endpoints,
and large-scale collection remain disabled until both data and method gates
pass.
