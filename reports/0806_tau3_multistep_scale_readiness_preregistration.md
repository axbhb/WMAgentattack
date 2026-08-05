# tau3 multi-step scale-readiness preregistration

The previous frozen comparison rejected immediate expansion of all three
one-step sources.  tau3 nevertheless showed replicated action-NLL signal, and
the simpler Semantic Markov representation outperformed Structured v3.  This
stage tests the smallest data-form and representation repair justified by that
counterevidence.

Before any new victim-model outcome, 48 tau3 tasks are selected across retail,
airline, and the previously unused telecom domain.  Selection requires at
least two reference assistant actions and is stratified by reference replay
mutation only to ensure structural multi-step coverage; reference calls,
states, and outcomes never enter model inputs.  Each task receives two frozen
Llama-3.1-70B seeds and at most five exact tool transitions.  Every executed
prefix is replayed twice from a fresh in-memory environment.

The data gate requires 96 complete episodes, at least 100 adjacent
transitions, broad task/tool/action coverage, both changed and unchanged exact
states, zero split leakage, deterministic replicas, and zero real endpoint
calls.  Failure completes this data-form candidate and forbids model selection
or scale-up.

If and only if the data gate passes, the accepted model candidate fills the
three previously unused Semantic Markov channels with the current visible
observation, exact receipt, and cumulative causal ledger summary.  It keeps
the same feature width and HybridSemanticWorldModel capacity.  Frozen controls
are the prior Semantic Markov, Structured v3, full history, frequency, and
TF-IDF.  The candidate must improve task-macro action NLL and accuracy, remain
noninferior to full history, improve two-step sequence consistency and
transition calibration, and replicate across seeds and tasks.  Planning,
attacks, Dreamer, and large-scale collection remain disabled until the whole
gate passes.
