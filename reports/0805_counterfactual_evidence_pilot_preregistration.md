# Clean counterfactual evidence execution pilot preregistration

Date: 2026-08-05

Status: frozen before manifest construction; the exact manifest hash must be committed before any counterfactual tool execution.

## Erratum that motivates v2

The preceding Candidate × Constraint v1 schema selected 12 balanced tasks, but its complete-cell audit inspected selected episode metadata rather than states that actually contributed executable queries. Travel-L1 task 2200 and Workspace-L3 task 2308 stopped without a tool call, so v1 contained supervised/query rows for only 10 of 12 suite × difficulty cells. The v1 archive remains immutable, training never started, and its schema GO is withdrawn for training authorization.

The corrected universe includes every causal prefix, including terminal prefixes. `STOP` is recorded as the victim decision but is never executed as an environment tool.

## Fixed outcome-blind manifest

The corrected development universe contains 12 tasks, 31 states, 12 terminal decisions, 19 observed transitions, 553 executable alternative state-action queries, and 6,079 candidate × constraint relations. Candidate arguments are bound without an LLM: use schema-valid empty/default arguments first, then argument-only donors from a different clean training task, then a same-task clean donor if no cross-task payload exists. Donor outputs and outcomes are forbidden.

Select exactly one read-only and one mutating query from each of the 12 suite × difficulty cells using the frozen hash seed. This yields 24 bound queries over 22 states and 20 tools. Each query will be executed twice from independently reconstructed clean prefix state, for a total budget of 48 synthetic sandbox tool calls.

## Gates

The collector passes only with exact sample counts, all 12 cells, no `STOP` execution, schema-valid arguments, exact prefix ledger/Semantic-v3 replay, no infrastructure failures or feature leakage, and identical paired replicas. Runtime tool errors are valid scientific outcomes rather than infrastructure failures.

Training remains separately gated at 25% bound-query coverage and at least five execution-error, conflict, and ambiguity events. If that readiness gate fails, the outcomes are retained but no model probe, attack generation, Dreamer run, or large training job is authorized.
