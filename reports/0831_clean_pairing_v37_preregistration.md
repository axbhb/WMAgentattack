# v37: clean feasibility and actual seed pairing

The first two rounds show sparse and domain-concentrated joint-success support,
not a validated architectural gain. Retain the same 20 tasks (all four suites),
but measure 60 clean episodes with seeds 601/607/613 on the server. These tasks
have been studied before; only the random draws are fresh. This study cannot
establish unseen-task generalization or attack-selector improvement.

One Slurm GPU allocation, one 4-bit Llama-3.1-70B model load, six hours maximum,
no retries. Retain the v34 base/function-tags victim configuration. The frozen
gate requires 60 complete raw traces, correct actual seeds, no runtime failures,
at least 12 stable (>=2/3 successful) tasks, and at least three suites with two
stable tasks. GPU failures/incomplete output are INVALID, not scientific NO-GO.

Runtime-integrity repair: arithmetic episode seeds use only the frozen user
task index and run seed. Variant-specific row IDs are excluded. This supports
future common-initial-RNG comparisons but cannot force equal token streams
after different prompts cause paths to diverge. No old v34 results are changed.

Only stock in-memory AgentDojo tools are available. The model is offline;
Python IPv4/IPv6 socket connect/sendto operations are denied and counted. This
is an application-level guard, not a claim of OS-level egress isolation.

If clean eligibility passes, freeze the stable task list and separately
preregister a small fixed-goal strategy-contrast pilot, with new independent
evaluation seeds. No attacks are part of v37. If it fails, report why and stop
the fixed cycle without selecting only favorable tasks after the result.

The intended longer-term method is a low-capacity, uncertainty-aware strategy
selector with four-cell outcomes and empirical feedback on a fixed goal.
Structured Markov remains the baseline; a large world model is not justified
by these two data audits.
