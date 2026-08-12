# Source-specific action head results

Decision: `NO_GO_SOURCE_SPECIFIC_HEAD_DOES_NOT_REPAIR_NEGATIVE_TRANSFER`

This fixed candidate removed the failed source-specific encoder residuals. The
state and candidate encoders remained fully shared, while only the final linear
action scorer was separated for AgentDojo, ToolSandbox, and InjecAgent. It used
the same 20-task five-fold OOF surface, seeds, source mass, and frozen parent
predictions as the preceding multi-source experiments.

| primary Semantic Markov comparison | NLL gain | accuracy gain | positive tasks |
|---|---:|---:|---:|
| source head vs raw pooled | +0.015143 | +0.004059 | 45% |
| source head vs AgentDojo-only | +0.032286 | -0.013281 | 50% |

Only 5/10 frozen clauses passed. Relative to raw pooling, only seed 29 crossed
the `+0.02` NLL threshold and no seed crossed the `+0.01` accuracy threshold.
The paired task NLL interval was `[-0.022393, 0.057355]` with 9 wins and 11
losses. The mechanism removes the severe degradation caused by dual residual
adapters, but does not establish a replicated benefit.

The preregistered Structured counterevidence was stronger: versus raw pooling,
NLL improved `0.018663`, accuracy improved `0.026692`, and 13/20 tasks improved;
versus AgentDojo-only, NLL improved `0.024531`, accuracy changed `-0.003090`, and
14/20 tasks improved. Its NLL interval still crosses zero and only one seed
crosses the raw-pooling threshold. Because Structured was not the primary arm,
this is a hypothesis for a genuinely fresh confirmation surface, not a
post-hoc GO.

The binding result is that source-head isolation is directionally useful but
has not solved multi-source transfer. No further tuning on the same 20-task
confirmation set is authorized.

- Slurm `6736`; 16 tests passed; zero runtime failures.
- Archive: `/share/guozhix/wmagentattack/0814/source_specific_head/formal_v1`
- Summary SHA256: `63f566c0787c63aa02eb02d35111600721a44ba42d2d4499f7d657988b4d05ab`
- Predictions SHA256: `93849e7deb87424e63c4b8700317e12e3e2af20bc37b25b7f427b45e57b38245`
- Full archive checksum verification: PASS.
