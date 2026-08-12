# Source residual adapter results

Decision: `NO_GO_SOURCE_RESIDUAL_ADAPTER_DOES_NOT_REPAIR_NEGATIVE_TRANSFER`

The fixed candidate kept shared state and candidate encoders and added separate
24-dimensional bottleneck residual adapters to both encoders for AgentDojo,
ToolSandbox, and InjecAgent. It used the frozen five AgentDojo folds, source
mass 0.50/0.25/0.25, and seeds 7/17/29 for 30 new fits. Frozen raw-pooled and
AgentDojo-only predictions were reused without rerunning baselines.

| comparison | NLL gain | accuracy gain | positive tasks |
|---|---:|---:|---:|
| adapter vs raw pooled | -0.121369 | -0.018261 | 35% |
| adapter vs AgentDojo-only | -0.104226 | -0.035601 | 25% |

Only 2/10 frozen clauses passed. The paired task NLL interval against raw
pooling was `[-0.229434, -0.038826]`; against AgentDojo-only it was
`[-0.202346, -0.028559]`. Thus the residual adapters materially degraded the
target source rather than merely producing an inconclusive effect.

The counterevidence indicates that placing independently initialized residuals
on both sides of the action scorer creates excessive source-local freedom and
weakens the shared representation. The candidate is discarded. The next
minimal source-isolation candidate retains fully shared encoders and separates
only the final source-specific action scorer.

- Slurm `6735`; 15 tests passed; zero runtime failures.
- Archive: `/share/guozhix/wmagentattack/0813/source_adapter/formal_v1`
- Summary SHA256: `8ca1db5451baeb7b15f43d373961b1fcd3e6343ae39ea1bec9c45e29768235f5`
- Predictions SHA256: `73788cf52ccd8ac8acb95736b90e366319898c02e83d2655b2c8fdcaf0010cd1`
- Full archive checksum verification: PASS.
