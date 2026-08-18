# Domain-routed affordance latent autoresearch v9

## Frozen rationale

The v8 interface-affordance state repaired Travel intent but produced domain-structured negative transfer, dominated by Slack task 1. This independent loop tests whether modular dynamics, rather than a larger shared adapter, resolves that conflict.

The design follows three primary-source findings. Mixture-of-World Models (ICLR 2026) combines a shared backbone with task-conditioned dynamics experts for heterogeneous multi-task world models. *Is a Modular Architecture Enough?* warns that generic learned routers may collapse or fail to specialize. *Mixture of Experts in a Mixture of RL settings* motivates capacity-controlled MoE comparisons under multi-task non-stationarity. These map to a shared interface-affordance encoder, four deterministic experts routed only by inference-visible AgentDojo `track`, and a dense adapter with matched total parameters.

Stage D1 keeps the v6 Structured Markov context and changes only adapter modularity. It must beat both v6 and the parameter-matched dense arm, preserve h1 and future four-cell prediction, improve at least three domains, and make both Slack and Travel nonnegative. Stage D2 removes Structured Markov v3 only if every D1 clause passes. No threshold, task, seed, or fit count may change after results are observed.

## Stage D1 formal result

Stage D1 completed under Slurm job 7083 with all 15 task-fold/seed units and all 45 frozen fits present (15 teacher, 15 dense-capacity, and 15 domain-expert fits). There were no runtime failures. Archive and frozen-source checksums pass, and the preregistered 17-test suite passes.

The formal decision is `NO_GO_DOMAIN_EXPERT_D1`: 7 of 15 gate clauses passed. The expert arm preserved h1 NLL relative to v6 by 0.00136, but h1 accuracy fell by 0.00275, beyond the 0.002 noninferiority allowance. H2--H5 NLL was 1.67591, worse than v6 (1.65745) by 0.01846 and worse than the parameter-matched dense arm (1.66318) by 0.01274. Future four-cell cross-entropy also worsened by 0.02292.

The negative result is not caused by a broken router or extra capacity. The expert/dense parameter gap is only 0.081%, every expert is used, routing depends only on inference-visible `causal_model_input.track`, and raw values or unmatched text are not encoded. Instead, the domain effects show continued structured negative transfer: Banking improves by 0.00258 and Travel by 0.00921, while Workspace degrades by 0.00978 and Slack degrades sharply by 0.07584. Only 40% of held-out tasks improve, and all three seeds miss the required multi-step gain.

This is direct counterevidence to the hypothesis that coarse domain routing is sufficient. The matched dense control is better than the expert arm, so the result cannot be credited to modularity or hidden extra capacity. A four-way domain label is too coarse to isolate the task-level tool and evidence dynamics that cause Slack interference.

## Frozen conclusion

Stage D2 is `NOT_AUTHORIZED`; no pure expert-latent replacement was trained. Structured Markov v3 plus the four-cell outcome head and zero-initialized multi-step residual dynamics (v6) remains the retained world model. The next independent research loop, if undertaken, should not add another coarse router. It should test a state-dependent sparse residual keyed by normalized tool/evidence relations, with v6 frozen and exact no-op initialization, against both a matched dense residual and the current domain expert as negative controls.

Formal archive: `/share/guozhix/wmagentattack/0818/domain_expert_affordance_v9/stage_d1/formal_v1`

Predictions SHA256: `bddcccb7504509ef1849a9ef8ebbb9ea94ea538da88ced2fa09952d825b05211`

Run-metrics SHA256: `b22250684fb79b3e4638dafe6dcb1af82a402565a516d8c4101f97a06e94745d`
