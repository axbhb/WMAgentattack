# Relational router and uncertainty direction-switch autoresearch v10

## Frozen question

The v9 domain experts were legal and parameter matched but worsened multi-step prediction, especially on Slack. This loop tests whether the failure came from routing an entire domain to one expert instead of routing the current relation state. If that mechanism fails the complete gate, the loop changes direction rather than adding another latent encoder: it tests whether a probabilistic v6 ensemble improves predictions and exposes epistemic uncertainty.

Stage E1 removes all lexical hash coordinates before learned encoding. Its 50-dimensional signature contains only node-type proportions, relation-type proportions, and means/maxima of numeric affordance and evidence summaries. Each task-disjoint fold standardizes these coordinates using training-task rows only and applies the frozen transform to confirmation rows. A learned router activates exactly two of four small basis residuals for each state. The dense control receives the identical signature and has matched parameters. Both are exact no-ops over v6 at initialization.

The design follows QORA's evidence for object-relational transfer, DSMoE's state-dependent sparse activation, and the modularity counterevidence that utilization alone does not establish specialization. Therefore E1 must beat v6, an equal-input dense control, and the rejected v9 domain expert; preserve one-step and four-cell predictions; improve broadly across tasks/domains; and pass router integrity and non-collapse checks.

Stage E2 is frozen now but authorized only if E1 is NO-GO. It is a genuine direction change based on PETS: three independently trained v6 models form each uniform predictive ensemble, and member disagreement estimates epistemic uncertainty. Confirmation labels cannot tune weights, temperatures, or thresholds. E2 advances only if the ensemble improves one-step, multi-step, and future joint distributions, retains accuracy, improves most held-out tasks in at least two independent ensemble groups, and assigns higher epistemic uncertainty to errors.

No attacks, planner, real endpoints, Dreamer training, task removal, threshold changes, or post-result reruns are authorized.

## Stage E1 formal conclusion

Slurm 7103 completed all 15 paired fold/seed units and all 45 frozen fits with zero runtime failures. Source and archive hashes pass, as do all 12 preregistered tests. The decision is `NO_GO_RELATIONAL_ROUTER_E1`: 10 of 17 clauses passed.

The mechanism has two real local signals. Future four-cell CE improves by 0.01042 and h1 accuracy improves by 0.00059. It also improves H2--H5 NLL over the rejected v9 domain experts by 0.01755. However, H2--H5 NLL is 1.65836 versus 1.65745 for v6 (a 0.00091 degradation) and 1.65615 for the identical-input dense control (a 0.00222 degradation). Only 55% of held-out tasks improve, Slack degrades by 0.01844, and normalized top-2 routing entropy is 0.46279 below the frozen 0.60 bar. The relation signature therefore contains useful outcome information, but sparse routing does not turn it into better dynamics.

This result rejects another latent/router escalation. Stage E2 is now authorized exactly as frozen: a uniform three-member probabilistic v6 ensemble with three independent ensemble groups. It changes the research target from representation capacity to predictive uncertainty. No E2 weight, temperature, threshold, task, seed, or gate has been selected from E1 outcomes.

Stage E1 archive: `/share/guozhix/wmagentattack/0818/relational_router_uncertainty_v10/stage_e1/formal_v1`
