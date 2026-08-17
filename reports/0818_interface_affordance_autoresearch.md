# Interface-aligned affordance latent autoresearch v8

## Frozen rationale

The v7 relational-slot encoder was non-collapsed and improved future four-cell outcome CE, but degraded h2--h5 action NLL in all three seeds, with the largest losses on Travel. The missing variable is not generic latent regularization: v7 removed interface-relevant concepts such as hotel, restaurant, rating, and price together with arbitrary raw text.

Stage C1 encodes only lexical intersections between visible goal/observation text and the currently legal tool interface. Concept nodes connect to matching tool nodes, and each tool receives goal/observation affordance strengths. Unmatched text and entity values remain invisible. The label-blind preaudit covered 98.388% of events, 96.880% had goal-to-tool overlap, and all 6,763 rows had zero arbitrary text encoding and zero truncation with the frozen 64-node/32-concept cap.

Stage C2 is pre-authorized after an integrity-valid C1 and adds direct discounted successor-action supervision. Pure replacement of Structured Markov v3 is authorized only after a complete performance gate passes.

## Stage C1 result

Slurm 7071 completed all 15 teacher/candidate fits with zero failures and valid checksums. Decision: `NO_GO_INTERFACE_AFFORDANCE_C1` (9/12 clauses passed). The candidate preserved h1 (NLL gain +0.000786; accuracy gain +0.002448) and future joint CE remained within noninferiority, but h2--h5 NLL worsened by 0.004152, only 8/20 tasks improved, and all three seed effects were negative.

The counterevidence differs materially from v7. Four of five Travel tasks now improve h2--h5, including task 10 (+0.029675), task 4 (+0.016144), and task 0 (+0.015939). The largest new losses are Slack task 1 (-0.078867) and Banking task 8 (-0.042846). Thus interface-aligned concepts repaired the previously missing Travel intent signal, but downstream rollout losses alone do not turn that signal into a broadly predictive state. All bottleneck checks passed: 27,052 fold-specific rows, maximum 43 nodes, zero raw values, zero unmatched text tokens, and zero truncation. C2 is authorized exactly as preregistered.

C1 archive: `/share/guozhix/wmagentattack/0818/interface_affordance_v8/stage_c1/formal_v1`.
