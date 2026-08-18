# Interface-aligned affordance latent autoresearch v8

## Frozen rationale

The v7 relational-slot encoder was non-collapsed and improved future four-cell outcome CE, but degraded h2--h5 action NLL in all three seeds, with the largest losses on Travel. The missing variable is not generic latent regularization: v7 removed interface-relevant concepts such as hotel, restaurant, rating, and price together with arbitrary raw text.

Stage C1 encodes only lexical intersections between visible goal/observation text and the currently legal tool interface. Concept nodes connect to matching tool nodes, and each tool receives goal/observation affordance strengths. Unmatched text and entity values remain invisible. The label-blind preaudit covered 98.388% of events, 96.880% had goal-to-tool overlap, and all 6,763 rows had zero arbitrary text encoding and zero truncation with the frozen 64-node/32-concept cap.

Stage C2 is pre-authorized after an integrity-valid C1 and adds direct discounted successor-action supervision. Pure replacement of Structured Markov v3 is authorized only after a complete performance gate passes.

## Stage C1 result

Slurm 7071 completed all 15 teacher/candidate fits with zero failures and valid checksums. Decision: `NO_GO_INTERFACE_AFFORDANCE_C1` (9/12 clauses passed). The candidate preserved h1 (NLL gain +0.000786; accuracy gain +0.002448) and future joint CE remained within noninferiority, but h2--h5 NLL worsened by 0.004152, only 8/20 tasks improved, and all three seed effects were negative.

The counterevidence differs materially from v7. Four of five Travel tasks now improve h2--h5, including task 10 (+0.029675), task 4 (+0.016144), and task 0 (+0.015939). The largest new losses are Slack task 1 (-0.078867) and Banking task 8 (-0.042846). Thus interface-aligned concepts repaired the previously missing Travel intent signal, but downstream rollout losses alone do not turn that signal into a broadly predictive state. All bottleneck checks passed: 27,052 fold-specific rows, maximum 43 nodes, zero raw values, zero unmatched text tokens, and zero truncation. C2 is authorized exactly as preregistered.

C1 archive: `/share/guozhix/wmagentattack/0818/interface_affordance_v8/stage_c1/formal_v1`.

## Stage C2 result

Slurm 7074 completed all 15 fixed teacher/candidate fits, 12 tests, and both archive and frozen-source checksum verification with zero runtime failures. Decision: `NO_GO_SUCCESSOR_AFFORDANCE_C2` (6/10 clauses passed).

Direct successor-action supervision improved h2--h5 NLL by 0.003710 relative to C1 and improved 13/20 tasks relative to C1. Relative to v6, however, h2--h5 NLL still changed by -0.000442, none of the three seeds met the +0.05 threshold, and one-step accuracy fell by 0.003461, outside the 0.002 noninferiority margin. Future joint CE improved by 0.002144 and 12/20 tasks improved h2--h5 relative to v6, but these local gains do not satisfy the frozen replacement gate.

The domain evidence remains structured. Four Travel tasks improve relative to v6 (task 4 +0.028645, task 10 +0.017657, task 0 +0.017387, task 5 +0.006748), confirming that interface affordances repaired the Travel intent omission. The dominant counterexample is Slack task 1 (-0.089262 h2--h5 gain), followed by Slack task 0 (-0.018247). The result rules out the claim that a shared discounted successor head alone can turn the interface graph into a task-disjoint replacement state.

All bottleneck checks passed: 27,052 fold-specific rows, maximum 43 nodes, zero raw values, zero unmatched text tokens, zero truncation, and all predictions legal. Stage C3 pure replacement is `NOT_AUTHORIZED`, because C2 did not pass the complete performance gate.

C2 archive: `/share/guozhix/wmagentattack/0818/interface_affordance_v8/stage_c2/formal_v1`.

## Fixed-budget conclusion

The v8 budget is exhausted with no scientifically supported replacement for Structured Markov v3. The retained production baseline remains Structured Markov v3 plus the four-cell outcome head and zero-initialized multi-step residual dynamics. Two mechanisms are preserved as research evidence rather than replacements: interface-aligned affordance slots repair Travel-specific intent, and successor-action supervision improves those slots relative to their unregularized C1 form. A future independently preregistered loop should address domain-conditional negative transfer, particularly Slack task 1, instead of removing Structured Markov v3 or tuning v8 after observing its test results.
