# Interface-aligned affordance latent autoresearch v8

## Frozen rationale

The v7 relational-slot encoder was non-collapsed and improved future four-cell outcome CE, but degraded h2--h5 action NLL in all three seeds, with the largest losses on Travel. The missing variable is not generic latent regularization: v7 removed interface-relevant concepts such as hotel, restaurant, rating, and price together with arbitrary raw text.

Stage C1 encodes only lexical intersections between visible goal/observation text and the currently legal tool interface. Concept nodes connect to matching tool nodes, and each tool receives goal/observation affordance strengths. Unmatched text and entity values remain invisible. The label-blind preaudit covered 98.388% of events, 96.880% had goal-to-tool overlap, and all 6,763 rows had zero arbitrary text encoding and zero truncation with the frozen 64-node/32-concept cap.

Stage C2 is pre-authorized after an integrity-valid C1 and adds direct discounted successor-action supervision. Pure replacement of Structured Markov v3 is authorized only after a complete performance gate passes.
