# Multi-source current-method suitability preregistration

This fixed-budget experiment asks whether ToolSandbox, InjecAgent, and tau3 are useful for the current Semantic Markov / Structured Semantic State v3 hybrid world-model method before collecting a materially larger dataset.

The primary surface is task-disjoint next-action prediction. The split unit is the normalized trusted goal within a source, not the row or InjecAgent pair ID. This prevents repeated instruction templates from crossing training and confirmation. Every InjecAgent clean/poison pair remains in one split.

The frozen comparison includes a legal-action frequency prior, a strong TF-IDF candidate scorer, and the three existing equal-capacity representations: Semantic Markov, Structured v3, and causal full visible history. The action gate requires Structured v3 to beat the frequency prior on both NLL and accuracy, replicate across seeds and tasks, and remain within 0.02 NLL of the lexical baseline.

Full world-model suitability is stricter. It additionally requires adjacent semantic transitions and adequate support for every evidence-delta label. The current records are allowed to return an action-only GO while failing this transition gate. Such a result authorizes targeted multi-step collection, not Dreamer, attack generation, utility/value training, or an unrestricted scale-up.

The fixed budget is 36 small neural runs, four TF-IDF fits, four frequency fits, two deterministic dataset builds, and zero new LLM or tool calls. No threshold or split may change after model results are observed.
