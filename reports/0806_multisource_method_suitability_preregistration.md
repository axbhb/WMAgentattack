# Multi-source current-method suitability preregistration

This fixed-budget experiment asks whether ToolSandbox, InjecAgent, and tau3 are useful for the current Semantic Markov / Structured Semantic State v3 hybrid world-model method before collecting a materially larger dataset.

The primary surface is task-disjoint next-action prediction. The split unit is the normalized trusted goal within a source, not the row or InjecAgent pair ID. This prevents repeated instruction templates from crossing training and confirmation. Every InjecAgent clean/poison pair remains in one split.

The frozen comparison includes a legal-action frequency prior, a strong TF-IDF candidate scorer, and the three existing equal-capacity representations: Semantic Markov, Structured v3, and causal full visible history. The action gate requires Structured v3 to beat the frequency prior on both NLL and accuracy, replicate across seeds and tasks, and remain within 0.02 NLL of the lexical baseline.

Full world-model suitability is stricter. It additionally requires adjacent semantic transitions and adequate support for every evidence-delta label. The current records are allowed to return an action-only GO while failing this transition gate. Such a result authorizes targeted multi-step collection, not Dreamer, attack generation, utility/value training, or an unrestricted scale-up.

Where the preflight contains at least 50 exact executions with adequate success/error support, the existing candidate-conditional evidence-error head is also tested. It must improve confirmation task-macro BCE over the training-frequency baseline by at least 0.01, replicate in at least two seeds, and improve at least half of confirmation tasks. This error-only probe cannot substitute for the full five-label transition gate.

The fixed budget is 36 small neural runs, four TF-IDF fits, four frequency fits, two deterministic dataset builds, and zero new LLM or tool calls. No threshold or split may change after model results are observed.

The frozen second preflight produced byte-identical independent builds (dataset SHA256 `3c2c3b60c757b17be14e52c45fa5bee79038d1d559f795727c9af70dd23f644b`) with zero task or causal-input fingerprint overlap. The immutable first preflight is retained as counterevidence: it exposed a tau3 same-name/different-schema candidate collision. The label-blind repair namespaces candidates by schema hash and was committed as `fe03a9e` before model training.
