# WMagentattack attack-focused research contract

This contract governs experiments whose primary question is attack selection,
not proof of a general-purpose world model.

1. Only AgentDojo's synthetic in-memory sandbox may be used. Real endpoints and
   non-sandbox tools are forbidden.
2. Candidate tasks must pass an independently measured clean-solvability gate
   before an attack manifest is frozen. A general Markov-sufficiency result is
   not an authorization prerequisite.
3. Attack variants must be paired by task, victim model, initial environment,
   injection goal, and random seed. A causal pilot changes one declared attack
   factor at a time.
4. Train/validation/test separation is by user task. Raw task IDs, payload text,
   final outcomes, and checker identifiers cannot be model inputs.
5. The primary endpoint is `P(task_success=1, attack_success=1)`. All four joint
   outcome cells remain supervised; ASR alone is diagnostic.
6. Budgets, seeds, tasks, controls, metrics, thresholds, and archive paths are
   frozen before execution. Failed gates are retained and never relaxed after
   results.
7. No content checksums are calculated, saved, verified, or reported. Runtime
   integrity is established by frozen row IDs, protocol IDs, exact counts,
   tests, Slurm job IDs, and immutable Git commits.
8. A successful pilot may authorize a separately preregistered data expansion.
   It does not directly authorize a planner, unrestricted attack generation, or
   a large world-model training run.
