# v37 status at 2026-08-31 07:33 UTC / 15:33 China time

Slurm **7562** was submitted once at 07:33:08 UTC, execution commit `ea4b6e4`.
State: **PENDING (Resources)**. No result exists yet; missing run logs while
pending is expected. The scheduler's provisional start estimate was September
1, 01:28 China time; this is not a guarantee.

Both local synthetic fixture tests and remote tests passed **23/23**. Remote
preflight verified all 20 stock task registrations, 60 unique task/seed
allocations, model config availability and imports without loading a model.
Runtime CUDA checks execute inside the eventual one-GPU Slurm allocation.

Budget: 20 familiar tasks × 3 new seeds (601/607/613), 60 clean episodes,
one 4-bit Llama-3.1-70B load, one GPU, six hours maximum, no attacks or fitting.
The archived preregistered protocol and exclusive submission/execution locks
prevent duplicate submissions and silent reruns. All actual data stay remote.

Archive: `/share/guozhix/wmagentattack/0831/clean_pairing_v37/formal_v1`.
Worktree: `/share/guozhix/WMagentattack-clean-pairing-aug31`.

The existing WMagentattack heartbeat has been updated to monitor this job and
continue only the next gate-authorized, separately preregistered sandbox pilot.
No architectural improvement has been demonstrated in v35/v36: those rounds
identified coverage/goal-comparison problems before fitting. Structured Markov
remains the retained baseline.
