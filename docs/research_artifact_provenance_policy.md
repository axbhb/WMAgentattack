# WMagentattack artifact provenance policy

Effective 2026-08-25, WMagentattack no longer calculates, verifies, stores, reports, or gates experiments on content checksums, including SHA256.

Future experiments establish provenance with:

- the Git branch and commit;
- an immutable, versioned artifact directory;
- copied dataset and experiment configurations with explicit schema versions;
- trajectory, transition, row, task, and split counts as applicable;
- Slurm job IDs, allocation details, exit states, and complete logs;
- exact launch commands, runtime settings, tests, and preregistered gate results.

New scripts, reports, configs, manifests, and ledgers must not add checksum generation or checksum fields. A checksum must not be used as an acceptance gate.

Completed historical experiments remain frozen and are not rewritten merely to remove earlier checksum records. Those old values are archival text only and must not be recomputed, checked, or propagated into new experiments.
