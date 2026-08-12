# Fresh integrated validation: infrastructure incident

This record does not change the preregistered task surface, seeds, victim-model
contract, model-training budget, or acceptance gates.

## Invalid first execution

- Slurm array `6769` launched the three frozen clean seeds but every shard
  failed before loading the Llama-3.1-70B model or creating a trajectory.
- All three failures reported that CUDA was unavailable. There were zero
  `result.json` files and zero raw traces.
- Dependent summary `6770` consequently wrote a 0/36 clean-gate file. That
  file is an infrastructure-invalid result, not a scientific clean-solvability
  NO-GO, and cannot authorize any model training.
- Frozen-input checksums remained valid.

## Label-blind diagnosis

- The Slurm allocation selected physical GPU 3 and set
  `SLURM_JOB_GPUS=3` / `CUDA_VISIBLE_DEVICES=3` in a batch probe.
- `nvidia-smi` could inspect GPU 3 and its compute mode was `Default`, but
  CUDA driver initialization returned `cuInit=100` (`CUDA_ERROR_NO_DEVICE`).
- PyTorch reported one device from visibility parsing but
  `torch.cuda.is_available() == false`; no model inference was attempted.
- GPUs 0--2 were occupied by other users and were not accessed.

## Permitted recovery

Wait for a healthy scheduled GPU. Before any retry, require a zero-model Slurm
probe with aligned Slurm/CUDA device IDs, `cuInit(0) == 0`, and a successful
PyTorch CUDA initialization. Only then retry exactly the three failed shards
once. Preserve the original logs and invalid gate separately. Do not change or
remove tasks, seeds, thresholds, prompts, or the model contract.
