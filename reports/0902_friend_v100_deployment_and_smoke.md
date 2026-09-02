# Friend V100 deployment and smoke validation

Date: 2026-09-02

## Deployment state

- SSH route: `friend-v100` through the configured Tailscale jump host.
- Target: `gpuserver`, user `pth`.
- GPU: 2 x Tesla V100-PCIE-32GB; both devices were idle during validation.
- Project: `/home/pth/projects/WMagentattack-v45`.
- Isolated environment: `/home/pth/venvs/wmagentattack_v45`.
- Formal dataset: `/home/pth/data/wmagentattack/v45/dataset.json`.
- Frozen local E5 model: `/home/pth/models/e5-base-v2`.
- Smoke outputs: `/home/pth/outputs/wmagentattack/v45_smoke`.

The isolated environment reuses the server's working CUDA PyTorch installation and
overrides only the packages needed by WMagentattack. The friend's original Conda
environments were not modified.

## Runtime versions

- Python 3.9
- PyTorch 2.2.2 + CUDA 12.1
- Transformers 4.46.3
- Pydantic 2.10.6
- Pytest 8.3.5

V100 semantic encoding uses FP16 because compute capability 7.0 does not provide
native BF16 execution.

## Validation results

1. The four v45 test modules completed with `14 passed`; the four emitted warnings
   concern an optional nested-tensor optimization and do not change correctness.
2. A full 160,120,334-parameter v45 neural model completed one FP16 forward and
   backward pass on one V100. The loss was finite and peak allocated memory was
   1,245.22 MiB for batch size 2 with 16 candidates.
3. The formal dataset loaded successfully with 6,763 events and 31 candidates.
4. A real eight-event semantic-cache smoke run loaded the transferred E5 model
   locally, encoded 40 structured state fields and all 31 candidates, and produced
   finite FP16 arrays of shapes `[8, 5, 768]` and `[31, 768]`.
5. The smoke path made zero real external endpoint calls and encoded neither task
   identifiers nor outcome fields.

## Execution note

The machine has Slurm client binaries but no usable Slurm configuration file.
Experiments therefore need to run directly, preferably inside `tmux`, with an
explicit `CUDA_VISIBLE_DEVICES` assignment. No formal cache build or 15-fit frozen
training experiment was started during deployment validation.
