<!-- markdownlint-disable -->
# CAE Benchmark — Scalability Sweep Pipeline

Main multi-GPU training sweep: generate `matrix.json`, dispatch with `submit_all.sh`,
run training on SLURM via the sbatch scripts.

**Nothing auto-runs the full chain.** Generate the matrix, then call `submit_all.sh`.

## Do these run on their own?

| Artifact | Standalone? | Notes |
|----------|-------------|-------|
| `matrix.json` | **No** | Data only. Generate with `matrix/benchmark_matrix.py`. |
| `submit_all.sh` | **Yes** | You run this on the login node to start a sweep. |
| `run_benchmark.sbatch` | **Yes** | Via `sbatch` — one matrix row / one config. Usually launched by `submit_all.sh`. |
| `run_benchmark_group.sbatch` | **Rarely** | Only with `submit_all.sh --group-nvme` (batched NVMe staging). |
| `stage_data.sh` | **Yes** | Standalone CLI for I/O debugging; normally called by sbatch when `STORAGE_MODE=nvme`. |
