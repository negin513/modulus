<!-- markdownlint-disable -->
# CAE Benchmark — Scalability Sweep Pipeline

Main multi-GPU training sweep: generate `matrix.json`, dispatch with `submit_all.sh`,
run training on SLURM via the sbatch scripts.

**Nothing auto-runs the full chain.** Generate the matrix, then call `submit_all.sh`.

## submit_all.sh

Login-node dispatcher: reads `matrix.json` and submits SLURM jobs for each row
you want to run. It does **not** train models itself — it only calls `sbatch` with
the right env vars (`MODEL`, `DATASET`, `NUM_GPUS`, `STORAGE_MODE`, `SUBSAMPLING`,
`RESULTS_DIR`, `RUN_ID`, etc.).

For each matrix row it:

1. Skips rows with `"skip": true` (e.g. DriveSim not configured yet).
2. Resolves `results_dir` to an absolute path under the recipe root and creates it.
3. Submits **Lustre** rows via `run_benchmark.sbatch` (one SLURM job per row).
4. Submits **NVMe** rows via `run_benchmark.sbatch` by default, or via
   `run_benchmark_group.sbatch` when you pass `--group-nvme` (one allocation,
   one `stage_data.sh`, then multiple training runs in sequence).
5. Appends `(jobid, run_id, …)` to `submitted_jobs.tsv` for `check_runs.sh`.

Common usage:

```bash
bash submit_all.sh --matrix matrix.json --dry-run          # preview sbatch lines
bash submit_all.sh --matrix matrix.json --max-concurrent 4 # cap parallel jobs
bash submit_all.sh --matrix matrix.json --group-nvme       # batched NVMe staging
```

Requires cluster env vars (`LUSTRE_DATA_ROOT_DRIVAER_ML`, `CONTAINER_IMAGE`,
`VENV_PATH`, `WORKDIR`, `CONTAINER_MOUNTS`) — see `run_benchmark.sbatch` header.

## Do these run on their own?

| Artifact | Standalone? | Notes |
|----------|-------------|-------|
| `matrix.json` | **No** | Data only. Example in `benchmarks/matrix.json` (3 rows: 1× Lustre, 2× NVMe). Regenerate with `matrix/benchmark_matrix.py`. |
| `submit_all.sh` | **Yes** | You run this on the login node to start a sweep. |
| `run_benchmark.sbatch` | **Yes** | Via `sbatch` — one matrix row / one config. Usually launched by `submit_all.sh`. |
| `run_benchmark_group.sbatch` | **Rarely** | Only with `submit_all.sh --group-nvme` (batched NVMe staging). |
| `stage_data.sh` | **Yes** | Standalone CLI for I/O debugging; normally called by sbatch when `STORAGE_MODE=nvme`. |
