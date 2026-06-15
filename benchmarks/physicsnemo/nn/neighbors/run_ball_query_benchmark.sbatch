#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

#SBATCH --account=coreai_modulus_cae
#SBATCH --partition=batch
#SBATCH --job-name=ball_query_bench
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --output=benchmarks/physicsnemo/nn/neighbors/slurm-%j.out
#SBATCH --error=benchmarks/physicsnemo/nn/neighbors/slurm-%j.err

set -euo pipefail

USER_LUSTRE="${USER_LUSTRE:-/lustre/fs1/portfolios/coreai/projects/coreai_modulus_cae/users/nsobhani}"
GROUP_LUSTRE="${GROUP_LUSTRE:-/lustre/fs1/portfolios/coreai/projects/coreai_modulus_cae}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-/lustre/fs1/portfolios/coreai/projects/coreai_modulus_cae/users/nsobhani/nvidia+pytorch+26.04-py3.sqsh}"
CONTAINER_MOUNTS="${CONTAINER_MOUNTS:-${USER_LUSTRE}:/user_data,${GROUP_LUSTRE}:/group_data,/lustre:/lustre,/home:/home,/tmp:/tmp}"
VENV_PATH="${VENV_PATH:-/lustre/fs1/portfolios/coreai/projects/coreai_modulus_cae/users/nsobhani/venvs/pytorch-26.04-py3-physnemo}"
REPO_ROOT="$(realpath "${SLURM_SUBMIT_DIR:-$(pwd)}")"
OUT_DIR="${REPO_ROOT}/benchmarks/physicsnemo/nn/neighbors"
JSONL="${OUT_DIR}/ball_query_benchmark.jsonl"

echo "[ball_query] job=${SLURM_JOB_ID:-?} node=${SLURMD_NODENAME:-?}"
echo "[ball_query] repo=${REPO_ROOT}"
echo "[ball_query] jsonl=${JSONL}"

srun --ntasks=1 --ntasks-per-node=1 \
     --container-image="${CONTAINER_IMAGE}" \
     --container-mounts="${CONTAINER_MOUNTS}" \
     bash -c "
        set -euo pipefail
        source ${VENV_PATH}/bin/activate
        cd ${REPO_ROOT}
        python benchmarks/physicsnemo/nn/neighbors/benchmark_ball_query.py \
            --query-sizes 512 1024 2048 4096 8192 16384 32768 65536 131072 262144 376531 \
            --target-sizes 32768 \
            --plot \
            --jsonl ${JSONL}
     "

echo "[ball_query] done"
