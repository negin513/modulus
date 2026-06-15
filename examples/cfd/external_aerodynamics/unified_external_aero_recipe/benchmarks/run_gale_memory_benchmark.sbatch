#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
#
# GALE-only subsample sweep (FLARE off) — peak memory vs subsampling on ShiftSUV.
# Skips GALE_FA runs; ~half the GPU time of run_flare_attention_benchmark.sbatch.
#
# Default subs 250k–500k (50k–200k GALE already in _flare_attention_shiftsuv from
# the full A/B job). Override FLARE_SUBS for a full 50k–500k GALE-only matrix.
#
# Optional env:
#   FLARE_SUBS, FLARE_MODEL, FLARE_DATASET, FLARE_NUM_EPOCHS, FLARE_RESULTS_DIR
#   LUSTRE_DATA_ROOT_SHIFT_SUV

#SBATCH --account=coreai_modulus_cae
#SBATCH --partition=batch
#SBATCH --job-name=gale_memory
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail

USER_LUSTRE="${USER_LUSTRE:-/lustre/fs1/portfolios/coreai/projects/coreai_modulus_cae/users/nsobhani}"
GROUP_LUSTRE="${GROUP_LUSTRE:-/lustre/fs1/portfolios/coreai/projects/coreai_modulus_cae}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-/lustre/fs1/portfolios/coreai/projects/coreai_modulus_cae/users/nsobhani/nvidia+pytorch+26.04-py3.sqsh}"
CONTAINER_MOUNTS="${CONTAINER_MOUNTS:-${USER_LUSTRE}:/user_data,${GROUP_LUSTRE}:/group_data,/lustre:/lustre,/home:/home,/tmp:/tmp}"
VENV_PATH="${VENV_PATH:-/lustre/fs1/portfolios/coreai/projects/coreai_modulus_cae/users/nsobhani/venvs/pytorch-26.04-py3-physnemo}"
WORKDIR_RAW="${WORKDIR:-${USER_LUSTRE}/workdir/physicsnemo_cae_benchmark/examples/cfd/external_aerodynamics/unified_external_aero_recipe}"
WORKDIR="$(realpath "${WORKDIR_RAW}")"
BENCHMARKS_DIR="${WORKDIR}/benchmarks"

FLARE_SUBS="${FLARE_SUBS:-250000,300000,350000,400000,450000,500000}"
FLARE_MODEL="${FLARE_MODEL:-geotransolver_surface}"
FLARE_DATASET="${FLARE_DATASET:-shift_suv_estate_surface}"
FLARE_NUM_EPOCHS="${FLARE_NUM_EPOCHS:-5}"
FLARE_EXCLUDE_EPOCHS="${FLARE_EXCLUDE_EPOCHS:-0}"
FLARE_RESULTS_DIR="${FLARE_RESULTS_DIR:-${WORKDIR}/results/_gale_memory_shiftsuv}"
LUSTRE_DATA_ROOT="${LUSTRE_DATA_ROOT:-${LUSTRE_DATA_ROOT_SHIFT_SUV:-${GROUP_LUSTRE}/datasets/PhysicsNeMo-ShiftSUV}}"

mkdir -p "${FLARE_RESULTS_DIR}"

echo "[gale_mem] job=${SLURM_JOB_ID:-?} node=${SLURMD_NODENAME:-?}"
echo "[gale_mem] workdir=${WORKDIR}"
echo "[gale_mem] dataset=${LUSTRE_DATA_ROOT}"
echo "[gale_mem] results=${FLARE_RESULTS_DIR}"
echo "[gale_mem] model=${FLARE_MODEL}  dataset=${FLARE_DATASET}"
echo "[gale_mem] subs=${FLARE_SUBS}  attention=gale (FLARE off)"

if [ -n "${LUSTRE_DATA_ROOT}" ]; then
    echo "[gale_mem] warming Lustre stat cache"
    srun --ntasks=1 --ntasks-per-node=1 \
         bash -c "find '${LUSTRE_DATA_ROOT}' -printf '' 2>/dev/null || true" || true
fi

srun --ntasks=1 --ntasks-per-node=1 \
     --container-image="${CONTAINER_IMAGE}" \
     --container-mounts="${CONTAINER_MOUNTS}" \
     bash -c "
        set -euo pipefail
        export DATASET_PATH_SHIFT_SUV='${LUSTRE_DATA_ROOT}'
        source ${VENV_PATH}/bin/activate
        cd ${WORKDIR}
        python ${BENCHMARKS_DIR}/sweeps/sweep_flare_attention.py \
            --results-dir ${FLARE_RESULTS_DIR} \
            --model ${FLARE_MODEL} \
            --dataset ${FLARE_DATASET} \
            --attention-modes gale \
            --storage lustre \
            --precision bfloat16 \
            --num-epochs ${FLARE_NUM_EPOCHS} \
            --exclude-epochs ${FLARE_EXCLUDE_EPOCHS} \
            --subs ${FLARE_SUBS} \
            --resume
        python ${BENCHMARKS_DIR}/plots/plot_flare_attention.py \
            --input ${FLARE_RESULTS_DIR}/flare_attention_results.json \
            --out-dir ${FLARE_RESULTS_DIR}/plots \
            --gale-only
     "

echo "[gale_mem] done"
ls -la "${FLARE_RESULTS_DIR}" || true
