#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
#
# VTK/VTU load benchmark @ ShiftSUV (manifest-aligned runs). CPU-only — no GPU.
#
# Default: 3 warm VTU loads + HSG cold-disk reference for epoch projection.
# Optional PNM comparison chart: E2E_COMPARE_PNM=1 sbatch ...
#
#   sbatch benchmarks/run_vtu_vs_pnm_e2e.sbatch

#SBATCH --account=coreai_modulus_cae
#SBATCH --partition=batch
#SBATCH --job-name=vtu_load
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm-vtu-load-%j.out
#SBATCH --error=slurm-vtu-load-%j.err

set -euo pipefail

USER_LUSTRE="${USER_LUSTRE:-/lustre/fs1/portfolios/coreai/projects/coreai_modulus_cae/users/nsobhani}"
GROUP_LUSTRE="${GROUP_LUSTRE:-/lustre/fs1/portfolios/coreai/projects/coreai_modulus_cae}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-/lustre/fs1/portfolios/coreai/projects/coreai_modulus_cae/users/nsobhani/nvidia+pytorch+26.04-py3.sqsh}"
CONTAINER_MOUNTS="${CONTAINER_MOUNTS:-${USER_LUSTRE}:/user_data,${GROUP_LUSTRE}:/group_data,/lustre:/lustre,/home:/home,/tmp:/tmp}"
VENV_PATH="${VENV_PATH:-/lustre/fs1/portfolios/coreai/projects/coreai_modulus_cae/users/nsobhani/venvs/pytorch-26.04-py3-physnemo}"
WORKDIR_RAW="${WORKDIR:-${USER_LUSTRE}/workdir/physicsnemo_cae_benchmark/examples/cfd/external_aerodynamics/unified_external_aero_recipe}"
WORKDIR="$(realpath "${WORKDIR_RAW}")"
BENCHMARKS_DIR="${WORKDIR}/benchmarks"

E2E_DATASET="${E2E_DATASET:-shift_suv_estate_surface}"
E2E_MODEL="${E2E_MODEL:-geotransolver_surface}"
E2E_SUBS="${E2E_SUBS:-200000}"
E2E_MAX_SAMPLES="${E2E_MAX_SAMPLES:-3}"
E2E_COMPARE_PNM="${E2E_COMPARE_PNM:-1}"

export DATASET_PATH_SHIFT_SUV="${DATASET_PATH_SHIFT_SUV:-${GROUP_LUSTRE}/datasets/PhysicsNeMo-ShiftSUV}"
export DATASET_PATH_SHIFT_SUV_VTU="${DATASET_PATH_SHIFT_SUV_VTU:-${GROUP_LUSTRE}/datasets/shift_suv/SUV/AeroSUV_full_scale_estate_transient}"

echo "[vtu_load] job=${SLURM_JOB_ID:-local} node=${SLURMD_NODENAME:-?}"
echo "[vtu_load] pnm_manifest=${DATASET_PATH_SHIFT_SUV}"
echo "[vtu_load] vtu=${DATASET_PATH_SHIFT_SUV_VTU}"
echo "[vtu_load] dataset=${E2E_DATASET} max_samples=${E2E_MAX_SAMPLES} compare_pnm=${E2E_COMPARE_PNM}"

COMPARE_ARGS=()
if [ "${E2E_COMPARE_PNM}" = "1" ]; then
    COMPARE_ARGS+=(--compare-pnm)
fi

srun --ntasks=1 \
    --container-image="${CONTAINER_IMAGE}" \
    --container-mounts="${CONTAINER_MOUNTS}" \
    bash -c "
        set -euo pipefail
        source ${VENV_PATH}/bin/activate
        cd ${WORKDIR}
        export DATASET_PATH_SHIFT_SUV='${DATASET_PATH_SHIFT_SUV}'
        export DATASET_PATH_SHIFT_SUV_VTU='${DATASET_PATH_SHIFT_SUV_VTU}'
        python ${BENCHMARKS_DIR}/sweeps/sweep_vtu_vs_pnm_e2e.py \
            --subs ${E2E_SUBS} \
            --model ${E2E_MODEL} \
            --dataset ${E2E_DATASET} \
            --max-samples ${E2E_MAX_SAMPLES} \
            ${COMPARE_ARGS[@]}
    "

echo "[vtu_load] done"
