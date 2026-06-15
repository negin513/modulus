#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
#
# Runtime body for FLARE attention subsample sweep (no #SBATCH directives).
# Invoked by run_flare_attention_benchmark.sbatch and run_flare_attention_drivaer_*.sbatch.
#
# HSG: use /lustre/fs1/... for WORKDIR/container; DrivAerML is often visible on compute
# at /lustre/fsw/.../datasets/PhysicsNeMo-DrivaerML (see successful flare volume 100k).

set -euo pipefail

USER_LUSTRE="${USER_LUSTRE:-/lustre/fs1/portfolios/coreai/projects/coreai_modulus_cae/users/nsobhani}"
GROUP_LUSTRE="${GROUP_LUSTRE:-/lustre/fs1/portfolios/coreai/projects/coreai_modulus_cae}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-/lustre/fs1/portfolios/coreai/projects/coreai_modulus_cae/users/nsobhani/nvidia+pytorch+26.04-py3.sqsh}"
CONTAINER_MOUNTS="${CONTAINER_MOUNTS:-${USER_LUSTRE}:/user_data,${GROUP_LUSTRE}:/group_data,/lustre:/lustre,/home:/home,/tmp:/tmp}"
VENV_PATH="${VENV_PATH:-/lustre/fs1/portfolios/coreai/projects/coreai_modulus_cae/users/nsobhani/venvs/pytorch-26.04-py3-physnemo}"
WORKDIR_RAW="${WORKDIR:-${USER_LUSTRE}/workdir/physicsnemo_cae_benchmark/examples/cfd/external_aerodynamics/unified_external_aero_recipe}"
WORKDIR="$(realpath "${WORKDIR_RAW}")"
BENCHMARKS_DIR="${WORKDIR}/benchmarks"

FLARE_SUBS="${FLARE_SUBS:-50000,100000,150000,200000,250000,300000,350000,400000,450000,500000}"
FLARE_MODEL="${FLARE_MODEL:-geotransolver_surface}"
FLARE_DATASET="${FLARE_DATASET:-shift_suv_estate_surface}"
FLARE_NUM_EPOCHS="${FLARE_NUM_EPOCHS:-5}"
FLARE_EXCLUDE_EPOCHS="${FLARE_EXCLUDE_EPOCHS:-0}"
FLARE_COMPILE="${FLARE_COMPILE:-true}"
FLARE_RETRY_FAILED="${FLARE_RETRY_FAILED:-false}"
FLARE_ATTENTION_MODES="${FLARE_ATTENTION_MODES:-both}"
FLARE_RESULTS_DIR="${FLARE_RESULTS_DIR:-${WORKDIR}/results/_flare_attention_shiftsuv}"
DRIVAER_LUSTRE_DEFAULT="${LUSTRE_DATA_ROOT_DRIVAER_ML:-/lustre/fsw/portfolios/coreai/projects/coreai_modulus_cae/datasets/PhysicsNeMo-DrivaerML}"
SHIFT_LUSTRE_DEFAULT="${LUSTRE_DATA_ROOT_SHIFT_SUV:-${GROUP_LUSTRE}/datasets/PhysicsNeMo-ShiftSUV}"

case "${FLARE_DATASET}" in
    drivaer_ml_surface|drivaer_ml_volume)
        DATAPATH_KEY="drivaer_ml"
        LUSTRE_DATA_ROOT="${LUSTRE_DATA_ROOT:-${DRIVAER_LUSTRE_DEFAULT}}"
        ;;
    highlift_surface|highlift_volume)
        DATAPATH_KEY="highlift_aero_ml"
        LUSTRE_DATA_ROOT="${LUSTRE_DATA_ROOT:-${LUSTRE_DATA_ROOT_HIGHLIFT_AERO_ML:-${GROUP_LUSTRE}/datasets/PhysicsNeMo-HighLiftAeroML}}"
        ;;
    shift_suv_*)
        DATAPATH_KEY="shift_suv"
        LUSTRE_DATA_ROOT="${LUSTRE_DATA_ROOT:-${SHIFT_LUSTRE_DEFAULT}}"
        ;;
    *)
        echo "[flare_bench] unknown FLARE_DATASET=${FLARE_DATASET}" >&2
        exit 2
        ;;
esac
DATAPATH_ENV="DATASET_PATH_$(printf '%s' "${DATAPATH_KEY}" | tr '[:lower:]' '[:upper:]')"

mkdir -p "${FLARE_RESULTS_DIR}"

echo "[flare_bench] job=${SLURM_JOB_ID:-?} node=${SLURMD_NODENAME:-?}"
echo "[flare_bench] workdir=${WORKDIR}"
echo "[flare_bench] dataset=${LUSTRE_DATA_ROOT}"
echo "[flare_bench] results=${FLARE_RESULTS_DIR}"
echo "[flare_bench] model=${FLARE_MODEL}  dataset=${FLARE_DATASET}"
echo "[flare_bench] datapath=${DATAPATH_ENV}=${LUSTRE_DATA_ROOT}"
echo "[flare_bench] subs=${FLARE_SUBS}"
echo "[flare_bench] attention_modes=${FLARE_ATTENTION_MODES}"
echo "[flare_bench] compile=${FLARE_COMPILE}  retry_failed=${FLARE_RETRY_FAILED}"

FLARE_COMPILE_FLAG="--compile"
if [ "${FLARE_COMPILE}" = "false" ] || [ "${FLARE_COMPILE}" = "0" ]; then
    FLARE_COMPILE_FLAG="--no-compile"
fi
FLARE_RETRY_FLAG=""
if [ "${FLARE_RETRY_FAILED}" = "true" ] || [ "${FLARE_RETRY_FAILED}" = "1" ]; then
    FLARE_RETRY_FLAG="--retry-failed"
fi

if [ -n "${LUSTRE_DATA_ROOT}" ]; then
    echo "[flare_bench] warming Lustre stat cache for ${LUSTRE_DATA_ROOT}"
    srun --ntasks=1 --ntasks-per-node=1 \
         bash -c "find '${LUSTRE_DATA_ROOT}' -printf '' 2>/dev/null || true" || true
    if ! srun --ntasks=1 --ntasks-per-node=1 \
         --container-image="${CONTAINER_IMAGE}" \
         --container-mounts="${CONTAINER_MOUNTS}" \
         bash -c "test -d '${LUSTRE_DATA_ROOT}'"; then
        echo "[flare_bench] ERROR: ${LUSTRE_DATA_ROOT} not visible in container on compute node" >&2
        srun --ntasks=1 --ntasks-per-node=1 \
             --container-image="${CONTAINER_IMAGE}" \
             --container-mounts="${CONTAINER_MOUNTS}" \
             bash -c "ls -la '$(dirname "${LUSTRE_DATA_ROOT}")' 2>&1 || true" >&2 || true
        exit 1
    fi
    echo "[flare_bench] dataset path OK in container"
fi

srun --ntasks=1 --ntasks-per-node=1 \
     --container-image="${CONTAINER_IMAGE}" \
     --container-mounts="${CONTAINER_MOUNTS}" \
     bash -c "
        set -euo pipefail
        export ${DATAPATH_ENV}='${LUSTRE_DATA_ROOT}'
        source ${VENV_PATH}/bin/activate
        cd ${WORKDIR}
        python ${BENCHMARKS_DIR}/sweeps/sweep_flare_attention.py \
            --results-dir ${FLARE_RESULTS_DIR} \
            --model ${FLARE_MODEL} \
            --dataset ${FLARE_DATASET} \
            --storage lustre \
            --precision bfloat16 \
            --num-epochs ${FLARE_NUM_EPOCHS} \
            --exclude-epochs ${FLARE_EXCLUDE_EPOCHS} \
            --subs ${FLARE_SUBS} \
            --attention-modes ${FLARE_ATTENTION_MODES} \
            ${FLARE_COMPILE_FLAG} \
            --resume \
            ${FLARE_RETRY_FLAG}
     "

echo "[flare_bench] done"
ls -la "${FLARE_RESULTS_DIR}" || true
