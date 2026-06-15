#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
#
# ---------------------------------------------------------------------------
# run_pytorch_measure.sbatch — measured PyTorch vs PhysicsNeMo train-step sweep
# ---------------------------------------------------------------------------
#
# Runs ``sweep_pytorch_measure.py`` on a single B200 GPU: for each subsampling
# level, ``profile_and_attribute.py --mode measure`` profiles one real
# GeoTransolver Volume training step with torch baseline forced, then with
# default PhysicsNeMo dispatch (torch RS rank=0, warp rank=1).
# ``results/_profile_attribute/pytorch_measured.json`` and optionally regenerate
# ``results/_scaling_snapshot/13_geotransolver_volume_training_latency.png``.
#
# Default dataset: DrivAerML volume (``DATASET_PATH_DRIVAER_ML`` on fs1 Lustre).
# Override with ``MEASURE_DATASET=shift_suv_estate_volume`` etc.
#
# Optional env vars:
#   MEASURE_MODEL         default geotransolver_volume
#   MEASURE_DATASET       default drivaer_ml_volume
#   MEASURE_SUBS          CSV subsampling levels (default 10000,50000,...,400000)
#   MEASURE_N_WARMUP      default 5
#   MEASURE_N_RECORD      default 10
#   MEASURE_REGEN_PLOT    default 1 (regenerate latency chart when done)
#   MEASURE_STORAGE       nvme|lustre for dataset + plot PNM bars (default lustre)
#   NVME_STAGE_ROOT       default /tmp/aero_data (same as run_benchmark.sbatch)
#   LUSTRE_DATA_ROOT      dataset source tree on Lustre (auto by MEASURE_DATASET)
#   VENV_PATH, WORKDIR, CONTAINER_IMAGE, CONTAINER_MOUNTS — same as run_benchmark.sbatch

#SBATCH --account=coreai_modulus_cae
#SBATCH --partition=batch
#SBATCH --job-name=pytorch_measure
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
PHYSICSNEMO_ROOT="${PHYSICSNEMO_ROOT:-$(realpath "${WORKDIR}/../../../..")}"

MEASURE_MODEL="${MEASURE_MODEL:-geotransolver_volume}"
MEASURE_DATASET="${MEASURE_DATASET:-drivaer_ml_volume}"
MEASURE_SUBS="${MEASURE_SUBS:-10000,50000,100000,200000,300000,400000}"
MEASURE_N_WARMUP="${MEASURE_N_WARMUP:-5}"
MEASURE_N_RECORD="${MEASURE_N_RECORD:-10}"
MEASURE_REGEN_PLOT="${MEASURE_REGEN_PLOT:-1}"
MEASURE_STORAGE="${MEASURE_STORAGE:-lustre}"
NVME_STAGE_ROOT="${NVME_STAGE_ROOT:-/tmp/aero_data}"
REPORTS_DIR="${REPORTS_DIR:-${WORKDIR}/results/_profile_attribute_drivaer}"

case "${MEASURE_DATASET}" in
    drivaer_ml_surface|drivaer_ml_volume)
        DATAPATH_KEY="drivaer_ml"
        NVME_STAGE_NAME="drivaer_ml"
        DEFAULT_LUSTRE="${LUSTRE_DATA_ROOT_DRIVAER_ML:-${GROUP_LUSTRE}/datasets/PhysicsNeMo-DrivaerML}"
        ;;
    highlift_surface|highlift_volume)
        DATAPATH_KEY="highlift_aero_ml"
        NVME_STAGE_NAME="highlift_aero_ml"
        DEFAULT_LUSTRE="${LUSTRE_DATA_ROOT_HIGHLIFT:-${GROUP_LUSTRE}/datasets/PhysicsNeMo-HighLiftAeroML}"
        ;;
    shift_suv_*)
        DATAPATH_KEY="shift_suv"
        NVME_STAGE_NAME="shift_suv"
        DEFAULT_LUSTRE="${LUSTRE_DATA_ROOT_SHIFT_SUV:-${GROUP_LUSTRE}/datasets/PhysicsNeMo-ShiftSUV}"
        ;;
    *)
        echo "[pytorch_measure] unknown MEASURE_DATASET=${MEASURE_DATASET}; cannot map to dataset_paths key" >&2
        exit 2
        ;;
esac

DATAPATH_ENV="DATASET_PATH_$(printf '%s' "${DATAPATH_KEY}" | tr '[:lower:]' '[:upper:]')"
LUSTRE_DATA_ROOT="${LUSTRE_DATA_ROOT:-${DEFAULT_LUSTRE}}"

mkdir -p "${REPORTS_DIR}"

echo "[pytorch_measure] job_id=${SLURM_JOB_ID:-?}  node=${SLURMD_NODENAME:-?}"
echo "[pytorch_measure] workdir=${WORKDIR}"
echo "[pytorch_measure] model=${MEASURE_MODEL}  dataset=${MEASURE_DATASET}"
echo "[pytorch_measure] lustre_data=${LUSTRE_DATA_ROOT}"
echo "[pytorch_measure] storage=${MEASURE_STORAGE}"
echo "[pytorch_measure] subs=${MEASURE_SUBS}"
echo "[pytorch_measure] reports_dir=${REPORTS_DIR}"
echo "[pytorch_measure] datapath_env=${DATAPATH_ENV}"

# Warm Lustre stat cache (same rationale as run_benchmark.sbatch).
if [ -n "${LUSTRE_DATA_ROOT}" ]; then
    echo "[pytorch_measure] warming Lustre stat cache for ${LUSTRE_DATA_ROOT}"
    srun --ntasks-per-node=1 --ntasks=1 \
         bash -c "find '${LUSTRE_DATA_ROOT}' -printf '' 2>/dev/null || true" || true
fi

DATASET_EXPORT=""
if [ "${MEASURE_STORAGE}" = "nvme" ]; then
    NVME_DEST="${NVME_STAGE_ROOT}/${NVME_STAGE_NAME}"
    MANIFEST_PATH="${LUSTRE_DATA_ROOT}/manifest.json"
    if [ ! -f "${MANIFEST_PATH}" ]; then
        echo "[pytorch_measure] ERROR: manifest not found at ${MANIFEST_PATH}" >&2
        exit 3
    fi
    echo "[pytorch_measure] staging ${LUSTRE_DATA_ROOT} -> ${NVME_DEST}"
    srun --ntasks-per-node=1 --ntasks=1 \
         --container-image="${CONTAINER_IMAGE}" \
         --container-mounts="${CONTAINER_MOUNTS}" \
         bash "${BENCHMARKS_DIR}/stage_data.sh" \
              "${LUSTRE_DATA_ROOT}" "${NVME_DEST}" "${MANIFEST_PATH}" \
              train val
    DATASET_EXPORT="export ${DATAPATH_ENV}=${NVME_DEST}"
    echo "[pytorch_measure] ${DATAPATH_ENV}=${NVME_DEST}"
else
    DATASET_EXPORT="export ${DATAPATH_ENV}=${LUSTRE_DATA_ROOT}"
    echo "[pytorch_measure] ${DATAPATH_ENV}=${LUSTRE_DATA_ROOT}"
fi

# shellcheck disable=SC2206
SUB_ARR=( ${MEASURE_SUBS//,/ } )
REGEN_FLAG=""
if [ "${MEASURE_REGEN_PLOT}" = "1" ]; then
    REGEN_FLAG="--regen-plot --storage ${MEASURE_STORAGE}"
fi

srun --ntasks=1 --ntasks-per-node=1 \
     --container-image="${CONTAINER_IMAGE}" \
     --container-mounts="${CONTAINER_MOUNTS}" \
     bash -c "
        set -euo pipefail
        ${DATASET_EXPORT}
        source ${VENV_PATH}/bin/activate
        cd ${WORKDIR}
        export PROFILE_MODEL=${MEASURE_MODEL}
        export PROFILE_DATASET=${MEASURE_DATASET}
        python ${BENCHMARKS_DIR}/sweeps/sweep_pytorch_measure.py \
            --device cuda \
            --model ${MEASURE_MODEL} \
            --dataset ${MEASURE_DATASET} \
            --n-warmup ${MEASURE_N_WARMUP} \
            --n-record ${MEASURE_N_RECORD} \
            --reports-dir ${REPORTS_DIR} \
            --asv-results ${PHYSICSNEMO_ROOT}/.asv/results \
            --subs ${SUB_ARR[*]} \
            ${REGEN_FLAG}
     "

echo "[pytorch_measure] done. Outputs:"
ls -la "${REPORTS_DIR}" || true
