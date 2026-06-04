#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ---------------------------------------------------------------------------
# run_benchmark_group.sbatch -- one allocation, many runs, ONE stage_data
# ---------------------------------------------------------------------------
#
# Sister of run_benchmark.sbatch.  Stages the dataset to local NVMe ONCE per
# allocation, then loops through a list of (model, dataset, sampling) entries
# and runs each one in sequence inside the same allocation.  This eliminates
# the redundant 25-30 min/run Lustre->NVMe stage cost for short benchmark
# matrices, and -- more importantly -- it stops 16 simultaneous jobs from
# thrashing the Lustre MDS by serializing the stage step inside one node's
# allocation.
#
# Required env vars (set by submit_all.sh --group-nvme):
#   RUN_GROUP_JSON     path to JSON array, one object per run, with keys:
#                        model, dataset, num_gpus, sampling, run_id,
#                        results_dir, num_epochs, precision, compile,
#                        extra_overrides (last 4 optional)
#   LUSTRE_DATA_ROOT   path to dataset on Lustre (the SHARED source for
#                      the whole group).  All entries in the group must
#                      resolve to the SAME LUSTRE_DATA_ROOT (i.e. they
#                      share a `dataset_family`).
#   GROUP_LABEL        human-readable group key (e.g. drivaer_ml_n1) for
#                      logging only.
#
# Optional env vars (with defaults inherited from run_benchmark.sbatch):
#   NVME_STAGE_ROOT, CONTAINER_IMAGE, CONTAINER_MOUNTS, VENV_PATH, WORKDIR,
#   USER_LUSTRE, GROUP_LUSTRE, CLEANUP_NVME (default 1)
#
# This script writes:
#   ${RESULTS_DIR}/slurm-${SLURM_JOB_ID}.{out,err}     per-entry, via tee
#   ${RUN_GROUP_DIR}/group_summary.tsv                 (jobid run_id rc)
#
# DESIGN NOTES (this script differs from run_benchmark.sbatch in that):
#   - the Lustre warmup + stage happen ONCE up front
#   - the train+summarize loop is wrapped in `set +e` per-iteration so one
#     OOM does not abort the whole group
#   - MASTER_PORT is rotated each iteration to avoid TCPStore collisions
#     when a previous run's torchrun teardown is slow
#   - cleanup of /tmp/aero_data happens at the end, AFTER all entries
#
# DO NOT remove the Lustre stat-cache warmup -- the [-d] guard removal in
# run_benchmark.sbatch is the same fix here.

#SBATCH --account=coreai_modulus_cae
#SBATCH --partition=batch
#SBATCH --job-name=cae_bench_group
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail

: "${RUN_GROUP_JSON:?RUN_GROUP_JSON must be set (path to group spec json)}"
: "${LUSTRE_DATA_ROOT:?LUSTRE_DATA_ROOT must be set for an NVMe group}"

GROUP_LABEL="${GROUP_LABEL:-unnamed_group}"
NVME_STAGE_ROOT="${NVME_STAGE_ROOT:-/tmp/aero_data}"
CLEANUP_NVME="${CLEANUP_NVME:-1}"

USER_LUSTRE="${USER_LUSTRE:-/lustre/fs1/portfolios/coreai/projects/coreai_modulus_cae/users/nsobhani}"
GROUP_LUSTRE_DEFAULT="/lustre/fs1/portfolios/coreai/projects/coreai_modulus_cae"
GROUP_LUSTRE="${GROUP_LUSTRE:-${GROUP_LUSTRE_DEFAULT}}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-/lustre/fs1/portfolios/coreai/projects/coreai_modulus_cae/users/nsobhani/nvidia+pytorch+26.04-py3.sqsh}"
CONTAINER_MOUNTS="${CONTAINER_MOUNTS:-${USER_LUSTRE}:/user_data,${GROUP_LUSTRE}:/group_data,/lustre:/lustre,/home:/home,/tmp:/tmp}"
VENV_PATH="${VENV_PATH:-/lustre/fs1/portfolios/coreai/projects/coreai_modulus_cae/users/nsobhani/venvs/pytorch-26.04-py3-physnemo}"
WORKDIR="${WORKDIR:-${USER_LUSTRE}/workdir/physicsnemo_cae_benchmark/examples/cfd/external_aerodynamics/unified_external_aero_recipe}"
BENCHMARKS_DIR="${WORKDIR}/benchmarks"

if [ ! -f "${RUN_GROUP_JSON}" ]; then
    echo "[run_group] ERROR: ${RUN_GROUP_JSON} not found" >&2
    exit 2
fi
if ! command -v jq &>/dev/null; then
    # Inside the container we have python; outside we need jq for parsing.
    # The login-node submit step already validated jq is present, but we
    # are now on a compute node, so fall back to a tiny python shim if
    # jq isn't available here.
    JQ() { python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(eval(sys.argv[2]))' "$@"; }
else
    JQ() { jq "$@"; }
fi

NUM_RUNS=$(jq 'length' "${RUN_GROUP_JSON}")
EXCLUDE_EPOCHS="${EXCLUDE_EPOCHS:-0}"
if [ "${NUM_RUNS}" -eq 0 ]; then
    echo "[run_group] WARN: empty RUN_GROUP_JSON, nothing to do" >&2
    exit 0
fi

echo "[run_group] ======================================================"
echo "[run_group] group=${GROUP_LABEL}  job_id=${SLURM_JOB_ID:-?}  nodes=${SLURM_JOB_NUM_NODES:-1}"
echo "[run_group] runs=${NUM_RUNS}  spec=${RUN_GROUP_JSON}"
echo "[run_group] lustre_src=${LUSTRE_DATA_ROOT}"
echo "[run_group] nvme_root=${NVME_STAGE_ROOT}"
echo "[run_group] ======================================================"

# Echo the run plan up front so the slurm log is self-describing.
jq -r '.[] | "[run_group] PLAN  \(.run_id)  model=\(.model)  dataset=\(.dataset)  gpus=\(.num_gpus)  sub=\(.sampling)"' \
    "${RUN_GROUP_JSON}"

# ---------------------------------------------------------------------------
# 1. Lustre stat-cache warmup (whole group, once).
#    Same logic as run_benchmark.sbatch -- must NOT be gated on [-d ...]
#    because cold MDS makes [-d] return false on a path that exists.
# ---------------------------------------------------------------------------
echo "[run_group] warming Lustre stat cache for ${LUSTRE_DATA_ROOT} on all nodes (full depth)"
T_WARM=$SECONDS
srun --ntasks-per-node=1 --ntasks="${SLURM_JOB_NUM_NODES:-1}" \
     bash -c "find '${LUSTRE_DATA_ROOT}' -printf '' 2>/dev/null || true" \
     || true
echo "[run_group] cache warm took $((SECONDS - T_WARM))s"
if [ ! -d "${LUSTRE_DATA_ROOT}" ]; then
    echo "[run_group] ERROR: ${LUSTRE_DATA_ROOT} not visible after Lustre warmup" >&2
    ls -la "$(dirname "${LUSTRE_DATA_ROOT}")" >&2 || true
    exit 4
fi

# ---------------------------------------------------------------------------
# 2. Resolve the dataset family from the FIRST entry, then assert all
#    entries share the same family.  We refuse to stage twice in one
#    allocation -- if the user mixed datasets, that's a submit_all.sh bug.
# ---------------------------------------------------------------------------
_dataset_to_key() {
    case "$1" in
        drivaer_ml_surface|drivaer_ml_volume) echo "drivaer_ml" ;;
        highlift_surface|highlift_volume)     echo "highlift_aero_ml" ;;
        drivesim_volume)                      echo "drivesim" ;;
        shift_suv_*)                          echo "shift_suv" ;;
        *) echo "" ;;
    esac
}

FIRST_DATASET=$(jq -r '.[0].dataset' "${RUN_GROUP_JSON}")
DATAPATH_KEY=$(_dataset_to_key "${FIRST_DATASET}")
if [ -z "${DATAPATH_KEY}" ]; then
    echo "[run_group] ERROR: unknown dataset family for ${FIRST_DATASET}" >&2
    exit 2
fi

while IFS= read -r d; do
    k=$(_dataset_to_key "${d}")
    if [ "${k}" != "${DATAPATH_KEY}" ]; then
        echo "[run_group] ERROR: mixed dataset families in group (${k} vs ${DATAPATH_KEY})" >&2
        exit 2
    fi
done < <(jq -r '.[].dataset' "${RUN_GROUP_JSON}")

DATAPATH_ENV="DATASET_PATH_$(printf '%s' "${DATAPATH_KEY}" | tr '[:lower:]' '[:upper:]')"
NVME_DEST="${NVME_STAGE_ROOT}/${DATAPATH_KEY}"
MANIFEST_PATH="${LUSTRE_DATA_ROOT}/manifest.json"
if [ ! -f "${MANIFEST_PATH}" ]; then
    echo "[run_group] ERROR: manifest not found at ${MANIFEST_PATH}" >&2
    exit 3
fi

# ---------------------------------------------------------------------------
# 3. Stage the dataset ONCE for the whole group.
# ---------------------------------------------------------------------------
echo "[run_group] STAGE  ${LUSTRE_DATA_ROOT} -> ${NVME_DEST} on ${SLURM_JOB_NUM_NODES:-1} node(s)"
T_STAGE=$SECONDS
srun --ntasks-per-node=1 --ntasks="${SLURM_JOB_NUM_NODES:-1}" \
     --container-image="${CONTAINER_IMAGE}" \
     --container-mounts="${CONTAINER_MOUNTS}" \
     bash "${BENCHMARKS_DIR}/stage_data.sh" \
          "${LUSTRE_DATA_ROOT}" "${NVME_DEST}" "${MANIFEST_PATH}" \
          train val
STAGE_DT=$(( SECONDS - T_STAGE ))
export "${DATAPATH_ENV}=${NVME_DEST}"
echo "[run_group] stage-in complete in ${STAGE_DT}s; ${DATAPATH_ENV}=${NVME_DEST}"

export MASTER_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST:-localhost}" | head -n 1)
MASTER_PORT_BASE="${MASTER_PORT_BASE:-29500}"

# Per-group summary table written into the parent dir of the spec file.
GROUP_SUMMARY="$(dirname "${RUN_GROUP_JSON}")/group_${SLURM_JOB_ID:-local}_summary.tsv"
printf "idx\trun_id\tmodel\tdataset\tnum_gpus\tsub\twallclock_s\trc\n" > "${GROUP_SUMMARY}"

# ---------------------------------------------------------------------------
# 4. Loop over the entries in the group spec.  We disable -e during the
#    inner srun so a failed run does not kill the next ones; we record
#    the rc in GROUP_SUMMARY and continue.
# ---------------------------------------------------------------------------
GROUP_FAILS=0

for IDX in $(seq 0 $((NUM_RUNS - 1))); do
    ENTRY=$(jq -c ".[${IDX}]" "${RUN_GROUP_JSON}")
    MODEL=$(echo "${ENTRY}"           | jq -r '.model')
    DATASET=$(echo "${ENTRY}"         | jq -r '.dataset')
    NUM_GPUS=$(echo "${ENTRY}"        | jq -r '.num_gpus')
    SUBSAMPLING=$(echo "${ENTRY}"     | jq -r '.sampling')
    RUN_ID=$(echo "${ENTRY}"          | jq -r '.run_id')
    RESULTS_DIR=$(echo "${ENTRY}"     | jq -r '.results_dir')
    NUM_EPOCHS=$(echo "${ENTRY}"      | jq -r '.num_epochs // 5')
    PRECISION=$(echo "${ENTRY}"       | jq -r '.precision // "bfloat16"')
    COMPILE=$(echo "${ENTRY}"         | jq -r '.compile // true')
    EXTRA_OVERRIDES=$(echo "${ENTRY}" | jq -r '.extra_overrides // ""')

    case "${RESULTS_DIR}" in
        /*) ;;
        *)  RESULTS_DIR="${WORKDIR}/${RESULTS_DIR}" ;;
    esac
    mkdir -p "${RESULTS_DIR}"

    # Per-entry logs land alongside each run's results_dir for traceability.
    PER_ENTRY_OUT="${RESULTS_DIR}/slurm-${SLURM_JOB_ID:-local}.out"
    PER_ENTRY_ERR="${RESULTS_DIR}/slurm-${SLURM_JOB_ID:-local}.err"

    {
        echo "[run_group] ------------------------------------------------------"
        echo "[run_group] ENTRY ${IDX}/${NUM_RUNS}  ${RUN_ID}"
        echo "[run_group]   model=${MODEL}  dataset=${DATASET}  gpus=${NUM_GPUS}  sub=${SUBSAMPLING}"
        echo "[run_group]   results_dir=${RESULTS_DIR}"
        echo "[run_group] ------------------------------------------------------"
    } | tee -a "${PER_ENTRY_OUT}"

    # Rotate MASTER_PORT so a slow TCPStore teardown from a previous
    # iteration cannot collide with the next iteration's bind.
    export MASTER_PORT=$(( MASTER_PORT_BASE + IDX ))

    HYDRA_OVERRIDES=(
        "model=${MODEL}"
        "dataset=${DATASET}"
        "sampling_resolution=${SUBSAMPLING}"
        "training.num_epochs=${NUM_EPOCHS}"
        "precision=${PRECISION}"
        "compile=${COMPILE}"
        "run_id=${RUN_ID}"
        "output_dir=${RESULTS_DIR}/runs"
    )
    if [ -n "${EXTRA_OVERRIDES}" ]; then
        # shellcheck disable=SC2206
        EXTRAS_ARR=(${EXTRA_OVERRIDES})
        HYDRA_OVERRIDES+=("${EXTRAS_ARR[@]}")
    fi

    # Match per-row srun sizing: NUM_GPUS<4 => true single-replica.
    # Note this only works when the GROUP'S allocation has at least
    # NUM_GPUS gpus per node; submit_all.sh sizes the group allocation
    # at min(4, max-num_gpus-in-group), so a mixed g=1+g=4 group still
    # has 4 GPUs/node and can run the g=1 entry on just 1 of them.
    if [ "${NUM_GPUS}" -lt 4 ]; then
        SRUN_TASKS_PER_NODE="${NUM_GPUS}"
    else
        SRUN_TASKS_PER_NODE=4
    fi
    T0=$SECONDS
    set +e
    srun --ntasks-per-node="${SRUN_TASKS_PER_NODE}" \
         --container-image="${CONTAINER_IMAGE}" \
         --container-mounts="${CONTAINER_MOUNTS}" \
         --export=ALL,MASTER_ADDR,MASTER_PORT,${DATAPATH_ENV} \
         bash -c "
            set -euo pipefail
            source ${VENV_PATH}/bin/activate
            cd ${WORKDIR}
            export RANK=\${SLURM_PROCID}
            export WORLD_SIZE=\${SLURM_NTASKS}
            export LOCAL_RANK=\${SLURM_LOCALID}
            python src/train.py ${HYDRA_OVERRIDES[*]}
         " > >(tee -a "${PER_ENTRY_OUT}") 2> >(tee -a "${PER_ENTRY_ERR}" >&2)
    TRAIN_RC=$?
    set -e
    DT=$(( SECONDS - T0 ))
    echo "[run_group] ENTRY ${IDX} ${RUN_ID} train rc=${TRAIN_RC} (${DT}s)" \
        | tee -a "${PER_ENTRY_OUT}"

    RUN_OUTPUT_DIR="${RESULTS_DIR}/runs/${RUN_ID}"
    METRICS_PATH="${RUN_OUTPUT_DIR}/metrics.jsonl"
    if [ "${TRAIN_RC}" -eq 0 ] && [ -f "${METRICS_PATH}" ]; then
        set +e
        srun --ntasks=1 --ntasks-per-node=1 \
             --container-image="${CONTAINER_IMAGE}" \
             --container-mounts="${CONTAINER_MOUNTS}" \
             bash -c "
                source ${VENV_PATH}/bin/activate
                python ${BENCHMARKS_DIR}/ingest/summarize_run.py \
                    --metrics ${METRICS_PATH} \
                    --run-id ${RUN_ID} \
                    --model ${MODEL} \
                    --dataset ${DATASET} \
                    --num-gpus ${NUM_GPUS} \
                    --storage nvme \
                    --sampling ${SUBSAMPLING} \
                    --num-epochs ${NUM_EPOCHS} \
                    --exclude-epochs ${EXCLUDE_EPOCHS}
             " > >(tee -a "${PER_ENTRY_OUT}") 2> >(tee -a "${PER_ENTRY_ERR}" >&2)
        SUM_RC=$?
        set -e
        if [ "${SUM_RC}" -ne 0 ]; then
            echo "[run_group] WARN ${RUN_ID} summarize_run.py rc=${SUM_RC}" \
                | tee -a "${PER_ENTRY_ERR}" >&2
        fi
    else
        echo "[run_group] SKIP summarize for ${RUN_ID}: train_rc=${TRAIN_RC} metrics=$([ -f "${METRICS_PATH}" ] && echo yes || echo no)" \
            | tee -a "${PER_ENTRY_ERR}" >&2
    fi

    printf "%d\t%s\t%s\t%s\t%s\t%s\t%d\t%d\n" \
        "${IDX}" "${RUN_ID}" "${MODEL}" "${DATASET}" \
        "${NUM_GPUS}" "${SUBSAMPLING}" "${DT}" "${TRAIN_RC}" \
        >> "${GROUP_SUMMARY}"

    if [ "${TRAIN_RC}" -ne 0 ]; then
        GROUP_FAILS=$((GROUP_FAILS + 1))
    fi
done

# ---------------------------------------------------------------------------
# 5. NVMe cleanup ONCE for the group.
# ---------------------------------------------------------------------------
if [ "${CLEANUP_NVME}" = "1" ]; then
    echo "[run_group] cleaning up ${NVME_DEST}"
    srun --ntasks-per-node=1 --ntasks="${SLURM_JOB_NUM_NODES:-1}" \
         bash -c "rm -rf '${NVME_DEST}'" || true
fi

echo "[run_group] ======================================================"
echo "[run_group] group=${GROUP_LABEL} done.  runs=${NUM_RUNS}  fails=${GROUP_FAILS}  stage=${STAGE_DT}s"
echo "[run_group] summary: ${GROUP_SUMMARY}"
cat "${GROUP_SUMMARY}"
echo "[run_group] ======================================================"

if [ "${GROUP_FAILS}" -gt 0 ]; then
    exit 1
fi
exit 0
