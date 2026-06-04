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
# run_benchmark.sbatch -- one matrix entry, one sbatch invocation
# ---------------------------------------------------------------------------
#
# This script runs ONE benchmark configuration (one row from
# benchmark_matrix.py).  Per Sanjay's directive it uses `srun` directly
# (no torchrun) with env-var rendezvous, one task per GPU, four GPUs
# per node on HSG.
#
# Resource sizing (--nodes, --ntasks-per-node, --gpus-per-node) is set
# at submit time by submit_all.sh because #SBATCH directives are
# parsed before the shell runs and cannot reference variables.  Defaults
# below are sized for a 1-GPU smoke run so the script is also
# submittable directly with `sbatch run_benchmark.sbatch`.
#
# Required env vars (set via `--export=ALL,MODEL=...,DATASET=...,...`):
#   MODEL              e.g. geotransolver_volume
#   DATASET            e.g. drivaer_ml_volume   (key in datasets/<NAME>.yaml)
#   NUM_GPUS           1, 4, 16, 32, 64
#   STORAGE_MODE       lustre | nvme
#   SUBSAMPLING        sampling_resolution override
#   RESULTS_DIR        absolute path; SLURM and run outputs land here
#   RUN_ID             stable run id for the recipe's output_dir
#
# Optional env vars (with defaults):
#   NUM_EPOCHS         default 5  (Sanjay's spec)
#   PRECISION          default bfloat16
#   COMPILE            default true
#   LUSTRE_DATA_ROOT   path to dataset on Lustre (REQUIRED for nvme staging)
#   NVME_STAGE_ROOT    default /tmp/aero_data
#   CONTAINER_IMAGE    sqsh path; defaults to /lustre/fs1/portfolios/coreai/projects/coreai_modulus_cae/users/nsobhani/nvidia+pytorch+26.04-py3.sqsh
#   CONTAINER_MOUNTS   default: $USER_LUSTRE:/user_data,$GROUP_LUSTRE:/group_data,/lustre:/lustre,/home:/home,/tmp:/tmp
#   VENV_PATH          virtualenv to activate inside container
#                      (default: /lustre/fs1/portfolios/coreai/projects/coreai_modulus_cae/users/nsobhani/venvs/pytorch-26.04-py3-physnemo)
#   WORKDIR            recipe directory (parent of src/train.py)
#   EXTRA_OVERRIDES    extra Hydra overrides (space-separated string)
#   CLEANUP_NVME       1|0; default 1 (rm -rf staged data on exit)
#
# Conventions on HSG (canonical /lustre/fs1 paths -- /lustre/fsw/... is a
# login-node-only symlink farm and does not resolve inside enroot containers):
#   USER_LUSTRE=/lustre/fs1/portfolios/coreai/projects/coreai_modulus_cae/users/nsobhani
#   GROUP_LUSTRE=/lustre/fs1/portfolios/coreai/projects/coreai_modulus_cae

#SBATCH --account=coreai_modulus_cae
#SBATCH --partition=batch
#SBATCH --job-name=cae_bench
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail

# ---------------------------------------------------------------------------
# Validate required env vars (fail fast with a useful message rather than
# letting the recipe blow up later with MissingMandatoryValue).
# ---------------------------------------------------------------------------
: "${MODEL:?MODEL must be set (e.g. geotransolver_volume)}"
: "${DATASET:?DATASET must be set (e.g. drivaer_ml_volume)}"
: "${NUM_GPUS:?NUM_GPUS must be set (e.g. 1, 4, 16, 32, 64)}"
: "${STORAGE_MODE:?STORAGE_MODE must be set (lustre | nvme)}"
: "${SUBSAMPLING:?SUBSAMPLING must be set (e.g. 100000)}"
: "${RESULTS_DIR:?RESULTS_DIR must be set}"
: "${RUN_ID:?RUN_ID must be set}"

NUM_EPOCHS="${NUM_EPOCHS:-5}"
EXCLUDE_EPOCHS="${EXCLUDE_EPOCHS:-0}"
PRECISION="${PRECISION:-bfloat16}"
COMPILE="${COMPILE:-true}"
NVME_STAGE_ROOT="${NVME_STAGE_ROOT:-/tmp/aero_data}"
CLEANUP_NVME="${CLEANUP_NVME:-1}"
EXTRA_OVERRIDES="${EXTRA_OVERRIDES:-}"

# Cluster locations (HSG-specific defaults; override via env if your layout differs).
# IMPORTANT: paths MUST resolve on compute nodes. /lustre/fsw/... is a symlink
# farm that does not resolve inside enroot containers; use the canonical
# /lustre/fs1/... mount for everything we mount or `cd` into.
USER_LUSTRE="${USER_LUSTRE:-/lustre/fs1/portfolios/coreai/projects/coreai_modulus_cae/users/nsobhani}"
GROUP_LUSTRE="${GROUP_LUSTRE:-/lustre/fs1/portfolios/coreai/projects/coreai_modulus_cae}"
# NGC PyTorch 26.04-py3 sqsh, layered with the physnemo virtualenv below.
CONTAINER_IMAGE="${CONTAINER_IMAGE:-/lustre/fs1/portfolios/coreai/projects/coreai_modulus_cae/users/nsobhani/nvidia+pytorch+26.04-py3.sqsh}"
# /home is mounted as an identity bind so symlinks under $USER_LUSTRE that
# chain back into NFS home (e.g. workdir/physicsnemo_cae_benchmark ->
# /home/nsobhani/forks/physicsnemo_cae_benchmark) can resolve inside the
# container.  Without this, `cd $WORKDIR` blows up with "No such file" the
# moment the symlink target leaves /lustre.
CONTAINER_MOUNTS="${CONTAINER_MOUNTS:-${USER_LUSTRE}:/user_data,${GROUP_LUSTRE}:/group_data,/lustre:/lustre,/home:/home,/tmp:/tmp}"
VENV_PATH="${VENV_PATH:-/lustre/fs1/portfolios/coreai/projects/coreai_modulus_cae/users/nsobhani/venvs/pytorch-26.04-py3-physnemo}"
WORKDIR="${WORKDIR:-${USER_LUSTRE}/workdir/physicsnemo_cae_benchmark/examples/cfd/external_aerodynamics/unified_external_aero_recipe}"

# IMPORTANT: do NOT compute SCRIPT_DIR via $(dirname "${BASH_SOURCE[0]}").
# SLURM copies the sbatch script into a per-job spool dir (e.g.
# /cm/local/apps/slurm/var/spool/job<JOBID>/) before executing it on the
# compute node, so BASH_SOURCE[0] resolves to the spool path -- where
# our sibling scripts (stage_data.sh, summarize_run.py) do NOT exist.
# That manifests as "python: can't open file '.../summarize_run.py'".
# We anchor to the canonical $WORKDIR/benchmarks instead.
BENCHMARKS_DIR="${WORKDIR}/benchmarks"

# Anchor a relative RESULTS_DIR to WORKDIR.  train.py runs after `cd
# $WORKDIR`, so a relative output_dir lands under WORKDIR; if our caller
# (smoke_test.sh / submit_all.sh) didn't already absolutise the path we
# do it here so the post-run `[ -f $METRICS_PATH ]` check on the login
# node sees the same physical file.
case "${RESULTS_DIR}" in
    /*) ;;
    *)  RESULTS_DIR="${WORKDIR}/${RESULTS_DIR}" ;;
esac

mkdir -p "${RESULTS_DIR}"

echo "[run_benchmark] -------------------------------------------------------"
echo "[run_benchmark] job_id=${SLURM_JOB_ID:-?}  nodes=${SLURM_JOB_NUM_NODES:-1}"
echo "[run_benchmark] model=${MODEL}  dataset=${DATASET}  gpus=${NUM_GPUS}"
echo "[run_benchmark] storage=${STORAGE_MODE}  sampling=${SUBSAMPLING}"
echo "[run_benchmark] precision=${PRECISION}  compile=${COMPILE}  epochs=${NUM_EPOCHS}"
echo "[run_benchmark] results_dir=${RESULTS_DIR}"
echo "[run_benchmark] run_id=${RUN_ID}"
echo "[run_benchmark] container=${CONTAINER_IMAGE}"
echo "[run_benchmark] -------------------------------------------------------"

# ---------------------------------------------------------------------------
# 1. NVMe stage-in (one rank per node copies in parallel; xargs -P fans out
#    inside each node).  Skipped on Lustre.
# ---------------------------------------------------------------------------
# `dataset_paths` is NOT a Hydra config; it lives in datasets/dataset_paths.yaml
# and is loaded programmatically by src/datasets.load_dataset_config().  Hydra
# CLI overrides like `dataset_paths.foo=...` are rejected.  Instead, that YAML
# resolves each entry from a DATASET_PATH_<KEY> env var (oc.env interpolation),
# so we just export the env var here and let OmegaConf pick it up.
case "${DATASET}" in
    drivaer_ml_surface|drivaer_ml_volume) DATAPATH_KEY="drivaer_ml" ;;
    highlift_surface|highlift_volume)     DATAPATH_KEY="highlift_aero_ml" ;;
    drivesim_volume)                      DATAPATH_KEY="drivesim" ;;
    shift_suv_*)                          DATAPATH_KEY="shift_suv" ;;
    *)
        echo "[run_benchmark] unknown DATASET=${DATASET}; cannot map to dataset_paths key" >&2
        exit 2
        ;;
esac

# Convert dataset key (drivaer_ml) to env-var name (DATASET_PATH_DRIVAER_ML).
DATAPATH_ENV="DATASET_PATH_$(printf '%s' "${DATAPATH_KEY}" | tr '[:lower:]' '[:upper:]')"

# Warm Lustre's stat cache for the dataset tree on every compute node.
# Without this, the first dataloader glob (`Path.glob('run_*/*.pdmsh/...')`)
# AND `stage_data.sh`'s `cp -a` can race with Lustre MDS lookups and
# return ENOENT for paths that actually exist.  We've seen two flavors:
#   - shallow miss: `Path.glob` returns 9 instead of 484 (depth ~3-4).
#   - deep miss: `cp -a` fails on `<run>/.pdmsh/_tensordict/boundaries/
#     vehicle/_tensordict/<point|cell|global>_data` (depth 6-8) -- this
#     killed job 2928080 with 4 phantom-missing runs and a 33-min walltime
#     burn.
# The first `find -maxdepth 4` was sufficient for the dataloader case but
# didn't reach the `cp` failure depth.  Walking the full tree on every
# node costs ~30s for DrivAerML (484 runs * ~25 entries each) but
# eliminates the cp-retry hot path.  Done before stage-in (NVMe) and
# before training (Lustre).
if [ -n "${LUSTRE_DATA_ROOT:-}" ]; then
    # IMPORTANT: do NOT gate this warmup on `[ -d "${LUSTRE_DATA_ROOT}" ]`.
    # That test runs on the launching compute node, which is exactly where
    # Lustre's cold stat-cache returns ENOENT for paths that DO exist on
    # disk -- which is the failure we are trying to repair.  Gating the
    # warmup on the cold-cache result skipped it entirely on jobs
    # 3029150-3029153 and silently produced "Skipping dataset ... does not
    # exist" warnings followed by SIGTERM at NCCL handshake (~1:45 wall).
    echo "[run_benchmark] warming Lustre stat cache for ${LUSTRE_DATA_ROOT} on all nodes (full depth)"
    T_WARM=$SECONDS
    srun --ntasks-per-node=1 --ntasks="${SLURM_JOB_NUM_NODES:-1}" \
         bash -c "find '${LUSTRE_DATA_ROOT}' -printf '' 2>/dev/null || true" \
         || true
    echo "[run_benchmark] cache warm took $((SECONDS - T_WARM))s"

    # Fail fast if the dir is still invisible after warmup -- otherwise
    # build_dataloaders only WARNs and skips, and the job runs zero-batch
    # until SLURM kills it (looks like an OOM/preemption but isn't).
    if [ ! -d "${LUSTRE_DATA_ROOT}" ]; then
        echo "[run_benchmark] ERROR: ${LUSTRE_DATA_ROOT} not visible after Lustre warmup" >&2
        ls -la "$(dirname "${LUSTRE_DATA_ROOT}")" >&2 || true
        exit 4
    fi
fi

if [ "${STORAGE_MODE}" = "nvme" ]; then
    : "${LUSTRE_DATA_ROOT:?LUSTRE_DATA_ROOT must be set when STORAGE_MODE=nvme}"
    NVME_DEST="${NVME_STAGE_ROOT}/${DATAPATH_KEY}"
    MANIFEST_PATH="${LUSTRE_DATA_ROOT}/manifest.json"

    if [ ! -f "${MANIFEST_PATH}" ]; then
        echo "[run_benchmark] ERROR: manifest not found at ${MANIFEST_PATH}" >&2
        echo "[run_benchmark] stage_data.sh requires <src>/manifest.json" >&2
        exit 3
    fi

    echo "[run_benchmark] Staging ${LUSTRE_DATA_ROOT} -> ${NVME_DEST} on ${SLURM_JOB_NUM_NODES:-1} node(s)"
    srun --ntasks-per-node=1 --ntasks="${SLURM_JOB_NUM_NODES:-1}" \
         --container-image="${CONTAINER_IMAGE}" \
         --container-mounts="${CONTAINER_MOUNTS}" \
         bash "${BENCHMARKS_DIR}/stage_data.sh" \
              "${LUSTRE_DATA_ROOT}" "${NVME_DEST}" "${MANIFEST_PATH}" \
              train val
    export "${DATAPATH_ENV}=${NVME_DEST}"
    echo "[run_benchmark] stage-in complete; ${DATAPATH_ENV}=${NVME_DEST}"
elif [ -n "${LUSTRE_DATA_ROOT:-}" ]; then
    export "${DATAPATH_ENV}=${LUSTRE_DATA_ROOT}"
    echo "[run_benchmark] lustre mode; ${DATAPATH_ENV}=${LUSTRE_DATA_ROOT}"
fi

# ---------------------------------------------------------------------------
# 2. Rendezvous + training launch (srun directly, one task per GPU)
# ---------------------------------------------------------------------------
export MASTER_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST:-localhost}" | head -n 1)
export MASTER_PORT="${MASTER_PORT:-29500}"

# Build Hydra overrides.  The recipe's run_id key is `run_id`, output_dir
# is `output_dir`; sampling_resolution and training.num_epochs map
# straight through.
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
# Append free-form extras last so they win over the standard knobs.
if [ -n "${EXTRA_OVERRIDES}" ]; then
    # shellcheck disable=SC2206
    EXTRAS_ARR=(${EXTRA_OVERRIDES})
    HYDRA_OVERRIDES+=("${EXTRAS_ARR[@]}")
fi

T0=$SECONDS
# Inner srun ranks-per-node matches the sbatch allocation: min(4, NUM_GPUS).
# Older convention always used 4 ranks even for NUM_GPUS<4, which spun up 4
# DDP processes on a 4-GPU node and divided throughput by num_gpus inside
# the recipe.  New convention: NUM_GPUS<4 => true single-replica (no DDP),
# matches what was actually requested.
if [ "${NUM_GPUS}" -lt 4 ]; then
    SRUN_TASKS_PER_NODE="${NUM_GPUS}"
else
    SRUN_TASKS_PER_NODE=4
fi
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
     "
TRAIN_RC=$?
DT=$(( SECONDS - T0 ))
echo "[run_benchmark] training finished in ${DT}s (rc=${TRAIN_RC})"

# ---------------------------------------------------------------------------
# 3. Post-run summary (rank-0 only after srun returns).  Reads the
#    metrics.jsonl that train.py just wrote and emits
#    benchmark_summary.{json,md} alongside it.
# ---------------------------------------------------------------------------
RUN_OUTPUT_DIR="${RESULTS_DIR}/runs/${RUN_ID}"
METRICS_PATH="${RUN_OUTPUT_DIR}/metrics.jsonl"

if [ "${TRAIN_RC}" -eq 0 ] && [ -f "${METRICS_PATH}" ]; then
    echo "[run_benchmark] post-processing ${METRICS_PATH}"
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
                --storage ${STORAGE_MODE} \
                --sampling ${SUBSAMPLING} \
                --num-epochs ${NUM_EPOCHS} \
                --exclude-epochs ${EXCLUDE_EPOCHS}
         " || echo "[run_benchmark] summarize_run.py failed (non-fatal)"
else
    echo "[run_benchmark] skipping summary: train_rc=${TRAIN_RC} metrics_exists=$([ -f "${METRICS_PATH}" ] && echo yes || echo no)"
fi

# ---------------------------------------------------------------------------
# 4. NVMe cleanup (best-effort).  CLEANUP_NVME=0 keeps the staged data
#    around for a follow-up run on the same node.
# ---------------------------------------------------------------------------
if [ "${STORAGE_MODE}" = "nvme" ] && [ "${CLEANUP_NVME}" = "1" ]; then
    echo "[run_benchmark] cleaning up ${NVME_DEST}"
    srun --ntasks-per-node=1 --ntasks="${SLURM_JOB_NUM_NODES:-1}" \
         bash -c "rm -rf '${NVME_DEST}'" || true
fi

exit "${TRAIN_RC}"
