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
# submit_all.sh -- iterate the benchmark matrix and sbatch each row
# ---------------------------------------------------------------------------
#
# Reads matrix.json (built by benchmark_matrix.py), dispatches ONE
# sbatch per row, and records (jobid, run_id) in submitted_jobs.tsv so
# check_runs.sh / triage_failures.py can sweep them later.  The sbatch
# script (run_benchmark.sbatch) handles the actual NVMe staging and
# srun-direct training launch.
#
# Usage:
#   ./submit_all.sh --matrix matrix.json
#   ./submit_all.sh --matrix matrix.json --dry-run
#   ./submit_all.sh --matrix matrix.json --filter-gpus 1,4
#   ./submit_all.sh --matrix matrix.json --filter-storage lustre
#   ./submit_all.sh --matrix matrix.json --max-concurrent 8
#
# Required env vars (or supply via --lustre-data-root etc.):
#   LUSTRE_DATA_ROOT_DRIVAER_ML  Lustre path to DrivAerML dataset
#   LUSTRE_DATA_ROOT_DRIVESIM    Lustre path to DriveSim dataset (optional)
#   CONTAINER_IMAGE              Path to .sqsh container image
#   CONTAINER_MOUNTS             Container mount string
#   VENV_PATH                    Path to the recipe virtualenv
#   WORKDIR                      Path to the recipe directory inside container
#
# Concurrency:
#   --max-concurrent N forces SLURM to run at most N jobs from this
#   submission set in parallel by chaining all jobs onto a single
#   `--job-name` and using `--dependency=singleton` for the (N+1)th
#   onwards.  Simpler than QoS gymnastics and surfaces clearly in
#   `squeue`.
#
#   --group-nvme partitions all NVMe rows by (dataset_family, num_nodes)
#   and submits ONE sbatch per group via run_benchmark_group.sbatch,
#   which stages Lustre->NVMe ONCE and then loops through every entry.
#   This eliminates the redundant 25-30 min/run stage cost and prevents
#   the Lustre MDS thrashing seen when many small NVMe jobs hit the
#   shared dataset path simultaneously.  Lustre rows are unaffected.
# ---------------------------------------------------------------------------

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SBATCH_TEMPLATE="${SCRIPT_DIR}/run_benchmark.sbatch"
SBATCH_GROUP_TEMPLATE="${SCRIPT_DIR}/run_benchmark_group.sbatch"
# Recipe root (parent of benchmarks/) is the natural anchor for any
# relative results_dir emitted by benchmark_matrix.py.  train.py runs
# after `cd $WORKDIR` (= recipe root), so a relative output_dir is
# interpreted there; the post-run summary check, however, runs on the
# login node from sbatch's submission cwd.  Resolving to absolute paths
# up front makes those two views agree -- otherwise metrics.jsonl ends
# up in $WORKDIR/results/... but the sbatch's `[ -f $METRICS_PATH ]`
# check looks under $PWD/results/... and silently skips the summary.
RECIPE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MATRIX_PATH=""
DRY_RUN=0
FILTER_GPUS=""
FILTER_STORAGE=""
FILTER_MODEL=""
FILTER_DATASET=""
FILTER_PHASE=""
EXCLUDE_PENDING=0
MAX_CONCURRENT=0
TRACK_FILE=""
GROUP_NVME=0
# Cap on a single group's walltime in minutes.  Groups whose summed
# per-entry walltime would exceed this get split into multiple sub-groups
# in submit-order.  Default 12h leaves slack vs HSG's 24h batch cap.
GROUP_MAX_MIN="${GROUP_MAX_MIN:-720}"

usage() {
    cat <<EOF
Usage: $0 --matrix FILE [options]

Required:
  --matrix FILE             benchmark_matrix.py output (JSON list of run specs)

Options:
  --dry-run                 print sbatch commands without submitting
  --filter-gpus 1,4         only submit rows where num_gpus is in this set
  --filter-storage MODE     only submit rows for this storage mode (lustre|nvme)
  --filter-model NAME       only submit rows for this model
  --filter-dataset NAME     only submit rows for this dataset
  --filter-phase N          only submit rows tagged with this phase (1..4)
  --exclude-pending         drop rows still gated on missing config
  --max-concurrent N        cap simultaneous in-flight jobs at N (uses
                            --dependency=singleton)
  --group-nvme              partition NVMe rows by (dataset_family,
                            num_nodes) and submit ONE sbatch per group
                            via run_benchmark_group.sbatch (stage_data
                            runs once per group instead of once per row).
                            Lustre rows are submitted as usual.
  --group-max-min N         cap a single group's est_walltime sum at N
                            minutes; auto-split larger groups (default
                            ${GROUP_MAX_MIN})
  --track-file FILE         output TSV (default: submitted_jobs.tsv next to matrix)
  -h, --help                this message
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        --matrix)            MATRIX_PATH="$2"; shift 2 ;;
        --dry-run)           DRY_RUN=1; shift ;;
        --filter-gpus)       FILTER_GPUS="$2"; shift 2 ;;
        --filter-storage)    FILTER_STORAGE="$2"; shift 2 ;;
        --filter-model)      FILTER_MODEL="$2"; shift 2 ;;
        --filter-dataset)    FILTER_DATASET="$2"; shift 2 ;;
        --filter-phase)      FILTER_PHASE="$2"; shift 2 ;;
        --exclude-pending)   EXCLUDE_PENDING=1; shift ;;
        --max-concurrent)    MAX_CONCURRENT="$2"; shift 2 ;;
        --group-nvme)        GROUP_NVME=1; shift ;;
        --group-max-min)     GROUP_MAX_MIN="$2"; shift 2 ;;
        --track-file)        TRACK_FILE="$2"; shift 2 ;;
        -h|--help)           usage; exit 0 ;;
        *)                   echo "unknown arg: $1" >&2; usage; exit 2 ;;
    esac
done

if [ -z "${MATRIX_PATH}" ]; then
    echo "[submit_all] --matrix is required" >&2
    usage
    exit 2
fi
if [ ! -f "${MATRIX_PATH}" ]; then
    echo "[submit_all] matrix file not found: ${MATRIX_PATH}" >&2
    exit 2
fi
if [ ! -f "${SBATCH_TEMPLATE}" ]; then
    echo "[submit_all] sbatch template not found: ${SBATCH_TEMPLATE}" >&2
    exit 2
fi
if [ "${GROUP_NVME}" = "1" ] && [ ! -f "${SBATCH_GROUP_TEMPLATE}" ]; then
    echo "[submit_all] --group-nvme requested but ${SBATCH_GROUP_TEMPLATE} not found" >&2
    exit 2
fi
if [ "${GROUP_NVME}" = "1" ] && [ "${MAX_CONCURRENT}" -gt 0 ]; then
    echo "[submit_all] WARN: --group-nvme already serializes runs within a group;" >&2
    echo "[submit_all]       --max-concurrent will only throttle across groups." >&2
fi

if [ -z "${TRACK_FILE}" ]; then
    TRACK_FILE="$(dirname "${MATRIX_PATH}")/submitted_jobs.tsv"
fi

if ! command -v jq &>/dev/null; then
    echo "[submit_all] this script needs jq for matrix.json parsing" >&2
    exit 2
fi

# Make sure the tracking file exists and has a header.
if [ ! -s "${TRACK_FILE}" ]; then
    printf "submit_ts\tjobid\trun_id\tmodel\tdataset\tnum_gpus\tstorage\tsampling\tphase\tresults_dir\n" > "${TRACK_FILE}"
fi

# Build a jq filter implementing the active --filter-* flags.
JQ_FILTER='.[]'
[ -n "${FILTER_GPUS}" ]    && JQ_FILTER+=" | select([${FILTER_GPUS}] | index(.num_gpus))"
[ -n "${FILTER_STORAGE}" ] && JQ_FILTER+=" | select(.storage == \"${FILTER_STORAGE}\")"
[ -n "${FILTER_MODEL}" ]   && JQ_FILTER+=" | select(.model == \"${FILTER_MODEL}\")"
[ -n "${FILTER_DATASET}" ] && JQ_FILTER+=" | select(.dataset == \"${FILTER_DATASET}\")"
[ -n "${FILTER_PHASE}" ]   && JQ_FILTER+=" | select(.phase == ${FILTER_PHASE})"
[ "${EXCLUDE_PENDING}" = "1" ] && JQ_FILTER+=" | select(.skip == false)"

# We use NUL separators between rows to keep newlines in fields safe.
ROW_COUNT=$(jq -c "${JQ_FILTER}" "${MATRIX_PATH}" | wc -l)
echo "[submit_all] matrix=${MATRIX_PATH}  rows_after_filter=${ROW_COUNT}"
if [ "${DRY_RUN}" = "1" ]; then
    echo "[submit_all] DRY RUN -- nothing will be submitted"
fi

# --max-concurrent uses a shared --job-name + --dependency=singleton so
# Slurm runs at most N jobs from this submission group at once.  We rotate
# across N distinct names so within-group concurrency stays <= N.
SUBMITTED=0
SKIPPED_PENDING=0
mapfile -t ROWS < <(jq -c "${JQ_FILTER}" "${MATRIX_PATH}")

# When --group-nvme is set we accumulate NVMe rows here (one entry per
# row, JSON-encoded) and emit one sbatch per (dataset_family,nodes) at
# the end of the per-row loop.  Lustre rows still flow through the
# existing per-row submit path inside the loop.
declare -A NVME_GROUPS
declare -A NVME_GROUP_LUSTREROOT
declare -A NVME_GROUP_WALLMIN

# Map a row's dataset string to the same DATAPATH_KEY that
# run_benchmark.sbatch uses, so two datasets that share a Lustre source
# (e.g. drivaer_ml_surface + drivaer_ml_volume both -> drivaer_ml) end
# up in the same group.
_dataset_family() {
    case "$1" in
        drivaer_ml_*) echo "drivaer_ml" ;;
        highlift_*)   echo "highlift_aero_ml" ;;
        drivesim_*)   echo "drivesim" ;;
        shift_suv_*)  echo "shift_suv" ;;
        *)            echo "" ;;
    esac
}

for raw in "${ROWS[@]}"; do
    if [ -z "${raw}" ]; then
        continue
    fi
    run_id=$(echo "${raw}"        | jq -r '.run_id')
    model=$(echo "${raw}"         | jq -r '.model')
    dataset=$(echo "${raw}"       | jq -r '.dataset')
    num_gpus=$(echo "${raw}"      | jq -r '.num_gpus')
    nodes=$(echo "${raw}"         | jq -r '.nodes')
    storage=$(echo "${raw}"       | jq -r '.storage')
    sampling=$(echo "${raw}"      | jq -r '.sampling_resolution')
    phase=$(echo "${raw}"         | jq -r '.phase')
    skip=$(echo "${raw}"          | jq -r '.skip')
    skip_reason=$(echo "${raw}"   | jq -r '.skip_reason // empty')
    results_dir=$(echo "${raw}"   | jq -r '.results_dir')
    # Anchor relative paths to the recipe root (see RECIPE_ROOT comment
    # above).  Absolute paths from the matrix are passed through.
    case "${results_dir}" in
        /*) ;;
        *)  results_dir="${RECIPE_ROOT}/${results_dir}" ;;
    esac
    walltime_min=$(echo "${raw}"  | jq -r '.est_walltime_min')

    if [ "${skip}" = "true" ]; then
        echo "[submit_all] SKIP ${run_id}: ${skip_reason}"
        SKIPPED_PENDING=$((SKIPPED_PENDING + 1))
        continue
    fi

    # Resolve the Lustre data root for this dataset family.
    case "${dataset}" in
        drivaer_ml_*) LUSTRE_ROOT="${LUSTRE_DATA_ROOT_DRIVAER_ML:-}" ;;
        drivesim_*)   LUSTRE_ROOT="${LUSTRE_DATA_ROOT_DRIVESIM:-}" ;;
        highlift_*)   LUSTRE_ROOT="${LUSTRE_DATA_ROOT_HIGHLIFT:-}" ;;
        shift_suv_*)  LUSTRE_ROOT="${LUSTRE_DATA_ROOT_SHIFT_SUV:-}" ;;
        *)            LUSTRE_ROOT="" ;;
    esac
    if [ "${storage}" = "nvme" ] && [ -z "${LUSTRE_ROOT}" ]; then
        echo "[submit_all] SKIP ${run_id}: nvme requires LUSTRE_DATA_ROOT_* for ${dataset}"
        SKIPPED_PENDING=$((SKIPPED_PENDING + 1))
        continue
    fi

    mkdir -p "${results_dir}"

    # --group-nvme: divert NVMe rows into per-(family,nodes) accumulators
    # and skip the per-row sbatch.  Emission happens after the loop ends.
    if [ "${GROUP_NVME}" = "1" ] && [ "${storage}" = "nvme" ]; then
        family=$(_dataset_family "${dataset}")
        if [ -z "${family}" ]; then
            echo "[submit_all] SKIP ${run_id}: cannot derive dataset family for ${dataset}"
            SKIPPED_PENDING=$((SKIPPED_PENDING + 1))
            continue
        fi
        gkey="${family}|${nodes}"
        # Build a per-entry JSON object that run_benchmark_group.sbatch
        # can consume directly.  Note: we keep the original (un-padded)
        # walltime_min as `est_walltime_min` for group-walltime budgeting.
        entry=$(jq -nc \
            --arg run_id "${run_id}" \
            --arg model "${model}" \
            --arg dataset "${dataset}" \
            --arg num_gpus "${num_gpus}" \
            --arg sampling "${sampling}" \
            --arg results_dir "${results_dir}" \
            --arg walltime "${walltime_min}" \
            --arg num_epochs "${NUM_EPOCHS:-5}" \
            --arg precision "${PRECISION:-bfloat16}" \
            --arg compile "${COMPILE:-true}" \
            --arg extra "${EXTRA_OVERRIDES:-}" \
            '{run_id:$run_id, model:$model, dataset:$dataset,
              num_gpus:($num_gpus|tonumber), sampling:($sampling|tonumber),
              results_dir:$results_dir,
              est_walltime_min:($walltime|tonumber),
              num_epochs:($num_epochs|tonumber),
              precision:$precision, compile:$compile,
              extra_overrides:$extra}')
        # Newline-delimited JSON keeps the per-key buffer simple to parse.
        NVME_GROUPS["${gkey}"]+="${entry}"$'\n'
        NVME_GROUP_LUSTREROOT["${gkey}"]="${LUSTRE_ROOT}"
        prev_min="${NVME_GROUP_WALLMIN[$gkey]:-0}"
        NVME_GROUP_WALLMIN["${gkey}"]=$(( prev_min + walltime_min ))
        # Also write the per-row line into submitted_jobs.tsv -- jobid is
        # filled in once we emit the group sbatch below.  We park a
        # placeholder for now so the row order in TSV matches submit
        # order.
        # (Actual TSV append happens at group-emit time so the jobid is
        # the GROUP's jobid, not a placeholder.)
        SUBMITTED=$((SUBMITTED + 1))
        continue
    fi

    # Pad walltime by 50% (min 30 min) to leave room for stage-in / stage-out.
    walltime_padded=$(( walltime_min + walltime_min / 2 ))
    if [ "${walltime_padded}" -lt 30 ]; then walltime_padded=30; fi
    hh=$(( walltime_padded / 60 ))
    mm=$(( walltime_padded % 60 ))
    timestr=$(printf "%02d:%02d:00" "${hh}" "${mm}")

    # Per-node sizing.  The HSG QOS enforces a minimum 4 GPUs per
    # allocation (QOSMinGRES), so we always request --gpus-per-node=4
    # regardless of num_gpus.  --ntasks-per-node, however, scales with
    # min(4, num_gpus): for num_gpus<4 only that many ranks start, so
    # a "g=1" run is a true single-replica benchmark (1 rank using
    # GPU 0; GPUs 1-3 idle) instead of 4 DDP ranks artificially
    # sharing a wider effective batch.  Throughput and memory numbers
    # become per-rank in the cleanest sense.  num_gpus >= 4 keeps the
    # 4/node fan-out unchanged.
    if [ "${num_gpus}" -lt 4 ]; then
        ranks_per_node="${num_gpus}"
    else
        ranks_per_node=4
    fi
    SBATCH_ARGS=(
        "--nodes=${nodes}"
        "--ntasks-per-node=${ranks_per_node}"
        "--gpus-per-node=4"
        "--time=${timestr}"
        "--output=${results_dir}/slurm-%j.out"
        "--error=${results_dir}/slurm-%j.err"
        "--job-name=cae_${model}_${dataset}_g${num_gpus}_${storage}_s${sampling}"
    )

    # --max-concurrent: rotate across N names (singleton enforces
    # one-per-name in flight).
    if [ "${MAX_CONCURRENT}" -gt 0 ]; then
        slot=$(( SUBMITTED % MAX_CONCURRENT ))
        SBATCH_ARGS+=(
            "--job-name=cae_bench_slot${slot}"
            "--dependency=singleton"
        )
    fi

    EXPORT_VARS="ALL"
    EXPORT_VARS+=",MODEL=${model}"
    EXPORT_VARS+=",DATASET=${dataset}"
    EXPORT_VARS+=",NUM_GPUS=${num_gpus}"
    EXPORT_VARS+=",STORAGE_MODE=${storage}"
    EXPORT_VARS+=",SUBSAMPLING=${sampling}"
    EXPORT_VARS+=",RESULTS_DIR=${results_dir}"
    EXPORT_VARS+=",RUN_ID=${run_id}"
    [ -n "${LUSTRE_ROOT}" ]               && EXPORT_VARS+=",LUSTRE_DATA_ROOT=${LUSTRE_ROOT}"
    [ -n "${CONTAINER_IMAGE:-}" ]         && EXPORT_VARS+=",CONTAINER_IMAGE=${CONTAINER_IMAGE}"
    # CONTAINER_MOUNTS deliberately NOT appended inline -- its value
    # contains commas (mount list syntax) and SLURM's `--export=KEY=VAL`
    # parser treats each comma as a separator between exports, so the
    # value gets truncated to the first mount.  We rely on `ALL` to
    # inherit it from the parent shell instead, or fall back to the
    # sbatch script's hardcoded default if the parent shell doesn't
    # export it.  This bug killed jobs 3031340/41/42 (group sbatches):
    # only the first mount made it into the container so /lustre,
    # /home, /tmp were missing and the workdir symlink couldn't
    # resolve, producing "stage_data.sh: No such file or directory".
    [ -n "${VENV_PATH:-}" ]               && EXPORT_VARS+=",VENV_PATH=${VENV_PATH}"
    [ -n "${WORKDIR:-}" ]                 && EXPORT_VARS+=",WORKDIR=${WORKDIR}"
    [ -n "${NUM_EPOCHS:-}" ]              && EXPORT_VARS+=",NUM_EPOCHS=${NUM_EPOCHS}"
    [ -n "${PRECISION:-}" ]               && EXPORT_VARS+=",PRECISION=${PRECISION}"
    [ -n "${COMPILE:-}" ]                 && EXPORT_VARS+=",COMPILE=${COMPILE}"
    # EXTRA_OVERRIDES may also contain commas; same treatment as
    # CONTAINER_MOUNTS -- inherit via ALL or fall back to the sbatch
    # script's default.
    SBATCH_ARGS+=("--export=${EXPORT_VARS}")

    cmd=(sbatch "${SBATCH_ARGS[@]}" "${SBATCH_TEMPLATE}")

    if [ "${DRY_RUN}" = "1" ]; then
        printf "[dry-run] %s\n" "${cmd[*]}"
    else
        out=$("${cmd[@]}")
        # `Submitted batch job 12345` is the standard sbatch output.
        jobid=$(echo "${out}" | awk '{print $NF}')
        ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)
        printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
            "${ts}" "${jobid}" "${run_id}" "${model}" "${dataset}" \
            "${num_gpus}" "${storage}" "${sampling}" "${phase}" "${results_dir}" \
            >> "${TRACK_FILE}"
        echo "[submit_all] OK  ${jobid}  ${run_id}"
    fi

    SUBMITTED=$((SUBMITTED + 1))
done

# ---------------------------------------------------------------------------
# Grouped-NVMe emission pass.  Iterates every (dataset_family, num_nodes)
# bucket built during the per-row loop, splits any bucket whose summed
# est_walltime exceeds GROUP_MAX_MIN, and submits ONE sbatch per
# resulting sub-bucket using run_benchmark_group.sbatch.
#
# Walltime budget per sub-bucket:
#   stage=30 + sum_per_entry(walltime*1.5) + cleanup=5
# (matches the per-row 50% padding used above; stage budget reflects the
# 25-30 min observed for a cold-cache DrivAerML stage on one node.)
# ---------------------------------------------------------------------------
if [ "${GROUP_NVME}" = "1" ] && [ "${#NVME_GROUPS[@]}" -gt 0 ]; then
    # Spec files include the matrix basename so successive submissions
    # of different matrices don't clobber each other's group specs.
    # Without this, two submit_all.sh invocations sharing the same
    # (family, nodes) key would emit two specs with identical names and
    # the first sbatch (still queued) would read the second's content.
    MATRIX_STEM="$(basename "${MATRIX_PATH}" .json)"
    GROUPS_DIR="$(dirname "${MATRIX_PATH}")/groups"
    mkdir -p "${GROUPS_DIR}"
    GROUP_STAGE_MIN=30
    GROUP_TAIL_MIN=5

    GROUP_JOBS=0
    for gkey in "${!NVME_GROUPS[@]}"; do
        family="${gkey%|*}"
        nodes="${gkey#*|}"
        lustre_root="${NVME_GROUP_LUSTREROOT[$gkey]}"

        # Parse the buffered entries (one JSON object per line) into a
        # bash array preserving submit order.
        mapfile -t entries < <(printf '%s' "${NVME_GROUPS[$gkey]}" | jq -c '.')

        # Greedy-pack into sub-buckets so each one stays under
        # GROUP_MAX_MIN total walltime (with stage+tail accounted for).
        sub_idx=0
        sub_entries=()
        sub_minutes=0
        flush_sub() {
            if [ "${#sub_entries[@]}" -eq 0 ]; then return; fi
            local spec="${GROUPS_DIR}/${MATRIX_STEM}__${family}_n${nodes}_$(printf '%02d' "${sub_idx}").json"
            printf '%s\n' "${sub_entries[@]}" | jq -s '.' > "${spec}"

            # Pad each entry's walltime by 50% (same convention as
            # per-row), then add stage + tail; cap to 23h59 to stay
            # under HSG batch limit.
            local padded=$(( sub_minutes + sub_minutes / 2 ))
            local total=$(( GROUP_STAGE_MIN + padded + GROUP_TAIL_MIN ))
            if [ "${total}" -gt 1439 ]; then total=1439; fi
            local hh=$(( total / 60 ))
            local mm=$(( total % 60 ))
            local timestr
            timestr=$(printf "%02d:%02d:00" "${hh}" "${mm}")

            local label="${MATRIX_STEM}_${family}_n${nodes}_$(printf '%02d' "${sub_idx}")"
            local glog_dir="${GROUPS_DIR}"
            local SBATCH_ARGS=(
                "--nodes=${nodes}"
                "--ntasks-per-node=4"
                "--gpus-per-node=4"
                "--time=${timestr}"
                "--output=${glog_dir}/slurm-%j.out"
                "--error=${glog_dir}/slurm-%j.err"
                "--job-name=cae_group_${label}"
            )
            if [ "${MAX_CONCURRENT}" -gt 0 ]; then
                local slot=$(( GROUP_JOBS % MAX_CONCURRENT ))
                SBATCH_ARGS+=(
                    "--job-name=cae_group_slot${slot}"
                    "--dependency=singleton"
                )
            fi

            local EXPORT_VARS="ALL"
            EXPORT_VARS+=",RUN_GROUP_JSON=${spec}"
            EXPORT_VARS+=",GROUP_LABEL=${label}"
            EXPORT_VARS+=",LUSTRE_DATA_ROOT=${lustre_root}"
            [ -n "${CONTAINER_IMAGE:-}" ]  && EXPORT_VARS+=",CONTAINER_IMAGE=${CONTAINER_IMAGE}"
            # CONTAINER_MOUNTS deliberately NOT appended inline -- see
            # comment in the per-row export block above for why.
            [ -n "${VENV_PATH:-}" ]        && EXPORT_VARS+=",VENV_PATH=${VENV_PATH}"
            [ -n "${WORKDIR:-}" ]          && EXPORT_VARS+=",WORKDIR=${WORKDIR}"
            SBATCH_ARGS+=("--export=${EXPORT_VARS}")

            local cmd=(sbatch "${SBATCH_ARGS[@]}" "${SBATCH_GROUP_TEMPLATE}")
            local n_runs="${#sub_entries[@]}"
            echo "[submit_all] GROUP ${label}  runs=${n_runs}  walltime=${timestr}  spec=${spec}"

            if [ "${DRY_RUN}" = "1" ]; then
                printf "[dry-run] %s\n" "${cmd[*]}"
                local jobid="DRYRUN-${family}-n${nodes}-${sub_idx}"
            else
                local out
                out=$("${cmd[@]}")
                local jobid
                jobid=$(echo "${out}" | awk '{print $NF}')
                echo "[submit_all] OK  group_jobid=${jobid}  ${label}"
            fi

            # Fan the group's jobid out into the tracking TSV: one row
            # per entry, all sharing the same jobid.  Downstream
            # check_runs.sh / triage_failures.py keep working unchanged.
            local ts
            ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)
            local entry
            for entry in "${sub_entries[@]}"; do
                local e_run e_model e_ds e_ng e_sub e_rdir
                e_run=$(echo "${entry}"   | jq -r '.run_id')
                e_model=$(echo "${entry}" | jq -r '.model')
                e_ds=$(echo "${entry}"    | jq -r '.dataset')
                e_ng=$(echo "${entry}"    | jq -r '.num_gpus')
                e_sub=$(echo "${entry}"   | jq -r '.sampling')
                e_rdir=$(echo "${entry}"  | jq -r '.results_dir')
                printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
                    "${ts}" "${jobid}" "${e_run}" "${e_model}" "${e_ds}" \
                    "${e_ng}" "nvme" "${e_sub}" "grouped" "${e_rdir}" \
                    >> "${TRACK_FILE}"
            done

            GROUP_JOBS=$((GROUP_JOBS + 1))
            sub_entries=()
            sub_minutes=0
            sub_idx=$((sub_idx + 1))
        }

        for entry in "${entries[@]}"; do
            local_walltime=$(echo "${entry}" | jq -r '.est_walltime_min')
            # If a single entry by itself would exceed the cap, give it
            # its own bucket and warn rather than refusing to submit --
            # the budget is best-effort.
            projected=$(( sub_minutes + local_walltime ))
            projected_total=$(( GROUP_STAGE_MIN + projected + projected / 2 + GROUP_TAIL_MIN ))
            if [ "${#sub_entries[@]}" -gt 0 ] && [ "${projected_total}" -gt "${GROUP_MAX_MIN}" ]; then
                flush_sub
            fi
            sub_entries+=("${entry}")
            sub_minutes=$(( sub_minutes + local_walltime ))
        done
        flush_sub
    done
    echo "[submit_all] grouped: emitted ${GROUP_JOBS} group sbatch(es) covering ${SUBMITTED} entries"
fi

echo "[submit_all] done. submitted=${SUBMITTED} skipped=${SKIPPED_PENDING} track=${TRACK_FILE}"
