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
# check_runs.sh -- summarize the state of jobs we submitted
# ---------------------------------------------------------------------------
#
# Reads submit_all.sh's tracking file (default: submitted_jobs.tsv) and
# for every recorded jobid:
#   * uses sacct to fetch the terminal state (or falls back to squeue
#     for in-flight jobs)
#   * counts pass / fail / running / pending
#   * prints a per-running-job tail (last 5 lines of slurm-*.out)
#   * estimates time remaining for running jobs
#
# Usage:
#   ./check_runs.sh                          # uses submitted_jobs.tsv next to script
#   ./check_runs.sh --track-file PATH        # custom tracking file
#   ./check_runs.sh --tail-lines 10          # change tail length
#   ./check_runs.sh --quiet                  # summary line only
# ---------------------------------------------------------------------------

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRACK_FILE="${SCRIPT_DIR}/submitted_jobs.tsv"
TAIL_LINES=5
QUIET=0

while [ $# -gt 0 ]; do
    case "$1" in
        --track-file) TRACK_FILE="$2"; shift 2 ;;
        --tail-lines) TAIL_LINES="$2"; shift 2 ;;
        --quiet)      QUIET=1; shift ;;
        -h|--help)
            sed -n '20,40p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'
            exit 0
            ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

if [ ! -f "${TRACK_FILE}" ]; then
    echo "[check_runs] no tracking file at ${TRACK_FILE}" >&2
    exit 2
fi
if ! command -v sacct &>/dev/null; then
    echo "[check_runs] sacct not on PATH; nothing to do." >&2
    exit 2
fi

# Pull jobids (skip header line).
mapfile -t LINES < <(tail -n +2 "${TRACK_FILE}")
if [ "${#LINES[@]}" -eq 0 ]; then
    echo "[check_runs] tracking file is empty; nothing to check."
    exit 0
fi

JOBIDS=()
declare -A RUN_ID RESULTS_DIR ELAPSED REMAINING STATE
for line in "${LINES[@]}"; do
    [ -z "${line}" ] && continue
    IFS=$'\t' read -r ts jobid run_id model dataset gpus storage sampling phase results_dir <<< "${line}"
    JOBIDS+=("${jobid}")
    RUN_ID["${jobid}"]="${run_id}"
    RESULTS_DIR["${jobid}"]="${results_dir}"
done

# sacct gives us the terminal state for COMPLETED/FAILED/CANCELLED/etc;
# squeue is more reliable for RUNNING/PENDING.  We query both and merge.
JOBID_CSV=$(IFS=,; echo "${JOBIDS[*]}")

# squeue for live jobs.  ` %j` adds a stable column delimiter.
declare -A LIVE_STATE LIVE_TIME LIVE_LIMIT
if command -v squeue &>/dev/null; then
    while IFS=$'\t' read -r jid st tm tl _; do
        [ -z "${jid}" ] && continue
        LIVE_STATE["${jid}"]="${st}"
        LIVE_TIME["${jid}"]="${tm}"
        LIVE_LIMIT["${jid}"]="${tl}"
    done < <(squeue --jobs="${JOBID_CSV}" -h -o "%i"$'\t'"%T"$'\t'"%M"$'\t'"%l" 2>/dev/null || true)
fi

# sacct for terminal jobs.  -X = job line only (skip step rows).
declare -A SACCT_STATE
while IFS='|' read -r jid st _; do
    [ -z "${jid}" ] && continue
    # sacct emits both '12345' and '12345.batch'; we only want the job row.
    base="${jid%%.*}"
    # First write wins; sacct outputs job row before step rows.
    if [ -z "${SACCT_STATE[$base]:-}" ]; then
        SACCT_STATE["${base}"]="${st%% *}"
    fi
done < <(sacct -j "${JOBID_CSV}" --format=JobID,State,ExitCode -X -p -n 2>/dev/null || true)

# Tally and per-job output.
N_PASS=0; N_FAIL=0; N_RUN=0; N_PEND=0; N_OTHER=0
for jid in "${JOBIDS[@]}"; do
    if [ -n "${LIVE_STATE[$jid]:-}" ]; then
        st="${LIVE_STATE[$jid]}"
        case "${st}" in
            RUNNING)   N_RUN=$((N_RUN+1)) ;;
            PENDING)   N_PEND=$((N_PEND+1)) ;;
            *)         N_OTHER=$((N_OTHER+1)) ;;
        esac
        STATE["${jid}"]="${st}"
        ELAPSED["${jid}"]="${LIVE_TIME[$jid]:-?}"
        REMAINING["${jid}"]="${LIVE_LIMIT[$jid]:-?}"
        continue
    fi
    st="${SACCT_STATE[$jid]:-UNKNOWN}"
    STATE["${jid}"]="${st}"
    case "${st}" in
        COMPLETED)         N_PASS=$((N_PASS+1)) ;;
        FAILED|CANCELLED|TIMEOUT|NODE_FAIL|OUT_OF_MEMORY|BOOT_FAIL|DEADLINE)
                           N_FAIL=$((N_FAIL+1)) ;;
        RUNNING)           N_RUN=$((N_RUN+1)) ;;
        PENDING)           N_PEND=$((N_PEND+1)) ;;
        *)                 N_OTHER=$((N_OTHER+1)) ;;
    esac
done

echo "[check_runs] tracked=${#JOBIDS[@]}  pass=${N_PASS}  fail=${N_FAIL}  run=${N_RUN}  pend=${N_PEND}  other=${N_OTHER}  (file: ${TRACK_FILE})"

if [ "${QUIET}" = "1" ]; then
    exit 0
fi

# Per-job detail.
printf '\n%-12s %-12s %-14s %-14s %s\n' "JOBID" "STATE" "ELAPSED" "LIMIT" "RUN_ID"
printf '%s\n' "----------------------------------------------------------------------------------------------------"
for jid in "${JOBIDS[@]}"; do
    st="${STATE[$jid]:-?}"
    el="${ELAPSED[$jid]:-?}"
    lim="${REMAINING[$jid]:-?}"
    printf '%-12s %-12s %-14s %-14s %s\n' "${jid}" "${st}" "${el}" "${lim}" "${RUN_ID[$jid]}"
    if [ "${st}" = "RUNNING" ] && [ "${TAIL_LINES}" -gt 0 ]; then
        rd="${RESULTS_DIR[$jid]:-}"
        if [ -n "${rd}" ]; then
            for f in "${rd}/slurm-${jid}.out" "${rd}"/slurm-*.out; do
                if [ -f "${f}" ]; then
                    echo "  -- tail ${f} --"
                    tail -n "${TAIL_LINES}" "${f}" | sed 's/^/    /'
                    break
                fi
            done
        fi
    fi
done
