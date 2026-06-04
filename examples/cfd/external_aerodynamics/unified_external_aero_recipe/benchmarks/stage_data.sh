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
# stage_data.sh -- parallel-copy dataset runs from Lustre to local storage
# ---------------------------------------------------------------------------
#
# Vendored from Corey's CAE benchmarking utilities; minor formatting
# only.  Copies run directories listed in a manifest.json from a shared
# Lustre dataset root to a fast node-local path (e.g. /tmp).  Uses
# `xargs -P` to fan out `cp -a` across all available CPU cores for
# maximum throughput.
#
# Usage:
#   stage_data.sh <src_root> <dest_root> <manifest.json> <split1> [split2 ...]
#
# Examples:
#   # Stage train+val splits to /tmp
#   stage_data.sh \
#       /lustre/fsw/portfolios/coreai/projects/coreai_modulus_cae/datasets/PhysicsNeMo-DrivaerML \
#       /tmp/aero_data/drivaer_ml \
#       /lustre/fsw/portfolios/coreai/projects/coreai_modulus_cae/datasets/PhysicsNeMo-DrivaerML/manifest.json \
#       train val
#
#   # Stage only the training split with 32 workers
#   STAGE_NPROC=32 stage_data.sh /lustre/.../data /tmp/data manifest.json train
#
# Environment:
#   STAGE_NPROC   Override worker count (default: nproc, i.e. all cores)
#
# Notes:
#   - Idempotent: run directories already present in dest_root are skipped.
#   - Requires jq for manifest parsing.  If jq is unavailable, set
#     STAGE_USE_PYTHON=1 to use a python3 fallback.
# ---------------------------------------------------------------------------

set -euo pipefail

if [ $# -lt 4 ]; then
    echo "Usage: $0 <src_root> <dest_root> <manifest.json> <split1> [split2 ...]" >&2
    exit 1
fi

SRC="$1"; DEST="$2"; MANIFEST="$3"; shift 3
SPLITS=("$@")
NPROC="${STAGE_NPROC:-$(nproc)}"

mkdir -p "$DEST"

# ---- Extract run directory names from the manifest -------------------------
# manifest.json format: {"train": ["run_1", ...], "val": ["run_4", ...], ...}
# We union all requested splits and deduplicate.

_extract_runs_jq() {
    # Build a jq filter like [.train, .val] | add | .[] for the requested splits
    local filter
    filter=$(printf '.%s, ' "${SPLITS[@]}")
    filter="[${filter%, }] | add | unique | .[]"
    jq -r "$filter" "$MANIFEST"
}

_extract_runs_python() {
    python3 -c "
import json, sys
data = json.load(open('$MANIFEST'))
runs = set()
for s in sys.argv[1:]:
    runs.update(data.get(s, []))
print('\n'.join(sorted(runs)))
" "${SPLITS[@]}"
}

if [ "${STAGE_USE_PYTHON:-0}" = "1" ] || ! command -v jq &>/dev/null; then
    RUNS=$(_extract_runs_python)
else
    RUNS=$(_extract_runs_jq)
fi

if [ -z "$RUNS" ]; then
    echo "[stage_data] No runs found for splits: ${SPLITS[*]}" >&2
    exit 1
fi

TOTAL=$(echo "$RUNS" | wc -l)
echo "[stage_data] Splits requested: ${SPLITS[*]}"
echo "[stage_data] Staging $TOTAL runs from $SRC -> $DEST (${NPROC} workers)"
echo "[stage_data] Manifest: $MANIFEST"
T0=$SECONDS

# ---- Parallel copy ---------------------------------------------------------
# One `cp -a` per run directory, up to NPROC at a time.  A `.staged_ok`
# marker file is written after each successful copy.  Runs whose marker
# already exists are skipped.  Incomplete copies (directory exists but
# no marker) are removed and re-copied.

export SRC DEST

XARGS_RC=0
echo "$RUNS" | xargs -P "$NPROC" -I{} bash -c '
    run="{}"
    marker="$DEST/$run/.staged_ok"
    if [ -f "$marker" ]; then
        echo "[stage_data]   SKIP $run (already staged)"
        exit 0
    fi
    if [ -d "$DEST/$run" ]; then
        echo "[stage_data]   RETRY $run (removing incomplete copy)"
        rm -rf "$DEST/$run"
    fi
    echo "[stage_data]   COPY $run ..."
    # Retry up to 3 times: Lustre MDS stat-cache misses on a cold compute
    # node occasionally make `cp -a` see ENOENT for files that demonstrably
    # exist (we see this at depth 6+ inside .pdmsh/_tensordict/boundaries/
    # vehicle/_tensordict/...).  A `find` over the source between attempts
    # primes the deep stat cache so the next `cp` walks cleanly.  This was
    # observed once per ~35 NVMe jobs in the Phase 2 sweep -- enough to
    # tank the run when it hits, rare enough that retry costs ~0 in
    # aggregate.
    for attempt in 1 2 3; do
        if cp -a "$SRC/$run" "$DEST/$run" 2>/tmp/stage_${run}_a${attempt}.err; then
            touch "$marker"
            if [ "$attempt" -gt 1 ]; then
                echo "[stage_data]   OK   $run (attempt ${attempt})"
            else
                echo "[stage_data]   OK   $run"
            fi
            rm -f /tmp/stage_${run}_a*.err
            exit 0
        fi
        echo "[stage_data]   WARN $run cp attempt ${attempt} failed; retrying after deep-stat warm" >&2
        rm -rf "$DEST/$run"
        find "$SRC/$run" -mindepth 1 -printf "" 2>/dev/null || true
        sleep 1
    done
    echo "[stage_data]   FAIL $run (cp failed after 3 attempts)" >&2
    cat /tmp/stage_${run}_a*.err >&2 2>/dev/null || true
    rm -f /tmp/stage_${run}_a*.err
    exit 1
' || XARGS_RC=$?

# Count results
STAGED=0; TOTAL_PRESENT=0; FAILED_RUNS=""
for run in $RUNS; do
    if [ -f "$DEST/$run/.staged_ok" ]; then
        TOTAL_PRESENT=$((TOTAL_PRESENT + 1))
    else
        FAILED_RUNS="${FAILED_RUNS} ${run}"
    fi
done

DT=$(( SECONDS - T0 ))
echo "[stage_data] Done in ${DT}s: $TOTAL_PRESENT/$TOTAL runs present in $DEST"

if [ -n "$FAILED_RUNS" ]; then
    echo "[stage_data] FAILED runs:${FAILED_RUNS}" >&2
fi

if [ "$XARGS_RC" -ne 0 ]; then
    echo "[stage_data] ERROR: staging exited with code $XARGS_RC" >&2
    exit "$XARGS_RC"
fi

# Copy the manifest itself into DEST so the training recipe's
# `resolve_manifest_spec` auto-derive (`${train_datadir}/manifest.json`)
# finds it next to the staged runs.  Without this, runs configured with
# `train_split` / `val_split` fail at startup because the local stage
# only contains run dirs, not the manifest that names the splits.
cp -a "$MANIFEST" "$DEST/manifest.json"

# Write a top-level completion marker (used as a barrier by other ranks)
touch "$DEST/.staging_complete"
