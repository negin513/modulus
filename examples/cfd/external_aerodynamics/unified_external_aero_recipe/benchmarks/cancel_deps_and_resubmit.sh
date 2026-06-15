#!/bin/bash
# Cancel all dep-blocked PENDING jobs in our queue and resubmit them
# without --max-concurrent (no singleton dependencies).  Run this when
# the slurm controller comes back online.
#
# What it does:
#   1. squeue -t PENDING --reason=Dependency,DependencyNeverSatisfied
#      to find every dep-blocked job (Phase 1/2 g64 + Option B singletons).
#   2. scancel those PIDs (leaves RUNNING jobs alone).
#   3. Resubmit the Phase 1/2 g=64 nvme pair (vol sub=10K, sub=50K) fresh.
#   4. Re-submit any Option B matrix WITHOUT --max-concurrent so SLURM's
#      own scheduler can run them as nodes free up.
#
# Pre-flight: parent shell must already have these env-vars exported
# (we check for them and fail loudly if missing).

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

# ---- 0. Sanity: require env vars and slurm health -------------------------
for v in LUSTRE_DATA_ROOT_DRIVAER_ML CONTAINER_IMAGE CONTAINER_MOUNTS \
         VENV_PATH WORKDIR; do
    if [ -z "${!v:-}" ]; then
        echo "ERROR: ${v} is not exported in this shell." >&2
        echo "       Re-export the canonical paths (see header of submit_all.sh)" >&2
        exit 2
    fi
done

echo "==> probing slurm controller (5s timeout)"
if ! timeout 5 squeue -u nsobhani -h -o '%i' >/dev/null 2>&1; then
    echo "ERROR: slurm controller still not responding -- aborting." >&2
    echo "       Re-run when squeue answers in <5s."                >&2
    exit 3
fi
echo "    slurm OK"

# ---- 1. Find dep-blocked PENDING jobs -------------------------------------
echo
echo "==> snapshotting current dep-blocked jobs"
DEP_PENDING=$(squeue -u nsobhani -t PENDING -h -o '%i %r %j' \
              | awk '$2 ~ /Dependency/ {print $1}')

if [ -z "${DEP_PENDING}" ]; then
    echo "    (none -- nothing to cancel)"
else
    echo "    cancelling $(echo "${DEP_PENDING}" | wc -l) dep-blocked jobs:"
    echo "${DEP_PENDING}" | sed 's/^/      /'
    scancel ${DEP_PENDING}
    sleep 5  # let scancel propagate before resubmitting
fi

# ---- 2. Re-submit Phase 1/2 g=64 nvme pair (no deps) ----------------------
# These are the original 3029564 / 3029565 dep-pending blocks.  Even if they
# were already running by the time the controller recovered, scancel above
# is a no-op for non-pending jobs, and the next sbatch creates a fresh pair.
# So we only resubmit if their summary file is still missing.
PHASE12_RESULTS_ROOT="${WORKDIR}/results/geotransolver_volume/drivaer_ml_volume/gpus_64/nvme"
NEED_PHASE12=()
for sub in 10000 50000; do
    f="${PHASE12_RESULTS_ROOT}/sub_${sub}/runs/geotransolver_volume__drivaer_ml_volume__g64__nvme__sub${sub}/benchmark_summary.json"
    if [ ! -s "${f}" ]; then
        NEED_PHASE12+=("${sub}")
    fi
done

if [ ${#NEED_PHASE12[@]} -gt 0 ]; then
    echo
    echo "==> resubmitting Phase 1/2 g=64 nvme (no deps): ${NEED_PHASE12[*]}"
    bash submit_all.sh --matrix matrix_phase12_g64_nodeep_resub.json 2>&1 \
        | tail -10
else
    echo
    echo "==> Phase 1/2 g=64 nvme summaries already present -- skipping resubmit"
fi

# ---- 3. Re-submit Option B matrices (no --max-concurrent) -----------------
# These are the 28 + 16 = 44 Option B jobs (matrix_optionB_*.json).  We
# unconditionally resubmit each matrix; submit_all.sh's idempotency
# already skips entries whose results_dir contains benchmark_summary.json,
# so completed runs (and cells the user wants to overwrite) are handled
# correctly.  The key change vs the original submission: NO --max-concurrent.
echo
echo "==> resubmitting Option B sweep without --max-concurrent"
for m in \
    matrix_optionB_phaseA_g1redo.json \
    matrix_optionB_phaseB_g1.json     \
    matrix_optionB_phaseB_g16.json    \
    matrix_optionB_phaseB_g32.json    \
    matrix_optionB_phaseB_g64.json
do
    if [ -f "${m}" ]; then
        echo
        echo "    --- ${m} ---"
        bash submit_all.sh --matrix "${m}" 2>&1 | tail -12
    fi
done

echo
echo "==> done.  squeue should now show many jobs in PENDING/RUNNING with no Dependency reason."
echo "    (Cluster scheduler will dispatch them as nodes free up.)"
