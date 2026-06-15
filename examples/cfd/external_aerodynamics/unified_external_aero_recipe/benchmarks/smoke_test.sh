#!/bin/bash
# smoke_test.sh -- single 1-GPU sbatch that validates the /home mount fix.
#
# Submits ONE 5-minute job that:
#   1. activates the venv inside the container
#   2. cd's into WORKDIR (i.e. follows the workdir symlink that now needs
#      the /home identity bind mount)
#   3. imports physicsnemo
#   4. invokes `src/train.py --help` (cheap; validates Hydra config tree
#      reachable + entry point importable)
#
# Then waits for the job to terminate (poll squeue) and prints a pass/fail
# verdict by inspecting the .out / .err.
#
# Usage:
#   ./smoke_test.sh                # default: g1 lustre, sampling 10000, surface
#   MODEL=geotransolver_volume DATASET=drivaer_ml_volume ./smoke_test.sh
#
# Required env (one only):
#   LUSTRE_DATA_ROOT_DRIVAER_ML   path on /lustre/fs1 to the dataset (only
#                                 needed if we ever decide to actually load
#                                 it; not needed for --help, but helps
#                                 surface other config errors).
#
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RECIPE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${SCRIPT_DIR}"

MODEL="${MODEL:-geotransolver_surface}"
DATASET="${DATASET:-drivaer_ml_surface}"
# HSG QOS requires whole-node (4-GPU) allocations; a 1-GPU request triggers
# QOSMinGRES at sbatch submission time.  We allocate the full node and let
# run_benchmark.sbatch's inner `srun --ntasks-per-node=4` use it.
NUM_GPUS=4
STORAGE_MODE=lustre
SUBSAMPLING=10000
RUN_ID="smoketest__${MODEL}__${DATASET}__$(date +%Y%m%d_%H%M%S)"
# Absolute path: train.py runs after `cd $WORKDIR` (= RECIPE_ROOT), so its
# relative output_dir would land under $WORKDIR while the sbatch's post-run
# `[ -f $METRICS_PATH ]` check runs from sbatch's submission cwd.  Anchoring
# RESULTS_DIR to RECIPE_ROOT up front makes both views resolve to the same
# physical location -- without this, metrics.jsonl is written but not seen
# and summarize_run.py is silently skipped.
RESULTS_DIR="${RECIPE_ROOT}/results/_smoketest/${RUN_ID}"
mkdir -p "$RESULTS_DIR"

# Use the same exports the user might already have, but force them sane.
unset USER_LUSTRE GROUP_LUSTRE VENV_PATH WORKDIR CONTAINER_MOUNTS
export LUSTRE_DATA_ROOT_DRIVAER_ML="${LUSTRE_DATA_ROOT_DRIVAER_ML:-/lustre/fs1/portfolios/coreai/projects/coreai_modulus_cae/datasets/PhysicsNeMo-DrivaerML}"

# Reuse the matrix sbatch with the smallest possible workload (1 epoch,
# sampling=10000) so we exercise the FULL code path: container start,
# /home bind mount, venv activate, cd into WORKDIR, src/train.py main(),
# OmegaConf load of datasets/dataset_paths.yaml (now env-var driven),
# pipeline build, one training step.  Anything that would break Phase 1
# also breaks this smoke run.
EXTRA_OVERRIDES=""

echo "[smoke] submitting ${RUN_ID}"
JOBID=$(sbatch --parsable \
    --nodes=1 --ntasks-per-node=4 --gpus-per-node=4 \
    --time=00:10:00 \
    --output="${RESULTS_DIR}/slurm-%j.out" \
    --error="${RESULTS_DIR}/slurm-%j.err" \
    --job-name=cae_smoketest \
    --export=ALL,MODEL="${MODEL}",DATASET="${DATASET}",NUM_GPUS=${NUM_GPUS},STORAGE_MODE=${STORAGE_MODE},SUBSAMPLING=${SUBSAMPLING},RESULTS_DIR="${RESULTS_DIR}",RUN_ID="${RUN_ID}",NUM_EPOCHS=1,EXTRA_OVERRIDES="${EXTRA_OVERRIDES}",LUSTRE_DATA_ROOT="${LUSTRE_DATA_ROOT_DRIVAER_ML}" \
    "$(pwd)/run_benchmark.sbatch")

echo "[smoke] jobid=${JOBID}  results=${RESULTS_DIR}"
echo "[smoke] waiting for job to finish..."

# Poll squeue until the job is gone.
for i in $(seq 1 360); do
    sleep 10
    state=$(squeue -h -j "${JOBID}" -o '%T' 2>/dev/null || true)
    if [ -z "${state}" ]; then
        break
    fi
    if [ $((i % 6)) -eq 0 ]; then
        echo "[smoke] still ${state} (elapsed ${i}0s)"
    fi
done

OUT="${RESULTS_DIR}/slurm-${JOBID}.out"
ERR="${RESULTS_DIR}/slurm-${JOBID}.err"

echo
echo "============================== STDOUT =============================="
[ -f "${OUT}" ] && tail -60 "${OUT}" || echo "(no stdout file)"
echo
echo "============================== STDERR =============================="
[ -f "${ERR}" ] && tail -60 "${ERR}" || echo "(no stderr file)"
echo
echo "===================================================================="

# Heuristic verdict.  Order matters: the most specific patterns first so
# we don't mis-classify (e.g. a "No such file" from inside a dataloader
# is NOT the same problem as "No such file" on `cd $WORKDIR`).
fail=0
if [ ! -s "${OUT}" ]; then
    echo "[smoke] FAIL: empty stdout"
    fail=1
elif grep -qE 'cd: .*: No such file or directory' "${ERR}" 2>/dev/null; then
    echo "[smoke] FAIL: bash 'cd $WORKDIR' failed -- /home mount or symlink chain still broken"
    fail=1
elif grep -qE 'ModuleNotFoundError|ImportError.*: No module named' "${ERR}" 2>/dev/null; then
    echo "[smoke] FAIL: import error in venv -- venv path or container layering is broken"
    fail=1
elif grep -qE 'MissingMandatoryValue|oc.env.*not set|Could not override' "${ERR}" 2>/dev/null; then
    echo "[smoke] FAIL: Hydra/OmegaConf config error -- likely a missing DATASET_PATH_* env var or a stale Hydra override"
    fail=1
elif grep -qE 'FileNotFoundError.*meta\.json|FileNotFoundError.*\.pdmsh' "${ERR}" 2>/dev/null; then
    echo "[smoke] FAIL: dataset layout mismatch -- the dataloader hit a missing tensordict file (likely a glob pattern hitting a hidden/cache tree)"
    fail=1
elif grep -qE 'CUDA out of memory|OutOfMemoryError' "${ERR}" 2>/dev/null; then
    echo "[smoke] FAIL: OOM at sampling=10000 -- something is very wrong, smoke shouldn't OOM"
    fail=1
elif grep -qE 'srun: error|Exited with exit code' "${ERR}" 2>/dev/null; then
    echo "[smoke] WARN: srun reported an error but no specific diagnostic pattern matched.  Inspect stderr."
    fail=1
fi

if [ "${fail}" -eq 0 ]; then
    echo "[smoke] PASS -- mount + venv + Hydra entry point are reachable.  Safe to run Phase 1."
    exit 0
else
    echo "[smoke] FAIL -- DO NOT RESUBMIT PHASE 1 yet.  Inspect ${OUT} and ${ERR}."
    exit 1
fi
