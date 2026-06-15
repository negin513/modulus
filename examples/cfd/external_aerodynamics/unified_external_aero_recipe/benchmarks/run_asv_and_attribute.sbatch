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
#
# ---------------------------------------------------------------------------
# run_asv_and_attribute.sbatch -- per-op micro-benches + end-to-end attribution
# ---------------------------------------------------------------------------
#
# Runs two consecutive jobs on a single node, one GPU:
#
#   1. asv run on the 8 dual-implementation FunctionSpecs (KNN, RadiusSearch,
#      GridToPointInterpolation, PointToGridInterpolation, UniformGridGradient,
#      RectilinearGridGradient, MeshLSQGradient, MeshGreenGaussGradient).
#      Uses --python=same so we stay inside the recipe's venv instead of
#      letting ASV build its own (which would re-install torch / warp / cuml
#      from scratch and almost always fails inside enroot).
#
#   2. python benchmarks/profile_and_attribute.py --mode estimate ...
#      Profiles the synthetic step once with default dispatch and
#      cross-references the ASV results from step 1 to produce a single
#      end-to-end speedup number plus a per-functional breakdown.
#
# Conventions mirror run_benchmark.sbatch (HSG / coreai_modulus_cae). Override
# any of the env vars below via `--export=ALL,FOO=bar,...`.
#
# Optional env vars:
#   ASV_PHASES         default "forward"          (matches PHYSICSNEMO_ASV_PHASES)
#   ASV_FUNCTIONALS    default the 8 head-to-head specs above (CSV)
#   PROFILE_MODE       default "estimate"         ("estimate" or "measure")
#   PROFILE_DEVICE     default "cuda"
#   PROFILE_OUTPUT     default <REPORTS_DIR>/profile_attribute
#   REPORTS_DIR        default <WORKDIR>/results/_profile_attribute
#   CONTAINER_IMAGE    default same NGC sqsh as run_benchmark.sbatch
#   VENV_PATH          default same physnemo venv as run_benchmark.sbatch
#   WORKDIR            recipe directory (parent of src/train.py)
#   PHYSICSNEMO_ROOT   repository root that contains asv.conf.json (auto-derived
#                      from WORKDIR if not set)

#SBATCH --account=coreai_modulus_cae
#SBATCH --partition=batch
#SBATCH --job-name=asv_and_attribute
#SBATCH --time=01:30:00
#SBATCH --nodes=1
# HSG QOS requires >=4 GPUs / job (QOSMinGRES). We only use 1 GPU but must
# allocate the minimum to satisfy the policy; the other 3 sit idle.
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail

# ---------------------------------------------------------------------------
# Resolved env / defaults (kept aligned with run_benchmark.sbatch).
# ---------------------------------------------------------------------------
USER_LUSTRE="${USER_LUSTRE:-/lustre/fs1/portfolios/coreai/projects/coreai_modulus_cae/users/nsobhani}"
GROUP_LUSTRE="${GROUP_LUSTRE:-/lustre/fs1/portfolios/coreai/projects/coreai_modulus_cae}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-/lustre/fs1/portfolios/coreai/projects/coreai_modulus_cae/users/nsobhani/nvidia+pytorch+26.04-py3.sqsh}"
CONTAINER_MOUNTS="${CONTAINER_MOUNTS:-${USER_LUSTRE}:/user_data,${GROUP_LUSTRE}:/group_data,/lustre:/lustre,/home:/home,/tmp:/tmp}"
VENV_PATH="${VENV_PATH:-/lustre/fs1/portfolios/coreai/projects/coreai_modulus_cae/users/nsobhani/venvs/pytorch-26.04-py3-physnemo}"
# /lustre/fsw is a login-node-only symlink farm that does NOT resolve inside
# enroot; resolve to the canonical /lustre/fs1 path before passing into srun.
WORKDIR_RAW="${WORKDIR:-${USER_LUSTRE}/workdir/physicsnemo_cae_benchmark/examples/cfd/external_aerodynamics/unified_external_aero_recipe}"
WORKDIR="$(realpath "${WORKDIR_RAW}")"

# Anchor benchmarks/ to WORKDIR (NOT BASH_SOURCE -- SLURM copies the sbatch
# script into a spool dir on the compute node where sibling files do not exist).
BENCHMARKS_DIR="${WORKDIR}/benchmarks"

# The asv project root is the physicsnemo repo (4 levels up from WORKDIR).
PHYSICSNEMO_ROOT="${PHYSICSNEMO_ROOT:-$(realpath "${WORKDIR}/../../../..")}"

# Default to the 8 head-to-head specs (each has both a fast impl + torch baseline).
DEFAULT_FUNCTIONALS="KNN,RadiusSearch,GridToPointInterpolation,PointToGridInterpolation,UniformGridGradient,RectilinearGridGradient,MeshLSQGradient,MeshGreenGaussGradient"
ASV_FUNCTIONALS="${ASV_FUNCTIONALS:-${DEFAULT_FUNCTIONALS}}"
ASV_PHASES="${ASV_PHASES:-forward}"

PROFILE_MODE="${PROFILE_MODE:-estimate}"
PROFILE_DEVICE="${PROFILE_DEVICE:-cuda}"
REPORTS_DIR="${REPORTS_DIR:-${WORKDIR}/results/_profile_attribute}"
PROFILE_OUTPUT="${PROFILE_OUTPUT:-${REPORTS_DIR}/profile_attribute}"

mkdir -p "${REPORTS_DIR}"

echo "[asv_attribute] -------------------------------------------------------"
echo "[asv_attribute] job_id=${SLURM_JOB_ID:-?}  node=${SLURMD_NODENAME:-?}"
echo "[asv_attribute] workdir=${WORKDIR}"
echo "[asv_attribute] physicsnemo_root=${PHYSICSNEMO_ROOT}"
echo "[asv_attribute] asv_functionals=${ASV_FUNCTIONALS}"
echo "[asv_attribute] asv_phases=${ASV_PHASES}"
echo "[asv_attribute] profile_mode=${PROFILE_MODE}  device=${PROFILE_DEVICE}"
echo "[asv_attribute] reports_dir=${REPORTS_DIR}"
echo "[asv_attribute] container=${CONTAINER_IMAGE}"
echo "[asv_attribute] -------------------------------------------------------"

# ---------------------------------------------------------------------------
# 1. ASV micro-benchmarks (single srun task -- ASV spawns its own processes).
# ---------------------------------------------------------------------------
# Notes:
#   * `--python=same` reuses the container's venv instead of building a fresh
#     virtualenv that would have to re-install torch+warp+cuml from scratch.
#   * The benchmark runner picks up PHYSICSNEMO_ASV_FUNCTIONALS to filter the
#     spec list (see benchmark_functionals.py).
#   * Results land in ${PHYSICSNEMO_ROOT}/.asv/results/<hostname>/.
#   * `asv machine --yes` registers the compute node (per-job hostname) so
#     'asv run' doesn't prompt interactively the first time.
T_ASV=$SECONDS
srun --ntasks=1 --ntasks-per-node=1 \
     --container-image="${CONTAINER_IMAGE}" \
     --container-mounts="${CONTAINER_MOUNTS}" \
     bash -c "
        set -euo pipefail
        source ${VENV_PATH}/bin/activate
        cd ${PHYSICSNEMO_ROOT}
        export PHYSICSNEMO_ASV_PHASES=${ASV_PHASES}
        export PHYSICSNEMO_ASV_FUNCTIONALS=${ASV_FUNCTIONALS}
        export PHYSICSNEMO_ASV_DEVICE=${PROFILE_DEVICE}
        # Make sure asv is installed (cheap if already present).
        python -m pip show asv >/dev/null 2>&1 || python -m pip install asv
        # Register this compute node so 'asv run' doesn't prompt or bail.
        asv machine --yes
        # --python=same reuses the venv so we don't rebuild torch/warp/cuml.
        # When using --python=same, asv only benchmarks the current checkout
        # and does NOT accept a commit-range argument (e.g. HEAD^! errors out
        # with 'No range spec may be specified'). Omit it.
        # --quick reduces ASV's repeat budget (fewer reps per case) for speed.
        # --show-stderr surfaces subprocess errors so failures aren't silent.
        #
        # IMPORTANT: a single failing param combo (e.g. scipy KNN on CUDA)
        # makes 'asv run' return a non-zero exit code even when ASV itself
        # successfully recorded results for every OTHER case. We must not
        # let that take down the whole job before phase 2 runs, so capture
        # the rc, then only abort if no result JSON file was actually
        # written under .asv/results/<host>/.
        ASV_RC=0
        asv run --launch-method spawn --python=same --quick --show-stderr || ASV_RC=\$?

        # Count result files (excluding metadata: machine.json, benchmarks.json).
        N_RESULTS=\$(find .asv/results -mindepth 2 -maxdepth 2 -name '*.json' \
                          ! -name 'machine.json' ! -name 'benchmarks.json' \
                          2>/dev/null | wc -l)
        if [ \"\${N_RESULTS}\" -eq 0 ]; then
            echo \"[asv_attribute] FATAL: ASV produced no result JSON (rc=\${ASV_RC})\" >&2
            exit 1
        fi
        echo \"[asv_attribute] ASV finished rc=\${ASV_RC}; \${N_RESULTS} result file(s) written\"
     "
echo "[asv_attribute] asv done in $((SECONDS - T_ASV))s"

# ---------------------------------------------------------------------------
# 2. Profile + attribute (single GPU, single rank).
# ---------------------------------------------------------------------------
T_PROF=$SECONDS
srun --ntasks=1 --ntasks-per-node=1 \
     --container-image="${CONTAINER_IMAGE}" \
     --container-mounts="${CONTAINER_MOUNTS}" \
     bash -c "
        set -euo pipefail
        source ${VENV_PATH}/bin/activate
        cd ${PHYSICSNEMO_ROOT}
        python ${BENCHMARKS_DIR}/sweeps/profile_and_attribute.py \
            --mode ${PROFILE_MODE} \
            --step synthetic \
            --device ${PROFILE_DEVICE} \
            --asv-results ${PHYSICSNEMO_ROOT}/.asv/results \
            --output ${PROFILE_OUTPUT}
     "
echo "[asv_attribute] profile_and_attribute done in $((SECONDS - T_PROF))s"

echo "[asv_attribute] Outputs:"
ls -la "${REPORTS_DIR}" || true
