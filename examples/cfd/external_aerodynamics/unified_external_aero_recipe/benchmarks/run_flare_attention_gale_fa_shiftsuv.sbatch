#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
#
# GALE_FA-only (FLARE on) subsample sweep — GeoTransolver surface × ShiftSUV.
# Use to extend _flare_attention_shiftsuv above 300k (GALE_FA rows only).
# 50k–300k GALE_FA already exist from the full A/B job unless --resume skips them.
#
# Optional env: FLARE_SUBS, FLARE_RESULTS_DIR, LUSTRE_DATA_ROOT_SHIFT_SUV

#SBATCH --account=coreai_modulus_cae
#SBATCH --partition=batch
#SBATCH --job-name=flare_fa_suv
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail

export FLARE_MODEL=geotransolver_surface
export FLARE_DATASET=shift_suv_estate_surface
export FLARE_ATTENTION_MODES=gale_fa
export FLARE_SUBS="${FLARE_SUBS:-350000,400000,450000,500000}"

USER_LUSTRE="${USER_LUSTRE:-/lustre/fs1/portfolios/coreai/projects/coreai_modulus_cae/users/nsobhani}"
GROUP_LUSTRE="${GROUP_LUSTRE:-/lustre/fs1/portfolios/coreai/projects/coreai_modulus_cae}"
export LUSTRE_DATA_ROOT="${LUSTRE_DATA_ROOT:-${LUSTRE_DATA_ROOT_SHIFT_SUV:-${GROUP_LUSTRE}/datasets/PhysicsNeMo-ShiftSUV}}"
export WORKDIR="${WORKDIR:-${USER_LUSTRE}/workdir/physicsnemo_cae_benchmark/examples/cfd/external_aerodynamics/unified_external_aero_recipe}"
export FLARE_RESULTS_DIR="${FLARE_RESULTS_DIR:-${WORKDIR}/results/_flare_attention_shiftsuv}"

BENCHMARKS_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
bash "${BENCHMARKS_DIR}/flare_attention_benchmark.sh"
