#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
#
# GALE_FA-only subsample sweep (FLARE on) — GeoTransolver surface × DrivAerML.
# Complements ordinary matrix runs (default GALE / FLARE off). Pair with
# geotransolver_surface/drivaer_ml_surface benchmark_summary for off-curve data.
#
# Optional env:
#   FLARE_SUBS              default 50000,100000,...,300000 (add 10000 for matrix parity)
#   Example 10k only: FLARE_SUBS=10000 FLARE_RETRY_FAILED=false sbatch ...
#   FLARE_NUM_EPOCHS        default 5
#   FLARE_RESULTS_DIR       default <WORKDIR>/results/_flare_attention_drivaer_surface
#   LUSTRE_DATA_ROOT_DRIVAER_ML  default fsw group datasets (flare volume 100k path)
#   FLARE_RETRY_FAILED           set true to re-run failed CSV rows

#SBATCH --account=coreai_modulus_cae
#SBATCH --partition=batch
#SBATCH --job-name=flare_fa_dm_s
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail

unset WORKDIR FLARE_RESULTS_DIR LUSTRE_DATA_ROOT 2>/dev/null || true

export FLARE_MODEL=geotransolver_surface
export FLARE_DATASET=drivaer_ml_surface
export FLARE_ATTENTION_MODES=gale_fa
export FLARE_RETRY_FAILED="${FLARE_RETRY_FAILED:-true}"
export FLARE_SUBS="${FLARE_SUBS:-50000,100000,150000,200000,250000,300000}"

USER_LUSTRE="${USER_LUSTRE:-/lustre/fs1/portfolios/coreai/projects/coreai_modulus_cae/users/nsobhani}"
export LUSTRE_DATA_ROOT_DRIVAER_ML="${LUSTRE_DATA_ROOT_DRIVAER_ML:-/lustre/fsw/portfolios/coreai/projects/coreai_modulus_cae/datasets/PhysicsNeMo-DrivaerML}"
export WORKDIR="${USER_LUSTRE}/workdir/physicsnemo_cae_benchmark/examples/cfd/external_aerodynamics/unified_external_aero_recipe"
export FLARE_RESULTS_DIR="${WORKDIR}/results/_flare_attention_drivaer_surface"

BENCHMARKS_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
bash "${BENCHMARKS_DIR}/flare_attention_benchmark.sh"
