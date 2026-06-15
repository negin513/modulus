#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
#
# GeoTransolver GALE vs GALE_FA (FLARE attention) subsample sweep.
#
# Defaults: ShiftSUV surface. Override FLARE_MODEL / FLARE_DATASET / LUSTRE_DATA_ROOT
# for DrivAerML volume (see run_flare_attention_drivaer_volume.sbatch).
#
# Optional env:
#   FLARE_SUBS              default 50000,100000,...,500000
#   FLARE_ATTENTION_MODES   default both (use both + FLARE_SUBS=10000 for 10k A/B)
#   Example 10k only: FLARE_SUBS=10000 FLARE_ATTENTION_MODES=both FLARE_RETRY_FAILED=false sbatch ...
#   FLARE_MODEL             default geotransolver_surface
#   FLARE_DATASET           default shift_suv_estate_surface
#   FLARE_NUM_EPOCHS        default 5
#   FLARE_EXCLUDE_EPOCHS    default 0
#   FLARE_RESULTS_DIR       default <WORKDIR>/results/_flare_attention_shiftsuv

#SBATCH --account=coreai_modulus_cae
#SBATCH --partition=batch
#SBATCH --job-name=flare_attn
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail
BENCHMARKS_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
bash "${BENCHMARKS_DIR}/flare_attention_benchmark.sh"
