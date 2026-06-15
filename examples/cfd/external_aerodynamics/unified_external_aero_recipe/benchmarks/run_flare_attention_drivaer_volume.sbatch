#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
#
# FLARE attention A/B: GeoTransolver Volume × DrivAerML (GALE vs GALE_FA).
#
# Requires readable DATASET_PATH_DRIVAER_ML on Lustre (PhysicsNeMo-DrivaerML .pdmsh).
# If permission denied, fix group ACLs on boundary memmaps before relying on results.
#
# Optional env:
#   FLARE_SUBS              default 50000,100000,150000,200000,250000,300000
#   FLARE_COMPILE           default true; use false for 50k volume (Inductor cdist bug)
#   FLARE_RETRY_FAILED      default false; set true to re-run failed rows with --resume
#   FLARE_NUM_EPOCHS        default 5
#   FLARE_RESULTS_DIR       default <WORKDIR>/results/_flare_attention_drivaer_volume
#   LUSTRE_DATA_ROOT        DrivAerML tree (default group Lustre path)

#SBATCH --account=coreai_modulus_cae
#SBATCH --partition=batch
#SBATCH --job-name=flare_vol_dm
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail

export FLARE_MODEL=geotransolver_volume
export FLARE_DATASET=drivaer_ml_volume
export FLARE_SUBS="${FLARE_SUBS:-50000,100000,150000,200000,250000,300000}"

USER_LUSTRE="${USER_LUSTRE:-/lustre/fs1/portfolios/coreai/projects/coreai_modulus_cae/users/nsobhani}"
GROUP_LUSTRE="${GROUP_LUSTRE:-/lustre/fs1/portfolios/coreai/projects/coreai_modulus_cae}"
# DrivAerML lives on fs1 (not fsw); explicit default avoids missing train_datadir on compute nodes.
export LUSTRE_DATA_ROOT_DRIVAER_ML="${LUSTRE_DATA_ROOT_DRIVAER_ML:-/lustre/fsw/portfolios/coreai/projects/coreai_modulus_cae/datasets/PhysicsNeMo-DrivaerML}"
export WORKDIR="${WORKDIR:-${USER_LUSTRE}/workdir/physicsnemo_cae_benchmark/examples/cfd/external_aerodynamics/unified_external_aero_recipe}"
export FLARE_RESULTS_DIR="${FLARE_RESULTS_DIR:-${WORKDIR}/results/_flare_attention_drivaer_volume}"

# SLURM copies this script to spool/; use submit directory to find shared runtime.
BENCHMARKS_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
bash "${BENCHMARKS_DIR}/flare_attention_benchmark.sh"
