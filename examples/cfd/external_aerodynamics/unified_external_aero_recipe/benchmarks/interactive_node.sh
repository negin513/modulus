#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
#
# Allocate an interactive compute node and open a shell matching benchmark jobs:
# enroot container, Lustre mounts, and the physnemo venv on Lustre.
#
# Usage (from login node):
#   ./interactive_node.sh
#   TIME=04:00:00 ./interactive_node.sh
#
# HSG QOS requires whole-node (4-GPU) allocations; fewer GPUs triggers QOSMinGRES.
#
# Usage (already inside an allocation — e.g. after manual salloc):
#   ./interactive_node.sh --shell-only
#
# Optional env (defaults match run_benchmark.sbatch / flare_attention_benchmark.sh):
#   SLURM_ACCOUNT, SLURM_PARTITION, GPUS_PER_NODE, TIME
#   USER_LUSTRE, GROUP_LUSTRE, CONTAINER_IMAGE, CONTAINER_MOUNTS, VENV_PATH, WORKDIR
#   LUSTRE_DATA_ROOT, LUSTRE_DATA_ROOT_DRIVAER_ML
#   INTERACTIVE_USE_HOME_WORKDIR=1  keep WORKDIR under $HOME/forks (default: Lustre workdir)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# /lustre/fsw is a login-node symlink farm; enroot and compute nodes need /lustre/fs1.
lustre_fs1_path() {
    local p="$1"
    case "${p}" in
        /lustre/fsw/*) printf '/lustre/fs1/%s' "${p#/lustre/fsw/}" ;;
        *) printf '%s' "${p}" ;;
    esac
}

lustre_fs1_mounts() {
    local mounts="$1"
    local entry src dst
    local -a normalized=()
    local IFS=,
    for entry in ${mounts}; do
        src="${entry%%:*}"
        dst="${entry#*:}"
        if [ "${src}" = "${entry}" ]; then
            normalized+=("$(lustre_fs1_path "${entry}")")
        else
            normalized+=("$(lustre_fs1_path "${src}"):${dst}")
        fi
    done
    local IFS=,
    printf '%s' "${normalized[*]}"
}

SLURM_ACCOUNT="${SLURM_ACCOUNT:-coreai_modulus_cae}"
SLURM_PARTITION="${SLURM_PARTITION:-batch}"
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
TIME="${TIME:-02:00:00}"

USER_LUSTRE="$(lustre_fs1_path "${USER_LUSTRE:-/lustre/fs1/portfolios/coreai/projects/coreai_modulus_cae/users/nsobhani}")"
GROUP_LUSTRE="$(lustre_fs1_path "${GROUP_LUSTRE:-/lustre/fs1/portfolios/coreai/projects/coreai_modulus_cae}")"
CONTAINER_IMAGE="$(lustre_fs1_path "${CONTAINER_IMAGE:-${USER_LUSTRE}/nvidia+pytorch+26.04-py3.sqsh}")"
CONTAINER_MOUNTS="$(lustre_fs1_mounts "${CONTAINER_MOUNTS:-${USER_LUSTRE}:/user_data,${GROUP_LUSTRE}:/group_data,/lustre:/lustre,/home:/home,/tmp:/tmp}")"
VENV_PATH="$(lustre_fs1_path "${VENV_PATH:-${USER_LUSTRE}/venvs/pytorch-26.04-py3-physnemo}")"
LUSTRE_WORKDIR="${USER_LUSTRE}/workdir/physicsnemo_cae_benchmark/examples/cfd/external_aerodynamics/unified_external_aero_recipe"
if [ -n "${WORKDIR:-}" ] && [ "${INTERACTIVE_USE_HOME_WORKDIR:-0}" != "1" ]; then
    case "${WORKDIR}" in
        /home/*|*/forks/*)
            echo "[interactive_node] ignoring WORKDIR=${WORKDIR}; using Lustre workdir (set INTERACTIVE_USE_HOME_WORKDIR=1 to keep HOME path)" >&2
            WORKDIR_RAW="${LUSTRE_WORKDIR}"
            ;;
        *) WORKDIR_RAW="$(lustre_fs1_path "${WORKDIR}")" ;;
    esac
else
    WORKDIR_RAW="$(lustre_fs1_path "${WORKDIR:-${LUSTRE_WORKDIR}}")"
fi
LUSTRE_DATA_ROOT_DRIVAER_ML="$(lustre_fs1_path "${LUSTRE_DATA_ROOT_DRIVAER_ML:-${GROUP_LUSTRE}/datasets/PhysicsNeMo-DrivAerML}")"
LUSTRE_DATA_ROOT="$(lustre_fs1_path "${LUSTRE_DATA_ROOT:-${LUSTRE_DATA_ROOT_DRIVAER_ML}}")"

export USER_LUSTRE GROUP_LUSTRE CONTAINER_IMAGE CONTAINER_MOUNTS VENV_PATH
export LUSTRE_DATA_ROOT LUSTRE_DATA_ROOT_DRIVAER_ML
export DATASET_PATH_DRIVAER_ML="${LUSTRE_DATA_ROOT_DRIVAER_ML}"

shell_only() {
    if [ ! -f "${CONTAINER_IMAGE}" ]; then
        echo "[interactive_node] missing container image: ${CONTAINER_IMAGE}" >&2
        exit 1
    fi
    if [ ! -f "${VENV_PATH}/bin/activate" ]; then
        echo "[interactive_node] missing venv activate: ${VENV_PATH}/bin/activate" >&2
        exit 1
    fi

    WORKDIR="$(realpath "${WORKDIR_RAW}" 2>/dev/null || echo "${WORKDIR_RAW}")"
    export WORKDIR

    NODE_NAME="${SLURMD_NODENAME:-$(srun --ntasks=1 hostname 2>/dev/null || echo '?')}"

    echo "[interactive_node] node=${NODE_NAME} job=${SLURM_JOB_ID:-?}"
    echo "[interactive_node] workdir=${WORKDIR}"
    echo "[interactive_node] dataset=${LUSTRE_DATA_ROOT}"
    echo "[interactive_node] container=${CONTAINER_IMAGE}"
    echo "[interactive_node] venv=${VENV_PATH}"

    # No bash -l: login profiles on compute nodes often break (e.g. 'IF: command not found').
    exec srun --pty --ntasks=1 --ntasks-per-node=1 \
        --gpus-per-node="${GPUS_PER_NODE}" \
        --container-image="${CONTAINER_IMAGE}" \
        --container-mounts="${CONTAINER_MOUNTS}" \
        bash -c "
            set -euo pipefail
            source '${VENV_PATH}/bin/activate'
            cd '${WORKDIR}'
            echo '[interactive_node] inside container; python=' \"\$(command -v python)\"
            export PS1='(cae-interactive) \w\$ '
            exec bash --noprofile --norc -i
        "
}

if [ "${1:-}" = "--shell-only" ]; then
    shell_only
fi

if [ -n "${SLURM_JOB_ID:-}" ]; then
    echo "[interactive_node] SLURM allocation detected; opening container shell."
    shell_only
fi

echo "[interactive_node] requesting ${GPUS_PER_NODE} GPU(s), time=${TIME}"
exec salloc \
    --account="${SLURM_ACCOUNT}" \
    --partition="${SLURM_PARTITION}" \
    --nodes=1 \
    --ntasks-per-node=1 \
    --gpus-per-node="${GPUS_PER_NODE}" \
    --time="${TIME}" \
    --job-name=interactive_cae \
    bash "${SCRIPT_DIR}/interactive_node.sh" --shell-only
