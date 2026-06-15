#!/bin/bash
#SBATCH -A coreai_modulus_cae
#SBATCH -J pytorch_measure_dbg
#SBATCH --time=02:00:00
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH -o /home/nsobhani/forks/physicsnemo_cae_benchmark/examples/cfd/external_aerodynamics/unified_external_aero_recipe/benchmarks/slurm-%j.out
#SBATCH -e /home/nsobhani/forks/physicsnemo_cae_benchmark/examples/cfd/external_aerodynamics/unified_external_aero_recipe/benchmarks/slurm-%j.err

# Debug 50k/100k crashes: CUDA_LAUNCH_BLOCKING vs no torch.compile.
# Each case logs separately; failures do not stop later cases.

set -uo pipefail

USER_LUSTRE="${USER_LUSTRE:-/lustre/fs1/portfolios/coreai/projects/coreai_modulus_cae/users/nsobhani}"
GROUP_LUSTRE="${GROUP_LUSTRE:-/lustre/fs1/portfolios/coreai/projects/coreai_modulus_cae}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-${USER_LUSTRE}/nvidia+pytorch+26.04-py3.sqsh}"
CONTAINER_MOUNTS="${CONTAINER_MOUNTS:-${USER_LUSTRE}:/user_data,${GROUP_LUSTRE}:/group_data,/lustre:/lustre,/home:/home,/tmp:/tmp}"
VENV_PATH="${VENV_PATH:-${USER_LUSTRE}/venvs/pytorch-26.04-py3-physnemo}"
WORKDIR="$(realpath "${WORKDIR:-/home/nsobhani/forks/physicsnemo_cae_benchmark/examples/cfd/external_aerodynamics/unified_external_aero_recipe}")"
BENCHMARKS_DIR="${WORKDIR}/benchmarks"
PHYSICSNEMO_ROOT="$(realpath "${WORKDIR}/../../../..")"
LUSTRE_DATA_ROOT="${LUSTRE_DATA_ROOT:-${GROUP_LUSTRE}/datasets/PhysicsNeMo-ShiftSUV}"
REPORTS_BASE="${REPORTS_BASE:-${WORKDIR}/results/_profile_attribute_debug}"

mkdir -p "${REPORTS_BASE}"

run_case() {
    local label="$1"
    local subs_csv="$2"
    local compile_flag="$3"
    local cuda_blocking="${4:-0}"
    local out_dir="${REPORTS_BASE}/${label}"
    mkdir -p "${out_dir}"

    echo ""
    echo "================================================================"
    echo "[debug] CASE=${label}  subs=${subs_csv}  compile=${compile_flag}  CUDA_LAUNCH_BLOCKING=${cuda_blocking}"
    echo "================================================================"

    local blocking_export=""
    if [ "${cuda_blocking}" = "1" ]; then
        blocking_export="export CUDA_LAUNCH_BLOCKING=1;"
    else
        blocking_export="unset CUDA_LAUNCH_BLOCKING;"
    fi

    # shellcheck disable=SC2206
    local sub_arr=( ${subs_csv//,/ } )

    if ! srun --ntasks=1 --ntasks-per-node=1 \
         --container-image="${CONTAINER_IMAGE}" \
         --container-mounts="${CONTAINER_MOUNTS}" \
         bash -c "
            set -uo pipefail
            ${blocking_export}
            export DATASET_PATH_SHIFT_SUV='${LUSTRE_DATA_ROOT}'
            export PROFILE_MODEL=geotransolver_volume
            export PROFILE_DATASET=shift_suv_estate_volume
            source '${VENV_PATH}/bin/activate'
            cd '${WORKDIR}'
            python '${BENCHMARKS_DIR}/sweeps/sweep_pytorch_measure.py' \
                --device cuda \
                --model geotransolver_volume \
                --dataset shift_suv_estate_volume \
                --n-warmup 3 \
                --n-record 5 \
                --reports-dir '${out_dir}' \
                --asv-results '${PHYSICSNEMO_ROOT}/.asv/results' \
                --subs ${sub_arr[*]} \
                ${compile_flag} \
                --continue-on-error
         "; then
        echo "[debug] CASE ${label}: srun FAILED (see slurm-%j.err)"
        return 1
    fi
    echo "[debug] CASE ${label}: done -> ${out_dir}"
    ls -la "${out_dir}" 2>/dev/null || true
    return 0
}

echo "[debug] job_id=${SLURM_JOB_ID:-?} node=${SLURMD_NODENAME:-?}"
echo "[debug] dataset=${LUSTRE_DATA_ROOT}"
echo "[debug] reports_base=${REPORTS_BASE}"

# 1) 50k with sync CUDA errors (compile on — reproduces prior failure mode)
run_case "50k_blocking_compile" "50000" "--compile" "1" || true

# 2) 50k no compile
run_case "50k_nocompile" "50000" "--no-compile" "0" || true

# 3) 100k no compile
run_case "100k_nocompile" "100000" "--no-compile" "0" || true

# 4) 100k with blocking + compile (optional control)
run_case "100k_blocking_compile" "100000" "--compile" "1" || true

echo ""
echo "[debug] All cases finished. Summary:"
for d in "${REPORTS_BASE}"/*/pytorch_measure_summary.json; do
    [ -f "$d" ] || continue
    echo "--- $d ---"
    cat "$d"
done
