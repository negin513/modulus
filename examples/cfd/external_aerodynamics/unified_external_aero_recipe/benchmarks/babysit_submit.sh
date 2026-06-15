#!/usr/bin/env bash
# babysit_submit.sh -- wait for slurmctld to recover, then auto-submit Phase 1.
#
# Polls `scontrol ping` every POLL_INTERVAL_S seconds. When *both* primary and
# backup controllers are UP, it invokes submit_all.sh on the Phase 1 matrix.
# All polling output and the final submission output are appended to the log
# file. The script exits 0 after a successful submission, exits 1 on a
# failed submission, and exits 2 if it gives up after MAX_WAIT_S without a
# recovery.
#
# Configurable via environment:
#   POLL_INTERVAL_S   default 60     (seconds between scontrol pings)
#   MAX_WAIT_S        default 21600  (6 hours)
#   MATRIX_FILE       default matrix_phase1.json
#   MAX_CONCURRENT    default 4
#   LUSTRE_DATA_ROOT_DRIVAER_ML  default /lustre/fs1/portfolios/coreai/projects/coreai_modulus_cae/datasets/PhysicsNeMo-DrivaerML
#
# Outputs (relative to this script's directory):
#   babysit.log              all polling + submission output
#   babysit.state            single line: "WAITING" | "SUBMITTING" | "DONE rc=N"
#
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

POLL_INTERVAL_S="${POLL_INTERVAL_S:-60}"
MAX_WAIT_S="${MAX_WAIT_S:-21600}"
MATRIX_FILE="${MATRIX_FILE:-matrix_phase1.json}"
MAX_CONCURRENT="${MAX_CONCURRENT:-4}"
export LUSTRE_DATA_ROOT_DRIVAER_ML="${LUSTRE_DATA_ROOT_DRIVAER_ML:-/lustre/fs1/portfolios/coreai/projects/coreai_modulus_cae/datasets/PhysicsNeMo-DrivaerML}"

LOG=babysit.log
STATE=babysit.state

_log() {
    printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$LOG"
}

_set_state() { printf '%s\n' "$*" > "$STATE"; }

_log "babysit start (interval=${POLL_INTERVAL_S}s, max_wait=${MAX_WAIT_S}s, matrix=${MATRIX_FILE})"
_set_state "WAITING"

deadline=$(( $(date +%s) + MAX_WAIT_S ))
attempt=0

while :; do
    attempt=$(( attempt + 1 ))
    if [ "$(date +%s)" -ge "$deadline" ]; then
        _log "GAVE UP after $((attempt - 1)) attempts; slurmctld never came back inside ${MAX_WAIT_S}s"
        _set_state "DONE rc=2"
        exit 2
    fi

    ping_out=$(timeout 10 scontrol ping 2>&1 || true)
    primary_state=$(printf '%s' "$ping_out" | awk '/Slurmctld\(primary\)/   {for (i=1;i<=NF;i++) if ($i=="UP" || $i=="DOWN") {print $i; exit}}')
    backup_state=$( printf '%s' "$ping_out" | awk '/Slurmctld\(backup\)/    {for (i=1;i<=NF;i++) if ($i=="UP" || $i=="DOWN") {print $i; exit}}')
    primary_state="${primary_state:-DOWN}"
    backup_state="${backup_state:-DOWN}"

    if [ "$primary_state" = "UP" ] || [ "$backup_state" = "UP" ]; then
        _log "controller recovered (primary=$primary_state backup=$backup_state) on attempt #$attempt -- verifying with sbatch --test-only"
        # extra confirmation: we want sbatch to actually be able to talk to it
        if probe=$(timeout 15 sbatch --test-only --account=coreai_modulus_cae --partition=batch --nodes=1 --ntasks=1 --time=00:01:00 --wrap="true" 2>&1) ; then
            _log "sbatch --test-only OK: ${probe}"
            break
        else
            _log "controller marked UP but sbatch --test-only failed (will keep polling): ${probe}"
        fi
    else
        if [ $((attempt % 5)) -eq 1 ]; then
            _log "still waiting (attempt #$attempt, primary=$primary_state, backup=$backup_state)"
        fi
    fi
    sleep "$POLL_INTERVAL_S"
done

_log "submitting Phase 1: ./submit_all.sh --matrix ${MATRIX_FILE} --max-concurrent ${MAX_CONCURRENT}"
_set_state "SUBMITTING"
set +e
./submit_all.sh --matrix "$MATRIX_FILE" --max-concurrent "$MAX_CONCURRENT" 2>&1 | tee -a "$LOG"
rc="${PIPESTATUS[0]}"
set -e

_log "submit_all.sh exited rc=${rc}"
_set_state "DONE rc=${rc}"

if [ "$rc" -eq 0 ]; then
    if [ -s submitted_jobs.tsv ]; then
        n=$(wc -l < submitted_jobs.tsv | tr -d ' ')
        _log "submitted_jobs.tsv has ${n} entries; tail:"
        tail -3 submitted_jobs.tsv | tee -a "$LOG" >/dev/null || true
    fi
fi

exit "$rc"
