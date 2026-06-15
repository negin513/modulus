#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
#
# Start/stop/status for watch_scaling_plots.py — auto-regenerates scaling PNGs
# (and the Cursor canvas) when new benchmark_summary.json files land.
#
# Usage:
#   ./watch_scaling_plots.sh start      # daemon (default poll 60s)
#   ./watch_scaling_plots.sh stop
#   ./watch_scaling_plots.sh status
#   ./watch_scaling_plots.sh once       # regenerate now if results changed
#   ./watch_scaling_plots.sh foreground # run in foreground (Ctrl-C to stop)
#
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

POLL_INTERVAL="${POLL_INTERVAL:-60}"
PIDFILE=watch_scaling.pid
LOG=watch_scaling.log
PY=tools/watch_scaling_plots.py

_is_running() {
    [ -f "$PIDFILE" ] || return 1
    local pid
    pid=$(tr -d '[:space:]' < "$PIDFILE")
    [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null
}

cmd="${1:-status}"

case "$cmd" in
    start)
        if _is_running; then
            echo "[watch] already running (pid $(cat "$PIDFILE"))"
            exit 0
        fi
        nohup python3 "$PY" --poll-interval "$POLL_INTERVAL" >> "$LOG" 2>&1 &
        echo $! > "$PIDFILE"
        echo "[watch] started pid=$! poll=${POLL_INTERVAL}s log=$LOG"
        ;;
    stop)
        if ! _is_running; then
            echo "[watch] not running"
            rm -f "$PIDFILE"
            exit 0
        fi
        pid=$(tr -d '[:space:]' < "$PIDFILE")
        kill "$pid" 2>/dev/null || true
        rm -f "$PIDFILE"
        echo "[watch] stopped pid=$pid"
        ;;
    status)
        if _is_running; then
            echo "[watch] running pid=$(cat "$PIDFILE") log=$LOG"
            tail -5 "$LOG" 2>/dev/null || true
        else
            echo "[watch] not running"
            rm -f "$PIDFILE" 2>/dev/null || true
        fi
        ;;
    once)
        python3 "$PY" --once
        ;;
    foreground|fg)
        exec python3 "$PY" --poll-interval "$POLL_INTERVAL"
        ;;
    *)
        echo "usage: $0 {start|stop|status|once|foreground}" >&2
        exit 2
        ;;
esac
