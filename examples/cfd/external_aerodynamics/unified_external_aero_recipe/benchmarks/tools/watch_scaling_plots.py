#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Poll results/ for new benchmark runs and regenerate scaling snapshot plots (+ canvas)."""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import sys
from pathlib import Path

_BENCHMARKS_ROOT = Path(__file__).resolve().parents[1]
if str(_BENCHMARKS_ROOT) not in sys.path:
    sys.path.insert(0, str(_BENCHMARKS_ROOT))
from paths import BENCHMARKS_DIR, RECIPE_ROOT, canvas_path

DEFAULT_RESULTS = RECIPE_ROOT / "results"
DEFAULT_OUT = RECIPE_ROOT / "results/_scaling_snapshot"
DEFAULT_CANVAS = canvas_path("e2e-training-performance.canvas.tsx")
DEFAULT_FULL_CANVAS = canvas_path("e2e-full-report.canvas.tsx")
DEFAULT_STATE = BENCHMARKS_DIR / "watch_scaling.state"
DEFAULT_LOG = BENCHMARKS_DIR / "watch_scaling.log"


def _log(msg: str, log_path: Path) -> None:
    line = f"[{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}] {msg}"
    print(line, flush=True)
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def _fingerprint(results_root: Path) -> tuple[str, int]:
    """Return (sha256 digest, run count) for benchmark summaries + profile JSON."""
    parts: list[str] = []
    count = 0
    for path in sorted(results_root.rglob("benchmark_summary.json")):
        if "_smoketest" in str(path):
            continue
        stat = path.stat()
        parts.append(f"{path.relative_to(results_root)}:{stat.st_mtime_ns}:{stat.st_size}")
        count += 1
    profile_json = results_root / "_profile_attribute" / "pytorch_measured.json"
    if profile_json.is_file():
        stat = profile_json.stat()
        parts.append(f"_profile_attribute/pytorch_measured.json:{stat.st_mtime_ns}:{stat.st_size}")
    digest = hashlib.sha256("\n".join(parts).encode()).hexdigest()[:16]
    return digest, count


def _load_state(state_path: Path) -> str | None:
    if not state_path.is_file():
        return None
    return state_path.read_text(encoding="utf-8").strip() or None


def _save_state(state_path: Path, fingerprint: str) -> None:
    state_path.write_text(fingerprint + "\n", encoding="utf-8")


def _regenerate(
    *,
    results: Path,
    out_dir: Path,
    canvas_out: Path | None,
    log_path: Path,
) -> None:
    plot_cmd = [
        sys.executable,
        str(BENCHMARKS_DIR / "plots/plot_scaling_snapshot.py"),
        "--results",
        str(results),
        "--out-dir",
        str(out_dir),
    ]
    _log(f"running: {' '.join(plot_cmd)}", log_path)
    subprocess.run(plot_cmd, check=True, cwd=RECIPE_ROOT)

    infer_cmd = [
        sys.executable,
        str(BENCHMARKS_DIR / "plots/plot_inference.py"),
        "--results",
        str(results),
        "--out-dir",
        str(out_dir),
    ]
    _log(f"running: {' '.join(infer_cmd)}", log_path)
    subprocess.run(infer_cmd, check=True, cwd=RECIPE_ROOT)

    latency_cmd = [
        sys.executable,
        str(BENCHMARKS_DIR / "plots/plot_model_latency.py"),
        "--results",
        str(results),
        "--out-dir",
        str(out_dir),
    ]
    pytorch_json = results / "_profile_attribute" / "pytorch_measured.json"
    if pytorch_json.is_file():
        latency_cmd.extend(["--pytorch-json", str(pytorch_json)])
    _log(f"running: {' '.join(latency_cmd)}", log_path)
    subprocess.run(latency_cmd, check=True, cwd=RECIPE_ROOT)

    if canvas_out is not None:
        for script, out_path in (
            ("canvas/build_canvas_report.py", canvas_out),
            ("canvas/build_full_canvas_report.py", DEFAULT_FULL_CANVAS),
        ):
            canvas_cmd = [
                sys.executable,
                str(BENCHMARKS_DIR / script),
                "--plots-dir",
                str(out_dir),
                "--results",
                str(results),
                "--out",
                str(out_path),
            ]
            _log(f"running: {' '.join(canvas_cmd)}", log_path)
            subprocess.run(canvas_cmd, check=True, cwd=RECIPE_ROOT)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--canvas-out",
        type=Path,
        default=DEFAULT_CANVAS,
        help="Canvas output path (pass --no-canvas to skip)",
    )
    parser.add_argument("--no-canvas", action="store_true")
    parser.add_argument("--poll-interval", type=float, default=60.0, help="Seconds between polls")
    parser.add_argument(
        "--debounce",
        type=float,
        default=20.0,
        help="Wait this long after a change before regenerating (avoids partial writes)",
    )
    parser.add_argument("--state-file", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--log-file", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--once", action="store_true", help="Regenerate once if changed, then exit")
    args = parser.parse_args()

    canvas_out = None if args.no_canvas else args.canvas_out
    args.log_file.parent.mkdir(parents=True, exist_ok=True)

    if not args.results.is_dir():
        _log(f"results directory missing: {args.results}", args.log_file)
        return 2

    last_fp = _load_state(args.state_file)
    pending_fp: str | None = None
    pending_since: float | None = None

    _log(
        f"watch start results={args.results} out={args.out_dir} "
        f"poll={args.poll_interval}s debounce={args.debounce}s "
        f"canvas={'off' if canvas_out is None else canvas_out}",
        args.log_file,
    )
    if last_fp:
        _, last_count = _fingerprint(args.results)
        _log(f"last known fingerprint={last_fp} ({last_count} runs on disk now)", args.log_file)
    else:
        _log("no prior state — will regenerate on first stable poll", args.log_file)
        pending_fp, _ = _fingerprint(args.results)
        pending_since = time.monotonic() - args.debounce

    while True:
        fp, count = _fingerprint(args.results)

        if fp == last_fp:
            pending_fp = None
            pending_since = None
            if args.once:
                _log(f"no change ({count} runs, fp={fp})", args.log_file)
                return 0
        elif pending_fp != fp:
            pending_fp = fp
            pending_since = time.monotonic()
            _log(f"change detected: {count} runs, fp={fp} (debouncing {args.debounce}s)", args.log_file)
            if args.once:
                time.sleep(args.debounce)
                try:
                    _regenerate(
                        results=args.results,
                        out_dir=args.out_dir,
                        canvas_out=canvas_out,
                        log_path=args.log_file,
                    )
                    _save_state(args.state_file, fp)
                    _log(f"regenerated plots for {count} runs (fp={fp})", args.log_file)
                except subprocess.CalledProcessError as exc:
                    _log(f"regeneration failed: rc={exc.returncode}", args.log_file)
                    return exc.returncode or 1
                return 0
        elif pending_since is not None and (time.monotonic() - pending_since) >= args.debounce:
            if fp == pending_fp:
                try:
                    _regenerate(
                        results=args.results,
                        out_dir=args.out_dir,
                        canvas_out=canvas_out,
                        log_path=args.log_file,
                    )
                    _save_state(args.state_file, fp)
                    last_fp = fp
                    _log(f"regenerated plots for {count} runs (fp={fp})", args.log_file)
                except subprocess.CalledProcessError as exc:
                    _log(f"regeneration failed: rc={exc.returncode}", args.log_file)
                    if args.once:
                        return exc.returncode or 1
                else:
                    if args.once:
                        return 0
                pending_fp = None
                pending_since = None

        time.sleep(args.poll_interval)


if __name__ == "__main__":
    raise SystemExit(main())
