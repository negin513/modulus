#!/usr/bin/env python3
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

# ---------------------------------------------------------------------------
# triage_failures.py -- bucket-sort failed runs by error category
# ---------------------------------------------------------------------------
#
# Walks the results/ tree (or whatever --results points at), and for
# every run that produced a slurm-*.{out,err} log without a
# benchmark_summary.json:
#
#   1. Match the log's tail against known signatures (OOM, NCCL, data
#      loader, config / import, other).
#   2. Print a categorized table.
#   3. Optionally dump the last N lines of each log under each
#      category for fast eyeballing.
#
# A run is considered "failed" when it has a slurm log but no
# benchmark_summary.json -- the sbatch wrapper writes that file via
# summarize_run.py only on a clean train run.
#
# Usage:
#   python triage_failures.py --results results/
#   python triage_failures.py --results results/ --tail 40
#   python triage_failures.py --results results/ --json failures.json
# ---------------------------------------------------------------------------

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

# Order matters -- first matching pattern wins.  Patterns are
# case-insensitive, multiline, applied to a tail of the log.
SIGNATURES: list[tuple[str, list[re.Pattern]]] = [
    (
        "oom",
        [
            re.compile(r"CUDA out of memory", re.IGNORECASE),
            re.compile(r"torch\.cuda\.OutOfMemoryError", re.IGNORECASE),
            re.compile(r"OutOfMemoryError", re.IGNORECASE),
            re.compile(r"OOM", re.IGNORECASE),
        ],
    ),
    (
        "nccl",
        [
            re.compile(r"NCCL error", re.IGNORECASE),
            re.compile(r"ncclSystemError", re.IGNORECASE),
            re.compile(r"ncclUnhandledCudaError", re.IGNORECASE),
            re.compile(r"NCCL WARN", re.IGNORECASE),
            re.compile(r"Watchdog caught collective operation timeout", re.IGNORECASE),
        ],
    ),
    (
        "data_loader",
        [
            re.compile(r"Worker .* exited unexpectedly", re.IGNORECASE),
            re.compile(r"DataLoader worker .* killed", re.IGNORECASE),
            re.compile(r"MissingMandatoryValue", re.IGNORECASE),
            re.compile(r"FileNotFoundError.*\.pdmsh", re.IGNORECASE),
            re.compile(r"manifest", re.IGNORECASE),
            re.compile(r"stage_data", re.IGNORECASE),
        ],
    ),
    (
        "config_import",
        [
            re.compile(r"hydra\.errors", re.IGNORECASE),
            re.compile(r"ImportError", re.IGNORECASE),
            re.compile(r"ModuleNotFoundError", re.IGNORECASE),
            re.compile(r"ConfigCompositionException", re.IGNORECASE),
            re.compile(r"omegaconf\.errors", re.IGNORECASE),
        ],
    ),
    (
        "timeout",
        [
            re.compile(r"DUE TO TIME LIMIT", re.IGNORECASE),
            re.compile(r"TIMEOUT", re.IGNORECASE),
        ],
    ),
]


@dataclass
class FailedRun:
    results_dir: str
    log_path: str
    category: str
    matched_pattern: str | None
    last_lines: list[str]


def _classify(tail_text: str) -> tuple[str, str | None]:
    for category, patterns in SIGNATURES:
        for pat in patterns:
            m = pat.search(tail_text)
            if m:
                return category, pat.pattern
    return "other", None


def _read_tail(path: Path, n: int) -> list[str]:
    try:
        with path.open("rb") as fh:
            # Cheap-ish tail: read last ~64KB.
            fh.seek(0, 2)
            size = fh.tell()
            fh.seek(max(0, size - 64 * 1024))
            data = fh.read().decode(errors="replace")
    except OSError:
        return []
    lines = data.splitlines()
    return lines[-n:]


def _has_summary(slurm_log: Path) -> bool:
    """`results_dir/runs/<run_id>/benchmark_summary.json` is the
    canonical 'this run finished cleanly' marker.  Search for any
    summary under the parent results_dir."""
    results_dir = slurm_log.parent
    return any(results_dir.rglob("benchmark_summary.json"))


def _walk_failures(results_root: Path, tail_n: int) -> list[FailedRun]:
    failures: list[FailedRun] = []
    for log in sorted(results_root.rglob("slurm-*.out")):
        if _has_summary(log):
            continue
        tail = _read_tail(log, tail_n)
        # Also consider the matching .err file (sbatch writes them in pairs).
        err = log.with_suffix(".err")
        err_tail = _read_tail(err, tail_n) if err.exists() else []
        full_tail_text = "\n".join(tail + err_tail)
        category, matched = _classify(full_tail_text)
        failures.append(
            FailedRun(
                results_dir=str(log.parent),
                log_path=str(log),
                category=category,
                matched_pattern=matched,
                last_lines=tail[-min(tail_n, 20):] if not err_tail else err_tail[-min(tail_n, 20):],
            )
        )
    return failures


def _print_table(failures: list[FailedRun], dump_logs: bool, tail_n: int) -> None:
    by_cat: dict[str, list[FailedRun]] = defaultdict(list)
    for f in failures:
        by_cat[f.category].append(f)

    print(f"[triage] failed runs detected: {len(failures)}")
    if not failures:
        return
    print()
    print(f"{'category':<14} {'count':>5}")
    print("-" * 22)
    for cat in sorted(by_cat):
        print(f"{cat:<14} {len(by_cat[cat]):>5}")
    print()

    if not dump_logs:
        return

    for cat, runs in sorted(by_cat.items()):
        print(f"\n=== {cat} ({len(runs)}) ===")
        for f in runs:
            print(f"\n-- {f.log_path}  (pattern: {f.matched_pattern or 'none'}) --")
            for line in f.last_lines[-tail_n:]:
                print(f"  {line}")


def main(argv: Iterable[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", default="results", type=Path, help="Root of results tree")
    p.add_argument("--tail", type=int, default=20, help="Last N lines of each failing log to dump (default: %(default)s)")
    p.add_argument("--json", type=Path, default=None, help="Also write classification result as JSON to PATH")
    p.add_argument("--no-dump", action="store_true", help="Skip the per-log tail dump (just show the bucket counts)")
    args = p.parse_args(list(argv) if argv is not None else None)

    if not args.results.exists():
        print(f"[triage] results dir not found: {args.results}")
        return 2

    failures = _walk_failures(args.results, args.tail)
    _print_table(failures, dump_logs=not args.no_dump, tail_n=args.tail)

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        with args.json.open("w") as fh:
            json.dump([asdict(f) for f in failures], fh, indent=2)
        print(f"\n[triage] wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
