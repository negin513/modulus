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
# summarize_run.py -- post-process metrics.jsonl into a benchmark summary
# ---------------------------------------------------------------------------
#
# Reads one training run's metrics.jsonl (written by src/train.py) and
# emits two artifacts next to it:
#
#   benchmark_summary.json   # machine-readable: aggregates + axis tags
#   benchmark_summary.md     # human-readable mirror of the JSON
#
# The recipe already writes per-step JSONL records carrying step_time_s
# and mem_gb, so this is a pure post-processing step -- no recipe edits
# required.  Aggregates follow Sanjay's metrics list:
#
#   * step time: p50, p95, p99, mean, RMS, std (train + val separately)
#   * throughput: samples/sec = batch_size * world_size / step_time
#   * peak GPU memory (max mem_gb in any train step) + steady-state mean
#   * total wall-clock from first dataset record to last train/val record
#
# Epoch 0 is excluded from all headline aggregates (step time, throughput,
# memory, wallclock train/val sums) by default because compile / allocator
# warm-up and first-epoch dataloader effects dominate.  Per-epoch rows still
# list epoch 0 for inspection.  Override with --exclude-epochs none.
#
# We do NOT drop a fixed number of steps at the head of each remaining
# epoch; epoch 0 is the sole warm-up exclusion policy.
# ---------------------------------------------------------------------------

from __future__ import annotations

import sys
from pathlib import Path

_BENCHMARKS_ROOT = Path(__file__).resolve().parents[1]
if str(_BENCHMARKS_ROOT) not in sys.path:
    sys.path.insert(0, str(_BENCHMARKS_ROOT))
import argparse
import json
import math
import os
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

DEFAULT_EXCLUDE_EPOCHS: tuple[int, ...] = (0,)

# Local import; same directory as this script.
from ingest.parse_metrics_timing import EpochTiming, LogData, parse_log


@dataclass
class StepTimeStats:
    n: int
    mean: float | None
    std: float | None
    rms: float | None
    p50: float | None
    p95: float | None
    p99: float | None
    min: float | None
    max: float | None
    total: float | None

    @classmethod
    def from_samples(cls, samples: list[float]) -> "StepTimeStats":
        n = len(samples)
        if n == 0:
            return cls(0, *([None] * 9))
        sorted_s = sorted(samples)
        mean = statistics.fmean(samples)
        std = statistics.stdev(samples) if n > 1 else 0.0
        rms = math.sqrt(sum(x * x for x in samples) / n)
        return cls(
            n=n,
            mean=mean,
            std=std,
            rms=rms,
            p50=_percentile(sorted_s, 0.50),
            p95=_percentile(sorted_s, 0.95),
            p99=_percentile(sorted_s, 0.99),
            min=sorted_s[0],
            max=sorted_s[-1],
            total=sum(samples),
        )


def _percentile(sorted_samples: list[float], q: float) -> float:
    """Linear-interpolation percentile.  Inputs must be sorted."""
    if not sorted_samples:
        raise ValueError("empty samples")
    if len(sorted_samples) == 1:
        return sorted_samples[0]
    pos = q * (len(sorted_samples) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_samples[lo]
    frac = pos - lo
    return sorted_samples[lo] * (1 - frac) + sorted_samples[hi] * frac


@dataclass
class MemoryStats:
    peak_gb: float | None
    steady_mean_gb: float | None
    steady_n: int

    @classmethod
    def from_records(cls, mem_values: list[float]) -> "MemoryStats":
        if not mem_values:
            return cls(None, None, 0)
        peak = max(mem_values)
        # "Steady state" = drop the first and last step in the aggregate
        # pool (sometimes carry allocator transients at epoch boundaries).
        steady = mem_values[1:-1] if len(mem_values) >= 4 else mem_values
        steady_mean = statistics.fmean(steady) if steady else peak
        return cls(peak_gb=peak, steady_mean_gb=steady_mean, steady_n=len(steady))


@dataclass
class RunSummary:
    run_id: str
    model: str
    dataset: str
    num_gpus: int
    storage: str
    sampling_resolution: int
    num_epochs: int
    excluded_epochs: list[int]
    num_epochs_aggregated: int

    # --- aggregates -------------------------------------------------------
    train: StepTimeStats
    val: StepTimeStats
    val_estimated: bool   # True when val timing came from val_duration/val_n

    throughput_samples_per_sec_p50: float | None
    throughput_samples_per_sec_mean: float | None

    memory: MemoryStats

    wallclock_total_s: float | None
    wallclock_train_s: float | None
    wallclock_val_s: float | None

    # --- provenance -------------------------------------------------------
    metrics_path: str
    n_train_steps_total: int
    n_train_steps_aggregated: int
    train_samples: int | None
    val_samples: int | None

    # --- per-epoch (small enough to fit alongside aggregates) -------------
    epochs: list[dict] = field(default_factory=list)


def parse_exclude_epochs(raw: str | None) -> tuple[int, ...]:
    """Parse ``--exclude-epochs`` (default ``0``; ``none`` excludes nothing)."""
    if raw is None:
        return DEFAULT_EXCLUDE_EPOCHS
    text = raw.strip().lower()
    if text in {"", "none"}:
        return ()
    return tuple(int(part.strip()) for part in raw.split(",") if part.strip())


def num_epochs_for_aggregate(summary: dict) -> int:
    """Epoch count for time-per-epoch = wallclock_train_s / N."""
    if "num_epochs_aggregated" in summary:
        return int(summary["num_epochs_aggregated"])
    excluded = summary.get("excluded_epochs")
    if excluded is not None:
        return int(summary.get("num_epochs", 5)) - len(excluded)
    return int(summary.get("num_epochs", 5))


def _aggregated_epochs(
    epochs: list[EpochTiming], exclude_epochs: frozenset[int]
) -> list[EpochTiming]:
    return [ep for ep in epochs if ep.epoch not in exclude_epochs]


def _estimated_val_times(
    data: LogData, exclude_epochs: frozenset[int]
) -> list[float]:
    """When the recipe didn't emit val_step records, fall back to one
    estimated per-step time per epoch from val_duration / val_n."""
    out: list[float] = []
    for ep in data.epochs:
        if ep.epoch in exclude_epochs:
            continue
        dur = ep.val_duration_s
        if dur is None:
            continue
        train_n = len(ep.train_step_times)
        if train_n == 0 or data.train_samples is None or data.val_samples is None:
            continue
        val_n = max(1, round(train_n * data.val_samples / max(data.train_samples, 1)))
        out.append(dur / val_n)
    return out


def _read_mem_gb(metrics_path: Path, exclude_epochs: frozenset[int]) -> list[float]:
    """Walk metrics.jsonl for `mem_gb` per train step; skip excluded epochs."""
    out: list[float] = []
    current_epoch = 0
    with metrics_path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            phase = rec.get("phase")
            if phase == "dataset":
                current_epoch = 0
                continue
            if phase == "step":
                if current_epoch in exclude_epochs:
                    continue
                mem = rec.get("mem_gb")
                if mem is not None:
                    out.append(float(mem))
            elif phase == "train":
                epoch_idx = rec.get("epoch")
                if epoch_idx is not None:
                    current_epoch = int(epoch_idx) + 1
                else:
                    current_epoch += 1
    return out


def _wallclock(
    data: LogData, exclude_epochs: frozenset[int]
) -> tuple[float | None, float | None, float | None]:
    """Return (total_s, train_s, val_s) for aggregated epochs only."""
    included = _aggregated_epochs(data.epochs, exclude_epochs)
    if not included:
        return None, None, None
    first_train_ts = next((e.train_ts for e in included if e.train_ts is not None), None)
    last_train_ts = next(
        (e.train_ts for e in reversed(included) if e.train_ts is not None), None
    )
    last_val_ts = next((e.val_ts for e in reversed(included) if e.val_ts is not None), None)
    end = last_val_ts or last_train_ts
    if first_train_ts is None or end is None:
        total = None
    else:
        total = (end - first_train_ts).total_seconds()

    train_total = sum(sum(e.train_step_times) for e in included) or None
    val_total = sum(
        (e.val_duration_s or 0.0) for e in included if e.val_duration_s is not None
    ) or None
    return total, train_total, val_total


def _epoch_rows(data: LogData, exclude_epochs: frozenset[int]) -> list[dict]:
    rows: list[dict] = []
    for ep in data.epochs:
        train_stats = StepTimeStats.from_samples(ep.train_step_times)
        val_stats = StepTimeStats.from_samples(ep.val_step_times)
        rows.append(
            {
                "epoch": ep.epoch,
                "excluded_from_aggregate": ep.epoch in exclude_epochs,
                "train": asdict(train_stats),
                "val": asdict(val_stats),
                "val_duration_s": ep.val_duration_s,
            }
        )
    return rows


def summarize(
    metrics_path: Path,
    *,
    run_id: str,
    model: str,
    dataset: str,
    num_gpus: int,
    storage: str,
    sampling_resolution: int,
    num_epochs: int,
    batch_size: int = 1,
    exclude_epochs: tuple[int, ...] = DEFAULT_EXCLUDE_EPOCHS,
) -> RunSummary:
    data = parse_log(metrics_path)
    exclude = frozenset(exclude_epochs)
    aggregated = _aggregated_epochs(data.epochs, exclude)
    num_epochs_aggregated = len(aggregated)

    # Aggregate train step times from epochs not in exclude_epochs.
    train_samples_all: list[float] = []
    for ep in aggregated:
        train_samples_all.extend(ep.train_step_times)
    n_train_total = sum(len(ep.train_step_times) for ep in data.epochs)

    train_stats = StepTimeStats.from_samples(train_samples_all)

    # Val: prefer measured per-step records; fall back to estimate.
    val_estimated = False
    val_samples_all: list[float] = []
    if data.has_val_steps:
        for ep in aggregated:
            val_samples_all.extend(ep.val_step_times)
    else:
        val_samples_all = _estimated_val_times(data, exclude)
        val_estimated = bool(val_samples_all)
    val_stats = StepTimeStats.from_samples(val_samples_all)

    # Throughput in samples/sec.  batch_size = 1 is hard-coded in the
    # recipe today; world_size = num_gpus.
    if train_stats.p50 and train_stats.p50 > 0:
        thr_p50 = batch_size * num_gpus / train_stats.p50
    else:
        thr_p50 = None
    if train_stats.mean and train_stats.mean > 0:
        thr_mean = batch_size * num_gpus / train_stats.mean
    else:
        thr_mean = None

    mem_values = _read_mem_gb(metrics_path, exclude)
    memory = MemoryStats.from_records(mem_values)

    wc_total, wc_train, wc_val = _wallclock(data, exclude)

    return RunSummary(
        run_id=run_id,
        model=model,
        dataset=dataset,
        num_gpus=num_gpus,
        storage=storage,
        sampling_resolution=sampling_resolution,
        num_epochs=num_epochs,
        excluded_epochs=list(exclude_epochs),
        num_epochs_aggregated=num_epochs_aggregated,
        train=train_stats,
        val=val_stats,
        val_estimated=val_estimated,
        throughput_samples_per_sec_p50=thr_p50,
        throughput_samples_per_sec_mean=thr_mean,
        memory=memory,
        wallclock_total_s=wc_total,
        wallclock_train_s=wc_train,
        wallclock_val_s=wc_val,
        metrics_path=str(metrics_path),
        n_train_steps_total=n_train_total,
        n_train_steps_aggregated=train_stats.n,
        train_samples=data.train_samples,
        val_samples=data.val_samples,
        epochs=_epoch_rows(data, exclude),
    )


def _fmt_seconds(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}"


def _fmt_count(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f}"


def render_markdown(s: RunSummary) -> str:
    lines: list[str] = []
    lines.append(f"# Benchmark summary: `{s.run_id}`")
    lines.append("")
    lines.append("## Configuration")
    lines.append("")
    lines.append("| Field | Value |")
    lines.append("|---|---|")
    lines.append(f"| model | {s.model} |")
    lines.append(f"| dataset | {s.dataset} |")
    lines.append(f"| num_gpus | {s.num_gpus} |")
    lines.append(f"| storage | {s.storage} |")
    lines.append(f"| sampling_resolution | {s.sampling_resolution} |")
    lines.append(f"| num_epochs | {s.num_epochs} |")
    lines.append(f"| excluded_epochs | {s.excluded_epochs or 'none'} |")
    lines.append(f"| num_epochs_aggregated | {s.num_epochs_aggregated} |")
    lines.append(f"| train_samples | {s.train_samples} |")
    lines.append(f"| val_samples | {s.val_samples} |")
    lines.append("")
    lines.append("## Step time (seconds)")
    lines.append("")
    if s.excluded_epochs:
        lines.append(
            f"Aggregates pool epochs **excluding** {s.excluded_epochs} "
            f"({s.num_epochs_aggregated} of {s.num_epochs} epochs)."
        )
        lines.append("")
    lines.append("| Phase | n | mean | std | rms | p50 | p95 | p99 | min | max | total |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for phase, st in (("train", s.train), ("val" + ("*" if s.val_estimated else ""), s.val)):
        lines.append(
            "| "
            + " | ".join(
                [
                    phase,
                    str(st.n),
                    _fmt_seconds(st.mean),
                    _fmt_seconds(st.std),
                    _fmt_seconds(st.rms),
                    _fmt_seconds(st.p50),
                    _fmt_seconds(st.p95),
                    _fmt_seconds(st.p99),
                    _fmt_seconds(st.min),
                    _fmt_seconds(st.max),
                    _fmt_seconds(st.total),
                ]
            )
            + " |"
        )
    if s.val_estimated:
        lines.append("")
        lines.append("(*) val per-step times estimated from val_duration / val_n.")
    lines.append("")
    lines.append("## Throughput (samples/sec)")
    lines.append("")
    lines.append(f"- p50: {_fmt_count(s.throughput_samples_per_sec_p50)}")
    lines.append(f"- mean: {_fmt_count(s.throughput_samples_per_sec_mean)}")
    lines.append("")
    lines.append("## GPU memory (GB, reserved)")
    lines.append("")
    lines.append(f"- peak: {_fmt_count(s.memory.peak_gb)}")
    lines.append(f"- steady-state mean: {_fmt_count(s.memory.steady_mean_gb)} (n={s.memory.steady_n})")
    lines.append("")
    lines.append("## Wall-clock (seconds)")
    lines.append("")
    lines.append(f"- total: {_fmt_seconds(s.wallclock_total_s)}")
    lines.append(f"- train (sum of step times): {_fmt_seconds(s.wallclock_train_s)}")
    lines.append(f"- val (sum of train_ts -> val_ts gaps): {_fmt_seconds(s.wallclock_val_s)}")
    lines.append("")
    lines.append("## Per-epoch")
    lines.append("")
    lines.append(
        "| epoch | in_aggregate | train_n | train_mean | train_p95 | val_n | val_mean | val_dur_s |"
    )
    lines.append("|---:|:---:|---:|---:|---:|---:|---:|---:|")
    for ep in s.epochs:
        in_agg = "no" if ep.get("excluded_from_aggregate") else "yes"
        lines.append(
            "| "
            + " | ".join(
                [
                    str(ep["epoch"]),
                    in_agg,
                    str(ep["train"]["n"]),
                    _fmt_seconds(ep["train"]["mean"]),
                    _fmt_seconds(ep["train"]["p95"]),
                    str(ep["val"]["n"]),
                    _fmt_seconds(ep["val"]["mean"]),
                    _fmt_seconds(ep["val_duration_s"]),
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append(f"_metrics: `{s.metrics_path}`_")
    return "\n".join(lines)


def main(argv: Iterable[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--metrics", required=True, type=Path, help="metrics.jsonl from train.py")
    p.add_argument("--run-id", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--num-gpus", required=True, type=int)
    p.add_argument("--storage", required=True, choices=["lustre", "nvme"])
    p.add_argument("--sampling", required=True, type=int, dest="sampling_resolution")
    p.add_argument("--num-epochs", required=True, type=int)
    p.add_argument(
        "--exclude-epochs",
        default="0",
        help="Comma-separated epoch indices excluded from aggregates (default: 0; use 'none' for all)",
    )
    p.add_argument("--batch-size", default=1, type=int)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directory for benchmark_summary.{json,md} (default: alongside metrics.jsonl)",
    )
    args = p.parse_args(list(argv) if argv is not None else None)

    if not args.metrics.exists():
        print(f"[summarize_run] missing metrics file: {args.metrics}")
        return 2

    summary = summarize(
        args.metrics,
        run_id=args.run_id,
        model=args.model,
        dataset=args.dataset,
        num_gpus=args.num_gpus,
        storage=args.storage,
        sampling_resolution=args.sampling_resolution,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        exclude_epochs=parse_exclude_epochs(args.exclude_epochs),
    )

    out_dir = args.out_dir or args.metrics.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "benchmark_summary.json"
    md_path = out_dir / "benchmark_summary.md"

    with json_path.open("w") as fh:
        json.dump(asdict(summary), fh, indent=2, default=str)
    with md_path.open("w") as fh:
        fh.write(render_markdown(summary))

    print(f"[summarize_run] wrote {json_path}")
    print(f"[summarize_run] wrote {md_path}")
    print(
        f"[summarize_run] train p50={_fmt_seconds(summary.train.p50)}s  "
        f"p95={_fmt_seconds(summary.train.p95)}s  "
        f"throughput p50={_fmt_count(summary.throughput_samples_per_sec_p50)} samples/s  "
        f"peak_mem={_fmt_count(summary.memory.peak_gb)} GB"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
