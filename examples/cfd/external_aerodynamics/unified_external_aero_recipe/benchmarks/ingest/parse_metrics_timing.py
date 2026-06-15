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

"""Parse a training metrics.jsonl log and report per-epoch timing.

Vendored from Corey's CAE benchmarking utilities.

The log is expected to contain one JSON record per line with a ``phase`` field.
Relevant phases:

- ``dataset``: emitted at run start and after every restart. Carries
  ``train_samples`` and ``val_samples`` (global counts). The first ``step``
  record following one of these is a torch.compile warm-up and is dropped.
  ``global_step`` resets after this event.
- ``step``: train-only per-step record with ``step_time_s`` and ``ts``.
  No explicit epoch field; epoch membership is inferred from the surrounding
  ``train`` markers.
- ``val_step``: val-only per-step record (added by the patched train.py)
  carrying an explicit ``epoch``, ``val_step`` index, and ``step_time_s``.
- ``train``: end-of-epoch train summary. Carries ``epoch`` and ``ts``. Marks
  the boundary between epochs.
- ``val``: end-of-epoch validation summary. Carries ``epoch`` and ``ts``.
- ``config``: ignored.

For each completed epoch the script reports:

- Train step-time distribution from the per-step ``step_time_s`` field
  (count, mean +/- std, median, min, max, total).
- Validation step-time distribution. If the log contains ``val_step``
  records, full per-step statistics (including a within-epoch std) are
  computed directly. Otherwise the per-step value is estimated as
  ``val_duration / val_n`` where ``val_n = round(train_n * val_samples
  / train_samples)``, and the uncertainty is the across-epoch std of that
  per-step value (i.e. scaled to per-step units).

Steps from failed restart attempts (i.e. those followed by another ``dataset``
event before a ``train`` event is reached) are discarded. Any step buffer left
over at end-of-file (an in-progress epoch with no ``train`` marker yet) is also
discarded.

Usage::

    python parse_metrics_timing.py path/to/metrics.jsonl [more.jsonl ...]
"""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path


@dataclass
class EpochTiming:
    epoch: int
    train_step_times: list[float] = field(default_factory=list)
    val_step_times: list[float] = field(default_factory=list)
    train_ts: datetime | None = None
    val_ts: datetime | None = None

    @property
    def val_duration_s(self) -> float | None:
        if self.train_ts is None or self.val_ts is None:
            return None
        return (self.val_ts - self.train_ts).total_seconds()


@dataclass
class LogData:
    epochs: list[EpochTiming]
    train_samples: int | None
    val_samples: int | None
    has_val_steps: bool


def _parse_ts(value: str | None) -> datetime | None:
    if value is None:
        return None
    return datetime.fromisoformat(value)


def parse_log(path: Path) -> LogData:
    """Parse a metrics.jsonl log into ``LogData``."""
    epochs: list[EpochTiming] = []
    by_epoch: dict[int, EpochTiming] = {}

    train_samples: int | None = None
    val_samples: int | None = None
    has_val_steps = False

    train_buffer: list[float] = []
    warmup_pending = False

    def _ensure(epoch_idx: int) -> EpochTiming:
        if epoch_idx not in by_epoch:
            entry = EpochTiming(epoch=epoch_idx)
            by_epoch[epoch_idx] = entry
            epochs.append(entry)
        return by_epoch[epoch_idx]

    with path.open() as f:
        for lineno, raw in enumerate(f, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                rec = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path}:{lineno}: malformed JSON ({exc.msg})"
                ) from exc

            phase = rec.get("phase")

            if phase == "dataset":
                # Restart: any pending train steps are from a failed attempt.
                train_buffer.clear()
                warmup_pending = True
                if train_samples is None and "train_samples" in rec:
                    train_samples = int(rec["train_samples"])
                if val_samples is None and "val_samples" in rec:
                    val_samples = int(rec["val_samples"])
            elif phase == "step":
                if warmup_pending:
                    # First step after a restart is torch.compile warm-up.
                    warmup_pending = False
                    continue
                t = rec.get("step_time_s")
                if t is not None:
                    train_buffer.append(float(t))
            elif phase == "val_step":
                # Per-step val record (patched train.py only). Use the
                # explicit ``epoch`` field rather than inferring from
                # surrounding markers, since val_step records sit between
                # the train and val phase markers of the same epoch.
                epoch_idx = rec.get("epoch")
                if epoch_idx is None:
                    raise ValueError(
                        f"{path}:{lineno}: 'val_step' record missing 'epoch' "
                        "field"
                    )
                t = rec.get("step_time_s")
                if t is not None:
                    _ensure(int(epoch_idx)).val_step_times.append(float(t))
                    has_val_steps = True
            elif phase == "train":
                epoch_idx = rec.get("epoch")
                if epoch_idx is None:
                    raise ValueError(
                        f"{path}:{lineno}: 'train' record missing 'epoch' field"
                    )
                epoch_idx = int(epoch_idx)
                entry = _ensure(epoch_idx)
                entry.train_step_times = list(train_buffer)
                entry.train_ts = _parse_ts(rec.get("ts"))
                train_buffer.clear()
                warmup_pending = False
            elif phase == "val":
                epoch_idx = rec.get("epoch")
                if epoch_idx is None:
                    raise ValueError(
                        f"{path}:{lineno}: 'val' record missing 'epoch' field"
                    )
                _ensure(int(epoch_idx)).val_ts = _parse_ts(rec.get("ts"))
            # 'config' and unknown phases are ignored.

    return LogData(
        epochs=epochs,
        train_samples=train_samples,
        val_samples=val_samples,
        has_val_steps=has_val_steps,
    )


def _estimate_val_n(
    train_n: int, train_samples: int | None, val_samples: int | None
) -> int | None:
    """Estimate val_steps_per_epoch from observed train_n and dataset sizes.

    Assumes val uses the same batch size and world size as train, so the
    per-rank step count scales with the sample count.
    """
    if train_n <= 0:
        return None
    if train_samples is None or val_samples is None or train_samples <= 0:
        return None
    return max(1, round(train_n * val_samples / train_samples))


def _fmt_train_table(epochs: list[EpochTiming]) -> str:
    header = (
        f"{'epoch':>6} {'n_steps':>8} {'mean(s)':>10} {'std(s)':>10} "
        f"{'median(s)':>11} {'min(s)':>9} {'max(s)':>9} {'total(s)':>10}"
    )
    lines = [header, "-" * len(header)]
    for ep in epochs:
        n = len(ep.train_step_times)
        if n == 0:
            lines.append(
                f"{ep.epoch:>6d} {0:>8d} {'-':>10} {'-':>10} "
                f"{'-':>11} {'-':>9} {'-':>9} {'-':>10}"
            )
            continue
        mean = statistics.fmean(ep.train_step_times)
        std = statistics.stdev(ep.train_step_times) if n > 1 else 0.0
        median = statistics.median(ep.train_step_times)
        lines.append(
            f"{ep.epoch:>6d} {n:>8d} {mean:>10.4f} {std:>10.4f} "
            f"{median:>11.4f} {min(ep.train_step_times):>9.4f} "
            f"{max(ep.train_step_times):>9.4f} "
            f"{sum(ep.train_step_times):>10.3f}"
        )
    return "\n".join(lines)


def _fmt_val_table_measured(epochs: list[EpochTiming]) -> str:
    """Per-epoch val table when we have real val_step records."""
    header = (
        f"{'epoch':>6} {'n_steps':>8} {'mean(s)':>10} {'std(s)':>10} "
        f"{'median(s)':>11} {'min(s)':>9} {'max(s)':>9} {'total(s)':>10}"
    )
    lines = [header, "-" * len(header)]
    for ep in epochs:
        n = len(ep.val_step_times)
        if n == 0:
            dur = ep.val_duration_s
            dur_str = f"{dur:>10.3f}" if dur is not None else f"{'-':>10}"
            lines.append(
                f"{ep.epoch:>6d} {0:>8d} {'-':>10} {'-':>10} "
                f"{'-':>11} {'-':>9} {'-':>9} {dur_str}"
            )
            continue
        mean = statistics.fmean(ep.val_step_times)
        std = statistics.stdev(ep.val_step_times) if n > 1 else 0.0
        median = statistics.median(ep.val_step_times)
        lines.append(
            f"{ep.epoch:>6d} {n:>8d} {mean:>10.4f} {std:>10.4f} "
            f"{median:>11.4f} {min(ep.val_step_times):>9.4f} "
            f"{max(ep.val_step_times):>9.4f} "
            f"{sum(ep.val_step_times):>10.3f}"
        )
    return "\n".join(lines)


def _fmt_val_table_estimated(
    epochs: list[EpochTiming],
    train_samples: int | None,
    val_samples: int | None,
) -> str:
    """Per-epoch val table when we have to estimate per-step time."""
    header = (
        f"{'epoch':>6} {'n_steps*':>9} {'mean(s)*':>10} {'total(s)':>10}"
    )
    lines = [header, "-" * len(header)]
    for ep in epochs:
        dur = ep.val_duration_s
        train_n = len(ep.train_step_times)
        val_n = _estimate_val_n(train_n, train_samples, val_samples)
        if dur is None:
            lines.append(
                f"{ep.epoch:>6d} {'-':>9} {'-':>10} {'-':>10}"
            )
            continue
        if val_n is None:
            lines.append(
                f"{ep.epoch:>6d} {'?':>9} {'?':>10} {dur:>10.3f}"
            )
            continue
        per_step = dur / val_n
        lines.append(
            f"{ep.epoch:>6d} {val_n:>9d} {per_step:>10.4f} {dur:>10.3f}"
        )
    return "\n".join(lines)


def _overall_summary(data: LogData) -> str:
    epochs = data.epochs
    all_train: list[float] = [t for ep in epochs for t in ep.train_step_times]
    lines: list[str] = []

    if all_train:
        n = len(all_train)
        mean = statistics.fmean(all_train)
        std = statistics.stdev(all_train) if n > 1 else 0.0
        total = sum(all_train)
        wall = str(timedelta(seconds=int(total)))
        lines.append(
            f"Train: {n} steps across {len(epochs)} epochs, "
            f"{mean:.4f} +/- {std:.4f} s/step, total {total:.2f} s ({wall})"
        )
    else:
        lines.append("Train: no completed step records found.")

    if data.has_val_steps:
        all_val: list[float] = [t for ep in epochs for t in ep.val_step_times]
        n_epochs_with_val = sum(1 for ep in epochs if ep.val_step_times)
        if all_val:
            n = len(all_val)
            mean = statistics.fmean(all_val)
            std = statistics.stdev(all_val) if n > 1 else 0.0
            total = sum(all_val)
            wall = str(timedelta(seconds=int(total)))
            lines.append(
                f"Val:   {n} steps across {n_epochs_with_val} epochs, "
                f"{mean:.4f} +/- {std:.4f} s/step, "
                f"total {total:.2f} s ({wall})"
            )
        else:
            lines.append("Val:   no val_step records present.")
        return "\n".join(lines)

    # Estimated path: per-epoch (val_duration / val_n) gives one
    # per-step number per epoch. Aggregate across epochs to a mean +/- std
    # in the same per-step units.
    val_per_step: list[float] = []
    val_total_dur = 0.0
    n_epochs_with_val = 0
    for ep in epochs:
        dur = ep.val_duration_s
        if dur is None:
            continue
        val_n = _estimate_val_n(
            len(ep.train_step_times), data.train_samples, data.val_samples
        )
        if val_n is None:
            continue
        val_per_step.append(dur / val_n)
        val_total_dur += dur
        n_epochs_with_val += 1

    if not val_per_step:
        lines.append("Val:   no usable val timing data (no train/val timestamp pairs).")
        return "\n".join(lines)

    mean = statistics.fmean(val_per_step)
    std = statistics.stdev(val_per_step) if len(val_per_step) > 1 else 0.0
    wall = str(timedelta(seconds=int(val_total_dur)))
    lines.append(
        f"Val*:  {n_epochs_with_val} epochs, "
        f"{mean:.4f} +/- {std:.4f} s/step (estimated from val_duration/val_n), "
        f"total {val_total_dur:.2f} s ({wall})"
    )
    return "\n".join(lines)


def report(path: Path) -> str:
    """Render the per-epoch + overall report for a single metrics.jsonl."""
    data = parse_log(path)
    blocks: list[str] = [f"== {path} =="]
    blocks.append("Train per epoch:")
    blocks.append(_fmt_train_table(data.epochs))
    blocks.append("")
    blocks.append("Val per epoch:")
    if data.has_val_steps:
        blocks.append(_fmt_val_table_measured(data.epochs))
    else:
        blocks.append(
            _fmt_val_table_estimated(
                data.epochs, data.train_samples, data.val_samples
            )
        )
        blocks.append("(* val timing estimated from val_duration/val_n)")
    blocks.append("")
    blocks.append(_overall_summary(data))
    return "\n".join(blocks)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("paths", nargs="+", type=Path, help="One or more metrics.jsonl paths")
    args = p.parse_args()
    for path in args.paths:
        if not path.exists():
            print(f"[parse_metrics_timing] missing: {path}")
            continue
        print(report(path))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
