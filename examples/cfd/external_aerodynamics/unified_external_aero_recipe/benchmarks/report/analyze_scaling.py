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
# analyze_scaling.py -- cross-run scaling analysis
# ---------------------------------------------------------------------------
#
# Walks the results/ tree, ingests every benchmark_summary.json produced
# by summarize_run.py, and emits:
#
#   analysis/summary.csv                       # one row per run
#   analysis/<key>/scaling.png                 # step-time, throughput, eff,
#                                              # mem vs num_gpus per axis key
#   analysis/<key>/storage_compare.png         # NVMe vs Lustre overlay
#   analysis/anomalies.md                      # flags for efficiency < 70%
#
# The "key" axis is (model, dataset, sampling_resolution).  Storage
# (lustre vs nvme) shows up as overlaid series on the same chart so we
# can see the data-tier impact at a glance.
#
# Scaling efficiency is computed against the smallest GPU count present
# for each (model, dataset, sampling, storage):
#
#     efficiency(N) = (throughput(N) / throughput(N_min)) * (N_min / N)
#
# i.e. perfect strong scaling = 1.0 (= 100%).  Sanjay's threshold for
# attention is < 70%.
# ---------------------------------------------------------------------------

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

# matplotlib is used only when --no-plots is not set; defer the import so
# the CSV export still works on machines without a display backend.

EFFICIENCY_THRESHOLD = 0.70


@dataclass(frozen=True)
class Key:
    model: str
    dataset: str
    sampling: int

    def slug(self) -> str:
        return f"{self.model}__{self.dataset}__sub{self.sampling}"


@dataclass
class Row:
    run_id: str
    model: str
    dataset: str
    num_gpus: int
    storage: str
    sampling: int
    train_p50: float | None
    train_p95: float | None
    train_mean: float | None
    throughput_p50: float | None
    throughput_mean: float | None
    peak_mem_gb: float | None
    steady_mem_gb: float | None
    wallclock_total_s: float | None
    val_p50: float | None
    n_train_steps: int
    metrics_path: str

    @classmethod
    def from_summary(cls, data: dict) -> "Row":
        return cls(
            run_id=data["run_id"],
            model=data["model"],
            dataset=data["dataset"],
            num_gpus=int(data["num_gpus"]),
            storage=data["storage"],
            sampling=int(data["sampling_resolution"]),
            train_p50=data["train"].get("p50"),
            train_p95=data["train"].get("p95"),
            train_mean=data["train"].get("mean"),
            throughput_p50=data.get("throughput_samples_per_sec_p50"),
            throughput_mean=data.get("throughput_samples_per_sec_mean"),
            peak_mem_gb=(data.get("memory") or {}).get("peak_gb"),
            steady_mem_gb=(data.get("memory") or {}).get("steady_mean_gb"),
            wallclock_total_s=data.get("wallclock_total_s"),
            val_p50=data["val"].get("p50"),
            n_train_steps=int(
                data.get(
                    "n_train_steps_aggregated",
                    data.get("n_train_steps_after_warmup", 0),
                )
            ),
            metrics_path=data.get("metrics_path", ""),
        )

    def key(self) -> Key:
        return Key(self.model, self.dataset, self.sampling)


def _walk_summaries(results_root: Path) -> list[Row]:
    rows: list[Row] = []
    for path in results_root.rglob("benchmark_summary.json"):
        try:
            with path.open() as fh:
                data = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"[analyze] skipping {path}: {exc}")
            continue
        rows.append(Row.from_summary(data))
    return rows


def _write_csv(rows: list[Row], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_id", "model", "dataset", "num_gpus", "storage", "sampling",
        "train_p50", "train_p95", "train_mean",
        "throughput_p50", "throughput_mean",
        "peak_mem_gb", "steady_mem_gb", "wallclock_total_s",
        "val_p50", "n_train_steps", "metrics_path",
    ]
    with out_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: getattr(r, k) for k in fieldnames})


def _compute_efficiency(rows: list[Row]) -> dict[tuple[Key, str], list[tuple[int, float]]]:
    """Return (key, storage) -> sorted [(num_gpus, efficiency)]."""
    by_key_storage: dict[tuple[Key, str], list[Row]] = defaultdict(list)
    for r in rows:
        if r.throughput_p50 is None or r.throughput_p50 <= 0:
            continue
        by_key_storage[(r.key(), r.storage)].append(r)
    out: dict[tuple[Key, str], list[tuple[int, float]]] = {}
    for k, group in by_key_storage.items():
        group.sort(key=lambda x: x.num_gpus)
        if not group:
            continue
        baseline = group[0]
        b_thr = baseline.throughput_p50 or 0.0
        b_n = baseline.num_gpus
        if b_thr == 0 or b_n == 0:
            continue
        out[k] = [(r.num_gpus, (r.throughput_p50 / b_thr) * (b_n / r.num_gpus)) for r in group]
    return out


def _plot_per_key(
    key: Key,
    rows: list[Row],
    efficiency: dict[tuple[Key, str], list[tuple[int, float]]],
    out_dir: Path,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_storage: dict[str, list[Row]] = defaultdict(list)
    for r in rows:
        by_storage[r.storage].append(r)
    for storage in by_storage:
        by_storage[storage].sort(key=lambda r: r.num_gpus)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle(f"{key.model} | {key.dataset} | sampling={key.sampling}")

    ax_step, ax_thr = axes[0]
    ax_eff, ax_mem = axes[1]

    for storage, group in by_storage.items():
        gpus = [r.num_gpus for r in group]
        ax_step.plot(gpus, [r.train_p50 for r in group], marker="o", label=f"{storage} p50")
        ax_step.plot(gpus, [r.train_p95 for r in group], marker="x", linestyle="--", label=f"{storage} p95")
        ax_thr.plot(gpus, [r.throughput_p50 for r in group], marker="o", label=storage)
        ax_mem.plot(gpus, [r.peak_mem_gb for r in group], marker="o", label=f"{storage} peak")

        eff_pairs = efficiency.get((key, storage), [])
        if eff_pairs:
            ex = [n for n, _ in eff_pairs]
            ey = [e * 100 for _, e in eff_pairs]
            ax_eff.plot(ex, ey, marker="o", label=storage)

    for ax, title, ylabel, log_x in (
        (ax_step, "Step Time Vs GPUs", "Step Time (S)", True),
        (ax_thr,  "Throughput Vs GPUs", "Samples/Sec", True),
        (ax_eff,  "Strong Scaling Efficiency", "% Of Perfect", True),
        (ax_mem,  "Peak GPU Memory Vs GPUs", "GB", True),
    ):
        ax.set_title(title)
        ax.set_xlabel("Num GPUs")
        ax.set_ylabel(ylabel)
        if log_x:
            ax.set_xscale("log", base=2)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    ax_eff.axhline(EFFICIENCY_THRESHOLD * 100, linestyle=":", linewidth=1.0, label="70% Threshold")
    ax_eff.legend(fontsize=8)

    out_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    out_path = out_dir / "scaling.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_storage_compare(
    key: Key,
    rows: list[Row],
    out_dir: Path,
) -> None:
    """Dedicated single-pane Lustre vs NVMe throughput overlay (good for slides)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_storage: dict[str, list[Row]] = defaultdict(list)
    for r in rows:
        by_storage[r.storage].append(r)
    if len(by_storage) < 2:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    for storage, group in by_storage.items():
        group.sort(key=lambda r: r.num_gpus)
        ax.plot(
            [r.num_gpus for r in group],
            [r.throughput_p50 for r in group],
            marker="o",
            label=storage,
        )
    ax.set_title(f"NVMe Vs Lustre Throughput\n{key.model} | {key.dataset} | Sampling={key.sampling}")
    ax.set_xlabel("Num GPUs")
    ax.set_ylabel("Samples/Sec (P50)")
    ax.set_xscale("log", base=2)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out_path = out_dir / "storage_compare.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _write_anomalies(
    efficiency: dict[tuple[Key, str], list[tuple[int, float]]],
    out_path: Path,
) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = ["# Scaling anomalies (efficiency < {:.0f}%)".format(EFFICIENCY_THRESHOLD * 100), ""]
    n_flags = 0
    for (key, storage), pairs in sorted(efficiency.items(), key=lambda kv: (kv[0][0].model, kv[0][0].dataset, kv[0][0].sampling, kv[0][1])):
        flagged = [(n, e) for n, e in pairs if e < EFFICIENCY_THRESHOLD]
        if not flagged:
            continue
        lines.append(f"## {key.model} | {key.dataset} | sampling={key.sampling} | {storage}")
        lines.append("")
        lines.append("| num_gpus | efficiency |")
        lines.append("|---:|---:|")
        for n, e in flagged:
            lines.append(f"| {n} | {e * 100:.1f}% |")
        lines.append("")
        n_flags += len(flagged)
    if n_flags == 0:
        lines.append("No anomalies detected -- all (model, dataset, sampling, storage) combinations scaled above threshold.")
    with out_path.open("w") as fh:
        fh.write("\n".join(lines))
    return n_flags


def main(argv: Iterable[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", default="results", type=Path, help="Root of results/ tree (default: %(default)s)")
    p.add_argument("--out-dir", default="analysis", type=Path, help="Output directory (default: %(default)s)")
    p.add_argument("--no-plots", action="store_true", help="Skip matplotlib output, write CSV + anomalies only")
    args = p.parse_args(list(argv) if argv is not None else None)

    if not args.results.exists():
        print(f"[analyze] results dir not found: {args.results}")
        return 2

    rows = _walk_summaries(args.results)
    print(f"[analyze] ingested {len(rows)} run summaries from {args.results}")
    if not rows:
        print("[analyze] nothing to do")
        return 0

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "summary.csv"
    _write_csv(rows, csv_path)
    print(f"[analyze] wrote {csv_path}")

    by_key: dict[Key, list[Row]] = defaultdict(list)
    for r in rows:
        by_key[r.key()].append(r)

    efficiency = _compute_efficiency(rows)

    if not args.no_plots:
        for key, group in by_key.items():
            sub = args.out_dir / key.slug()
            _plot_per_key(key, group, efficiency, sub)
            _plot_storage_compare(key, group, sub)
            print(f"[analyze] wrote charts to {sub}")

    n_anom = _write_anomalies(efficiency, args.out_dir / "anomalies.md")
    print(f"[analyze] anomalies flagged: {n_anom} -> {args.out_dir / 'anomalies.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
