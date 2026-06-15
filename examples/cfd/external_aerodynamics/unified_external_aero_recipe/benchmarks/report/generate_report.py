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
# generate_report.py -- compose the final markdown report
# ---------------------------------------------------------------------------
#
# Reads `analysis/summary.csv` (from analyze_scaling.py) and produces:
#
#   report/REPORT.md             # full report with embedded charts
#   report/SANJAY_ONE_PAGER.md   # exec summary suitable for sending up
#   report/figures/              # scaling charts copied here for slide reuse
#
# This script is purely a layout / aggregation step -- it does NOT
# recompute aggregates or re-read raw metrics.jsonl.  Run
# analyze_scaling.py first.
# ---------------------------------------------------------------------------

from __future__ import annotations

import argparse
import csv
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


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


def _f(value: str) -> float | None:
    if value is None or value == "" or value == "None":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _i(value: str) -> int:
    try:
        return int(value)
    except (ValueError, TypeError):
        return 0


def _read_csv(path: Path) -> list[Row]:
    rows: list[Row] = []
    with path.open() as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            rows.append(
                Row(
                    run_id=r["run_id"],
                    model=r["model"],
                    dataset=r["dataset"],
                    num_gpus=_i(r["num_gpus"]),
                    storage=r["storage"],
                    sampling=_i(r["sampling"]),
                    train_p50=_f(r.get("train_p50", "")),
                    train_p95=_f(r.get("train_p95", "")),
                    train_mean=_f(r.get("train_mean", "")),
                    throughput_p50=_f(r.get("throughput_p50", "")),
                    throughput_mean=_f(r.get("throughput_mean", "")),
                    peak_mem_gb=_f(r.get("peak_mem_gb", "")),
                    steady_mem_gb=_f(r.get("steady_mem_gb", "")),
                    wallclock_total_s=_f(r.get("wallclock_total_s", "")),
                    val_p50=_f(r.get("val_p50", "")),
                    n_train_steps=_i(r.get("n_train_steps", "0")),
                )
            )
    return rows


def _fmt(value: float | None, places: int = 4) -> str:
    return f"{value:.{places}f}" if value is not None else "n/a"


def _matrix_table(rows: list[Row]) -> str:
    """One row per (model, dataset, storage) showing #runs, GPU range, sampling range."""
    by_bucket: dict[tuple[str, str, str], list[Row]] = defaultdict(list)
    for r in rows:
        by_bucket[(r.model, r.dataset, r.storage)].append(r)
    lines = ["| model | dataset | storage | #runs | GPU counts | sampling | runs total wall-clock (s) |",
             "|---|---|---|---:|---|---|---:|"]
    for (model, dataset, storage), group in sorted(by_bucket.items()):
        gpus = sorted({r.num_gpus for r in group})
        sampling = sorted({r.sampling for r in group})
        wall = sum(r.wallclock_total_s or 0.0 for r in group)
        lines.append(
            f"| {model} | {dataset} | {storage} | {len(group)} | "
            f"{','.join(map(str, gpus))} | {','.join(map(str, sampling))} | {wall:.1f} |"
        )
    return "\n".join(lines)


def _scaling_section(
    rows: list[Row],
    figures_dir: Path,
    rel_figures_dir: str,
) -> str:
    """For each (model, dataset, sampling) emit a heading + a table + a link to scaling.png."""
    by_key: dict[tuple[str, str, int], list[Row]] = defaultdict(list)
    for r in rows:
        by_key[(r.model, r.dataset, r.sampling)].append(r)

    out: list[str] = []
    for (model, dataset, sampling), group in sorted(by_key.items()):
        slug = f"{model}__{dataset}__sub{sampling}"
        out.append(f"### {model} | {dataset} | sampling={sampling}")
        out.append("")
        out.append("| num_gpus | storage | step p50 (s) | step p95 (s) | thr p50 (samples/s) | peak mem (GB) | wall (s) |")
        out.append("|---:|---|---:|---:|---:|---:|---:|")
        for r in sorted(group, key=lambda r: (r.num_gpus, r.storage)):
            out.append(
                "| "
                + " | ".join(
                    [
                        str(r.num_gpus),
                        r.storage,
                        _fmt(r.train_p50),
                        _fmt(r.train_p95),
                        _fmt(r.throughput_p50, places=2),
                        _fmt(r.peak_mem_gb, places=2),
                        _fmt(r.wallclock_total_s, places=1),
                    ]
                )
                + " |"
            )
        png_src = figures_dir.parent / slug / "scaling.png"
        if png_src.exists():
            png_dest = figures_dir / f"{slug}__scaling.png"
            shutil.copy(png_src, png_dest)
            out.append("")
            out.append(f"![scaling]({rel_figures_dir}/{png_dest.name})")
        store_src = figures_dir.parent / slug / "storage_compare.png"
        if store_src.exists():
            store_dest = figures_dir / f"{slug}__storage_compare.png"
            shutil.copy(store_src, store_dest)
            out.append("")
            out.append(f"![storage compare]({rel_figures_dir}/{store_dest.name})")
        out.append("")
    return "\n".join(out)


def _exec_summary_bullets(rows: list[Row], anomalies_md: str | None) -> list[str]:
    """3-4 bullet points for the exec summary, derived from the data."""
    bullets: list[str] = []
    if not rows:
        return ["No runs ingested."]

    # Best throughput overall
    best = max(
        (r for r in rows if r.throughput_p50 is not None),
        key=lambda r: r.throughput_p50,
        default=None,
    )
    if best is not None:
        bullets.append(
            f"Best observed p50 throughput: **{best.throughput_p50:.1f} samples/s** "
            f"on `{best.model}` / `{best.dataset}` at {best.num_gpus} GPUs / {best.storage} / "
            f"sampling={best.sampling}."
        )

    # Storage uplift summary (where both lustre and nvme exist for the same key)
    by_key_storage: dict[tuple[str, str, int, int], dict[str, Row]] = defaultdict(dict)
    for r in rows:
        by_key_storage[(r.model, r.dataset, r.num_gpus, r.sampling)][r.storage] = r
    uplifts: list[float] = []
    for d in by_key_storage.values():
        if "lustre" in d and "nvme" in d:
            l = d["lustre"].throughput_p50 or 0.0
            n = d["nvme"].throughput_p50 or 0.0
            if l > 0:
                uplifts.append((n - l) / l)
    if uplifts:
        avg = 100 * sum(uplifts) / len(uplifts)
        bullets.append(
            f"NVMe vs Lustre: average **{avg:+.1f}%** throughput delta across "
            f"{len(uplifts)} matched run pairs."
        )

    # Memory ceiling
    peak = max((r.peak_mem_gb for r in rows if r.peak_mem_gb is not None), default=None)
    if peak is not None:
        bullets.append(f"Peak GPU memory observed across the matrix: **{peak:.1f} GB** reserved.")

    # Anomalies
    if anomalies_md and "No anomalies" not in anomalies_md:
        bullets.append("Strong-scaling efficiency dropped below 70% in at least one configuration -- see Anomalies section.")
    elif anomalies_md:
        bullets.append("All configurations scaled at or above 70% strong-scaling efficiency.")

    return bullets


def _render_full_report(
    rows: list[Row],
    out_dir: Path,
    analysis_dir: Path,
) -> None:
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    rel_figures = "figures"

    anomalies_path = analysis_dir / "anomalies.md"
    anomalies_text = anomalies_path.read_text() if anomalies_path.exists() else None

    bullets = _exec_summary_bullets(rows, anomalies_text)

    body: list[str] = []
    body.append("# CAE Benchmark Report")
    body.append("")
    body.append(f"Generated from `{analysis_dir}/summary.csv` ({len(rows)} runs).")
    body.append("")
    body.append("## Executive summary")
    body.append("")
    for b in bullets:
        body.append(f"- {b}")
    body.append("")
    body.append("## Benchmark matrix")
    body.append("")
    body.append(_matrix_table(rows))
    body.append("")
    body.append("## Scalability per (model, dataset, sampling)")
    body.append("")
    body.append(_scaling_section(rows, figures_dir, rel_figures))
    body.append("")
    body.append("## Anomalies")
    body.append("")
    if anomalies_text:
        body.append(anomalies_text)
    else:
        body.append("No anomalies file present (run analyze_scaling.py first).")
    body.append("")
    body.append("## Provenance")
    body.append("")
    body.append(f"- `summary.csv`: `{analysis_dir / 'summary.csv'}`")
    body.append(f"- `anomalies.md`: `{anomalies_path}`")
    body.append("- per-run JSON / md / metrics.jsonl: under `results/<model>/<dataset>/gpus_<N>/<storage>/sub_<S>/runs/<run_id>/`")
    body.append("")

    (out_dir / "REPORT.md").write_text("\n".join(body))


def _render_one_pager(rows: list[Row], out_dir: Path, anomalies_path: Path) -> None:
    anomalies_text = anomalies_path.read_text() if anomalies_path.exists() else None
    bullets = _exec_summary_bullets(rows, anomalies_text)

    body: list[str] = []
    body.append("# CAE benchmark - one-pager (for Sanjay)")
    body.append("")
    body.append("## Highlights")
    body.append("")
    for b in bullets:
        body.append(f"- {b}")
    body.append("")
    body.append("## Quick numbers")
    body.append("")
    by_key: dict[tuple[str, str, int], dict[str, Row]] = defaultdict(dict)
    for r in rows:
        by_key.setdefault((r.model, r.dataset, r.sampling), {})[f"g{r.num_gpus}_{r.storage}"] = r

    body.append("| model | dataset | sampling | best (storage, gpus) | thr p50 (samples/s) | peak mem (GB) |")
    body.append("|---|---|---:|---|---:|---:|")
    for (model, dataset, sampling), bucket in sorted(by_key.items()):
        best = max(bucket.values(), key=lambda r: r.throughput_p50 or 0.0)
        body.append(
            f"| {model} | {dataset} | {sampling} | "
            f"{best.storage} / {best.num_gpus} | "
            f"{_fmt(best.throughput_p50, places=2)} | {_fmt(best.peak_mem_gb, places=2)} |"
        )
    body.append("")
    body.append("Full charts + per-run breakdown in `report/REPORT.md`.")

    (out_dir / "SANJAY_ONE_PAGER.md").write_text("\n".join(body))


def main(argv: Iterable[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--analysis", default="analysis", type=Path, help="analyze_scaling.py output dir")
    p.add_argument("--out-dir", default="report", type=Path, help="report output dir")
    args = p.parse_args(list(argv) if argv is not None else None)

    csv_path = args.analysis / "summary.csv"
    if not csv_path.exists():
        print(f"[generate_report] missing {csv_path}; run analyze_scaling.py first.")
        return 2

    rows = _read_csv(csv_path)
    if not rows:
        print("[generate_report] empty summary.csv; nothing to render.")
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    _render_full_report(rows, args.out_dir, args.analysis)
    _render_one_pager(rows, args.out_dir, args.analysis / "anomalies.md")

    print(f"[generate_report] wrote {args.out_dir / 'REPORT.md'}")
    print(f"[generate_report] wrote {args.out_dir / 'SANJAY_ONE_PAGER.md'}")
    print(f"[generate_report] figures copied to {args.out_dir / 'figures'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
