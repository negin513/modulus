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

"""Generate slide-ready scaling snapshot plots from benchmark_summary.json files."""

from __future__ import annotations

import sys
from pathlib import Path

_BENCHMARKS_ROOT = Path(__file__).resolve().parents[1]
if str(_BENCHMARKS_ROOT) not in sys.path:
    sys.path.insert(0, str(_BENCHMARKS_ROOT))
import argparse
import json
from collections import defaultdict
from pathlib import Path

from ingest.summarize_run import num_epochs_for_aggregate

# Fixed subsampling grid for g=1 baseline plots (05–07). Same x-axis on Lustre and NVMe.
SINGLE_GPU_BASELINE_SUBS: tuple[int, ...] = (10_000, 50_000, 100_000, 200_000, 300_000)
SINGLE_GPU_BASELINE_SUBS_SET = frozenset(SINGLE_GPU_BASELINE_SUBS)

NVIDIA_COLORS = [
    "#76B900",  # NVIDIA green
    "#00B140",  # secondary green
    "#484848",  # dark gray
    "#007A3D",  # teal
    "#59595B",  # light gray
    "#B5BD00",  # lime
]

COL_NVME = NVIDIA_COLORS[0]
COL_LUSTRE = NVIDIA_COLORS[2]
COL_IDEAL = NVIDIA_COLORS[4]
COL_SURFACE = NVIDIA_COLORS[0]
COL_VOLUME = NVIDIA_COLORS[3]
COL_LIMIT = NVIDIA_COLORS[5]
COL_70PCT = NVIDIA_COLORS[2]

TINY_N = 10
SCALING_GPUS: tuple[int, ...] = (1, 4, 8, 16, 32, 64)
SCALING_GPUS_SET = frozenset(SCALING_GPUS)
SCALING_XLIM = (0, max(SCALING_GPUS) + 4)
SCALING_SUBS: tuple[int, ...] = (50_000, 100_000, 200_000)
B200_HBM_GB = 192  # NVIDIA B200 HBM3e per GPU
SUPTITLE_FONTSIZE = 16
SUPTITLE_Y = 0.985
SUBTITLE_FONTSIZE = 14
SUBTITLE_LINESPACING = 1.7
SUBTITLE_Y = 0.905
SUBTITLE_Y_MULTILINE = 0.888
SUBPLOT_TITLE_SIZE = 12
AXES_LABELSIZE = 12
TICK_LABELSIZE = 9
TICK_MAJORSIZE = 3.5
LAYOUT_TOP_WITH_SUBTITLE = 0.925
LAYOUT_TOP_WITH_SUBTITLE_MULTILINE = 0.905
LAYOUT_TOP_TITLE_ONLY = 0.975


def _fig_suptitle(fig, text: str, *, y: float | None = None) -> None:
    fig.suptitle(text, fontsize=SUPTITLE_FONTSIZE, fontweight="bold", y=y if y is not None else SUPTITLE_Y)


def _fig_subtitle(
    fig,
    text: str,
    y: float | None = None,
    multiline: bool = False,
    linespacing: float | None = None,
    *,
    bold: bool = False,
    italic: bool = True,
) -> None:
    """Second title line — below suptitle with explicit vertical gap."""
    if y is None:
        y = SUBTITLE_Y_MULTILINE if multiline else SUBTITLE_Y
    fig.text(
        0.5,
        y,
        text,
        ha="center",
        fontsize=SUBTITLE_FONTSIZE,
        fontweight="bold" if bold else "normal",
        linespacing=linespacing if linespacing is not None else SUBTITLE_LINESPACING,
        color="#484848",
        style="italic" if italic else "normal",
    )


def _bold_tick_labels(ax) -> None:
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")
        label.set_fontsize(TICK_LABELSIZE)


def _bold_legend(ax) -> None:
    leg = ax.get_legend()
    if leg is None:
        return
    for text in leg.get_texts():
        text.set_fontweight("bold")


def _finalize_axes(ax) -> None:
    for spine in ax.spines.values():
        spine.set_visible(True)
    ax.tick_params(
        axis="both",
        which="major",
        labelsize=TICK_LABELSIZE,
        length=TICK_MAJORSIZE,
        width=0.9,
        top=True,
        right=True,
        labeltop=False,
        labelright=False,
    )
    _bold_tick_labels(ax)


def _finalize_figure(fig) -> None:
    for ax in fig.get_axes():
        _finalize_axes(ax)


def _apply_style() -> None:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    plt.style.use("seaborn-v0_8-whitegrid")
    mpl.rcParams.update(
        {
            "figure.dpi": 150,
            "axes.titlesize": SUBPLOT_TITLE_SIZE,
            "axes.labelsize": AXES_LABELSIZE,
            "xtick.labelsize": TICK_LABELSIZE,
            "ytick.labelsize": TICK_LABELSIZE,
            "xtick.major.size": TICK_MAJORSIZE,
            "ytick.major.size": TICK_MAJORSIZE,
            "legend.fontsize": 11,
            "axes.titleweight": "bold",
            "axes.labelweight": "bold",
            "axes.edgecolor": "#333333",
            "axes.linewidth": 1.2,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linestyle": "--",
            "axes.facecolor": "#FAFAFA",
            "savefig.bbox": "tight",
            "savefig.transparent": False,
            "font.family": "DejaVu Sans",
        }
    )


def _load_rows(results_root: Path) -> list[dict]:
    rows: list[dict] = []
    for path in results_root.rglob("benchmark_summary.json"):
        if "_smoketest" in str(path):
            continue
        summary = json.loads(path.read_text())
        rows.append(
            {
                "model": summary["model"],
                "g": summary["num_gpus"],
                "storage": summary["storage"],
                "sub": summary["sampling_resolution"],
                "thr": summary["throughput_samples_per_sec_p50"],
                "mem": summary["memory"]["peak_gb"],
                "n_steps": summary.get(
                    "n_train_steps_aggregated",
                    summary.get("n_train_steps_after_warmup", 0),
                ),
                "wall_train": summary.get("wallclock_train_s"),
                "epochs": num_epochs_for_aggregate(summary),
            }
        )
    return rows


def _model_label(model: str) -> str:
    if model == "geotransolver_surface":
        return "GeoTransolver Surface"
    if model == "geotransolver_volume":
        return "GeoTransolver Volume"
    return model.replace("geotransolver_", "GeoTransolver ").title()


def _scaling_points(pts: list[dict]) -> list[dict]:
    """Keep only g ∈ SCALING_GPUS for weak-scaling plots."""
    return [r for r in pts if r["g"] in SCALING_GPUS_SET]


def _set_gpu_count_axis(ax, *, show_xlabel: bool) -> None:
    """Label weak-scaling x-axis with GPU counts from SCALING_GPUS."""
    ax.set_xticks(list(SCALING_GPUS))
    ax.set_xticklabels([str(g) for g in SCALING_GPUS])
    ax.set_xlim(*SCALING_XLIM)
    ax.tick_params(axis="x", labelbottom=True)
    for label in ax.get_xticklabels():
        label.set_visible(True)
    if show_xlabel:
        ax.set_xlabel("# GPUs")


def _plot_throughput(groups, out: Path) -> None:
    import matplotlib.pyplot as plt

    model = "geotransolver_volume"
    subs = list(SCALING_SUBS)
    storages = [("nvme", COL_NVME, "s"), ("lustre", COL_LUSTRE, "o")]
    n_subs = len(subs)

    fig, axes = plt.subplots(1, n_subs, figsize=(4.5 * n_subs, 4.5), sharex=True, sharey=True)
    if n_subs == 1:
        axes = [axes]
    _fig_suptitle(fig, "DrivAerML Weak Scaling — Throughput Vs GPU Count", y=0.99)
    _fig_subtitle(
        fig,
        "Throughput (P50): median training throughput (samples/s) from the per-step time median at each GPU count · B200",
        y=0.875,
        bold=True,
        italic=False,
    )

    for j, sub in enumerate(subs):
        ax = axes[j]

        for storage, color, marker in storages:
            pts = _scaling_points(groups.get((model, sub, storage), []))
            if not pts:
                continue
            xs = [r["g"] for r in pts]
            ys = [r["thr"] for r in pts]
            for x, y in zip(xs, ys):
                ax.scatter(
                    x,
                    y,
                    color=color,
                    marker=marker,
                    s=70,
                    facecolor=color,
                    edgecolor=color,
                    lw=1.5,
                    zorder=4,
                )
            label = "NVMe" if storage == "nvme" else "Lustre"
            ax.plot(xs, ys, color=color, lw=2.0, label=label, alpha=0.95, zorder=3)

        _set_gpu_count_axis(ax, show_xlabel=True)
        ax.set_ylim(0, 200)
        ax.set_title(f"{_model_label(model)}  |  Sub={sub:,}", fontsize=SUBPLOT_TITLE_SIZE)
        if j == 0:
            ax.set_ylabel("Throughput (Samples/S)")
        if j == n_subs - 1:
            ax.legend(loc="upper left", framealpha=0.95)

    fig.tight_layout(rect=[0, 0, 1, 0.915])
    _finalize_figure(fig)
    fig.savefig(out / "01_throughput_vs_gpus.png")
    plt.close(fig)


def _plot_efficiency(groups, out: Path) -> None:
    import matplotlib.pyplot as plt

    models = ["geotransolver_volume", "geotransolver_surface"]
    subs = [50_000, 100_000, 200_000]
    storages = [("nvme", COL_NVME, "s"), ("lustre", COL_LUSTRE, "o")]

    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5), sharex=True, sharey=True)
    _fig_suptitle(fig, "DrivAerML Weak Scaling — Parallel Efficiency Vs GPU Count")
    _fig_subtitle(
        fig,
        "Parallel efficiency η(N) = Φ(N) / (N · Φ(1)) × 100%, using g = 1 throughput as baseline. 100% = perfect weak scaling.",
        bold=True,
        italic=False,
    )

    for i, model in enumerate(models):
        for j, sub in enumerate(subs):
            ax = axes[i, j]
            ax.axhline(100, color=COL_IDEAL, ls="--", lw=1.2, alpha=0.85)

            for storage, color, marker in storages:
                pts = _scaling_points(groups.get((model, sub, storage), []))
                if len(pts) < 2:
                    continue
                base = next((r for r in pts if r["g"] == 1), pts[0])
                xs = [r["g"] for r in pts]
                effs = [(r["thr"] / base["thr"]) / (r["g"] / base["g"]) * 100 for r in pts]
                for x, e in zip(xs, effs):
                    ax.scatter(
                        x,
                        e,
                        color=color,
                        marker=marker,
                        s=70,
                        facecolor=color,
                        edgecolor=color,
                        lw=1.5,
                        zorder=4,
                    )
                label = "NVMe" if storage == "nvme" else "Lustre"
                ax.plot(xs, effs, color=color, lw=2.0, label=label, alpha=0.95, zorder=3)

            ax.set_xticks(list(SCALING_GPUS))
            ax.set_xticklabels([str(g) for g in SCALING_GPUS])
            ax.set_xlim(*SCALING_XLIM)
            ax.set_ylim(0, 120)
            ax.set_title(f"{_model_label(model)}  |  Sub={sub:,}", fontsize=SUBPLOT_TITLE_SIZE)
            if j == 0:
                ax.set_ylabel("Parallel Efficiency (%)")
            if i == 1:
                ax.set_xlabel("GPUs")
            if i == 0 and j == 2:
                ax.legend(loc="upper left", framealpha=0.95)

    fig.tight_layout(rect=[0, 0, 1, LAYOUT_TOP_WITH_SUBTITLE])
    _finalize_figure(fig)
    fig.savefig(out / "02_efficiency_vs_gpus.png")
    plt.close(fig)


def _plot_storage_compare(groups, out: Path) -> None:
    import matplotlib.pyplot as plt

    models = ["geotransolver_surface", "geotransolver_volume"]
    subs = [10000, 50000, 100000, 200000]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    _fig_suptitle(fig, "NVMe Stage-In Vs Lustre At 16 GPUs")

    x_labels = [f"{s:,}" for s in subs]
    xpos = list(range(len(subs)))
    width = 0.38

    for ax, model in zip(axes, models):
        lustre_thr, nvme_thr = [], []
        for sub in subs:
            lustre = next((r for r in groups.get((model, sub, "lustre"), []) if r["g"] == 16), None)
            nvme = next((r for r in groups.get((model, sub, "nvme"), []) if r["g"] == 16), None)
            lustre_thr.append(lustre["thr"] if lustre else 0)
            nvme_thr.append(nvme["thr"] if nvme else 0)

        ax.bar(
            [x - width / 2 for x in xpos],
            lustre_thr,
            width,
            label="Lustre",
            color=COL_LUSTRE,
            edgecolor="white",
            lw=1,
        )
        ax.bar(
            [x + width / 2 for x in xpos],
            nvme_thr,
            width,
            label="NVMe",
            color=COL_NVME,
            edgecolor="white",
            lw=1,
        )

        for x, lustre, nvme in zip(xpos, lustre_thr, nvme_thr):
            if lustre > 0 and nvme > 0:
                gain = (nvme / lustre - 1) * 100
                ax.annotate(
                    f"+{gain:.0f}%",
                    xy=(x + width / 2, nvme),
                    xytext=(0, 6),
                    textcoords="offset points",
                    ha="center",
                    fontsize=12,
                    color=COL_NVME,
                    fontweight="bold",
                )

        ax.set_xticks(xpos)
        ax.set_xticklabels(x_labels)
        ax.set_xlabel("Sampling Resolution")
        ax.set_ylabel("Throughput (Samples/S)")
        ax.set_title(f"{_model_label(model)} Model", fontsize=SUBPLOT_TITLE_SIZE)
        ax.legend(loc="upper right", framealpha=0.95)

    fig.tight_layout(rect=[0, 0, 1, LAYOUT_TOP_TITLE_ONLY])
    _finalize_figure(fig)
    fig.savefig(out / "03_nvme_vs_lustre_16gpu.png")
    plt.close(fig)


def _plot_memory(rows, out: Path, num_gpus: int = 4) -> None:
    import matplotlib.pyplot as plt

    exclude_subs = {250_000, 400_000}

    fig, ax = plt.subplots(figsize=(9, 5.8))
    _fig_suptitle(fig, "Peak GPU Memory Vs Subsampling Resolution")
    _fig_subtitle(fig, "1 Node · 4 B200 GPUs · Per-Rank Peak Reserved VRAM")

    gpu_rows = [r for r in rows if r["g"] == num_gpus and r["sub"] not in exclude_subs]
    x_max = 0
    y_max = 0.0

    for model, color in [
        ("geotransolver_surface", COL_SURFACE),
        ("geotransolver_volume", COL_VOLUME),
    ]:
        per_sub: dict[int, float] = {}
        for r in gpu_rows:
            if r["model"] != model:
                continue
            per_sub[r["sub"]] = r["mem"]
        if not per_sub:
            continue
        xs = sorted(per_sub)
        ys = [per_sub[s] for s in xs]
        x_max = max(x_max, xs[-1])
        y_max = max(y_max, max(ys))
        ax.plot(xs, ys, "o-", color=color, lw=2.5, ms=10,
                label=_model_label(model), zorder=4)

    xlim_max = max(350_000, int(x_max * 1.08)) if x_max else 350_000
    xticks = [t for t in (0, 100_000, 200_000, 300_000) if t <= xlim_max]
    if x_max and x_max not in xticks:
        xticks.append(x_max)
    xticks = sorted(set(xticks))
    ax.set_xlabel("Subsampling Resolution")
    ax.set_ylabel("Peak GPU Memory (GB)")
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{x:,}" for x in xticks])
    ax.set_xlim(0, xlim_max)
    ax.set_ylim(0, y_max * 1.12 if y_max else B200_HBM_GB)
    ax.legend(loc="lower right", framealpha=0.95, fontsize=9, ncol=1, prop={"weight": "bold"})
    fig.tight_layout(rect=[0, 0, 1, LAYOUT_TOP_WITH_SUBTITLE])
    _finalize_figure(fig)
    _bold_legend(ax)
    fig.savefig(out / "04_memory_vs_sampling.png")
    plt.close(fig)


def _single_gpu_sub_sweep_complete(
    by_model: dict[str, list[dict]],
    subs: tuple[int, ...] = (10_000, 50_000, 100_000, 200_000),
) -> bool:
    """True when at least one model has all standard single-GPU subsampling points."""
    for pts in by_model.values():
        if not pts:
            continue
        have = {p["sub"] for p in pts}
        if all(s in have for s in subs):
            return True
    return False


def _single_gpu_by_storage(rows: list[dict], storage: str) -> dict[str, list[dict]]:
    """Return model -> sorted single-GPU points for a storage tier (≤300k grid only)."""
    out: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if r["g"] != 1 or r["storage"] != storage:
            continue
        if r["sub"] not in SINGLE_GPU_BASELINE_SUBS_SET:
            continue
        time_per_epoch = None
        if r.get("wall_train") and r["wall_train"] > 0 and r.get("epochs"):
            time_per_epoch = r["wall_train"] / r["epochs"]
        out[r["model"]].append({**r, "time_per_epoch": time_per_epoch})
    for model in out:
        out[model].sort(key=lambda x: x["sub"])
    return out


def _gpu_by_storage(rows: list[dict], storage: str, num_gpus: int) -> dict[str, list[dict]]:
    """Return model -> sorted points for a fixed (storage, num_gpus) pair."""
    out: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if r["g"] != num_gpus or r["storage"] != storage:
            continue
        time_per_epoch = None
        if r.get("wall_train") and r["wall_train"] > 0 and r.get("epochs"):
            time_per_epoch = r["wall_train"] / r["epochs"]
        out[r["model"]].append({**r, "time_per_epoch": time_per_epoch})
    for model in out:
        out[model].sort(key=lambda x: x["sub"])
    return out


def _single_gpu_baseline_xaxis() -> tuple[list[int], list[str]]:
    """Uniform x-axis ticks/labels for single-GPU baseline charts (linear subsampling scale)."""
    subs = list(SINGLE_GPU_BASELINE_SUBS)
    return subs, [f"{s:,}" for s in subs]


def _plot_throughput_epochtime_panels(
    by_model: dict[str, list[dict]],
    out_path: Path,
    suptitle: str,
    subtitle: str,
) -> bool:
    """Plot samples/s (left) and time/epoch (right). Returns False if no data."""
    import matplotlib.pyplot as plt

    if not any(by_model.values()):
        return False

    subs, xlabels = _single_gpu_baseline_xaxis()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    _fig_suptitle(fig, suptitle)
    _fig_subtitle(fig, subtitle)

    for ax, metric, ylabel, title in (
        (axes[0], "thr", "Throughput (Samples/S)", "Samples Per Second"),
        (axes[1], "time_per_epoch", "Time Per Epoch (S)", "Time Per Epoch"),
    ):
        for model, color in [("geotransolver_surface", COL_SURFACE), ("geotransolver_volume", COL_VOLUME)]:
            pts = by_model.get(model, [])
            if not pts:
                continue
            plotted = [
                p for p in pts
                if p.get(metric) is not None and p["sub"] in SINGLE_GPU_BASELINE_SUBS_SET
            ]
            plotted.sort(key=lambda p: p["sub"])
            if not plotted:
                continue
            xs = [p["sub"] for p in plotted]
            ys = [p[metric] for p in plotted]
            ax.plot(xs, ys, "o-", color=color, lw=2.5, ms=11, label=_model_label(model))
        ax.set_xscale("linear")
        ax.set_xticks(subs)
        ax.set_xticklabels(xlabels, fontsize=AXES_LABELSIZE)
        ax.set_xlim(min(subs) * 0.85, max(subs) * 1.05)
        ax.set_xlabel("Subsampling (Points/Sample)", fontsize=AXES_LABELSIZE)
        ax.set_ylabel(ylabel, fontsize=AXES_LABELSIZE)
        ax.set_title(title, fontsize=SUBPLOT_TITLE_SIZE)
        ax.set_ylim(bottom=0)
        ax.legend(loc="upper right" if metric == "thr" else "upper left", framealpha=0.95)

    fig.tight_layout(rect=[0, 0, 1, LAYOUT_TOP_WITH_SUBTITLE])
    _finalize_figure(fig)
    fig.savefig(out_path)
    plt.close(fig)
    return True


def _plot_single_gpu_throughput_epochtime(rows: list[dict], out: Path) -> None:
    lustre = _single_gpu_by_storage(rows, "lustre")
    _plot_throughput_epochtime_panels(
        lustre,
        out / "05_single_gpu_throughput_epochtime_lustre.png",
        "Single GPU Baseline — Throughput vs. Subsampling",
        "batch_size = 1 · 5 epochs · 1 GPU · B200 · Lustre",
    )

    nvme_g1 = _single_gpu_by_storage(rows, "nvme")
    if _single_gpu_sub_sweep_complete(nvme_g1) and _plot_throughput_epochtime_panels(
        nvme_g1,
        out / "07_single_gpu_throughput_epochtime_nvme.png",
        "Single GPU Baseline — Throughput vs. Subsampling",
        "batch_size = 1 · 5 epochs · 1 GPU · B200 · NVMe",
    ):
        return

    # Phase 3 g=1 NVMe sweep incomplete — plot minimum NVMe config (g=16) for comparison.
    nvme_g16 = _gpu_by_storage(rows, "nvme", 16)
    _plot_throughput_epochtime_panels(
        nvme_g16,
        out / "07_single_gpu_throughput_epochtime_nvme.png",
        "NVMe Staged Baseline — Throughput Vs Sample Size (B200, 16 GPUs)",
        "batch_size = 1 · 5 epochs · NVMe stage-in · g=1 NVMe Phase 3 pending — showing g=16 NVMe",
    )


def _plot_storage_tier_bar_grid(
    rows: list[dict],
    out: Path,
    *,
    num_gpus: int,
    filename: str,
    suptitle: str,
    subtitle: str,
) -> None:
    """2×2 bar grid: Lustre vs NVMe at fixed GPU count."""
    import matplotlib.pyplot as plt
    import numpy as np

    lustre = _gpu_by_storage(rows, "lustre", num_gpus)
    nvme = _gpu_by_storage(rows, "nvme", num_gpus)
    subs = [10000, 50000, 100000, 200000]
    xlabels = [f"{s:,}" for s in subs]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    _fig_suptitle(fig, suptitle)
    _fig_subtitle(fig, subtitle, multiline=True)

    for row_i, model in enumerate(["geotransolver_surface", "geotransolver_volume"]):
        for col_i, metric, ylabel in (
            (0, "thr", "Throughput (Samples/S)"),
            (1, "time_per_epoch", "Time Per Epoch (S)"),
        ):
            ax = axes[row_i, col_i]
            x = np.arange(len(subs))
            width = 0.35

            def vals(src, m):
                pts = {p["sub"]: p.get(m) for p in src.get(model, [])}
                return [pts.get(s) for s in subs]

            series = [
                (f"Lustre g={num_gpus}", vals(lustre, metric), COL_LUSTRE),
                (f"NVMe g={num_gpus}", vals(nvme, metric), COL_NVME),
            ]
            for j, (label, data, color) in enumerate(series):
                heights = [d if d is not None else 0 for d in data]
                offset = (j - 0.5) * width
                ax.bar(
                    x + offset,
                    heights,
                    width,
                    label=label,
                    color=color,
                    edgecolor="white",
                )
            ax.set_xticks(x)
            ax.set_xticklabels(xlabels)
            ax.set_xlabel("Sampling Resolution")
            ax.set_ylabel(ylabel)
            ax.set_title(f"{_model_label(model)} — {ylabel.split('(')[0].strip()}")
            ax.set_ylim(bottom=0)
            if row_i == 0 and col_i == 1:
                ax.legend(fontsize=9, loc="upper left")

    fig.tight_layout(rect=[0, 0, 1, LAYOUT_TOP_WITH_SUBTITLE_MULTILINE])
    _finalize_figure(fig)
    fig.savefig(out / filename)
    plt.close(fig)


def _plot_lustre_vs_nvme_io_compare(rows: list[dict], out: Path) -> None:
    """Bar comparison: g=1 Lustre vs g=16 Lustre vs g=16 NVMe at each sampling."""
    import matplotlib.pyplot as plt
    import numpy as np

    lustre_g1 = _single_gpu_by_storage(rows, "lustre")
    lustre_g16 = _gpu_by_storage(rows, "lustre", 16)
    nvme_g16 = _gpu_by_storage(rows, "nvme", 16)
    subs = [10000, 50000, 100000, 200000]
    xlabels = [f"{s:,}" for s in subs]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    _fig_suptitle(fig, "Storage Tier Impact — Lustre g=1 vs Multi-GPU g=16")
    _fig_subtitle(
        fig,
        "Top Row: GeoTransolver Surface · Bottom Row: GeoTransolver Volume\nLeft: Throughput (Samples/S) · Right: Time Per Epoch (S)",
        multiline=True,
    )

    for row_i, model in enumerate(["geotransolver_surface", "geotransolver_volume"]):
        for col_i, metric, ylabel in (
            (0, "thr", "Throughput (Samples/S)"),
            (1, "time_per_epoch", "Time Per Epoch (S)"),
        ):
            ax = axes[row_i, col_i]
            x = np.arange(len(subs))
            width = 0.25

            def vals(src, m):
                pts = {p["sub"]: p.get(m) for p in src.get(model, [])}
                return [pts.get(s) for s in subs]

            series = [
                ("Lustre g=1", vals(lustre_g1, metric), COL_LUSTRE),
                ("Lustre g=16", vals(lustre_g16, metric), COL_IDEAL),
                ("NVMe g=16", vals(nvme_g16, metric), COL_NVME),
            ]
            for j, (label, data, color) in enumerate(series):
                heights = [d if d is not None else 0 for d in data]
                ax.bar(x + (j - 1) * width, heights, width, label=label, color=color, edgecolor="white")
            ax.set_xticks(x)
            ax.set_xticklabels(xlabels)
            ax.set_xlabel("Sampling Resolution")
            ax.set_ylabel(ylabel)
            ax.set_title(f"{_model_label(model)} — {ylabel.split('(')[0].strip()}")
            ax.set_ylim(bottom=0)
            if row_i == 0 and col_i == 1:
                ax.legend(fontsize=9, loc="upper left")

    fig.tight_layout(rect=[0, 0, 1, LAYOUT_TOP_WITH_SUBTITLE_MULTILINE])
    _finalize_figure(fig)
    fig.savefig(out / "08_lustre_vs_nvme_storage_compare.png")
    plt.close(fig)


def _plot_storage_tier_g16_compare(rows: list[dict], out: Path) -> None:
    _plot_storage_tier_bar_grid(
        rows,
        out,
        num_gpus=16,
        filename="09_lustre_vs_nvme_g16.png",
        suptitle="Storage Tier Impact — Lustre g=16 vs NVMe g=16",
        subtitle=(
            "Top Row: GeoTransolver Surface · Bottom Row: GeoTransolver Volume\n"
            "Left: Throughput (Samples/S) · Right: Time Per Epoch (S)"
        ),
    )


def _plot_storage_tier_g1_g16_by_metric(
    rows: list[dict],
    out: Path,
    *,
    metric: str,
    ylabel: str,
    filename: str,
    suptitle: str,
) -> None:
    """2×2 Lustre vs NVMe bars: rows = model, columns = 1 GPU vs 16 GPUs."""
    import matplotlib.pyplot as plt
    import numpy as np

    subs = [10000, 50000, 100000, 200000]
    xlabels = [f"{s:,}" for s in subs]
    gpu_cols = (1, 16)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    _fig_suptitle(fig, suptitle)
    _fig_subtitle(
        fig,
        "Top Row: GeoTransolver Surface · Bottom Row: GeoTransolver Volume\n"
        "Left: 1 GPU · Right: 16 GPUs · Bars: Lustre vs NVMe",
        multiline=True,
    )

    for row_i, model in enumerate(["geotransolver_surface", "geotransolver_volume"]):
        for col_i, num_gpus in enumerate(gpu_cols):
            ax = axes[row_i, col_i]
            lustre = _gpu_by_storage(rows, "lustre", num_gpus)
            nvme = _gpu_by_storage(rows, "nvme", num_gpus)
            x = np.arange(len(subs))
            width = 0.35

            def vals(src, m):
                pts = {p["sub"]: p.get(m) for p in src.get(model, [])}
                return [pts.get(s) for s in subs]

            series = [
                (f"Lustre g={num_gpus}", vals(lustre, metric), COL_LUSTRE),
                (f"NVMe g={num_gpus}", vals(nvme, metric), COL_NVME),
            ]
            for j, (label, data, color) in enumerate(series):
                heights = [d if d is not None else 0 for d in data]
                offset = (j - 0.5) * width
                ax.bar(
                    x + offset,
                    heights,
                    width,
                    label=label,
                    color=color,
                    edgecolor="white",
                )
            ax.set_xticks(x)
            ax.set_xticklabels(xlabels)
            ax.set_xlabel("Sampling Resolution")
            ax.set_ylabel(ylabel)
            gpu_label = "1 GPU" if num_gpus == 1 else "16 GPUs"
            ax.set_title(f"{_model_label(model)} — {gpu_label}")
            ax.set_ylim(bottom=0)
            if row_i == 0 and col_i == 1:
                ax.legend(fontsize=9, loc="upper left")

    fig.tight_layout(rect=[0, 0, 1, LAYOUT_TOP_WITH_SUBTITLE_MULTILINE])
    _finalize_figure(fig)
    fig.savefig(out / filename)
    plt.close(fig)


def _plot_storage_tier_g1_compare(rows: list[dict], out: Path) -> None:
    _plot_storage_tier_g1_g16_by_metric(
        rows,
        out,
        metric="thr",
        ylabel="Throughput (Samples/S)",
        filename="10_lustre_vs_nvme_throughput_g1_g16.png",
        suptitle="Storage Tier Impact — Throughput (Lustre vs NVMe)",
    )
    _plot_storage_tier_g1_g16_by_metric(
        rows,
        out,
        metric="time_per_epoch",
        ylabel="Time Per Epoch (S)",
        filename="10_lustre_vs_nvme_epochtime_g1_g16.png",
        suptitle="Storage Tier Impact — Time Per Epoch (Lustre vs NVMe)",
    )
    legacy = out / "10_lustre_vs_nvme_g1.png"
    if legacy.exists():
        legacy.unlink()


def _plot_single_gpu_memory(rows: list[dict], out: Path) -> None:
    import matplotlib.pyplot as plt

    subs, xlabels = _single_gpu_baseline_xaxis()
    by_model = _single_gpu_by_storage(rows, "lustre")

    fig, ax = plt.subplots(figsize=(8, 5.5))
    _fig_suptitle(fig, "Single-GPU Baseline — Peak Memory Vs Sample Size (B200)")
    _fig_subtitle(
        fig,
        "batch_size = 1 (Fixed) · Memory Swept via Subsampling Resolution",
    )

    for model, color in [("geotransolver_surface", COL_SURFACE), ("geotransolver_volume", COL_VOLUME)]:
        pts = [p for p in by_model.get(model, []) if p["sub"] in SINGLE_GPU_BASELINE_SUBS_SET]
        if not pts:
            continue
        pts.sort(key=lambda p: p["sub"])
        xs = [p["sub"] for p in pts]
        ys = [p["mem"] for p in pts]
        ax.plot(xs, ys, "o-", color=color, lw=2.5, ms=11, label=_model_label(model))

    ax.set_xscale("linear")
    ax.set_xticks(subs)
    ax.set_xticklabels(xlabels, fontsize=AXES_LABELSIZE)
    ax.set_xlim(min(subs) * 0.85, max(subs) * 1.05)
    ax.set_xlabel("Subsampling Resolution", fontsize=AXES_LABELSIZE)
    ax.set_ylabel("Peak GPU Memory (GB)", fontsize=AXES_LABELSIZE)
    ax.set_ylim(0, 155)
    ax.legend(loc="upper left", framealpha=0.95)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    _finalize_figure(fig)
    fig.savefig(out / "06_single_gpu_memory.png")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"), help="Results root directory")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/_scaling_snapshot"),
        help="Output directory for PNG plots",
    )
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    _apply_style()

    rows = _load_rows(args.results)
    if not rows:
        print(f"[plot] no benchmark summaries found under {args.results}")
        return 1

    groups: dict[tuple[str, int, str], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["model"], row["sub"], row["storage"])].append(row)
    for key in groups:
        groups[key].sort(key=lambda r: r["g"])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    _plot_throughput(groups, args.out_dir)
    _plot_efficiency(groups, args.out_dir)
    _plot_storage_compare(groups, args.out_dir)
    _plot_memory(rows, args.out_dir)
    _plot_single_gpu_throughput_epochtime(rows, args.out_dir)
    _plot_lustre_vs_nvme_io_compare(rows, args.out_dir)
    _plot_storage_tier_g16_compare(rows, args.out_dir)
    _plot_storage_tier_g1_compare(rows, args.out_dir)
    _plot_single_gpu_memory(rows, args.out_dir)

    print(f"[plot] wrote 11 charts from {len(rows)} runs -> {args.out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
