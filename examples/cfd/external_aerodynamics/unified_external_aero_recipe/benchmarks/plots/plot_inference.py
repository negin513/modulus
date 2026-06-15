#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Generate inference latency plots from validation-step timings in benchmark summaries."""

from __future__ import annotations

import sys
from pathlib import Path

_BENCHMARKS_ROOT = Path(__file__).resolve().parents[1]
if str(_BENCHMARKS_ROOT) not in sys.path:
    sys.path.insert(0, str(_BENCHMARKS_ROOT))
import argparse
import json
from pathlib import Path

from plots.inference_metrics import (
    INFERENCE_BOX_SUBS,
    INFERENCE_THROUGHPUT_SUBS,
    MODEL_LABELS,
    MODEL_SURFACE,
    MODEL_VOLUME,
    ValLatencyBox,
    ValThroughputBox,
    collect_inference_points,
    collect_val_latency_boxes,
    collect_val_throughput_boxes,
    load_summaries,
    points_for_model,
)
from plots.plot_scaling_snapshot import (
    COL_SURFACE,
    COL_VOLUME,
    LAYOUT_TOP_WITH_SUBTITLE,
    _apply_style,
    _bold_legend,
    _bold_tick_labels,
    _fig_subtitle,
    _fig_suptitle,
    _finalize_figure,
)

OUT_FILES = {
    "latency": "23_inference_latency_vs_size.png",
    "breakdown": "24_inference_breakdown_stacked.png",
    "memory": "25_inference_memory_vs_size.png",
    "inference_box_nvme": "26_inference_boxplot_nvme.png",
    "inference_box_lustre": "27_inference_boxplot_lustre.png",
    "volume_box_nvme": "28_inference_volume_boxplot_nvme.png",
    "volume_throughput_nvme": "29_inference_volume_throughput_nvme.png",
    "surface_throughput_nvme": "30_inference_surface_throughput_nvme.png",
}

BOX_GREEN = "#76B900"
BOX_GREEN_LIGHT = "#B5E08A"
BOX_MEDIAN = "#1A1A1A"


def _storage_label(storage: str) -> str:
    return storage.upper() if storage.lower() == "nvme" else storage.title()


def _hardware_subtitle(
    *,
    gpu_type: str,
    storage: str,
    batch_size: int = 1,
) -> str:
    """Shared subtitle line: GPU SKU, storage tier, batch size."""
    return (
        f"{gpu_type} · {_storage_label(storage)} · batch_size = {batch_size}"
    )


def _plot_latency(points, out: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5.8))
    _fig_suptitle(fig, "Inference Latency Vs Subsampling Resolution")
    _fig_subtitle(fig, "B200 · g = 1 · Lustre · Validation Pass · batch_size = 1")

    for model, color, marker in (
        (MODEL_SURFACE, COL_SURFACE, "o"),
        (MODEL_VOLUME, COL_VOLUME, "s"),
    ):
        pts = points_for_model(points, model)
        if not pts:
            continue
        xs = [p.subsampling for p in pts]
        label = MODEL_LABELS[model]
        ax.plot(
            xs,
            [p.p50_ms for p in pts],
            f"{marker}-",
            color=color,
            lw=2.5,
            ms=9,
            label=f"{label} P50",
        )
        ax.plot(
            xs,
            [p.p95_ms for p in pts],
            f"{marker}--",
            color=color,
            lw=1.5,
            ms=7,
            alpha=0.75,
            label=f"{label} P95",
        )
        ax.plot(
            xs,
            [p.p99_ms for p in pts],
            f"{marker}:",
            color=color,
            lw=1.2,
            ms=6,
            alpha=0.6,
            label=f"{label} P99",
        )

    ax.set_xlabel("Subsampling Resolution")
    ax.set_ylabel("Validation Step Latency (ms)")
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left", framealpha=0.95, fontsize=8, ncol=2, prop={"weight": "bold"})
    fig.tight_layout(rect=[0, 0, 1, LAYOUT_TOP_WITH_SUBTITLE])
    _finalize_figure(fig)
    _bold_legend(ax)
    fig.savefig(out)
    plt.close(fig)


def _plot_breakdown(points, out: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    all_subs = sorted({p.subsampling for p in points})
    if not all_subs:
        return

    fig, ax = plt.subplots(figsize=(11, 5.8))
    _fig_suptitle(fig, "Inference Latency Breakdown (P50)")
    _fig_subtitle(
        fig,
        "Stacked: min validation step (lower bound) + P50 − min (dataloader variance & tail) · g = 1 · Lustre",
    )

    x = np.arange(len(all_subs))
    width = 0.36
    xlabels = [f"{s:,}" for s in all_subs]

    for offset, model, color in (
        (-width / 2, MODEL_SURFACE, COL_SURFACE),
        (width / 2, MODEL_VOLUME, COL_VOLUME),
    ):
        pts = {p.subsampling: p for p in points_for_model(points, model)}
        mins = [pts[s].min_ms if s in pts else 0 for s in all_subs]
        over = [pts[s].overhead_ms if s in pts else 0 for s in all_subs]
        label = MODEL_LABELS[model]
        ax.bar(
            x + offset,
            mins,
            width,
            label=f"{label} min step",
            color=color,
            alpha=0.45,
            edgecolor="white",
        )
        ax.bar(
            x + offset,
            over,
            width,
            bottom=mins,
            label=f"{label} P50 − min",
            color=color,
            edgecolor="white",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(xlabels)
    ax.set_xlabel("Subsampling Resolution")
    ax.set_ylabel("Latency (ms)")
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left", framealpha=0.95, fontsize=8, prop={"weight": "bold"})
    fig.tight_layout(rect=[0, 0, 1, LAYOUT_TOP_WITH_SUBTITLE])
    _finalize_figure(fig)
    _bold_legend(ax)
    fig.savefig(out)
    plt.close(fig)


def _plot_memory(points, out: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5.8))
    _fig_suptitle(fig, "Peak GPU Memory During Training (Inference Proxy)")
    _fig_subtitle(fig, "Per-rank peak reserved VRAM · g = 1 · Lustre · same runs as validation pass")

    for model, color in (
        (MODEL_SURFACE, COL_SURFACE),
        (MODEL_VOLUME, COL_VOLUME),
    ):
        pts = points_for_model(points, model)
        if not pts:
            continue
        ax.plot(
            [p.subsampling for p in pts],
            [p.peak_gb for p in pts],
            "o-",
            color=color,
            lw=2.5,
            ms=10,
            label=MODEL_LABELS[model],
        )

    ax.axhline(192, color="#59595B", ls="--", lw=1.2, alpha=0.7, label="B200 192 GB ceiling")
    ax.set_xlabel("Subsampling Resolution")
    ax.set_ylabel("Peak GPU Memory (GB)")
    ax.set_ylim(0, 210)
    ax.legend(loc="upper left", framealpha=0.95, fontsize=9, prop={"weight": "bold"})
    fig.tight_layout(rect=[0, 0, 1, LAYOUT_TOP_WITH_SUBTITLE])
    _finalize_figure(fig)
    _bold_legend(ax)
    fig.savefig(out)
    plt.close(fig)



def _box_legend_handles(*, include_p99: bool = True):
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle

    handles = [
        Rectangle(
            (0, 0),
            1,
            1,
            facecolor=BOX_GREEN_LIGHT,
            edgecolor=BOX_GREEN,
            label="IQR (P25–P75)",
        ),
        Line2D(
            [0],
            [0],
            color=BOX_MEDIAN,
            lw=2.5 if not include_p99 else 1.2,
            ls="-" if not include_p99 else "--",
            label="P50 (median)",
        ),
        Line2D([0], [0], color=BOX_MEDIAN, lw=1.2, label="P5-P95 (Whiskers)"),
    ]
    if include_p99:
        handles.insert(
            0,
            Line2D(
                [0],
                [0],
                marker="D",
                color="w",
                markerfacecolor=BOX_GREEN,
                markeredgecolor=BOX_GREEN,
                markersize=8,
                label="P99",
            ),
        )
    return handles


def _box_width(subs_order: tuple[int, ...]) -> float:
    if len(subs_order) < 2:
        return float(subs_order[0]) * 0.35 if subs_order else 5000.0
    span = max(subs_order) - min(subs_order)
    return span * 0.085


def _linear_xlim(subs_order: tuple[int, ...], box_w: float) -> tuple[float, float]:
    """Pad x limits so outer boxes (10k, 300k) and whisker caps are fully visible."""
    x_min, x_max = min(subs_order), max(subs_order)
    half = box_w / 2
    cap = box_w * 0.22  # whisker cap half-width (matches _draw_latency_boxes)
    extra = box_w * 0.15  # margin beyond box + caps
    left = max(0.0, x_min - half - cap - extra)
    right = x_max + half + cap + extra
    return left, right


def _apply_boxplot_yscale(ax, boxes: list[ValLatencyBox]) -> None:
    """Set y limits so IQR boxes stay visible across wide latency ranges.

    Uses a log scale when whisker span exceeds ~2.5× so boxes at low subsampling
    resolutions (small absolute IQR in ms) are not crushed by high-resolution tails.
    """
    import math

    from matplotlib.ticker import FuncFormatter

    p5_min = min(b.p5_ms for b in boxes)
    p99_max = max(b.p99_ms for b in boxes)
    ratio = p99_max / max(p5_min, 1e-6)

    if ratio >= 2.5:
        ax.set_yscale("log")
        y_lo = p5_min * 0.88
        y_hi = p99_max * 1.10
        ax.set_ylim(y_lo, y_hi)

        lo_exp = math.floor(math.log10(y_lo))
        hi_exp = math.ceil(math.log10(y_hi))
        ticks = [
            m * 10**e
            for e in range(lo_exp, hi_exp + 1)
            for m in (1, 2, 5)
            if y_lo <= m * 10**e <= y_hi
        ]
        ax.set_yticks(ticks)
        ax.yaxis.set_major_formatter(
            FuncFormatter(lambda v, _: f"{int(round(v)):,}")
        )
        ax.grid(axis="y", which="major", linestyle="--", alpha=0.35, zorder=0)
        ax.grid(axis="y", which="minor", linestyle=":", alpha=0.2, zorder=0)
    else:
        span = max(p99_max - p5_min, 1.0)
        ax.set_ylim(max(0.0, p5_min - 0.10 * span), p99_max + 0.12 * span)
        ax.grid(axis="y", linestyle="--", alpha=0.35, zorder=0)


def _draw_latency_boxes(
    ax,
    boxes: list[ValLatencyBox],
    subs_order: tuple[int, ...],
    *,
    box_w: float | None = None,
) -> None:
    """Draw box plots at linear subsampling-resolution x positions."""
    from matplotlib.patches import Rectangle

    if box_w is None:
        box_w = _box_width(subs_order)

    by_sub = {b.subsampling: b for b in boxes}
    cap = box_w * 0.22
    for sub in subs_order:
        box = by_sub.get(sub)
        if box is None:
            continue
        x = float(sub)
        ax.vlines(x, box.p5_ms, box.p95_ms, color=BOX_MEDIAN, lw=1.2, zorder=2)
        ax.hlines(box.p5_ms, x - cap, x + cap, color=BOX_MEDIAN, lw=1.2, zorder=2)
        ax.hlines(box.p95_ms, x - cap, x + cap, color=BOX_MEDIAN, lw=1.2, zorder=2)
        iqr_h = max(box.p75_ms - box.p25_ms, 1.0)
        ax.add_patch(
            Rectangle(
                (x - box_w / 2, box.p25_ms),
                box_w,
                iqr_h,
                facecolor=BOX_GREEN_LIGHT,
                edgecolor=BOX_GREEN,
                lw=1.4,
                zorder=3,
            )
        )
        ax.hlines(
            box.p50_ms,
            x - box_w / 2,
            x + box_w / 2,
            color=BOX_MEDIAN,
            lw=1.2,
            ls="--",
            zorder=4,
        )
        ax.plot(
            x,
            box.p99_ms,
            marker="D",
            ms=8,
            color=BOX_GREEN,
            markeredgecolor=BOX_GREEN,
            markeredgewidth=1.0,
            linestyle="none",
            zorder=5,
        )


def _plot_inference_boxplots(
    volume_boxes: list[ValLatencyBox],
    surface_boxes: list[ValLatencyBox],
    subs_order: tuple[int, ...],
    out: Path,
    *,
    num_gpus: int = 1,
    gpu_type: str = "B200",
    storage: str = "nvme",
) -> None:
    """Two-panel box plot: Volume (left) and Surface (right)."""
    import matplotlib.pyplot as plt

    if not volume_boxes and not surface_boxes:
        return

    fig, (ax_vol, ax_surf) = plt.subplots(
        1,
        2,
        figsize=(12, 5.5),
        sharex=True,
        gridspec_kw={"wspace": 0.28},
    )
    _fig_suptitle(fig, "GeoTransolver Inference Latency (validation pass)")
    _fig_subtitle(
        fig,
        _hardware_subtitle(gpu_type=gpu_type, storage=storage),
        y=0.905,
        bold=False,
        italic=False,
    )

    box_w = _box_width(subs_order)
    x_left, x_right = _linear_xlim(subs_order, box_w)

    if volume_boxes:
        _draw_latency_boxes(ax_vol, volume_boxes, subs_order, box_w=box_w)
        _apply_boxplot_yscale(ax_vol, volume_boxes)
    ax_vol.set_ylabel("Latency (ms)")
    ax_vol.set_title("GeoTransolver Volume", fontsize=12, fontweight="bold", pad=10)

    if surface_boxes:
        _draw_latency_boxes(ax_surf, surface_boxes, subs_order, box_w=box_w)
        _apply_boxplot_yscale(ax_surf, surface_boxes)
    ax_surf.set_ylabel("Latency (ms)")
    ax_surf.set_title("GeoTransolver Surface", fontsize=12, fontweight="bold", pad=10)

    for ax in (ax_vol, ax_surf):
        ax.set_xscale("linear")
        ax.set_xticks(list(subs_order))
        ax.set_xticklabels([f"{s:,}" for s in subs_order])
        ax.set_xlim(x_left, x_right)
        ax.set_xlabel("Subsampling Resolution")

    ax_vol.legend(
        handles=_box_legend_handles(),
        loc="upper left",
        framealpha=0.95,
        fontsize=9,
        prop={"weight": "bold"},
    )

    fig.subplots_adjust(top=0.80, wspace=0.28)
    _finalize_figure(fig)
    _bold_tick_labels(ax_vol)
    _bold_tick_labels(ax_surf)
    fig.savefig(out)
    plt.close(fig)


def _plot_volume_boxplot_single(
    volume_boxes: list[ValLatencyBox],
    subs_order: tuple[int, ...],
    out: Path,
    *,
    num_gpus: int = 1,
    gpu_type: str = "B200",
    storage: str = "nvme",
) -> None:
    """Single-panel Volume box plot with shared x/y axes across all resolutions."""
    import matplotlib.pyplot as plt

    if not volume_boxes:
        return

    fig, ax = plt.subplots(figsize=(8, 6.5))
    _fig_suptitle(fig, "GeoTransolver Volume Inference Latency (validation pass)")
    _fig_subtitle(
        fig,
        _hardware_subtitle(gpu_type=gpu_type, storage=storage),
        y=0.905,
        bold=False,
        italic=False,
    )

    box_w = _box_width(subs_order)
    x_left, x_right = _linear_xlim(subs_order, box_w)
    _draw_latency_boxes(ax, volume_boxes, subs_order, box_w=box_w)
    _apply_boxplot_yscale(ax, volume_boxes)
    ax.set_ylabel("Latency (ms)")
    ax.set_xscale("linear")
    ax.set_xticks(list(subs_order))
    ax.set_xticklabels([f"{s:,}" for s in subs_order])
    ax.set_xlim(x_left, x_right)
    ax.set_xlabel("Subsampling Resolution")
    ax.legend(
        handles=_box_legend_handles(),
        loc="upper left",
        framealpha=0.95,
        fontsize=9,
        prop={"weight": "bold"},
    )

    fig.subplots_adjust(top=0.82)
    _finalize_figure(fig)
    _bold_tick_labels(ax)
    fig.savefig(out)
    plt.close(fig)


def _draw_throughput_boxes(
    ax,
    boxes: list[ValThroughputBox],
    subs_order: tuple[int, ...],
    *,
    box_w: float = 0.55,
) -> None:
    """Categorical box plot for validation-step throughput (samples/s)."""
    from matplotlib.patches import Rectangle

    by_sub = {b.subsampling: b for b in boxes}
    cap = box_w * 0.22
    for i, sub in enumerate(subs_order):
        box = by_sub.get(sub)
        if box is None:
            continue
        x = float(i)
        ax.vlines(x, box.p5, box.p95, color=BOX_MEDIAN, lw=1.2, zorder=2)
        ax.hlines(box.p5, x - cap, x + cap, color=BOX_MEDIAN, lw=1.2, zorder=2)
        ax.hlines(box.p95, x - cap, x + cap, color=BOX_MEDIAN, lw=1.2, zorder=2)
        iqr_h = max(box.p75 - box.p25, 1e-6)
        ax.add_patch(
            Rectangle(
                (x - box_w / 2, box.p25),
                box_w,
                iqr_h,
                facecolor=BOX_GREEN_LIGHT,
                edgecolor=BOX_GREEN,
                lw=1.4,
                zorder=3,
            )
        )
        ax.hlines(
            box.p50,
            x - box_w / 2,
            x + box_w / 2,
            color=BOX_MEDIAN,
            lw=2.5,
            zorder=4,
        )


def _apply_throughput_yscale(ax, boxes: list[ValThroughputBox]) -> None:
    p5_min = min(b.p5 for b in boxes)
    p95_max = max(b.p95 for b in boxes)
    span = max(p95_max - p5_min, 0.1)
    ax.set_ylim(max(0.0, p5_min - 0.12 * span), p95_max + 0.15 * span)
    ax.grid(axis="y", linestyle="--", alpha=0.35, zorder=0)
    ax.grid(axis="x", linestyle="--", alpha=0.25, zorder=0)


def _plot_inference_throughput_single(
    boxes: list[ValThroughputBox],
    subs_order: tuple[int, ...],
    out: Path,
    *,
    model: str,
    gpu_type: str = "B200",
    storage: str = "nvme",
) -> None:
    """Single-panel inference throughput box plot (validation pass)."""
    import matplotlib.pyplot as plt

    if not boxes:
        return

    label = MODEL_LABELS[model]
    fig, ax = plt.subplots(figsize=(8, 6.5))
    _fig_suptitle(fig, f"{label} Inference Throughput (validation pass)")
    _fig_subtitle(
        fig,
        _hardware_subtitle(gpu_type=gpu_type, storage=storage),
        y=0.905,
        bold=False,
        italic=False,
    )

    present_subs = tuple(s for s in subs_order if any(b.subsampling == s for b in boxes))
    _draw_throughput_boxes(ax, boxes, present_subs)
    _apply_throughput_yscale(ax, boxes)

    ax.set_xticks(range(len(present_subs)))
    ax.set_xticklabels([f"{s:,}" for s in present_subs])
    ax.set_xlabel("Subsampling Resolution")
    ax.set_ylabel("Throughput (samples/s)")
    ax.legend(
        handles=_box_legend_handles(include_p99=False),
        loc="upper left",
        framealpha=0.95,
        fontsize=9,
        prop={"weight": "bold"},
    )

    fig.subplots_adjust(top=0.82)
    _finalize_figure(fig)
    _bold_tick_labels(ax)
    fig.savefig(out)
    plt.close(fig)


def _write_box_json(
    volume_boxes: list[ValLatencyBox],
    surface_boxes: list[ValLatencyBox],
    out: Path,
) -> None:
    def _rows(boxes: list[ValLatencyBox]) -> list[dict]:
        return [
            {
                "subsampling": b.subsampling,
                "p5_ms": round(b.p5_ms, 2),
                "p25_ms": round(b.p25_ms, 2),
                "p50_ms": round(b.p50_ms, 2),
                "p75_ms": round(b.p75_ms, 2),
                "p95_ms": round(b.p95_ms, 2),
                "p99_ms": round(b.p99_ms, 2),
                "n_steps": b.n_steps,
            }
            for b in boxes
        ]

    payload = {
        "volume": _rows(volume_boxes),
        "surface": _rows(surface_boxes),
    }
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_throughput_json(
    volume_boxes: list[ValThroughputBox],
    surface_boxes: list[ValThroughputBox],
    out: Path,
) -> None:
    def _rows(boxes: list[ValThroughputBox]) -> list[dict]:
        return [
            {
                "subsampling": b.subsampling,
                "p5_samples_per_sec": round(b.p5, 3),
                "p25_samples_per_sec": round(b.p25, 3),
                "p50_samples_per_sec": round(b.p50, 3),
                "p75_samples_per_sec": round(b.p75, 3),
                "p95_samples_per_sec": round(b.p95, 3),
                "n_steps": b.n_steps,
            }
            for b in boxes
        ]

    payload = {
        "volume": _rows(volume_boxes),
        "surface": _rows(surface_boxes),
    }
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_json(points, out: Path) -> None:
    payload = [
        {
            "model": p.model,
            "model_label": p.model_label,
            "subsampling": p.subsampling,
            "p50_ms": round(p.p50_ms, 2),
            "p95_ms": round(p.p95_ms, 2),
            "p99_ms": round(p.p99_ms, 2),
            "min_ms": round(p.min_ms, 2),
            "overhead_ms": round(p.overhead_ms, 2),
            "train_p50_ms": round(p.train_p50_ms, 2),
            "peak_gb": round(p.peak_gb, 2),
            "val_estimated": p.val_estimated,
        }
        for p in points
    ]
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/_scaling_snapshot"),
    )
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument(
        "--gpu-type",
        default="B200",
        help="GPU SKU label for plot subtitles (default: B200)",
    )
    parser.add_argument("--storage", default="lustre")
    args = parser.parse_args()

    recipe_root = Path(__file__).resolve().parent.parent
    results_root = args.results if args.results.is_absolute() else recipe_root / args.results
    out_dir = args.out_dir if args.out_dir.is_absolute() else recipe_root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    _apply_style()
    summaries = load_summaries(results_root)
    points = collect_inference_points(
        summaries, num_gpus=args.num_gpus, storage=args.storage
    )
    if not points and args.storage == "lustre":
        points = collect_inference_points(
            summaries, num_gpus=args.num_gpus, storage="nvme"
        )

    vol_boxes = collect_val_latency_boxes(
        results_root,
        model=MODEL_VOLUME,
        num_gpus=args.num_gpus,
        storage="nvme",
        subs=INFERENCE_BOX_SUBS,
    )
    surf_boxes = collect_val_latency_boxes(
        results_root,
        model=MODEL_SURFACE,
        num_gpus=args.num_gpus,
        storage="nvme",
        subs=INFERENCE_BOX_SUBS,
    )

    if not points and not vol_boxes and not surf_boxes:
        lustre_probe = collect_val_latency_boxes(
            results_root,
            model=MODEL_VOLUME,
            num_gpus=args.num_gpus,
            storage="lustre",
            subs=INFERENCE_BOX_SUBS,
        )
        if not lustre_probe:
            raise SystemExit(
                "no inference data found (need g=1 summaries with val_step data)"
            )

    if points:
        _plot_latency(points, out_dir / OUT_FILES["latency"])
        _plot_breakdown(points, out_dir / OUT_FILES["breakdown"])
        _plot_memory(points, out_dir / OUT_FILES["memory"])
        _write_json(points, out_dir / "inference_metrics.json")

    if vol_boxes or surf_boxes:
        _plot_inference_boxplots(
            vol_boxes,
            surf_boxes,
            INFERENCE_BOX_SUBS,
            out_dir / OUT_FILES["inference_box_nvme"],
            num_gpus=args.num_gpus,
            gpu_type=args.gpu_type,
            storage="nvme",
        )
        _write_box_json(vol_boxes, surf_boxes, out_dir / "inference_box_nvme.json")
        _plot_volume_boxplot_single(
            vol_boxes,
            INFERENCE_BOX_SUBS,
            out_dir / OUT_FILES["volume_box_nvme"],
            num_gpus=args.num_gpus,
            gpu_type=args.gpu_type,
            storage="nvme",
        )

    vol_thr = collect_val_throughput_boxes(
        results_root,
        model=MODEL_VOLUME,
        num_gpus=args.num_gpus,
        storage="nvme",
        subs=INFERENCE_THROUGHPUT_SUBS,
    )
    surf_thr = collect_val_throughput_boxes(
        results_root,
        model=MODEL_SURFACE,
        num_gpus=args.num_gpus,
        storage="nvme",
        subs=INFERENCE_THROUGHPUT_SUBS,
    )
    if vol_thr:
        _plot_inference_throughput_single(
            vol_thr,
            INFERENCE_THROUGHPUT_SUBS,
            out_dir / OUT_FILES["volume_throughput_nvme"],
            model=MODEL_VOLUME,
            gpu_type=args.gpu_type,
            storage="nvme",
        )
    if surf_thr:
        _plot_inference_throughput_single(
            surf_thr,
            INFERENCE_THROUGHPUT_SUBS,
            out_dir / OUT_FILES["surface_throughput_nvme"],
            model=MODEL_SURFACE,
            gpu_type=args.gpu_type,
            storage="nvme",
        )
    if vol_thr or surf_thr:
        _write_throughput_json(
            vol_thr, surf_thr, out_dir / "inference_throughput_nvme.json"
        )

    lustre_vol = collect_val_latency_boxes(
        results_root,
        model=MODEL_VOLUME,
        num_gpus=args.num_gpus,
        storage="lustre",
        subs=INFERENCE_BOX_SUBS,
    )
    lustre_surf = collect_val_latency_boxes(
        results_root,
        model=MODEL_SURFACE,
        num_gpus=args.num_gpus,
        storage="lustre",
        subs=INFERENCE_BOX_SUBS,
    )
    if lustre_vol or lustre_surf:
        _plot_inference_boxplots(
            lustre_vol,
            lustre_surf,
            INFERENCE_BOX_SUBS,
            out_dir / OUT_FILES["inference_box_lustre"],
            num_gpus=args.num_gpus,
            gpu_type=args.gpu_type,
            storage="lustre",
        )
        _write_box_json(lustre_vol, lustre_surf, out_dir / "inference_box_lustre.json")

    n_surface = len(points_for_model(points, MODEL_SURFACE)) if points else 0
    n_volume = len(points_for_model(points, MODEL_VOLUME)) if points else 0
    box_notes: list[str] = []
    if vol_boxes or surf_boxes:
        box_notes.append(f"NVMe box (Vol={len(vol_boxes)}, Surf={len(surf_boxes)})")
    if vol_thr or surf_thr:
        box_notes.append(
            f"NVMe throughput (Vol={len(vol_thr)}, Surf={len(surf_thr)})"
        )
    if lustre_vol or lustre_surf:
        box_notes.append(f"Lustre box (Vol={len(lustre_vol)}, Surf={len(lustre_surf)})")
    box_note = f", {', '.join(box_notes)}" if box_notes else ""
    print(
        f"[inference] wrote 3–8 plots + inference_metrics.json "
        f"(Surface={n_surface} subs, Volume={n_volume} subs{box_note}) -> {out_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
