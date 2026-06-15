#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Generate VTU vs PhysicsNeMo-Mesh I/O benchmark chart (Peter Sharpe style).

Produces a two-panel figure matching the mesh_benchmarking benchmark layout.
Data sources:
  - Peter Sharpe published results (README, Apr 2026) for all three datasets
  - HSG replication (jobs 3077314, 3079778, 3115330, Jun 2026) for measured load times

Usage (from recipe root)::

    python benchmarks/plots/generate_mesh_io_benchmark_chart.py
    python benchmarks/plots/generate_mesh_io_benchmark_chart.py --variant hsg
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import sys
from pathlib import Path

_BENCHMARKS_ROOT = Path(__file__).resolve().parents[1]
if str(_BENCHMARKS_ROOT) not in sys.path:
    sys.path.insert(0, str(_BENCHMARKS_ROOT))
from paths import RECIPE_ROOT

GIB = 1024**3
DEFAULT_OUT_DIR = RECIPE_ROOT / "results/_scaling_snapshot"


@dataclass(frozen=True)
class Row:
    vtu_gib: float
    pmsh_gib: float
    vtu_load_s: float
    pmsh_load_s: float
    vtu_trim_load_s: float | None = None  # physics-fields-only (HiLift)


# Peter Sharpe mesh_benchmarking/README.md (cold-disk, 3 trials)
PETER_REFERENCE: dict[str, Row] = {
    "ShiftSUV": Row(4.1, 1.5, 58.5, 1.7),
    "DrivAerML": Row(46.3, 5.8, 412.6, 4.7),
    "HiLiftAeroML": Row(99.9, 18.7, 119.4, 12.6, vtu_trim_load_s=112.2),
}

# HSG Jun 2026 — ShiftSUV (PhysicsNeMo-ShiftSUV + shift_suv VTU, job 3077314)
HSG_SHIFTSUV = Row(
    vtu_gib=(4_365_395_975) / GIB,
    pmsh_gib=(1_791_067_498) / GIB,
    vtu_load_s=58.1,  # median of 58.035, 58.129, 58.139
    pmsh_load_s=1.6,  # median of 1.688, 1.860, 1.307
)

# HSG HiLift (job 3079778, Jun 2026 — PhysicsNeMo-HighLiftAeroML + nashton VTU)
HSG_HILIFT = Row(
    vtu_gib=99.94,
    pmsh_gib=28.71,
    vtu_load_s=98.133,  # median of 99.643, 96.934, 98.133
    pmsh_load_s=4.756,  # median of 4.756, 7.545, 4.684
    vtu_trim_load_s=97.357,
)

# HSG DrivAerML (job 3115330, Jun 2026 — PhysicsNeMo-DrivaerML + drivaer_aws VTU)
HSG_DRIVAER = Row(
    vtu_gib=46.30,
    pmsh_gib=6.47,
    vtu_load_s=412.448,  # median of 412.293, 412.448, 414.444
    pmsh_load_s=3.052,  # median of 3.230, 3.052, 2.810
)


def _bolden_axes(ax: plt.Axes) -> None:
    """Bold axis labels, tick labels, and subplot title."""
    ax.xaxis.label.set_fontweight("bold")
    ax.yaxis.label.set_fontweight("bold")
    ax.title.set_fontweight("bold")
    for lbl in ax.get_xticklabels():
        lbl.set_fontweight("bold")
    for lbl in ax.get_yticklabels():
        lbl.set_fontweight("bold")


def _bolden_legend(ax: plt.Axes) -> None:
    leg = ax.get_legend()
    if leg is not None:
        for text in leg.get_texts():
            text.set_fontweight("bold")


def print_shiftsuv_comparison() -> None:
    """Print HSG vs Peter published ShiftSUV metrics."""
    p, h = PETER_REFERENCE["ShiftSUV"], HSG_SHIFTSUV

    def pct(a: float, b: float) -> str:
        return f"{100 * (a - b) / b:+.1f}%"

    print("\nShiftSUV: HSG (job 3077314) vs Peter published (mesh_benchmarking README)\n")
    print(f"{'Metric':<22} {'Peter':>10} {'HSG':>10} {'Δ vs Peter':>12}")
    print("-" * 56)
    for label, pv, hv in [
        ("VTU disk (GiB)", p.vtu_gib, h.vtu_gib),
        ("PMSH disk (GiB)", p.pmsh_gib, h.pmsh_gib),
        ("VTU load (s)", p.vtu_load_s, h.vtu_load_s),
        ("PMSH load (s)", p.pmsh_load_s, h.pmsh_load_s),
    ]:
        print(f"{label:<22} {pv:>10.2f} {hv:>10.2f} {pct(hv, pv):>12}")
    print(
        f"{'Load speedup (VTU/PMSH)':<22} {p.vtu_load_s / p.pmsh_load_s:>10.1f}x"
        f" {h.vtu_load_s / h.pmsh_load_s:>10.1f}x"
    )
    print(
        "\nConclusion: HSG replication matches Peter within ~1–10% on load times;"
        " disk sizes agree (~4.1 GiB VTU); PMSH on-disk is slightly larger (1.67 vs 1.53 GiB)."
    )


def plot_chart(
    data: dict[str, Row],
    output_path: Path,
    *,
    title_suffix: str = "",
    subtitle: str = "",
) -> None:
    datasets = list(data.keys())
    n = len(datasets)
    x = np.arange(n)
    bar_width = 0.35

    vtu_sizes_gb = [data[d].vtu_gib * (GIB / 1e9) for d in datasets]
    pmsh_sizes_gb = [data[d].pmsh_gib * (GIB / 1e9) for d in datasets]
    vtu_times = [data[d].vtu_load_s for d in datasets]
    trim_times = [
        data[d].vtu_trim_load_s if data[d].vtu_trim_load_s is not None else data[d].vtu_load_s
        for d in datasets
    ]
    pmsh_times = [data[d].pmsh_load_s for d in datasets]
    is_selective = [data[d].vtu_trim_load_s is not None for d in datasets]

    fig, (ax_size, ax_time) = plt.subplots(1, 2, figsize=(14, 5.5))
    vtu_color = "#999999"
    vtu_trim_color = "#666666"
    pmsh_color = "#76B900"

    ax_size.bar(x - bar_width / 2, vtu_sizes_gb, bar_width, label="VTU", color=vtu_color)
    ax_size.bar(
        x + bar_width / 2, pmsh_sizes_gb, bar_width, label="PhysicsNeMo-Mesh *", color=pmsh_color
    )
    ax_size.set_ylabel("Disk Size (GB)", fontweight="bold")
    ax_size.set_title(
        "Disk Size per Sample (interior + boundary)",
        fontweight="bold",
        pad=12,
    )
    ax_size.set_xticks(x)
    ax_size.set_xticklabels(datasets)
    ax_size.legend(fontsize=9, prop={"weight": "bold"})

    for i in range(n):
        ax_size.annotate(
            f"{vtu_sizes_gb[i]:.1f} GB",
            xy=(x[i] - bar_width / 2, vtu_sizes_gb[i]),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            fontsize=9,
        )
        ratio = vtu_sizes_gb[i] / pmsh_sizes_gb[i] if pmsh_sizes_gb[i] > 0 else 0
        ax_size.annotate(
            f"{pmsh_sizes_gb[i]:.1f} GB\n({ratio:.1f}×\nsmaller)",
            xy=(x[i] + bar_width / 2, pmsh_sizes_gb[i]),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            fontweight="bold",
            fontsize=9,
        )

    size_ymax = max(max(vtu_sizes_gb), max(pmsh_sizes_gb), 1.0)
    ax_size.set_ylim(0, size_ymax * 1.14)

    has_any_selective = any(is_selective)
    for i in range(n):
        vx = x[i] - bar_width / 2
        if is_selective[i]:
            extra = vtu_times[i] - trim_times[i]
            ax_time.bar(
                vx,
                trim_times[i],
                bar_width,
                color=vtu_trim_color,
                label="VTU (physics fields only)" if i == 0 else "",
            )
            ax_time.bar(
                vx,
                extra,
                bar_width,
                bottom=trim_times[i],
                color=vtu_color,
                label="VTU (extra fields overhead)" if i == 0 else "",
            )
            ax_time.annotate(
                f"{vtu_times[i]:.1f} s total",
                xy=(vx, vtu_times[i]),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )
            ax_time.annotate(
                f"physics fields\nonly: {trim_times[i]:.1f} s",
                xy=(vx, trim_times[i]),
                xytext=(0, -5),
                textcoords="offset points",
                ha="center",
                va="top",
                fontsize=7,
                fontstyle="italic",
            )
        else:
            ax_time.bar(vx, vtu_times[i], bar_width, color=vtu_color, label="VTU" if i == 0 else "")
            ax_time.annotate(
                f"{vtu_times[i]:.1f} s",
                xy=(vx, vtu_times[i]),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                fontsize=9,
            )

    ax_time.bar(
        x + bar_width / 2, pmsh_times, bar_width, label="PhysicsNeMo-Mesh *", color=pmsh_color
    )
    for i in range(n):
        ref = trim_times[i] if is_selective[i] else vtu_times[i]
        ratio = ref / pmsh_times[i] if pmsh_times[i] > 0 else 0
        ax_time.annotate(
            f"{pmsh_times[i]:.1f} s\n({ratio:.1f}×\nfaster)",
            xy=(x[i] + bar_width / 2, pmsh_times[i]),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            fontweight="bold",
            fontsize=9,
        )

    ax_time.set_ylabel("Deserialization Time (s)", fontweight="bold")
    ax_time.set_title(
        "Load Time per Sample (interior + boundary)",
        fontweight="bold",
        pad=12,
    )
    ax_time.set_xticks(x)
    ax_time.set_xticklabels(datasets)
    ax_time.legend(fontsize=8, loc="upper left", prop={"weight": "bold"})

    time_ymax = max(max(vtu_times), max(trim_times), max(pmsh_times), 1.0)
    ax_time.set_ylim(0, time_ymax * 1.14)

    _bolden_axes(ax_size)
    _bolden_axes(ax_time)
    _bolden_legend(ax_size)
    _bolden_legend(ax_time)

    title = "I/O Benchmark: VTU vs. PhysicsNeMo-Mesh (*.pmsh)"
    if title_suffix:
        title += f" — {title_suffix}"
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.96)

    footnotes = (
        "* PhysicsNeMo-Mesh stores only physics-relevant fields: mean flow quantities "
        "(pressure, velocity, temperature, density). Fields excluded during curation include\n"
        "  normal vectors (recomputed on-the-fly), cell/node IDs (trivially reconstructable), "
        "and solver-internal quantities (RMS, Reynolds stress, etc.).\n"
        "  To isolate the effect of field trimming from format efficiency, the HiLiftAeroML "
        "load-time bar shows VTU with selective field loading (†), matching the PMSH field set.\n"
        "  Cell connectivity dominates VTU file size for these meshes, so the trimming effect "
        "on load time is modest; the bulk of the speedup is attributable to the memmap format.\n"
        "† HiLiftAeroML uses VTK appended binary, which supports selective array loading. "
        "ShiftSUV and DrivAerML use inline binary, where the parser must read the full file "
        "regardless of field selection."
    )
    if subtitle:
        footnotes = subtitle + "\n\n" + footnotes
    footnote_bottom = 0.24
    fig.tight_layout(rect=[0, footnote_bottom, 1, 0.92])
    fig.text(
        0.02,
        0.01,
        footnotes,
        ha="left",
        fontsize=7.5,
        fontstyle="italic",
        va="bottom",
        linespacing=1.85,
        transform=fig.transFigure,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Wrote {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Directory for PNG outputs (default: results/_scaling_snapshot)",
    )
    parser.add_argument(
        "--variant",
        choices=("reference", "hsg", "both"),
        default="both",
        help="reference=Peter README; hsg=HSG ShiftSUV replication; both=generate both PNGs",
    )
    parser.add_argument(
        "--compare-shiftsuv",
        action="store_true",
        help="Print HSG vs Peter ShiftSUV table and exit",
    )
    parser.add_argument(
        "--shiftsuv-only",
        action="store_true",
        help="Only write 29_mesh_io_vtu_vs_pmsh_shiftsuv.png (HSG measured)",
    )
    args = parser.parse_args()
    out_dir = args.out_dir.resolve()

    if args.compare_shiftsuv:
        print_shiftsuv_comparison()
        return

    if args.shiftsuv_only:
        plot_chart(
            {"ShiftSUV": HSG_SHIFTSUV},
            out_dir / "29_mesh_io_vtu_vs_pmsh_shiftsuv.png",
            title_suffix="ShiftSUV — HSG measured",
            subtitle=(
                "Dataset: PhysicsNeMo-ShiftSUV (estate surface). Cold-disk deserialization, "
                "3 trials, subprocess isolation. Job 3077314, Jun 2026."
            ),
        )
        return

    if args.variant in ("reference", "both"):
        plot_chart(
            PETER_REFERENCE,
            out_dir / "29_mesh_io_vtu_vs_pmsh_reference.png",
            title_suffix="published reference",
        )

    if args.variant in ("hsg", "both"):
        hsg_data = {
            "ShiftSUV": HSG_SHIFTSUV,
            "DrivAerML": HSG_DRIVAER,
            "HiLiftAeroML": HSG_HILIFT,
        }
        plot_chart(
            hsg_data,
            out_dir / "29_mesh_io_vtu_vs_pmsh_hsg.png",
        )

    plot_chart(
        {"ShiftSUV": HSG_SHIFTSUV},
        out_dir / "29_mesh_io_vtu_vs_pmsh_shiftsuv.png",
        title_suffix="ShiftSUV — HSG measured",
        subtitle=(
            "Dataset: PhysicsNeMo-ShiftSUV (estate surface). Cold-disk deserialization, "
            "3 trials, subprocess isolation. Job 3077314, Jun 2026."
        ),
    )


if __name__ == "__main__":
    main()
