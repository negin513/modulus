#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""DoMINO-style training-step latency chart: PhysicsNeMo vs PyTorch baseline.

PhysicsNeMo bars use measured median train-step time (P50, epochs 1–4) from
``benchmark_summary.json``.  PyTorch bars come from either:

* ``--pytorch-json`` — explicit ``{subsampling: ms}`` measurements (e.g. from
  ``profile_and_attribute.py --mode measure``), or
* default Amdahl estimate using profiler kernel fraction + ASV layer speedup
  (see ``END_TO_END_TRAINING_PERFORMANCE_REPORT.md`` §1.2).
"""

from __future__ import annotations

import sys
from pathlib import Path

_BENCHMARKS_ROOT = Path(__file__).resolve().parents[1]
if str(_BENCHMARKS_ROOT) not in sys.path:
    sys.path.insert(0, str(_BENCHMARKS_ROOT))
import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

from plots.plot_scaling_snapshot import (
    LAYOUT_TOP_WITH_SUBTITLE,
    _apply_style,
    _fig_subtitle,
    _fig_suptitle,
    _finalize_figure,
)

COL_PYTORCH = "#111111"
COL_PNM = "#76B900"
COL_SPEEDUP = "#76B900"

DEFAULT_SUBS = (10_000, 50_000, 100_000, 200_000, 300_000)
OUT_FILE = "13_geotransolver_volume_training_latency.png"


@dataclass(frozen=True)
class LatencyPoint:
    subsampling: int
    pnm_ms: float
    pytorch_ms: float
    pytorch_estimated: bool

    @property
    def speedup(self) -> float:
        if self.pnm_ms <= 0:
            return float("nan")
        return self.pytorch_ms / self.pnm_ms


def _load_pnm_train_p50_ms(
    results_root: Path,
    *,
    model: str,
    num_gpus: int,
    storage: str,
) -> dict[int, float]:
    """Return ``{subsampling -> train P50 ms}`` from benchmark summaries."""

    out: dict[int, float] = {}
    for path in results_root.rglob("benchmark_summary.json"):
        if "_smoketest" in str(path):
            continue
        summary = json.loads(path.read_text())
        if summary.get("model") != model:
            continue
        if summary.get("num_gpus") != num_gpus:
            continue
        if summary.get("storage") != storage:
            continue
        sub = int(summary["sampling_resolution"])
        p50_s = summary.get("train", {}).get("p50")
        if p50_s is None:
            continue
        out[sub] = float(p50_s) * 1e3
    return out


def _estimate_pytorch_ms(
    pnm_ms: float,
    subsampling: int,
    *,
    kernel_fraction_ref: float,
    kernel_speedup: float,
    ref_sub: int,
) -> float:
    """Amdahl estimate: spatial-kernel fraction grows with subsampling."""

    scale = subsampling / ref_sub
    f_kernel = min(max(kernel_fraction_ref * (scale**0.85), 0.06), 0.42)
    overall = 1.0 / (1.0 - f_kernel + f_kernel / max(kernel_speedup, 1.0))
    return pnm_ms * overall


def _load_pytorch_json(path: Path) -> dict[int, float]:
    payload = json.loads(path.read_text())
    if isinstance(payload, list):
        return {int(row["subsampling"]): float(row["pytorch_ms"]) for row in payload}
    return {int(k): float(v) for k, v in payload.items()}


def _load_measure_summary(path: Path, subs: tuple[int, ...]) -> list[LatencyPoint]:
    """Build latency points from ``sweep_pytorch_measure`` summary rows."""

    payload = json.loads(path.read_text())
    if not isinstance(payload, list):
        raise ValueError(f"expected list in {path}")
    by_sub = {int(row["subsampling"]): row for row in payload}
    points: list[LatencyPoint] = []
    for sub in subs:
        row = by_sub.get(sub)
        if row is None:
            continue
        points.append(
            LatencyPoint(
                subsampling=sub,
                pnm_ms=float(row["pnm_ms"]),
                pytorch_ms=float(row["pytorch_ms"]),
                pytorch_estimated=False,
            )
        )
    return points


def collect_points(
    results_root: Path,
    *,
    model: str,
    num_gpus: int,
    storage: str,
    subs: tuple[int, ...],
    pytorch_json: Path | None,
    kernel_fraction_ref: float,
    kernel_speedup: float,
    ref_sub: int,
) -> list[LatencyPoint]:
    pnm_by_sub = _load_pnm_train_p50_ms(
        results_root, model=model, num_gpus=num_gpus, storage=storage
    )
    pytorch_measured = _load_pytorch_json(pytorch_json) if pytorch_json else {}

    points: list[LatencyPoint] = []
    for sub in subs:
        if sub not in pnm_by_sub:
            continue
        pnm_ms = pnm_by_sub[sub]
        if sub in pytorch_measured:
            pytorch_ms = pytorch_measured[sub]
            estimated = False
        else:
            pytorch_ms = _estimate_pytorch_ms(
                pnm_ms,
                sub,
                kernel_fraction_ref=kernel_fraction_ref,
                kernel_speedup=kernel_speedup,
                ref_sub=ref_sub,
            )
            estimated = True
        points.append(
            LatencyPoint(
                subsampling=sub,
                pnm_ms=pnm_ms,
                pytorch_ms=pytorch_ms,
                pytorch_estimated=estimated,
            )
        )
    return points


def _format_speedup(x: float) -> str:
    if not math.isfinite(x) or x < 1.05:
        return ""
    if x >= 10:
        return f"{x:.0f}x"
    return f"{x:.1f}x"


def _format_ms(x: float) -> str:
    if x >= 10_000:
        return f"{x / 1000:.0f}k"
    if x >= 1000:
        return f"{x / 1000:.1f}k"
    return f"{x:.0f}"


def plot_volume_latency(
    points: list[LatencyPoint],
    out_path: Path,
    *,
    title: str,
    subtitle: str,
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    if not points:
        raise SystemExit("no latency points to plot")

    subs = [p.subsampling for p in points]
    x = np.arange(len(subs))
    width = 0.36

    fig, ax = plt.subplots(figsize=(11, 6.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    pytorch_vals = [p.pytorch_ms for p in points]
    pnm_vals = [p.pnm_ms for p in points]

    ax.bar(
        x - width / 2,
        pytorch_vals,
        width,
        color=COL_PYTORCH,
        label="PyTorch",
        zorder=3,
    )
    ax.bar(
        x + width / 2,
        pnm_vals,
        width,
        color=COL_PNM,
        label="PhysicsNeMo",
        zorder=3,
    )

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s:,}" for s in subs])
    ax.set_xlabel("Number of Points")
    ax.set_ylabel("Time (ms)")
    ax.grid(axis="y", which="both", linestyle=":", color="#CCCCCC", alpha=0.8)
    ax.legend(frameon=False, loc="upper left")

    y_top = max(pytorch_vals + pnm_vals) * 1.35
    y_lo = min(pnm_vals) * 0.55
    ax.set_ylim(y_lo, y_top)

    for i, pt in enumerate(points):
        ax.text(
            x[i] + width / 2,
            pt.pnm_ms * 1.08,
            _format_ms(pt.pnm_ms),
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color=COL_PNM,
        )
        label = _format_speedup(pt.speedup)
        if label:
            ax.text(
                x[i] - width / 2,
                pt.pytorch_ms * 1.08,
                label,
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
                color=COL_SPEEDUP,
            )

    _fig_suptitle(fig, title)
    ax.set_title("")
    _fig_subtitle(fig, subtitle, bold=False, italic=True)
    fig.subplots_adjust(top=LAYOUT_TOP_WITH_SUBTITLE)
    _finalize_figure(fig)
    fig.savefig(out_path, dpi=150, facecolor="white")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/_scaling_snapshot"),
    )
    parser.add_argument("--model", default="geotransolver_volume")
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--storage", default="nvme", choices=("nvme", "lustre"))
    parser.add_argument(
        "--subs",
        type=int,
        nargs="+",
        default=list(DEFAULT_SUBS),
        help="Subsampling levels to include (must exist in summaries)",
    )
    parser.add_argument(
        "--measure-summary",
        type=Path,
        default=None,
        help=(
            "Use measured PNM + PyTorch bars from pytorch_measure_summary.json "
            "(output of sweep_pytorch_measure.py)."
        ),
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Figure suptitle override",
    )
    parser.add_argument(
        "--dataset-label",
        default="DrivAerML",
        help="Dataset name shown in subtitle",
    )
    parser.add_argument(
        "--pytorch-json",
        type=Path,
        default=None,
        help=(
            "Measured PyTorch step times: {subsampling: ms} or "
            "[{subsampling, pytorch_ms}]. Default: results/_profile_attribute/"
            "pytorch_measured.json when present."
        ),
    )
    parser.add_argument(
        "--kernel-fraction-ref",
        type=float,
        default=0.33,
        help=(
            "Profiler CUDA fraction for all PNM-accelerated kernels @ ref_sub "
            "(default: 0.33 — Ball Query + BVH + interp + grad)"
        ),
    )
    parser.add_argument(
        "--kernel-speedup",
        type=float,
        default=15.0,
        help="Effective geom-mean ASV speedup vs torch @ ref_sub (default: 15)",
    )
    parser.add_argument(
        "--ref-sub",
        type=int,
        default=200_000,
        help="Subsampling level for kernel_fraction_ref (default: 200k)",
    )
    args = parser.parse_args()

    recipe_root = Path(__file__).resolve().parent.parent
    results_root = args.results if args.results.is_absolute() else recipe_root / args.results
    out_dir = args.out_dir if args.out_dir.is_absolute() else recipe_root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.pytorch_json is None:
        default_pytorch = results_root / "_profile_attribute" / "pytorch_measured.json"
        if default_pytorch.is_file():
            args.pytorch_json = default_pytorch

    _apply_style()
    subs = tuple(args.subs)
    if args.measure_summary is not None:
        measure_path = (
            args.measure_summary
            if args.measure_summary.is_absolute()
            else recipe_root / args.measure_summary
        )
        points = _load_measure_summary(measure_path, subs)
    else:
        points = collect_points(
            results_root,
            model=args.model,
            num_gpus=args.num_gpus,
            storage=args.storage,
            subs=subs,
            pytorch_json=args.pytorch_json,
            kernel_fraction_ref=args.kernel_fraction_ref,
            kernel_speedup=args.kernel_speedup,
            ref_sub=args.ref_sub,
        )
    if not points:
        raise SystemExit(
            f"no latency points for subs={subs} "
            f"(measure_summary={args.measure_summary}, results={results_root})"
        )

    any_estimated = any(p.pytorch_estimated for p in points)
    storage_label = args.storage.upper()
    subtitle = (
        f"Measured train-step wall time · 1× B200 · {args.dataset_label} · {storage_label} · "
        f"batch_size=1 · "
        + (
            "PyTorch bars estimated via Amdahl (PNM-accelerated kernel fraction × ASV speedup)."
            if any_estimated
            else "PyTorch = torch RS baseline; PhysicsNeMo = warp RS default (rank 0/1)."
        )
    )
    title = args.title or "GeoTransolver Volume Model Latency (Training)"

    out_path = out_dir / OUT_FILE
    plot_volume_latency(points, out_path, title=title, subtitle=subtitle)

    payload = [
        {
            "subsampling": p.subsampling,
            "pnm_ms": round(p.pnm_ms, 2),
            "pytorch_ms": round(p.pytorch_ms, 2),
            "speedup": round(p.speedup, 2),
            "pytorch_estimated": p.pytorch_estimated,
        }
        for p in points
    ]
    json_path = out_dir / OUT_FILE.replace(".png", ".json")
    json_path.write_text(json.dumps(payload, indent=2))

    print(f"[plot_model_latency] wrote {out_path} ({len(points)} points)")
    print(f"[plot_model_latency] wrote {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
