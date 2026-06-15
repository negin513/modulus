#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Plot GALE vs GALE_FA sweep results (memory and step time vs subsample)."""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import statistics
from pathlib import Path

import sys
from pathlib import Path

_BENCHMARKS_ROOT = Path(__file__).resolve().parents[1]
if str(_BENCHMARKS_ROOT) not in sys.path:
    sys.path.insert(0, str(_BENCHMARKS_ROOT))
from paths import RECIPE_ROOT

import matplotlib.pyplot as plt

from plots.plot_scaling_snapshot import (
    AXES_LABELSIZE,
    COL_SURFACE,
    COL_VOLUME,
    LAYOUT_TOP_WITH_SUBTITLE,
    _apply_style,
    _bold_legend,
    _fig_subtitle,
    _fig_suptitle,
    _finalize_figure,
)

# GALE (FLARE off) = surface green; GALE_FA (FLARE on) = volume teal (scaling snapshot palette).
COL_GALE = COL_SURFACE
COL_GALE_FA = COL_VOLUME

_RUN_ID_RE = re.compile(
    r"flare_bench__(?P<model>.+)__sub(?P<sub>\d+)__(?P<tag>gale|gale_fa)__g1$"
)

# Slide filenames under results/_scaling_snapshot/ (Milestone 1 deck).
FLARE_SCALING_SNAPSHOT_NAMES: dict[str, str] = {
    "shift_suv_estate_surface": "31_flare_peak_memory_gale_vs_gale_fa_surface.png",
    "drivaer_ml_surface": "33_flare_peak_memory_gale_vs_gale_fa_drivaer_surface.png",
}


def _flare_scaling_snapshot_name(provenance: dict) -> str | None:
    dataset = str(provenance.get("dataset") or "")
    return FLARE_SCALING_SNAPSHOT_NAMES.get(dataset)


def _load_rows(path: Path) -> tuple[dict, list[dict]]:
    if path.suffix == ".csv":
        with path.open(newline="") as fh:
            return {}, list(csv.DictReader(fh))
    payload = json.loads(path.read_text())
    return payload.get("provenance") or {}, list(payload.get("rows") or [])


def _rows_from_csv(csv_path: Path) -> list[dict]:
    with csv_path.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    for row in rows:
        row["subsampling"] = int(row["subsampling"])
        row["flare_attention"] = row["flare_attention"].strip().lower() in {
            "true",
            "1",
            "yes",
        }
        row["oom"] = row["oom"].strip().lower() in {"true", "1", "yes"}
        if row.get("peak_mem_gb"):
            row["peak_mem_gb"] = float(row["peak_mem_gb"])
        if row.get("train_step_p50_s"):
            row["train_step_p50_s"] = float(row["train_step_p50_s"])
    return rows


def _partial_rows(results_dir: Path, *, done_ids: set[str]) -> list[dict]:
    """Build in-progress rows from metrics.jsonl when benchmark_summary is missing."""
    runs_dir = results_dir / "runs"
    if not runs_dir.is_dir():
        return []
    partial: list[dict] = []
    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        run_id = run_dir.name
        if run_id in done_ids:
            continue
        if (run_dir / "benchmark_summary.json").exists():
            continue
        metrics_path = run_dir / "metrics.jsonl"
        if not metrics_path.exists():
            continue
        m = _RUN_ID_RE.match(run_id)
        if not m:
            continue
        mems: list[float] = []
        steps: list[float] = []
        for line in metrics_path.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("phase") != "step":
                continue
            mems.append(float(rec["mem_gb"]))
            steps.append(float(rec["step_time_s"]))
        if not mems:
            continue
        flare = m.group("tag") == "gale_fa"
        partial.append(
            {
                "run_id": run_id,
                "modality": "surface" if "surface" in m.group("model") else "volume",
                "model": m.group("model"),
                "subsampling": int(m.group("sub")),
                "flare_attention": flare,
                "attention_type": "GALE_FA" if flare else "GALE",
                "status": "in_progress",
                "oom": False,
                "peak_mem_gb": max(mems),
                "train_step_p50_s": statistics.median(steps),
            }
        )
    return partial


def _series(
    rows: list[dict],
    *,
    modality: str,
    flare: bool,
    y_key: str,
) -> tuple[list[int], list[float | None], list[bool]]:
    filt = [
        r
        for r in rows
        if r.get("modality") == modality and r.get("flare_attention") is flare
    ]
    filt.sort(key=lambda r: int(r["subsampling"]))
    xs = [int(r["subsampling"]) for r in filt]
    ys: list[float | None] = []
    ooms: list[bool] = []
    for r in filt:
        if r.get("oom"):
            ys.append(None)
            ooms.append(True)
        elif r.get("status") in ("ok", "in_progress"):
            ys.append(r.get(y_key))
            ooms.append(False)
        else:
            ys.append(None)
            ooms.append(False)
    return xs, ys, ooms


def _linear_subs_ticks(max_sub: int, *, step: int = 50_000) -> list[int]:
    """Evenly spaced subsampling ticks from 0 through max_sub (linear x-axis)."""
    if max_sub <= 0:
        return [0]
    ticks = list(range(0, max_sub + 1, step))
    if ticks[-1] != max_sub:
        ticks.append(max_sub)
    return ticks


def _format_subs_axis(
    ax,
    subs: list[int],
    *,
    max_sub: int | None = None,
    linear_from_zero: bool = False,
    uniform_linear_ticks: bool = False,
    tick_step: int = 50_000,
) -> None:
    """Format subsampling x-axis (linear scale; optional uniform tick spacing)."""
    if not subs and max_sub is None:
        return
    x_hi = max_sub if max_sub is not None else max(subs)
    if linear_from_zero or uniform_linear_ticks:
        xticks = _linear_subs_ticks(x_hi, step=tick_step)
        ax.set_xscale("linear")
        ax.set_xticks(xticks)
        ax.set_xticklabels([f"{x:,}" for x in xticks], fontsize=AXES_LABELSIZE)
        ax.set_xlim(0, x_hi * 1.05)
        return
    xs = sorted(set(subs))
    if max_sub is not None:
        xs = [x for x in xs if x <= max_sub]
    if not xs:
        return
    ax.set_xscale("linear")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{x:,}" for x in xs], fontsize=AXES_LABELSIZE)
    ax.set_xlim(min(xs) * 0.92, max(xs) * 1.05)


def _load_matrix_gale_rows(
    results_root: Path,
    *,
    model: str,
    dataset: str,
    num_gpus: int = 1,
    storage: str = "lustre",
) -> list[dict]:
    """Load GALE (FLARE off) rows from phase-1 matrix benchmark_summary.json files."""
    base = results_root / model / dataset / f"gpus_{num_gpus}" / storage
    if not base.is_dir():
        print(f"[plot_flare] matrix results not found: {base}")
        return []
    modality = "surface" if "surface" in model else "volume"
    rows: list[dict] = []
    for sub_dir in sorted(base.glob("sub_*")):
        sub = int(sub_dir.name.replace("sub_", ""))
        for summary_path in sorted(sub_dir.rglob("benchmark_summary.json")):
            if "_flare_attention" in str(summary_path):
                continue
            summary = json.loads(summary_path.read_text())
            if summary.get("model") != model or summary.get("dataset") != dataset:
                continue
            rows.append(
                {
                    "run_id": f"matrix__{model}__sub{sub}__gale__g1",
                    "modality": modality,
                    "model": model,
                    "dataset": dataset,
                    "subsampling": sub,
                    "flare_attention": False,
                    "attention_type": "GALE",
                    "status": "ok",
                    "oom": False,
                    "peak_mem_gb": float(summary["memory"]["peak_gb"]),
                    "train_step_p50_s": float(summary["train"]["p50"]),
                }
            )
            break
    rows.sort(key=lambda r: int(r["subsampling"]))
    print(f"[plot_flare] +{len(rows)} matrix GALE row(s) from {base}")
    return rows


def _provenance_subtitle(provenance: dict) -> str:
    gpu = provenance.get("gpu_name", "B200")
    prec = provenance.get("precision", "bfloat16")
    storage = provenance.get("storage_tier", "lustre")
    dataset = provenance.get("dataset", "shift_suv_estate_surface")
    return f"{dataset} · 1 GPU · {gpu} · {prec} · {storage}"


def _plot_series_on_ax(
    ax,
    rows: list[dict],
    *,
    modality: str,
    flare: bool,
    y_key: str,
    color: str,
    label: str,
) -> None:
    ok_rows = sorted(
        [
            r
            for r in rows
            if r.get("modality") == modality
            and r.get("flare_attention") is flare
            and r.get("status") == "ok"
        ],
        key=lambda r: int(r["subsampling"]),
    )
    partial_rows = sorted(
        [
            r
            for r in rows
            if r.get("modality") == modality
            and r.get("flare_attention") is flare
            and r.get("status") == "in_progress"
        ],
        key=lambda r: int(r["subsampling"]),
    )
    if ok_rows:
        ax.plot(
            [int(r["subsampling"]) for r in ok_rows],
            [r.get(y_key) for r in ok_rows],
            "o-",
            color=color,
            lw=2.5,
            ms=10,
            label=label,
            zorder=4,
        )
    if partial_rows:
        # Extend the series with a dashed line from the last finalized point.
        if ok_rows:
            tail = [ok_rows[-1], *partial_rows]
        else:
            tail = partial_rows
        ax.plot(
            [int(r["subsampling"]) for r in tail],
            [r.get(y_key) for r in tail],
            "o-",
            color=color,
            lw=2.5,
            ms=10,
            label=f"{label} (in progress)" if ok_rows else label,
            zorder=3,
        )


def _plot_metric(
    rows: list[dict],
    *,
    modality: str,
    y_key: str,
    ylabel: str,
    title: str,
    out_path: Path,
    provenance: dict,
    max_sub: int | None = None,
    uniform_linear_ticks: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.8))
    _fig_suptitle(fig, title)
    _fig_subtitle(fig, _provenance_subtitle(provenance))

    for flare, label, color in (
        (False, "GALE (FLARE off)", COL_GALE),
        (True, "GALE_FA (FLARE on)", COL_GALE_FA),
    ):
        _plot_series_on_ax(
            ax,
            rows,
            modality=modality,
            flare=flare,
            y_key=y_key,
            color=color,
            label=label,
        )

    all_subs = [
        int(r["subsampling"])
        for r in rows
        if r.get("modality") == modality and r.get("status") in ("ok", "in_progress")
    ]
    _format_subs_axis(
        ax, all_subs, max_sub=max_sub, uniform_linear_ticks=uniform_linear_ticks
    )

    oom_labeled = False
    ax.relim()
    ax.autoscale_view()
    y_top = ax.get_ylim()[1]
    for flare, _label, _color in (
        (False, "GALE (FLARE off)", COL_GALE),
        (True, "GALE_FA (FLARE on)", COL_GALE_FA),
    ):
        xs, ys, ooms = _series(rows, modality=modality, flare=flare, y_key=y_key)
        for x, y, is_oom in zip(xs, ys, ooms):
            if is_oom:
                ax.scatter(
                    [x],
                    [y_top * 0.98],
                    marker="x",
                    s=120,
                    color="red",
                    zorder=5,
                    label="OOM (failed)" if not oom_labeled else None,
                )
                ax.annotate("OOM", (x, y_top * 0.94), fontsize=8, color="red", ha="center")
                oom_labeled = True

    ax.set_xlabel("Subsampling Resolution", fontsize=AXES_LABELSIZE)
    ax.set_ylabel(ylabel, fontsize=AXES_LABELSIZE)
    ax.legend(loc="lower right" if y_key == "peak_mem_gb" else "best", framealpha=0.95, prop={"weight": "bold"})
    fig.tight_layout(rect=[0, 0, 1, LAYOUT_TOP_WITH_SUBTITLE])
    _finalize_figure(fig)
    _bold_legend(ax)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_gale_only_memory(
    rows: list[dict],
    *,
    modality: str,
    out_path: Path,
    provenance: dict,
    max_sub: int | None = None,
    uniform_linear_ticks: bool = False,
) -> None:
    """Single-series peak memory vs subsample (GALE / FLARE off only)."""

    def _within_max(r: dict) -> bool:
        return max_sub is None or int(r["subsampling"]) <= max_sub

    gale_rows = [
        r
        for r in rows
        if r.get("modality") == modality
        and r.get("flare_attention") is False
        and r.get("status") == "ok"
        and _within_max(r)
    ]
    gale_rows.sort(key=lambda r: int(r["subsampling"]))
    partial = [
        r
        for r in rows
        if r.get("modality") == modality
        and r.get("flare_attention") is False
        and r.get("status") == "in_progress"
        and _within_max(r)
    ]
    if not gale_rows and not partial:
        print(f"[plot_flare] no GALE rows for modality={modality}")
        return

    fig, ax = plt.subplots(figsize=(9, 5.8))
    _fig_suptitle(fig, "Peak GPU Memory Vs Subsampling Resolution")
    _fig_subtitle(fig, f"{_provenance_subtitle(provenance)} · GALE only (FLARE off)")

    if gale_rows:
        xs = [int(r["subsampling"]) for r in gale_rows]
        ys = [float(r["peak_mem_gb"]) for r in gale_rows]
        ax.plot(xs, ys, "o-", color=COL_GALE, lw=2.5, ms=10, label="GALE (FLARE off)", zorder=4)
    if partial:
        if gale_rows:
            tail = [gale_rows[-1], *partial]
        else:
            tail = partial
        ax.plot(
            [int(r["subsampling"]) for r in tail],
            [float(r["peak_mem_gb"]) for r in tail],
            "o-",
            color=COL_GALE,
            lw=2.5,
            ms=10,
            label="GALE (in progress)" if gale_rows else "GALE (FLARE off)",
            zorder=3,
        )

    all_subs = [int(r["subsampling"]) for r in gale_rows + partial]
    _format_subs_axis(
        ax, all_subs, max_sub=max_sub, uniform_linear_ticks=uniform_linear_ticks
    )
    ax.set_xlabel("Subsampling Resolution", fontsize=AXES_LABELSIZE)
    ax.set_ylabel("Peak GPU Memory (GB)", fontsize=AXES_LABELSIZE)
    ax.legend(loc="lower right", framealpha=0.95, prop={"weight": "bold"})
    fig.tight_layout(rect=[0, 0, 1, LAYOUT_TOP_WITH_SUBTITLE])
    _finalize_figure(fig)
    _bold_legend(ax)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results/_flare_attention/flare_attention_results.json"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/_flare_attention/plots"),
    )
    parser.add_argument(
        "--gale-only",
        action="store_true",
        help="Only plot GALE (FLARE off) peak memory vs subsample.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Sweep results root (for --include-partial metrics scan).",
    )
    parser.add_argument(
        "--include-partial",
        action="store_true",
        help="Add in-progress runs from metrics.jsonl (dashed markers).",
    )
    parser.add_argument(
        "--max-sub",
        type=int,
        default=None,
        metavar="N",
        help="Omit subsampling resolutions above N (e.g. 300000).",
    )
    parser.add_argument(
        "--matrix-results",
        type=Path,
        default=None,
        metavar="DIR",
        help="Recipe results root; supplement GALE (FLARE off) from matrix benchmark_summary.json.",
    )
    parser.add_argument(
        "--uniform-linear-x",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Evenly spaced x ticks (0, 50k, 100k, …) on a linear axis (default: on).",
    )
    parser.add_argument(
        "--scaling-snapshot-dir",
        type=Path,
        default=None,
        help="Copy surface peak-memory plot here with slide filename (default: results/_scaling_snapshot)",
    )
    parser.add_argument(
        "--no-scaling-snapshot",
        action="store_true",
        help="Skip copying peak-memory plot into the scaling snapshot directory",
    )
    args = parser.parse_args()
    results_dir = (args.results_dir or args.input.parent).resolve()
    json_path = args.input
    if args.input.suffix == ".csv":
        provenance_path = results_dir / "provenance.json"
        provenance = (
            json.loads(provenance_path.read_text())
            if provenance_path.exists()
            else {}
        )
        rows = _rows_from_csv(args.input)
    else:
        provenance, rows = _load_rows(args.input)
        json_path = args.input
    if args.include_partial:
        done_ids = {r["run_id"] for r in rows}
        partial = _partial_rows(results_dir, done_ids=done_ids)
        if partial:
            print(f"[plot_flare] +{len(partial)} in-progress row(s) from metrics")
            rows = rows + partial
    if args.matrix_results is not None:
        model = provenance.get("model") or rows[0].get("model")
        dataset = provenance.get("dataset") or rows[0].get("dataset")
        storage = provenance.get("storage_tier", "lustre")
        num_gpus = int(provenance.get("gpu_count", 1))
        if model and dataset:
            matrix_rows = _load_matrix_gale_rows(
                args.matrix_results.resolve(),
                model=str(model),
                dataset=str(dataset),
                num_gpus=num_gpus,
                storage=str(storage),
            )
            gale_subs = {
                int(r["subsampling"])
                for r in rows
                if not r.get("flare_attention")
            }
            for mr in matrix_rows:
                if int(mr["subsampling"]) not in gale_subs:
                    rows.append(mr)
        else:
            print("[plot_flare] skip --matrix-results (missing model/dataset)")
    if not rows:
        print(f"[plot_flare] no rows in {args.input}")
        return 1
    if args.max_sub is not None:
        rows = [r for r in rows if int(r["subsampling"]) <= args.max_sub]

    _apply_style()
    modalities = sorted({r.get("modality", "volume") for r in rows})
    for mod in modalities:
        tag = mod
        if args.gale_only:
            _plot_gale_only_memory(
                rows,
                modality=mod,
                out_path=args.out_dir / f"peak_memory_gale_only_{tag}.png",
                provenance=provenance,
                max_sub=args.max_sub,
                uniform_linear_ticks=args.uniform_linear_x,
            )
            continue
        _plot_metric(
            rows,
            modality=mod,
            y_key="peak_mem_gb",
            ylabel="Peak GPU memory (GB)",
            title="FLARE Attention — Peak GPU Memory Vs Subsampling Resolution",
            out_path=args.out_dir / f"peak_memory_vs_subsample_{tag}.png",
            provenance=provenance,
            max_sub=args.max_sub,
            uniform_linear_ticks=args.uniform_linear_x,
        )
        _plot_metric(
            rows,
            modality=mod,
            y_key="train_step_p50_s",
            ylabel="Train step time P50 (s)",
            title="FLARE Attention — Train Step Time P50 Vs Subsampling Resolution",
            out_path=args.out_dir / f"step_time_vs_subsample_{tag}.png",
            provenance=provenance,
            max_sub=args.max_sub,
            uniform_linear_ticks=args.uniform_linear_x,
        )

    if not args.gale_only and not args.no_scaling_snapshot:
        snap_name = _flare_scaling_snapshot_name(provenance)
        if snap_name:
            snap_dir = args.scaling_snapshot_dir or (RECIPE_ROOT / "results/_scaling_snapshot")
            src = args.out_dir / "peak_memory_vs_subsample_surface.png"
            if src.is_file():
                dest = snap_dir / snap_name
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dest)
                print(f"[plot_flare] scaling snapshot: {dest}")

    print(f"[plot_flare] plots under {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
