#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Measure VTK/VTU load time for manifest-aligned training runs.

Uses an **available** PhysicsNeMo-Mesh dataset (default: ShiftSUV on HSG/Lustre)
only to read ``manifest.json`` and pick run IDs. All timed work is VTK/VTU I/O:

* ``pyvista.read`` + ``from_pyvista`` for interior ``*.vtu`` + boundary ``*.vtp``.

Optional ``--compare-pnm`` adds reference PNM.mesh train-step numbers from
existing benchmark artifacts (no live PNM GPU reruns).

Example (CPU node is fine — no GPU required)::

    export DATASET_PATH_SHIFT_SUV=/lustre/.../PhysicsNeMo-ShiftSUV
    export DATASET_PATH_SHIFT_SUV_VTU=/lustre/.../shift_suv/SUV/AeroSUV_full_scale_estate_transient
    python benchmarks/sweeps/sweep_vtu_vs_pnm_e2e.py --max-samples 3

Cluster::

    sbatch benchmarks/run_vtu_vs_pnm_e2e.sbatch
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# HSG cold-disk medians (jobs 3077314 / 3115330, Jun 2026)
REFERENCE_COLD_VTU_S: dict[str, float] = {
    "drivaer_ml_volume": 412.448,
    "drivaer_ml_surface": 412.448,
    "shift_suv_estate_volume": 58.1,
    "shift_suv_estate_surface": 58.1,
}

# PNM train-step references for optional comparison charts (no live reruns).
REFERENCE_PNM_TRAIN_STEP_S: dict[str, float] = {
    "geotransolver_volume__drivaer_ml_volume": 0.810155,
    "geotransolver_surface__drivaer_ml_surface": 0.263248,
    "geotransolver_surface__shift_suv_estate_surface": 0.201015,
    "geotransolver_volume__shift_suv_estate_volume": 0.35,  # conservative placeholder
}


@dataclass(frozen=True)
class DatasetProfile:
    """Maps a Hydra dataset name to on-disk PNM + VTU layouts."""

    manifest_subdir: str
    vtu_layout: str  # "drivaer" | "shift_suv"
    default_pnm_env: str
    default_vtu_env: str
    default_pnm_path: str
    default_vtu_path: str


DATASET_PROFILES: dict[str, DatasetProfile] = {
    "shift_suv_estate_volume": DatasetProfile(
        manifest_subdir="estate",
        vtu_layout="shift_suv",
        default_pnm_env="DATASET_PATH_SHIFT_SUV",
        default_vtu_env="DATASET_PATH_SHIFT_SUV_VTU",
        default_pnm_path="/lustre/fs1/portfolios/coreai/projects/coreai_modulus_cae/datasets/PhysicsNeMo-ShiftSUV",
        default_vtu_path=(
            "/lustre/fs1/portfolios/coreai/projects/coreai_modulus_cae/datasets/shift_suv/"
            "SUV/AeroSUV_full_scale_estate_transient"
        ),
    ),
    "shift_suv_estate_surface": DatasetProfile(
        manifest_subdir="estate",
        vtu_layout="shift_suv",
        default_pnm_env="DATASET_PATH_SHIFT_SUV",
        default_vtu_env="DATASET_PATH_SHIFT_SUV_VTU",
        default_pnm_path="/lustre/fs1/portfolios/coreai/projects/coreai_modulus_cae/datasets/PhysicsNeMo-ShiftSUV",
        default_vtu_path=(
            "/lustre/fs1/portfolios/coreai/projects/coreai_modulus_cae/datasets/shift_suv/"
            "SUV/AeroSUV_full_scale_estate_transient"
        ),
    ),
    "drivaer_ml_volume": DatasetProfile(
        manifest_subdir=".",
        vtu_layout="drivaer",
        default_pnm_env="DATASET_PATH_DRIVAER_ML",
        default_vtu_env="DATASET_PATH_DRIVAER_VTU",
        default_pnm_path="/lustre/fs1/portfolios/coreai/projects/coreai_modulus_cae/datasets/PhysicsNeMo-DrivaerML",
        default_vtu_path=(
            "/lustre/fs1/portfolios/coreai/projects/coreai_modulus_cae/datasets/"
            "drivaer_aws/drivaer_data_full"
        ),
    ),
    "drivaer_ml_surface": DatasetProfile(
        manifest_subdir=".",
        vtu_layout="drivaer",
        default_pnm_env="DATASET_PATH_DRIVAER_ML",
        default_vtu_env="DATASET_PATH_DRIVAER_VTU",
        default_pnm_path="/lustre/fs1/portfolios/coreai/projects/coreai_modulus_cae/datasets/PhysicsNeMo-DrivaerML",
        default_vtu_path=(
            "/lustre/fs1/portfolios/coreai/projects/coreai_modulus_cae/datasets/"
            "drivaer_aws/drivaer_data_full"
        ),
    ),
}


def _recipe_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _profile(dataset: str) -> DatasetProfile:
    if dataset not in DATASET_PROFILES:
        known = ", ".join(sorted(DATASET_PROFILES))
        raise ValueError(f"unknown dataset {dataset!r}; supported: {known}")
    return DATASET_PROFILES[dataset]


def _manifest_train_runs(pnm_root: Path, manifest_subdir: str, n: int) -> list[str]:
    manifest_dir = pnm_root if manifest_subdir == "." else pnm_root / manifest_subdir
    manifest = manifest_dir / "manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError(f"manifest.json not found under {manifest_dir}")
    data = json.loads(manifest.read_text())
    train = data.get("train", [])
    if not train:
        raise ValueError(f"empty train split in {manifest}")
    return [str(x).strip("/") for x in train[:n]]


def _drivaer_vtu_paths(vtu_root: Path, run_name: str) -> tuple[Path, Path | None]:
    """Map ``run_1`` -> ``.../run_1/volume_1.vtu`` and optional boundary VTP."""
    m = re.match(r"run_(\d+)$", run_name)
    if not m:
        raise ValueError(f"unexpected DrivAer run name: {run_name}")
    idx = m.group(1)
    run_dir = vtu_root / run_name
    vol = run_dir / f"volume_{idx}.vtu"
    bnd = run_dir / f"boundary_{idx}.vtp"
    if not vol.is_file():
        raise FileNotFoundError(vol)
    return vol, bnd if bnd.is_file() else None


def _shift_suv_vtu_paths(vtu_root: Path, run_name: str) -> tuple[Path, Path | None]:
    """Map ``run_00001`` -> ``.../run_00001/merged_volumes.vtu`` + ``merged_surfaces.vtp``."""
    run_dir = vtu_root / run_name
    vol = run_dir / "merged_volumes.vtu"
    bnd = run_dir / "merged_surfaces.vtp"
    if not vol.is_file():
        raise FileNotFoundError(vol)
    return vol, bnd if bnd.is_file() else None


def _resolve_vtu_paths(profile: DatasetProfile, vtu_root: Path, run_name: str) -> tuple[Path, Path | None]:
    if profile.vtu_layout == "drivaer":
        return _drivaer_vtu_paths(vtu_root, run_name)
    if profile.vtu_layout == "shift_suv":
        return _shift_suv_vtu_paths(vtu_root, run_name)
    raise ValueError(f"unsupported vtu_layout: {profile.vtu_layout}")


def _timed_vtu_load(vol: Path, bnd: Path | None) -> float:
    import pyvista as pv
    from physicsnemo.mesh.io.io_pyvista import from_pyvista

    t0 = time.perf_counter()
    pv_vol = pv.read(str(vol))
    from_pyvista(pv_vol, manifold_dim="auto")
    if bnd is not None:
        pv_bnd = pv.read(str(bnd))
        from_pyvista(pv_bnd, manifold_dim="auto")
    return time.perf_counter() - t0


def _stats_seconds(times: list[float]) -> dict[str, float]:
    if not times:
        return {}
    return {
        "n": len(times),
        "mean_s": statistics.mean(times),
        "median_s": statistics.median(times),
        "p95_s": sorted(times)[int(0.95 * (len(times) - 1))] if len(times) > 1 else times[0],
        "total_s": sum(times),
    }


def _count_train_samples(pnm_root: Path, manifest_subdir: str) -> int:
    manifest_dir = pnm_root if manifest_subdir == "." else pnm_root / manifest_subdir
    manifest = manifest_dir / "manifest.json"
    data = json.loads(manifest.read_text())
    return len(data.get("train", []))


def _load_pnm_reference(model: str, dataset: str, recipe_results: Path) -> dict[str, Any] | None:
    """Load existing PNM benchmark_summary or embedded reference (comparison only)."""
    run_id = f"{model}__{dataset}__g1__lustre__sub200000"
    candidates = list(recipe_results.glob(f"**/{run_id}/benchmark_summary.json"))
    if candidates:
        data = json.loads(candidates[0].read_text())
        return {
            "source": str(candidates[0]),
            "train_step_p50_s": data["train"]["p50"],
            "train_samples": data.get("train_samples"),
        }
    key = f"{model}__{dataset}"
    if key not in REFERENCE_PNM_TRAIN_STEP_S:
        return None
    return {
        "source": "embedded_reference",
        "train_step_p50_s": REFERENCE_PNM_TRAIN_STEP_S[key],
        "train_samples": None,
    }


@dataclass
class E2ERow:
    format: str
    ms_per_sample: float
    s_per_sample: float
    epoch_s_for_n: float
    n_train_samples: int
    notes: str


def _compose_vtu_rows(
    *,
    n_samples: int,
    vtu_load_s: float,
    vtu_label: str,
    pnm_train_s: float | None = None,
    pnm_datapipe_fraction: float = 0.15,
) -> list[E2ERow]:
    rows: list[E2ERow] = [
        E2ERow(
            format=f"VTU load ({vtu_label})",
            ms_per_sample=vtu_load_s * 1e3,
            s_per_sample=vtu_load_s,
            epoch_s_for_n=n_samples * vtu_load_s,
            n_train_samples=n_samples,
            notes=f"pyvista.read + from_pyvista (interior + boundary)",
        )
    ]
    if pnm_train_s is not None:
        compute_s = max(0.0, pnm_train_s * (1.0 - pnm_datapipe_fraction))
        vtu_e2e = vtu_load_s + compute_s
        rows.append(
            E2ERow(
                format="PNM.mesh (reference train step)",
                ms_per_sample=pnm_train_s * 1e3,
                s_per_sample=pnm_train_s,
                epoch_s_for_n=n_samples * pnm_train_s,
                n_train_samples=n_samples,
                notes="existing benchmark; not remeasured",
            )
        )
        rows.append(
            E2ERow(
                format=f"VTU + compute ({vtu_label})",
                ms_per_sample=vtu_e2e * 1e3,
                s_per_sample=vtu_e2e,
                epoch_s_for_n=n_samples * vtu_e2e,
                n_train_samples=n_samples,
                notes=f"vtu_load={vtu_load_s:.3f}s + compute≈{compute_s:.3f}s",
            )
        )
    return rows


def _write_plot(out_dir: Path, rows: list[E2ERow], subs: int, title_suffix: str) -> Path:
    import matplotlib.pyplot as plt

    labels = [r.format for r in rows]
    epoch_h = [r.epoch_s_for_n / 3600 for r in rows]
    palette = ["#999999", "#76B900", "#9ACD32"]
    colors = palette[: len(rows)]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    x = range(len(labels))

    axes[0].bar(x, [r.s_per_sample for r in rows], color=colors)
    axes[0].set_ylabel("Seconds / sample")
    axes[0].set_title(f"Per-sample time @ sub={subs:,}")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=15, ha="right")
    for i, r in enumerate(rows):
        axes[0].text(i, r.s_per_sample, f"{r.s_per_sample:.2f}s", ha="center", va="bottom", fontsize=8)

    axes[1].bar(x, epoch_h, color=colors)
    axes[1].set_ylabel("Hours / train epoch")
    axes[1].set_title(f"One epoch (N={rows[0].n_train_samples} train samples)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=15, ha="right")
    for i, r in enumerate(rows):
        axes[1].text(i, r.epoch_s_for_n / 3600, f"{r.epoch_s_for_n/3600:.2f}h", ha="center", va="bottom", fontsize=8)

    fig.suptitle(f"VTK/VTU load — {title_suffix}", fontsize=11)
    fig.tight_layout()
    out = out_dir / "16_end_to_end_vtu_vs_pnm.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subs", type=int, default=200_000, help="Subsampling level (for plot labels only)")
    parser.add_argument("--model", default="geotransolver_surface")
    parser.add_argument("--dataset", default="shift_suv_estate_surface")
    parser.add_argument("--max-samples", type=int, default=3, help="Number of VTU loads to time")
    parser.add_argument("--reference-cold-vtu", action="store_true", default=True)
    parser.add_argument("--no-reference-cold-vtu", action="store_false", dest="reference_cold_vtu")
    parser.add_argument(
        "--compare-pnm",
        action="store_true",
        help="Add reference PNM.mesh train-step rows to charts (no GPU reruns)",
    )
    parser.add_argument(
        "--pnm-root",
        type=Path,
        default=None,
        help="PNM.mesh root for manifest only (default: DATASET_PATH_SHIFT_SUV)",
    )
    parser.add_argument(
        "--vtu-root",
        type=Path,
        default=None,
        help="VTU tree (default: DATASET_PATH_SHIFT_SUV_VTU or dataset default)",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Output dir (default: results/_vtu_load_benchmark)",
    )
    args = parser.parse_args()

    recipe_root = _recipe_root()
    profile = _profile(args.dataset)
    results_dir = args.results_dir or (recipe_root / "results/_vtu_load_benchmark")
    results_dir.mkdir(parents=True, exist_ok=True)

    pnm_root = args.pnm_root or Path(
        os.environ.get(profile.default_pnm_env, profile.default_pnm_path)
    )
    vtu_root = args.vtu_root or Path(
        os.environ.get(profile.default_vtu_env, profile.default_vtu_path)
    )

    runs = _manifest_train_runs(pnm_root, profile.manifest_subdir, args.max_samples)
    n_train = _count_train_samples(pnm_root, profile.manifest_subdir)

    report: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "vtu_vtk_only",
        "subsampling": args.subs,
        "model": args.model,
        "dataset": args.dataset,
        "storage": "lustre",
        "pnm_root_manifest_only": str(pnm_root),
        "vtu_root": str(vtu_root),
        "manifest_runs": runs,
        "n_train_samples_per_epoch": n_train,
        "compare_pnm": args.compare_pnm,
    }

    pnm_ref: dict[str, Any] | None = None
    pnm_train_s: float | None = None
    if args.compare_pnm:
        pnm_ref = _load_pnm_reference(args.model, args.dataset, recipe_root / "results")
        if pnm_ref is None:
            print(
                f"[vtu] --compare-pnm: no reference for {args.model}/{args.dataset}; "
                "chart will show VTU loads only",
                file=sys.stderr,
            )
        else:
            pnm_train_s = float(pnm_ref["train_step_p50_s"])
            report["pnm_reference"] = pnm_ref
            print(f"[vtu] PNM reference train step: {pnm_train_s:.3f} s  source={pnm_ref['source']}")

    warm_times: list[float] = []
    per_run: list[dict[str, Any]] = []
    for i, run in enumerate(runs):
        vol, bnd = _resolve_vtu_paths(profile, vtu_root, run)
        label = f"{run} ({vol.name}" + (f" + {bnd.name})" if bnd else ")")
        print(f"[vtu] load {i + 1}/{len(runs)}: {label}", flush=True)
        dt = _timed_vtu_load(vol, bnd)
        warm_times.append(dt)
        per_run.append(
            {
                "run": run,
                "volume": str(vol),
                "boundary": str(bnd) if bnd else None,
                "load_s": dt,
            }
        )
        print(f"[vtu]   loaded in {dt:.2f} s", flush=True)

    report["vtu_warm"] = _stats_seconds(warm_times)
    report["vtu_per_run"] = per_run

    vtu_warm_s = report["vtu_warm"]["median_s"]
    vtu_cold_s = (
        REFERENCE_COLD_VTU_S.get(args.dataset)
        if args.reference_cold_vtu
        else vtu_warm_s
    )
    if vtu_cold_s is None:
        vtu_cold_s = vtu_warm_s
        report["vtu_cold_note"] = "no HSG cold reference; using warm median"
    else:
        report["vtu_cold_reference_s"] = vtu_cold_s

    rows_warm = _compose_vtu_rows(
        n_samples=n_train,
        vtu_load_s=vtu_warm_s,
        vtu_label="warm page cache",
        pnm_train_s=pnm_train_s,
    )
    rows_cold = _compose_vtu_rows(
        n_samples=n_train,
        vtu_load_s=vtu_cold_s,
        vtu_label="cold Lustre (HSG ref)" if args.reference_cold_vtu else "cold",
        pnm_train_s=pnm_train_s,
    )
    report["rows_warm_vtu"] = [r.__dict__ for r in rows_warm]
    report["rows_cold_vtu"] = [r.__dict__ for r in rows_cold]

    out_json = results_dir / "vtu_load_benchmark.json"
    out_json.write_text(json.dumps(report, indent=2))
    print(f"[vtu] wrote {out_json}")

    snap = recipe_root / "results/_scaling_snapshot"
    snap.mkdir(parents=True, exist_ok=True)
    plot_cold = _write_plot(
        snap,
        rows_cold,
        args.subs,
        f"{args.dataset} / Lustre (VTU cold)",
    )
    plot_warm = _write_plot(
        results_dir,
        rows_warm,
        args.subs,
        f"{args.dataset} / Lustre (VTU warm)",
    )
    print(f"[vtu] plot (cold): {plot_cold}")
    print(f"[vtu] plot (warm): {plot_warm}")

    vtu_row = rows_cold[0]
    print(
        f"\n[vtu] SUMMARY @ {args.dataset}\n"
        f"  warm median: {vtu_warm_s:.2f} s/sample  ({n_train} samples → {n_train * vtu_warm_s / 3600:.2f} h/epoch)\n"
        f"  cold ref:    {vtu_cold_s:.2f} s/sample  ({n_train} samples → {vtu_row.epoch_s_for_n/3600:.2f} h/epoch)"
    )
    if pnm_train_s is not None and len(rows_cold) > 2:
        pnm_row = next(r for r in rows_cold if "PNM.mesh" in r.format)
        e2e_row = next(r for r in rows_cold if "VTU + compute" in r.format)
        print(
            f"  PNM ref:     {pnm_train_s:.2f} s/sample  ({pnm_row.epoch_s_for_n/60:.1f} min/epoch)\n"
            f"  VTU+cold vs PNM: {e2e_row.epoch_s_for_n / pnm_row.epoch_s_for_n:.0f}x slower"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
