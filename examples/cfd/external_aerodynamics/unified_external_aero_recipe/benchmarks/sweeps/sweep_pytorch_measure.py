#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Run ``profile_and_attribute.py --mode measure`` across subsampling levels.

For each subsampling point, profiles one real GeoTransolver Surface training
step twice (torch baseline forced, then default PhysicsNeMo dispatch) and
records the measured PyTorch wall time.  Aggregates into
``pytorch_measured.json`` for ``plot_model_latency.py --pytorch-json``.

Example (login node with GPU + dataset env set)::

    export DATASET_PATH_SHIFT_SUV=/lustre/.../PhysicsNeMo-ShiftSUV
    python benchmarks/sweeps/sweep_pytorch_measure.py --device cuda

Cluster::

    sbatch benchmarks/run_pytorch_measure.sbatch
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

DEFAULT_SUBS = (10_000, 50_000, 100_000, 200_000, 300_000, 400_000)
PYTORCH_JSON_NAME = "pytorch_measured.json"


def _run_measure(
    *,
    benchmarks_dir: Path,
    recipe_root: Path,
    physicsnemo_root: Path,
    subsampling: int,
    output_stem: Path,
    device: str,
    n_warmup: int,
    n_record: int,
    model: str,
    dataset: str,
    asv_results: Path,
    compile_model: bool = True,
) -> dict:
    env = os.environ.copy()
    env["PROFILE_SUBSAMPLING"] = str(subsampling)
    env["PROFILE_MODEL"] = model
    env["PROFILE_DATASET"] = dataset
    env["PROFILE_COMPILE"] = "true" if compile_model else "false"
    env["PYTHONPATH"] = (
        str(benchmarks_dir)
        + os.pathsep
        + env.get("PYTHONPATH", "")
    )

    cmd = [
        sys.executable,
        str(benchmarks_dir / "sweeps/profile_and_attribute.py"),
        "--mode",
        "measure",
        "--step",
        "sweeps.recipe_train_step:make_step",
        "--device",
        device,
        "--n-warmup",
        str(n_warmup),
        "--n-record",
        str(n_record),
        "--asv-results",
        str(asv_results),
        "--output",
        str(output_stem),
    ]
    print(f"[sweep] sub={subsampling:,}  ->  {output_stem}.json", flush=True)
    subprocess.run(cmd, check=True, cwd=recipe_root, env=env)

    report = json.loads(output_stem.with_suffix(".json").read_text())
    return {
        "subsampling": subsampling,
        "pytorch_ms": round(float(report["baseline_total_s"]) * 1e3, 2),
        "pnm_ms": round(float(report["fast_total_s"]) * 1e3, 2),
        "speedup": round(float(report["overall_speedup"]), 3),
        "report_json": str(output_stem.with_suffix(".json")),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--subs",
        type=int,
        nargs="+",
        default=list(DEFAULT_SUBS),
        help="Subsampling levels to profile",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-warmup", type=int, default=5)
    parser.add_argument("--n-record", type=int, default=10)
    parser.add_argument("--model", default="geotransolver_surface")
    parser.add_argument("--dataset", default="shift_suv_estate_surface")
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=Path("results/_profile_attribute"),
    )
    parser.add_argument(
        "--asv-results",
        type=Path,
        default=None,
        help="ASV results root (default: <recipe>/../../../.. /.asv/results)",
    )
    parser.add_argument(
        "--regen-plot",
        action="store_true",
        help="Regenerate 13_geotransolver_volume_training_latency.png after sweep",
    )
    parser.add_argument(
        "--storage",
        default="nvme",
        choices=("nvme", "lustre"),
        help="Storage tag passed to plot_model_latency for PNM bars",
    )
    parser.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable torch.compile on the model (PROFILE_COMPILE)",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Log failures and continue remaining subsampling levels",
    )
    args = parser.parse_args()

    benchmarks_dir = Path(__file__).resolve().parent
    recipe_root = benchmarks_dir.parent
    physicsnemo_root = recipe_root.parent.parent.parent
    reports_dir = (
        args.reports_dir
        if args.reports_dir.is_absolute()
        else recipe_root / args.reports_dir
    )
    reports_dir.mkdir(parents=True, exist_ok=True)
    asv_results = args.asv_results or (physicsnemo_root / ".asv/results")

    rows: list[dict] = []
    for sub in args.subs:
        stem = reports_dir / f"measure_sub{sub}"
        try:
            row = _run_measure(
                benchmarks_dir=benchmarks_dir,
                recipe_root=recipe_root,
                physicsnemo_root=physicsnemo_root,
                subsampling=sub,
                output_stem=stem,
                device=args.device,
                n_warmup=args.n_warmup,
                n_record=args.n_record,
                model=args.model,
                dataset=args.dataset,
                asv_results=asv_results,
                compile_model=args.compile,
            )
        except subprocess.CalledProcessError as exc:
            print(f"[sweep] FAILED sub={sub:,}: {exc}", flush=True)
            if not args.continue_on_error:
                raise
            continue
        rows.append(row)
        print(
            f"[sweep] sub={sub:,}  compile={args.compile}  "
            f"pytorch={row['pytorch_ms']:.1f} ms  "
            f"pnm={row['pnm_ms']:.1f} ms  speedup={row['speedup']:.2f}x",
            flush=True,
        )

    pytorch_map = {str(r["subsampling"]): r["pytorch_ms"] for r in rows}
    out_json = reports_dir / PYTORCH_JSON_NAME
    out_json.write_text(json.dumps(pytorch_map, indent=2))
    summary_path = reports_dir / "pytorch_measure_summary.json"
    summary_path.write_text(json.dumps(rows, indent=2))

    print(f"[sweep] wrote {out_json}")
    print(f"[sweep] wrote {summary_path}")

    if args.regen_plot:
        summary_path = reports_dir / "pytorch_measure_summary.json"
        plot_cmd = [
            sys.executable,
            str(benchmarks_dir / "plots/plot_model_latency.py"),
            "--out-dir",
            str(recipe_root / "results/_scaling_snapshot"),
            "--measure-summary",
            str(summary_path),
            "--subs",
            *[str(s) for s in args.subs],
        ]
        print(f"[sweep] running: {' '.join(plot_cmd)}", flush=True)
        subprocess.run(plot_cmd, check=True, cwd=recipe_root)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
