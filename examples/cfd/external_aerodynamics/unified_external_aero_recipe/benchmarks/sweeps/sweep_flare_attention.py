#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Sweep GeoTransolver GALE vs GALE_FA (FLARE attention) on ShiftSUV (surface).

Default dataset is ``shift_suv_estate_surface`` (readable on HSG). DrivAerML
volume runs are opt-in via ``--dataset drivaer_ml_volume --model geotransolver_volume``.

Runs one training job per (modality, subsample, flare_on/off), records OOM
explicitly, aggregates metrics via summarize_run.py, and writes CSV/JSON.

Example (inside recipe container venv on a B200 node)::

    export DATASET_PATH_DRIVAER_ML=/lustre/fsw/.../PhysicsNeMo-DrivaerML
    python benchmarks/sweeps/sweep_flare_attention.py --device cuda

Cluster::

    sbatch benchmarks/run_flare_attention_benchmark.sbatch
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_SUBS = (
    50_000,
    100_000,
    150_000,
    200_000,
    250_000,
    300_000,
    350_000,
    400_000,
    450_000,
    500_000,
)

OOM_PATTERNS = (
    re.compile(r"CUDA out of memory", re.I),
    re.compile(r"OutOfMemoryError", re.I),
    re.compile(r"CUBLAS_STATUS_ALLOC_FAILED", re.I),
)
PERMISSION_PATTERNS = (
    re.compile(r"Permission denied", re.I),
    re.compile(r"permission denied", re.I),
)


@dataclass
class SweepCell:
    modality: str  # volume | surface
    model: str
    dataset: str
    subsampling: int
    flare_attention: bool
    attention_type: str  # GALE | GALE_FA

    @property
    def run_id(self) -> str:
        flare_tag = "gale_fa" if self.flare_attention else "gale"
        return (
            f"flare_bench__{self.model}__sub{self.subsampling}__{flare_tag}__g1"
        )


def _recipe_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _benchmarks_dir() -> Path:
    return Path(__file__).resolve().parent


def _run_cmd(cmd: list[str], *, cwd: Path, env: dict[str, str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log_f:
        log_f.write(f"# cmd: {' '.join(cmd)}\n\n")
        log_f.flush()
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            env=env,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return proc.returncode


def _tail_text(path: Path, n: int = 80) -> str:
    if not path.exists():
        return ""
    lines = path.read_text(errors="replace").splitlines()
    return "\n".join(lines[-n:])


def _classify_failure(rc: int, log_path: Path) -> tuple[str, str | None]:
    """Return (status, error_detail). status in ok|oom|data_denied|failed."""
    if rc == 0:
        return "ok", None
    tail = _tail_text(log_path)
    for pat in OOM_PATTERNS:
        if pat.search(tail):
            return "oom", "CUDA OOM (detected in train log tail)"
    for pat in PERMISSION_PATTERNS:
        if pat.search(tail):
            return "data_denied", "Dataset file permission denied"
    return "failed", f"train exited rc={rc}"


def _gpu_name() -> str:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True,
            timeout=30,
        )
        names = [ln.strip() for ln in out.strip().splitlines() if ln.strip()]
        return names[0] if names else "unknown"
    except (subprocess.SubprocessError, FileNotFoundError):
        return "unknown"


def _pkg_version(dist: str) -> str | None:
    try:
        out = subprocess.check_output(
            [sys.executable, "-m", "pip", "show", dist],
            text=True,
            timeout=60,
        )
        for line in out.splitlines():
            if line.startswith("Version:"):
                return line.split(":", 1)[1].strip()
    except subprocess.SubprocessError:
        return None
    return None


def collect_provenance(
    *,
    args: argparse.Namespace,
    results_dir: Path,
) -> dict[str, Any]:
    import torch

    physicsnemo_ver = _pkg_version("nvidia-physicsnemo") or _pkg_version(
        "physicsnemo"
    )
    cmd = (
        f"python benchmarks/sweeps/sweep_flare_attention.py "
        f"--results-dir {results_dir} "
        f"--storage {args.storage} "
        f"--precision {args.precision} "
        f"--num-epochs {args.num_epochs} "
        f"--exclude-epochs {args.exclude_epochs} "
        f"--subs {','.join(str(s) for s in args.subs_list)}"
        f" --model {args.model} --dataset {args.dataset}"
        f" --attention-modes {args.attention_modes}"
    )
    args.modalities = ["surface" if "surface" in args.model else "volume"]
    mode_labels = {
        "both": ["GALE (FLARE off)", "GALE_FA (FLARE on)"],
        "gale": ["GALE (FLARE off)"],
        "gale_fa": ["GALE_FA (FLARE on)"],
        "flare": ["GALE_FA (FLARE on)"],
        "fa": ["GALE_FA (FLARE on)"],
    }
    prov = {
        "benchmark": "geotransolver_flare_attention",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "gpu_name": _gpu_name(),
        "gpu_count": 1,
        "precision": args.precision,
        "storage_tier": args.storage,
        "compile": args.compile,
        "dataset": args.dataset,
        "model": args.model,
        "software": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "physicsnemo": physicsnemo_ver,
        },
        "training": {
            "num_epochs": args.num_epochs,
            "exclude_epochs": list(args.exclude_epochs_list),
            "batch_size": 1,
        },
        "sweep": {
            "subsampling_points": args.subs_list,
            "attention_modes": args.attention_modes,
            "flare_modes": mode_labels.get(
                args.attention_modes.strip().lower(),
                [args.attention_modes],
            ),
            "modalities": args.modalities,
        },
        "command": cmd,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "hostname": os.environ.get("SLURMD_NODENAME") or os.environ.get("HOSTNAME"),
    }
    return prov


def _val_loss_from_metrics(
    metrics_path: Path, exclude_epochs: frozenset[int]
) -> dict[str, float | None]:
    losses: list[tuple[int, float]] = []
    with metrics_path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("phase") != "val":
                continue
            epoch = rec.get("epoch")
            loss = rec.get("loss")
            if epoch is None or loss is None:
                continue
            losses.append((int(epoch), float(loss)))
    if not losses:
        return {"val_loss_last": None, "val_loss_mean": None}
    agg = [(e, v) for e, v in losses if e not in exclude_epochs]
    if not agg:
        agg = losses
    values = [v for _, v in agg]
    return {
        "val_loss_last": losses[-1][1],
        "val_loss_mean": sum(values) / len(values),
    }


def _row_from_summary(
    cell: SweepCell,
    *,
    status: str,
    error_detail: str | None,
    summary_path: Path | None,
    metrics_path: Path | None,
    train_rc: int,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    exclude = frozenset(provenance["training"]["exclude_epochs"])
    row: dict[str, Any] = {
        "run_id": cell.run_id,
        "modality": cell.modality,
        "model": cell.model,
        "dataset": cell.dataset,
        "subsampling": cell.subsampling,
        "flare_attention": cell.flare_attention,
        "attention_type": cell.attention_type,
        "status": status,
        "oom": status == "oom",
        "train_exit_code": train_rc,
        "error_detail": error_detail,
        "gpu_name": provenance["gpu_name"],
        "precision": provenance["precision"],
        "storage_tier": provenance["storage_tier"],
        "peak_mem_gb": None,
        "train_step_p50_s": None,
        "train_step_mean_s": None,
        "throughput_samples_per_sec_p50": None,
        "val_loss_last": None,
        "val_loss_mean": None,
    }
    if metrics_path and metrics_path.exists():
        row.update(_val_loss_from_metrics(metrics_path, exclude))
    if summary_path and summary_path.exists():
        summary = json.loads(summary_path.read_text())
        mem = summary.get("memory") or {}
        train = summary.get("train") or {}
        row["peak_mem_gb"] = mem.get("peak_gb")
        row["train_step_p50_s"] = train.get("p50")
        row["train_step_mean_s"] = train.get("mean")
        row["throughput_samples_per_sec_p50"] = summary.get(
            "throughput_samples_per_sec_p50"
        )
        if row["val_loss_last"] is None:
            # Epoch rows may carry loss in future; summary lacks it today.
            pass
    return row


def _flare_flags_for_modes(attention_modes: str) -> tuple[bool, ...]:
    """Map CLI ``--attention-modes`` to (flare_attention,) sweep flags."""
    modes = attention_modes.strip().lower()
    if modes == "both":
        return (False, True)
    if modes == "gale":
        return (False,)
    if modes in ("gale_fa", "flare", "fa"):
        return (True,)
    raise ValueError(
        f"Unknown attention-modes {attention_modes!r}; use both, gale, or gale_fa"
    )


def build_cells(
    subs: tuple[int, ...],
    *,
    model: str,
    dataset: str,
    attention_modes: str = "both",
) -> list[SweepCell]:
    cells: list[SweepCell] = []
    if "volume" in model:
        modality = "volume"
    else:
        modality = "surface"
    modalities = [(modality, model, dataset)]
    flare_flags = _flare_flags_for_modes(attention_modes)
    for modality, model_name, dataset_name in modalities:
        for sub in subs:
            for flare in flare_flags:
                cells.append(
                    SweepCell(
                        modality=modality,
                        model=model_name,
                        dataset=dataset_name,
                        subsampling=sub,
                        flare_attention=flare,
                        attention_type="GALE_FA" if flare else "GALE",
                    )
                )
    return cells


def run_cell(
    cell: SweepCell,
    *,
    args: argparse.Namespace,
    results_dir: Path,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    recipe = _recipe_root()
    benchmarks = _benchmarks_dir()
    run_output = results_dir / "runs" / cell.run_id
    metrics_path = run_output / "metrics.jsonl"
    summary_path = run_output / "benchmark_summary.json"
    train_log = results_dir / "logs" / f"{cell.run_id}.log"

    hydra = [
        f"model={cell.model}",
        f"dataset={cell.dataset}",
        f"sampling_resolution={cell.subsampling}",
        f"training.num_epochs={args.num_epochs}",
        f"precision={args.precision}",
        f"compile={str(args.compile).lower()}",
        f"run_id={cell.run_id}",
        f"output_dir={results_dir / 'runs'}",
    ]
    if cell.flare_attention:
        hydra.append("+model.attention_type=GALE_FA")

    env = os.environ.copy()
    env["RANK"] = "0"
    env["WORLD_SIZE"] = "1"
    env["LOCAL_RANK"] = "0"
    env["MASTER_ADDR"] = "127.0.0.1"
    env["MASTER_PORT"] = str(29500 + (cell.subsampling % 1000))

    train_cmd = [sys.executable, "src/train.py", *hydra]
    print(
        f"[flare] {cell.run_id}  sub={cell.subsampling:,}  "
        f"attention={cell.attention_type}",
        flush=True,
    )
    if args.dry_run:
        return _row_from_summary(
            cell,
            status="dry_run",
            error_detail=None,
            summary_path=None,
            metrics_path=None,
            train_rc=0,
            provenance=provenance,
        )

    rc = _run_cmd(train_cmd, cwd=recipe, env=env, log_path=train_log)
    status, error_detail = _classify_failure(rc, train_log)

    if metrics_path.exists() and status == "ok":
        sum_cmd = [
            sys.executable,
            str(benchmarks / "ingest/summarize_run.py"),
            "--metrics",
            str(metrics_path),
            "--run-id",
            cell.run_id,
            "--model",
            cell.model,
            "--dataset",
            cell.dataset,
            "--num-gpus",
            "1",
            "--storage",
            args.storage,
            "--sampling",
            str(cell.subsampling),
            "--num-epochs",
            str(args.num_epochs),
            "--exclude-epochs",
            args.exclude_epochs,
        ]
        _run_cmd(
            sum_cmd,
            cwd=recipe,
            env=env,
            log_path=results_dir / "logs" / f"{cell.run_id}.summarize.log",
        )
    elif metrics_path.exists() and status == "oom":
        # Partial metrics may exist; still summarize for peak mem seen before OOM.
        sum_cmd = [
            sys.executable,
            str(benchmarks / "ingest/summarize_run.py"),
            "--metrics",
            str(metrics_path),
            "--run-id",
            cell.run_id,
            "--model",
            cell.model,
            "--dataset",
            cell.dataset,
            "--num-gpus",
            "1",
            "--storage",
            args.storage,
            "--sampling",
            str(cell.subsampling),
            "--num-epochs",
            str(args.num_epochs),
            "--exclude-epochs",
            args.exclude_epochs,
        ]
        _run_cmd(
            sum_cmd,
            cwd=recipe,
            env=env,
            log_path=results_dir / "logs" / f"{cell.run_id}.summarize.log",
        )

    return _row_from_summary(
        cell,
        status=status,
        error_detail=error_detail,
        summary_path=summary_path if summary_path.exists() else None,
        metrics_path=metrics_path if metrics_path.exists() else None,
        train_rc=rc,
        provenance=provenance,
    )


def write_results(
    path_json: Path, path_csv: Path, provenance: dict[str, Any], rows: list[dict]
) -> None:
    path_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {"provenance": provenance, "rows": rows}
    path_json.write_text(json.dumps(payload, indent=2))
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sweep GALE vs GALE_FA on GeoTransolver × DrivAerML."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/_flare_attention"),
    )
    parser.add_argument(
        "--subs",
        default=",".join(str(s) for s in DEFAULT_SUBS),
        help="Comma-separated subsample point counts.",
    )
    parser.add_argument(
        "--model",
        default="geotransolver_surface",
        help="Hydra model template (default: geotransolver_surface for ShiftSUV).",
    )
    parser.add_argument(
        "--dataset",
        default="shift_suv_estate_surface",
        help="Hydra dataset (default: shift_suv_estate_surface).",
    )
    parser.add_argument("--storage", default="lustre")
    parser.add_argument("--precision", default="bfloat16")
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--num-epochs", type=int, default=5)
    parser.add_argument(
        "--exclude-epochs",
        default="0",
        help="Epochs excluded from aggregates (default 0 = drop warmup epoch).",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-plot", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip cells whose run_id already appears in results JSON.",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help=(
            "With --resume, only skip rows with status=ok; re-run failed/oom rows "
            "and replace their entries in the results file."
        ),
    )
    parser.add_argument(
        "--attention-modes",
        choices=("both", "gale", "gale_fa"),
        default="both",
        help=(
            "both: GALE + GALE_FA (default A/B). "
            "gale: GALE only (FLARE off) for memory scaling. "
            "gale_fa: GALE_FA only (FLARE on)."
        ),
    )
    args = parser.parse_args()
    args.subs_list = [int(x.strip()) for x in args.subs.split(",") if x.strip()]
    if args.exclude_epochs.strip().lower() in {"", "none"}:
        args.exclude_epochs_list: list[int] = []
    else:
        args.exclude_epochs_list = [
            int(x) for x in args.exclude_epochs.split(",") if x.strip()
        ]

    results_dir = args.results_dir.resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    provenance = collect_provenance(args=args, results_dir=results_dir)
    (results_dir / "provenance.json").write_text(json.dumps(provenance, indent=2))

    json_path = results_dir / "flare_attention_results.json"
    csv_path = results_dir / "flare_attention_results.csv"
    existing_rows: list[dict] = []
    done_ids: set[str] = set()
    if args.resume and json_path.exists():
        prior = json.loads(json_path.read_text())
        existing_rows = list(prior.get("rows") or [])
        if args.retry_failed:
            done_ids = {r["run_id"] for r in existing_rows if r.get("status") == "ok"}
        else:
            done_ids = {r["run_id"] for r in existing_rows}

    cells = build_cells(
        tuple(args.subs_list),
        model=args.model,
        dataset=args.dataset,
        attention_modes=args.attention_modes,
    )
    rows = list(existing_rows)
    for cell in cells:
        if args.resume and cell.run_id in done_ids:
            print(f"[flare] skip (resume) {cell.run_id}", flush=True)
            continue
        row = run_cell(cell, args=args, results_dir=results_dir, provenance=provenance)
        if args.retry_failed:
            rows = [r for r in rows if r["run_id"] != cell.run_id]
        rows.append(row)
        write_results(json_path, csv_path, provenance, rows)

    write_results(json_path, csv_path, provenance, rows)
    print(f"[flare] wrote {json_path}", flush=True)
    print(f"[flare] wrote {csv_path}", flush=True)

    if not args.skip_plot and not args.dry_run:
        plot_script = _benchmarks_dir() / "plots/plot_flare_attention.py"
        if plot_script.exists():
            subprocess.run(
                [
                    sys.executable,
                    str(plot_script),
                    "--input",
                    str(json_path),
                    "--out-dir",
                    str(results_dir / "plots"),
                ],
                check=False,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
