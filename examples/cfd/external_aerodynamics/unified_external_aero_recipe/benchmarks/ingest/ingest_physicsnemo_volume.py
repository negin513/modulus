#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Ingest GeoTransolver Volume full runs from physicsnemo external-aero results."""

from __future__ import annotations

import sys
from pathlib import Path

_BENCHMARKS_ROOT = Path(__file__).resolve().parents[1]
if str(_BENCHMARKS_ROOT) not in sys.path:
    sys.path.insert(0, str(_BENCHMARKS_ROOT))
import argparse
import json
import re
import shutil
from dataclasses import asdict
from pathlib import Path

from ingest.summarize_run import render_markdown, summarize

MODEL = "geotransolver_volume"
DATASET = "drivaer_ml_volume"
MIN_AGGREGATED_STEPS = 11


def _read_config(metrics_path: Path) -> dict:
    for line in metrics_path.read_text().splitlines():
        rec = json.loads(line)
        if rec.get("phase") == "config":
            params = rec.get("params") or {}
            return {
                "sampling_resolution": int(params.get("sampling_resolution", 0)),
                "num_epochs": int(params.get("training.num_epochs", 5)),
                "model": params.get("dataset", DATASET),
            }
    return {}


def _parse_run_dir(name: str) -> tuple[int, str] | None:
    m = re.match(
        r"bench__geotransolver_volume__(?:samp\d+__)?gpu(\d+)__(lustre|nvme)__full__",
        name,
    )
    if m:
        return int(m.group(1)), m.group(2)
    m2 = re.match(r"bench__geotransolver_volume__(lustre|nvme)__full__", name)
    if m2:
        return 1, m2.group(1)
    return None


def _dest_dir(
    results_root: Path,
    *,
    num_gpus: int,
    storage: str,
    sampling: int,
    run_id: str,
) -> Path:
    return (
        results_root
        / MODEL
        / DATASET
        / f"gpus_{num_gpus}"
        / storage
        / f"sub_{sampling}"
        / "runs"
        / run_id
    )


def _ingest_one(
    src_dir: Path,
    results_root: Path,
    *,
    dry_run: bool,
) -> bool:
    metrics_path = src_dir / "metrics.jsonl"
    if not metrics_path.is_file() or "io_only" in src_dir.name:
        return False

    parsed = _parse_run_dir(src_dir.name)
    if parsed is None:
        print(f"[ingest] skip (unparsed name): {src_dir.name}")
        return False
    num_gpus, storage = parsed

    cfg = _read_config(metrics_path)
    sampling = cfg.get("sampling_resolution", 0)
    num_epochs = cfg.get("num_epochs", 5)
    if sampling <= 0:
        print(f"[ingest] skip (no sampling in config): {src_dir.name}")
        return False

    run_id = f"{MODEL}__{DATASET}__g{num_gpus}__{storage}__sub{sampling}"
    summary = summarize(
        metrics_path,
        run_id=run_id,
        model=MODEL,
        dataset=DATASET,
        num_gpus=num_gpus,
        storage=storage,
        sampling_resolution=sampling,
        num_epochs=num_epochs,
    )
    if summary.throughput_samples_per_sec_p50 is None:
        print(f"[ingest] skip (no throughput): {src_dir.name}")
        return False
    if summary.n_train_steps_aggregated < MIN_AGGREGATED_STEPS:
        print(
            f"[ingest] skip (only {summary.n_train_steps_aggregated} aggregated steps): "
            f"{src_dir.name}"
        )
        return False

    dest = _dest_dir(
        results_root,
        num_gpus=num_gpus,
        storage=storage,
        sampling=sampling,
        run_id=run_id,
    )
    if dry_run:
        print(
            f"[ingest] dry-run {src_dir.name} -> g={num_gpus} sub={sampling} {storage} "
            f"thr={summary.throughput_samples_per_sec_p50:.2f}"
        )
        return True

    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy2(metrics_path, dest / "metrics.jsonl")
    (dest / "benchmark_summary.json").write_text(
        json.dumps(asdict(summary), indent=2, default=str) + "\n"
    )
    (dest / "benchmark_summary.md").write_text(render_markdown(summary))
    (dest / "provenance.json").write_text(
        json.dumps(
            {
                "source": "physicsnemo_external_aerodynamics",
                "source_dir": str(src_dir),
                "source_run_id": src_dir.name,
            },
            indent=2,
        )
        + "\n"
    )
    print(
        f"[ingest] g={num_gpus:3d} sub={sampling:7d} {storage:5s} "
        f"thr={summary.throughput_samples_per_sec_p50:7.2f} -> {dest}"
    )
    return True


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--external-results",
        type=Path,
        required=True,
        help="Root directory of external geotransolver_volume bench__* run folders",
    )
    p.add_argument("--results-root", type=Path, default=Path("results"))
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    external_results = args.external_results.expanduser().resolve()
    if not external_results.is_dir():
        print(f"[ingest] missing external results: {external_results}")
        return 2

    # Latest run wins per (sub, g, storage).
    candidates: dict[tuple[int, int, str], Path] = {}
    for src_dir in external_results.glob("bench__geotransolver_volume__*"):
        metrics_path = src_dir / "metrics.jsonl"
        if not metrics_path.is_file() or "io_only" in src_dir.name:
            continue
        parsed = _parse_run_dir(src_dir.name)
        if parsed is None:
            continue
        num_gpus, storage = parsed
        cfg = _read_config(src_dir / "metrics.jsonl")
        sampling = cfg.get("sampling_resolution", 0)
        if sampling <= 0:
            continue
        key = (sampling, num_gpus, storage)
        if key not in candidates or src_dir.name > candidates[key].name:
            candidates[key] = src_dir

    ok = 0
    for src_dir in sorted(candidates.values(), key=lambda p: p.name):
        if _ingest_one(src_dir, args.results_root, dry_run=args.dry_run):
            ok += 1

    print(f"[ingest] processed {ok}/{len(candidates)} volume cells")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
