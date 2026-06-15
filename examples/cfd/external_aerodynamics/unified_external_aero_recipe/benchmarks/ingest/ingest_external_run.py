#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Copy an external metrics.jsonl into the CAE results tree and summarize it."""

from __future__ import annotations

import sys
from pathlib import Path

_BENCHMARKS_ROOT = Path(__file__).resolve().parents[1]
if str(_BENCHMARKS_ROOT) not in sys.path:
    sys.path.insert(0, str(_BENCHMARKS_ROOT))
import argparse
import json
import shutil
from dataclasses import asdict
from pathlib import Path

from ingest.summarize_run import render_markdown, summarize


def _run_dir(
    results_root: Path,
    *,
    model: str,
    dataset: str,
    num_gpus: int,
    storage: str,
    sampling: int,
    run_id: str,
) -> Path:
    return (
        results_root
        / model
        / dataset
        / f"gpus_{num_gpus}"
        / storage
        / f"sub_{sampling}"
        / "runs"
        / run_id
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--metrics", type=Path, required=True, help="Source metrics.jsonl")
    p.add_argument("--results-root", type=Path, default=Path("results"))
    p.add_argument("--model", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--num-gpus", type=int, required=True)
    p.add_argument("--storage", choices=["lustre", "nvme"], required=True)
    p.add_argument("--sampling", type=int, required=True)
    p.add_argument("--num-epochs", type=int, default=5)
    p.add_argument(
        "--run-id",
        default=None,
        help="Canonical run_id (default: geotransolver_* matrix naming)",
    )
    args = p.parse_args()

    if not args.metrics.is_file():
        print(f"[ingest] missing metrics: {args.metrics}")
        return 2

    run_id = args.run_id or (
        f"{args.model}__{args.dataset}__g{args.num_gpus}__{args.storage}__sub{args.sampling}"
    )
    dest = _run_dir(
        args.results_root,
        model=args.model,
        dataset=args.dataset,
        num_gpus=args.num_gpus,
        storage=args.storage,
        sampling=args.sampling,
        run_id=run_id,
    )
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.metrics, dest / "metrics.jsonl")

    summary = summarize(
        dest / "metrics.jsonl",
        run_id=run_id,
        model=args.model,
        dataset=args.dataset,
        num_gpus=args.num_gpus,
        storage=args.storage,
        sampling_resolution=args.sampling,
        num_epochs=args.num_epochs,
    )
    (dest / "benchmark_summary.json").write_text(
        json.dumps(asdict(summary), indent=2, default=str) + "\n"
    )
    (dest / "benchmark_summary.md").write_text(render_markdown(summary))
    print(f"[ingest] wrote {dest / 'benchmark_summary.json'}")
    print(
        f"[ingest] thr_p50={summary.throughput_samples_per_sec_p50:.2f} samples/s  "
        f"peak_mem={summary.memory.peak_gb:.1f} GB"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
