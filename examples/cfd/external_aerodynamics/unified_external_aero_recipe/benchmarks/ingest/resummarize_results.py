#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Re-run summarize_run.py on existing benchmark runs (no retrain required)."""

from __future__ import annotations

import sys
from pathlib import Path

_BENCHMARKS_ROOT = Path(__file__).resolve().parents[1]
if str(_BENCHMARKS_ROOT) not in sys.path:
    sys.path.insert(0, str(_BENCHMARKS_ROOT))
import argparse
import json
from dataclasses import asdict
from pathlib import Path

from ingest.summarize_run import parse_exclude_epochs, render_markdown, summarize


def _resummarize_one(
    summary_path: Path,
    *,
    exclude_epochs: tuple[int, ...],
    dry_run: bool,
) -> bool:
    old = json.loads(summary_path.read_text())
    metrics_path = Path(old.get("metrics_path", summary_path.parent / "metrics.jsonl"))
    if not metrics_path.is_file():
        # Fall back to colocated metrics.jsonl when metrics_path is stale.
        local_metrics = summary_path.parent / "metrics.jsonl"
        if local_metrics.is_file():
            metrics_path = local_metrics
        else:
            print(f"[resummarize] skip (no metrics): {summary_path}")
            return False

    new = summarize(
        metrics_path,
        run_id=old["run_id"],
        model=old["model"],
        dataset=old["dataset"],
        num_gpus=int(old["num_gpus"]),
        storage=old["storage"],
        sampling_resolution=int(old["sampling_resolution"]),
        num_epochs=int(old["num_epochs"]),
        exclude_epochs=exclude_epochs,
    )
    if dry_run:
        print(
            f"[resummarize] dry-run {old['run_id']}: "
            f"thr {old.get('throughput_samples_per_sec_p50')} -> "
            f"{new.throughput_samples_per_sec_p50}"
        )
        return True

    out_dir = summary_path.parent
    json_path = out_dir / "benchmark_summary.json"
    md_path = out_dir / "benchmark_summary.md"
    json_path.write_text(json.dumps(asdict(new), indent=2, default=str) + "\n")
    md_path.write_text(render_markdown(new))
    print(f"[resummarize] wrote {json_path}")
    return True


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--results-root",
        type=Path,
        default=Path("results"),
        help="Root directory to walk for benchmark_summary.json",
    )
    p.add_argument(
        "--exclude-epochs",
        default="0",
        help="Same as summarize_run.py (default: 0)",
    )
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    exclude = parse_exclude_epochs(args.exclude_epochs)
    root = args.results_root
    if not root.is_dir():
        print(f"[resummarize] missing results root: {root}")
        return 2

    ok = 0
    skip = 0
    for path in sorted(root.rglob("benchmark_summary.json")):
        if "_smoketest" in str(path):
            continue
        if _resummarize_one(path, exclude_epochs=exclude, dry_run=args.dry_run):
            ok += 1
        else:
            skip += 1

    print(f"[resummarize] updated={ok} skipped={skip}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
