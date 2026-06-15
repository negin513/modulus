#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ---------------------------------------------------------------------------
# benchmark_matrix.py -- generate the CAE benchmark run matrix
# ---------------------------------------------------------------------------
#
# Emits one entry per concrete benchmark run as JSON, suitable for
# `submit_all.sh` to consume.  The matrix axes follow Sanjay's directive:
#
#   model              {geotransolver_surface, geotransolver_volume}
#   dataset            {drivaer_ml_surface, drivaer_ml_volume, drivesim_volume}
#   num_gpus           [1, 4, 16, 32, 64]   (4 GPUs per node on HSG)
#   storage            [lustre, nvme]
#   sampling_resolution [10000, 50000, 100000, 200000]
#
# Surface models are paired only with surface datasets and likewise for
# volume.  DriveSim entries are emitted with `skip=true` until
# `dataset_paths.drivesim` is configured (i.e. Muhammad has confirmed
# the path).
#
# Phase tags follow the rollout ordering in the plan:
#   1: smoke   (1, 4 GPUs / DrivAerML / Lustre)
#   2: scale   (16, 32, 64 GPUs / DrivAerML / both storage)
#   3: nvme_small (1, 4 GPUs / DrivAerML / NVMe)
#   4: drivesim (everything DriveSim, gated)
#
# Usage:
#   python benchmark_matrix.py --output matrix.json
#   python benchmark_matrix.py --filter-gpus 1,4 --filter-storage lustre
#   python benchmark_matrix.py --phase 1 --output matrix_phase1.json
#   python benchmark_matrix.py --exclude-pending --output matrix_runnable.json
# ---------------------------------------------------------------------------

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

GPUS_PER_NODE = 4
DEFAULT_OUTPUT = "matrix.json"

MODELS = ["geotransolver_surface", "geotransolver_volume"]
DATASETS = ["drivaer_ml_surface", "drivaer_ml_volume", "drivesim_volume"]
GPU_SCALES = [1, 4, 16, 32, 64]
STORAGE_MODES = ["lustre", "nvme"]
SAMPLING_RESOLUTIONS = [10000, 50000, 100000, 200000]

### Datasets that are not yet wired up on disk get `skip=true` rows so
### `submit_all.sh` can either skip them silently or print a reminder.
PENDING_DATASETS = {"drivesim_volume"}


@dataclass
class RunSpec:
    run_id: str
    model: str
    dataset: str
    num_gpus: int
    nodes: int
    storage: str
    sampling_resolution: int
    phase: int
    skip: bool
    skip_reason: str | None
    results_dir: str
    est_walltime_min: int
    extra_overrides: list[str] = field(default_factory=list)


def _modality(name: str) -> str:
    """Return 'surface' or 'volume' suffix from a model/dataset name."""
    if name.endswith("_surface"):
        return "surface"
    if name.endswith("_volume"):
        return "volume"
    raise ValueError(f"cannot infer modality from {name!r}")


def _is_valid_pair(model: str, dataset: str) -> bool:
    return _modality(model) == _modality(dataset)


def _phase_for(num_gpus: int, dataset: str, storage: str) -> int:
    """Phase tag per the plan's rollout ordering."""
    if dataset in PENDING_DATASETS:
        return 4
    is_small = num_gpus in (1, 4)
    if is_small and storage == "lustre":
        return 1
    if not is_small:
        return 2
    return 3  # small + nvme


### Very rough wall-clock estimate per run, used as a planning
### placeholder so submit_all.sh can sanity-check Slurm time limits.
### Updated empirically after Phase 1 runs.
_BASE_WALLTIME_MIN = {
    "surface": 20,
    "volume": 35,
}


def _est_walltime(model: str, dataset: str, num_gpus: int, sampling: int) -> int:
    base = _BASE_WALLTIME_MIN[_modality(model)]
    # weak scaling assumption for 5-epoch run; small overhead at large GPU
    # counts because of the all-reduce on a tiny per-rank workload.
    sampling_factor = sampling / 100000.0
    gpu_factor = max(1.0, math.log2(num_gpus + 1) / 3.0)
    return int(round(base * sampling_factor * gpu_factor)) + 5  # +5 for stage_in / startup


def _results_dir(model: str, dataset: str, num_gpus: int, storage: str, sampling: int) -> str:
    return os.path.join(
        "results",
        model,
        dataset,
        f"gpus_{num_gpus}",
        storage,
        f"sub_{sampling}",
    )


def _run_id(model: str, dataset: str, num_gpus: int, storage: str, sampling: int) -> str:
    return f"{model}__{dataset}__g{num_gpus}__{storage}__sub{sampling}"


def _generate_specs() -> list[RunSpec]:
    specs: list[RunSpec] = []
    for model in MODELS:
        for dataset in DATASETS:
            if not _is_valid_pair(model, dataset):
                continue
            for num_gpus in GPU_SCALES:
                for storage in STORAGE_MODES:
                    for sampling in SAMPLING_RESOLUTIONS:
                        skip = dataset in PENDING_DATASETS
                        skip_reason = (
                            "dataset_paths.drivesim not yet configured "
                            "(awaiting Muhammad)"
                            if skip
                            else None
                        )
                        specs.append(
                            RunSpec(
                                run_id=_run_id(model, dataset, num_gpus, storage, sampling),
                                model=model,
                                dataset=dataset,
                                num_gpus=num_gpus,
                                nodes=max(1, math.ceil(num_gpus / GPUS_PER_NODE)),
                                storage=storage,
                                sampling_resolution=sampling,
                                phase=_phase_for(num_gpus, dataset, storage),
                                skip=skip,
                                skip_reason=skip_reason,
                                results_dir=_results_dir(model, dataset, num_gpus, storage, sampling),
                                est_walltime_min=_est_walltime(model, dataset, num_gpus, sampling),
                            )
                        )
    return specs


def _filter_specs(
    specs: list[RunSpec],
    *,
    gpus: set[int] | None,
    storage: set[str] | None,
    models: set[str] | None,
    datasets: set[str] | None,
    phase: int | None,
    exclude_pending: bool,
) -> list[RunSpec]:
    out: list[RunSpec] = []
    for s in specs:
        if gpus is not None and s.num_gpus not in gpus:
            continue
        if storage is not None and s.storage not in storage:
            continue
        if models is not None and s.model not in models:
            continue
        if datasets is not None and s.dataset not in datasets:
            continue
        if phase is not None and s.phase != phase:
            continue
        if exclude_pending and s.skip:
            continue
        out.append(s)
    return out


def _format_summary(specs: list[RunSpec]) -> str:
    """Compact table showing how many runs hit each (model,dataset,storage)
    bucket, broken down by GPU count and sampling resolution."""
    if not specs:
        return "(empty matrix)"
    by_bucket: dict[tuple[str, str, str], list[RunSpec]] = {}
    for s in specs:
        key = (s.model, s.dataset, s.storage)
        by_bucket.setdefault(key, []).append(s)

    lines: list[str] = []
    lines.append(f"Total runs: {len(specs)}  (skipped: {sum(1 for s in specs if s.skip)})")
    lines.append("")
    header = (
        f"{'model':<26} {'dataset':<22} {'storage':<7} "
        f"{'#runs':>6} {'#gpus':<18} {'#sampling':<22} {'phase':<6} {'~walltime_min':>14}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for (model, dataset, storage), rows in sorted(by_bucket.items()):
        gpus = sorted({r.num_gpus for r in rows})
        sampling = sorted({r.sampling_resolution for r in rows})
        phases = sorted({r.phase for r in rows})
        total_walltime = sum(r.est_walltime_min for r in rows)
        marker = " *" if rows[0].skip else ""
        lines.append(
            f"{model:<26} {dataset:<22} {storage:<7} {len(rows):>6} "
            f"{','.join(map(str, gpus)):<18} {','.join(map(str, sampling)):<22} "
            f"{','.join(map(str, phases)):<6} {total_walltime:>14}{marker}"
        )
    if any(s.skip for s in specs):
        lines.append("")
        lines.append("(*) marked rows are skipped until upstream config lands")
    return "\n".join(lines)


def _parse_csv(value: str | None, cast=str) -> set | None:
    if value is None:
        return None
    return {cast(v.strip()) for v in value.split(",") if v.strip()}


def main(argv: Iterable[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", default=DEFAULT_OUTPUT, help="Output JSON path (default: %(default)s)")
    p.add_argument("--filter-gpus", type=str, default=None, help="Comma-separated GPU counts to keep")
    p.add_argument("--filter-storage", type=str, default=None, help="Comma-separated storage modes to keep")
    p.add_argument("--filter-model", type=str, default=None, help="Comma-separated model names to keep")
    p.add_argument("--filter-dataset", type=str, default=None, help="Comma-separated dataset names to keep")
    p.add_argument("--phase", type=int, default=None, help="Only emit runs with the given phase tag (1..4)")
    p.add_argument("--exclude-pending", action="store_true", help="Drop rows still gated on missing config")
    p.add_argument("--quiet", action="store_true", help="Skip the human-readable summary table")
    args = p.parse_args(list(argv) if argv is not None else None)

    specs = _generate_specs()
    specs = _filter_specs(
        specs,
        gpus=_parse_csv(args.filter_gpus, int),
        storage=_parse_csv(args.filter_storage),
        models=_parse_csv(args.filter_model),
        datasets=_parse_csv(args.filter_dataset),
        phase=args.phase,
        exclude_pending=args.exclude_pending,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fh:
        json.dump([asdict(s) for s in specs], fh, indent=2)

    if not args.quiet:
        print(_format_summary(specs))
        print()
        print(f"[matrix] wrote {len(specs)} runs to {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
