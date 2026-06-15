#!/usr/bin/env python3
"""Build matrix JSON files for the "Option B" full-sweep submission:

  Phase A: 16 g=1 redos (existing g=1 cells, re-run with --ntasks-per-node=1
           so they're true single-replica measurements).
  Phase B: 28 "--" cells -- high-resolution memcheck-style sweeps at all
           GPU counts that we did not previously schedule.

Outputs (writes into the benchmarks/ directory):

  matrix_optionB_phaseA_g1redo.json     -- 16 entries, all g=1
  matrix_optionB_phaseB_g1.json         -- 7 entries, new high-sub g=1
  matrix_optionB_phaseB_g16.json        -- 8 entries
  matrix_optionB_phaseB_g32.json        -- 8 entries
  matrix_optionB_phaseB_g64.json        -- 8 entries (some likely OOM)

Run:  python build_full_sweep_matrices.py
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCHMARKS_DIR = HERE.parent
RECIPE_ROOT = BENCHMARKS_DIR.parent

# Walltime estimates (minutes) keyed by (num_gpus, sampling).  Conservative
# upper bounds; submit_all.sh adds a 20% safety margin and rounds up.
# The numbers come from observed Phase 1/2/3 runs:
#   * g=1 ranges ~5 min @ sub=10K up to ~25 min @ sub=400K
#   * g=4 (kept for completeness, though phase A has no g=4) ~5-30 min
#   * g=16 dominated by stage_data on NVMe + ~5-10 min train
#   * g=32 / g=64 dominated by allocation/comm overhead, ~10-15 min train
def walltime_min(num_gpus: int, sampling: int, storage: str) -> int:
    # Train time grows ~linearly with sampling and ~inversely with num_gpus.
    # Use an empirical formula calibrated to existing runs.
    base = {1: 28, 4: 8, 16: 5, 32: 5, 64: 5}[num_gpus]
    sub_factor = sampling / 100_000  # 1.0 at sub=100K, scales linearly
    train_min = base * max(sub_factor, 0.3)
    # Stage time (NVMe only): ~5-8 min at low concurrency
    stage_min = 8 if storage == "nvme" else 0
    # Overhead: allocation + warmup + summarize
    overhead_min = 5 if num_gpus < 32 else 8
    total = train_min + stage_min + overhead_min
    # Round up to nearest 5 min, with floors/caps
    rounded = max(15, int(total + 4) // 5 * 5)
    return min(rounded, 90)  # 90 min hard cap per row


def make_entry(*, model: str, dataset: str, num_gpus: int,
               storage: str, sampling: int, phase_label: str) -> dict:
    nodes = max(1, num_gpus // 4)
    run_id = (
        f"{model}__{dataset}__g{num_gpus}__{storage}__sub{sampling}"
    )
    results_dir = (
        RECIPE_ROOT / "results" / model / dataset
        / f"gpus_{num_gpus}" / storage / f"sub_{sampling}"
    )
    return {
        "run_id": run_id,
        "model": model,
        "dataset": dataset,
        "num_gpus": num_gpus,
        "nodes": nodes,
        "storage": storage,
        "sampling_resolution": sampling,
        "phase": 0,
        "skip": False,
        "skip_reason": None,
        "results_dir": str(results_dir),
        "est_walltime_min": walltime_min(num_gpus, sampling, storage),
        "extra_overrides": [],
        "_optionB_label": phase_label,
    }


def main() -> None:
    matrices: dict[str, list[dict]] = {}

    # ---------- Phase A: 16 g=1 redos (existing g=1 cells) ----------
    # Existing g=1 cells, all 4 sub levels in {10K, 50K, 100K, 200K},
    # both surf+vol, both lustre+nvme.  Submit with the new
    # --ntasks-per-node=1 convention via the (already-modified)
    # submit_all.sh + run_benchmark.sbatch.
    phase_a: list[dict] = []
    for storage in ("lustre", "nvme"):
        for kind in ("surface", "volume"):
            for sub in (10_000, 50_000, 100_000, 200_000):
                phase_a.append(make_entry(
                    model=f"geotransolver_{kind}",
                    dataset=f"drivaer_ml_{kind}",
                    num_gpus=1,
                    storage=storage,
                    sampling=sub,
                    phase_label="A_g1_redo",
                ))
    matrices["matrix_optionB_phaseA_g1redo.json"] = phase_a

    # ---------- Phase B cells (28 total) ----------
    # Decomposition by (storage, kind, sub):
    #   lustre surf 250K, 300K @ g=1, 16, 32, 64           = 8
    #   lustre vol  250K, 300K, 400K, 500K @ g=1, 16, 32, 64 = 16
    #   nvme  vol  400K @ g=1, 16, 32, 64                  = 4
    # Total = 28.

    phase_b_g1: list[dict] = []
    phase_b_g16: list[dict] = []
    phase_b_g32: list[dict] = []
    phase_b_g64: list[dict] = []

    target_specs = [
        # (storage, kind, sub levels)
        ("lustre", "surface", [250_000, 300_000]),
        ("lustre", "volume",  [250_000, 300_000, 400_000, 500_000]),
        ("nvme",   "volume",  [400_000]),
    ]
    for storage, kind, sub_list in target_specs:
        for sub in sub_list:
            for ngpu, bucket in (
                (1, phase_b_g1), (16, phase_b_g16),
                (32, phase_b_g32), (64, phase_b_g64),
            ):
                bucket.append(make_entry(
                    model=f"geotransolver_{kind}",
                    dataset=f"drivaer_ml_{kind}",
                    num_gpus=ngpu,
                    storage=storage,
                    sampling=sub,
                    phase_label=f"B_g{ngpu}_new",
                ))

    matrices["matrix_optionB_phaseB_g1.json"]  = phase_b_g1
    matrices["matrix_optionB_phaseB_g16.json"] = phase_b_g16
    matrices["matrix_optionB_phaseB_g32.json"] = phase_b_g32
    matrices["matrix_optionB_phaseB_g64.json"] = phase_b_g64

    # Write all + summary
    print("=" * 70)
    print(f"{'Matrix':<40} {'Cells':>6} {'Σ walltime (h)':>16}")
    print("-" * 70)
    grand_cells = 0
    grand_minutes = 0
    for fname, entries in matrices.items():
        path = BENCHMARKS_DIR / fname
        with path.open("w") as f:
            json.dump(entries, f, indent=2)
        total_min = sum(e["est_walltime_min"] for e in entries)
        cells = len(entries)
        grand_cells += cells
        grand_minutes += total_min
        print(f"{fname:<40} {cells:>6} {total_min/60:>16.2f}")
    print("-" * 70)
    print(f"{'TOTAL':<40} {grand_cells:>6} {grand_minutes/60:>16.2f}")
    print("=" * 70)
    print()
    print("Suggested submission order (avoid cluster contention):")
    print("  1. matrix_optionB_phaseA_g1redo.json   (cheap, validates new convention)")
    print("  2. matrix_optionB_phaseB_g1.json       (cheap, completes g=1 sweep)")
    print("  3. matrix_optionB_phaseB_g16.json      (medium)")
    print("  4. matrix_optionB_phaseB_g32.json      (heavy)")
    print("  5. matrix_optionB_phaseB_g64.json      (heaviest; some OOM expected)")


if __name__ == "__main__":
    main()
