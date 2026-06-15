#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Rebuild e2e-training-performance.canvas.tsx with embedded plot PNGs."""

from __future__ import annotations

import argparse
import base64
import json
from datetime import datetime, timezone
from pathlib import Path

import sys
from pathlib import Path

_BENCHMARKS_ROOT = Path(__file__).resolve().parents[1]
if str(_BENCHMARKS_ROOT) not in sys.path:
    sys.path.insert(0, str(_BENCHMARKS_ROOT))
from paths import canvas_path
from ingest.summarize_run import num_epochs_for_aggregate

PLOT_FILES = {
    "IMG_THROUGHPUT": "01_throughput_vs_gpus.png",
    "IMG_EFFICIENCY": "02_efficiency_vs_gpus.png",
    "IMG_NVME_16": "03_nvme_vs_lustre_16gpu.png",
    "IMG_MEMORY": "04_memory_vs_sampling.png",
    "IMG_LUSTRE_G1": "05_single_gpu_throughput_epochtime_lustre.png",
    "IMG_SINGLE_GPU_MEM": "06_single_gpu_memory.png",
    "IMG_NVME_G16": "07_single_gpu_throughput_epochtime_nvme.png",
    "IMG_STORAGE_CMP": "08_lustre_vs_nvme_storage_compare.png",
}

IMPORTS = """import {
  Callout, Divider, Grid, H1, H2, H3, Stack, Stat, Table, Text, useHostTheme,
} from "cursor/canvas";
"""

BODY = r'''
const GT_SURFACE = "GeoTransolver Surface";
const GT_VOLUME = "GeoTransolver Volume";

const lustreG1 = [
  ["10,000", "5.25", "24.7", "0.190", "3.6", "2.79", "50.4", "0.358", "4.6"],
  ["50,000", "4.26", "30.6", "0.235", "15.5", "2.06", "61.0", "0.486", "20.2"],
  ["100,000", "3.69", "34.8", "0.271", "31.2", "1.58", "79.4", "0.635", "39.9"],
  ["200,000", "2.86", "45.6", "0.350", "60.1", "1.03", "114.2", "0.972", "77.6"],
];

const nvmeG16 = [
  ["10,000", "95.14", "6.2", "0.168", "53.48", "14.9", "0.299"],
  ["50,000", "83.21", "9.0", "0.192", "43.05", "13.0", "0.372"],
  ["100,000", "74.52", "13.1", "0.215", "31.21", "18.7", "0.513"],
  ["200,000", "59.10", "12.3", "0.271", "19.27", "26.6", "0.830"],
];

const compare100k = [
  ["Lustre g=1", "3.69", "34.8", "0.271", "1.58", "79.4", "0.635"],
  ["Lustre g=16", "39.27", "13.6", "0.407", "16.69", "29.1", "0.958"],
  ["NVMe g=16", "74.52", "13.1", "0.215", "31.21", "18.7", "0.513"],
];

const memRows = [
  ["10,000", "3.6", "98%", "4.6", "98%"],
  ["50,000", "15.5", "92%", "20.2", "89%"],
  ["100,000", "31.2", "84%", "39.9", "79%"],
  ["200,000", "60.1", "69%", "77.6", "60%"],
  ["250,000 ⓘ", "75.84", "61%", "96.71", "50%"],
  ["300,000 ⓘ", "90.71", "53%", "115.79", "40%"],
  ["400,000 ⓘⓘ", "—", "—", "154.21", "20%"],
];

const eff100kLustre = [
  ["1", "3.69", "100%", "1.58", "100%"],
  ["4", "14.84", "101%", "6.37", "101%"],
  ["16", "39.27", "67%", "16.69", "66%"],
  ["32", "85.22", "72%", "33.39", "66%"],
  ["64", "101.63", "43%", "63.70", "63%"],
];

const nvmeGain16Surface = [
  ["10,000", "51", "95", "+86%"],
  ["50,000", "50", "83", "+67%"],
  ["100,000", "39", "75", "+90%"],
  ["200,000", "31", "59", "+92%"],
];

const peakRuns = [
  ["1", GT_SURFACE, "64", "NVMe", "200,000", "212.3"],
  ["2", GT_SURFACE, "64", "NVMe", "10,000", "189.2"],
  ["3", GT_SURFACE, "64", "Lustre", "200,000", "170.7"],
  ["4", GT_SURFACE, "64", "NVMe", "50,000", "158.2"],
  ["5", GT_VOLUME, "64", "Lustre", "100,000", "63.7"],
];

function Figure({ n, src, caption }: { n: number; src: string; caption: string }) {
  const theme = useHostTheme();
  return (
    <Stack gap={8} style={{ marginBottom: 24 }}>
      <div style={{ border: `1px solid ${theme.stroke.tertiary}`, borderRadius: 4, overflow: "hidden" }}>
        <img src={src} alt={`Figure ${n}`} style={{ width: "100%", display: "block" }} />
      </div>
      <Text tone="secondary" size="small" style={{ textAlign: "center", lineHeight: 1.6 }}>
        <Text weight="semibold" as="span">Figure {n}. </Text>{caption}
      </Text>
    </Stack>
  );
}

function Para({ children }: { children: any }) {
  return <Text style={{ lineHeight: 1.75, maxWidth: 780 }}>{children}</Text>;
}

function TC({ n, children }: { n: number; children: any }) {
  return (
    <Text tone="secondary" size="small" style={{ marginTop: 8, lineHeight: 1.6 }}>
      <Text weight="semibold" as="span">Table {n}. </Text>{children}
    </Text>
  );
}

function Section({ n, title }: { n: number; title: string }) {
  return <H2 style={{ fontFamily: "inherit", marginTop: 8 }}>{n}. {title}</H2>;
}

function Sub({ id, title }: { id: string; title: string }) {
  return <H3 style={{ fontFamily: "inherit", fontSize: 15 }}>{id} {title}</H3>;
}

export default function CAEBenchmarkE2ETrainingReport() {
  const theme = useHostTheme();
  return (
    <div style={{ maxWidth: 780, margin: "0 auto", padding: "48px 32px 64px", fontFamily: "Georgia, serif" }}>
      <Stack gap={28}>
        <Stack gap={12} style={{ borderBottom: `1px solid ${theme.stroke.tertiary}`, paddingBottom: 24 }}>
          <Text tone="secondary" size="small">CAE Benchmarking · Milestone 1 — Internal Perf Benchmark</Text>
          <H1 style={{ fontFamily: "inherit" }}>
            End-to-End Model Training Performance: GeoTransolver Surface and GeoTransolver Volume on DrivAerML
          </H1>
          <Text tone="secondary">NVIDIA B200 (HSG) · May 29, 2026 · {{RUN_COUNT}} runs · Deadline 6/7/2026 · updated {{LAST_UPDATED}}</Text>
        </Stack>

        <Grid columns={4} gap={12}>
          <Stat value="212" label="Peak samples/s (GeoTransolver Surface, g=64, NVMe)" />
          <Stat value="77.6 GB" label="Peak VRAM (GeoTransolver Volume @ 200k)" tone="warning" />
          <Stat value="+67–131%" label="NVMe gain vs Lustre @ g=16" tone="success" />
          <Stat value="Pending" label="RTX 6000 Pro baseline" tone="neutral" />
        </Grid>

        <Section n={0} title="Problem Statement and Objectives" />
        <Para>
          PhysicsNeMo CAE benchmarking supports the value proposition: faster training, farther scaling, smaller
          hardware, preserved CAE accuracy, with reproducible measurements and clear cost efficiency. Milestone 1
          targets internal performance baselines for marketing collateral, gap identification, and NV hardware
          positioning.
        </Para>
        <Callout tone="info" title="This report — Milestone 1, Ask 3">
          Pick a model and measure training performance on single GPU as well as scaling across multi-GPU.
          Single GPU: throughput (samples/s), time per epoch, peak memory vs sample size. Multi-GPU: aggregate
          throughput and weak-scaling efficiency. GPU SKUs: B200 (complete), RTX 6000 Pro (pending).
        </Callout>

        <Sub id="0.1" title="Scope (Sanjay / Mohammad)" />
        <Table
          headers={["Dimension", "In scope for this artifact"]}
          rows={[
            ["Models", "GeoTransolver Surface, GeoTransolver Volume"],
            ["Dataset", "DrivAerML only"],
            ["Storage", "Lustre · NVMe stage-in"],
            ["Subsample sweep", "10k · 50k · 100k · 200k (Phase 1) · 250k · 300k (memcheck)"],
            ["Out of scope", "Layer PyTorch comparison, ETL+ datapipe, inference PBR, optimization ablations"],
          ]}
          striped
        />

        <H2 style={{ fontFamily: "inherit" }}>Abstract</H2>
        <Para>
          We measured end-to-end training for GeoTransolver Surface and GeoTransolver Volume on DrivAerML on B200
          from single-GPU baselines through 64-GPU multi-node runs. At sub = 100k on Lustre g = 1: GeoTransolver
          Surface 3.69 samples/s (34.8 s/epoch); GeoTransolver Volume 1.58 samples/s (79.4 s/epoch). Within-node
          scaling (g = 1→4) is ~100% efficient; multi-node Lustre drops below 70% at g ≥ 16 (I/O bound). NVMe
          stage-in at g = 16 yields +67% to +131% throughput. GeoTransolver Volume @ 200k uses 77.6 GB of 192 GB
          HBM3e (60% headroom on B200).
        </Para>

        <Section n={1} title="Benchmark Configuration" />
        <Table
          headers={["Model", "Hydra Config", "Dataset"]}
          rows={[
            [GT_SURFACE, "geotransolver_surface", "drivaer_ml_surface"],
            [GT_VOLUME, "geotransolver_volume", "drivaer_ml_volume"],
          ]}
          striped
        />
        <Table
          headers={["Axis", "Values"]}
          rows={[
            ["GPUs", "1, 4, 16, 32, 64"],
            ["Storage", "Lustre (network FS) · NVMe (node-local stage-in via stage_data.sh)"],
            ["Sampling resolution", "10k · 50k · 100k · 200k points/sample"],
            ["Epochs", "5"],
            ["batch_size", "1 per GPU (fixed — memory swept via sampling resolution)"],
          ]}
          striped
        />

        <Section n={2} title="Methodology" />
        <Para>
          Each run produces benchmark_summary.json. Throughput = batch_size × num_gpus / step_time_P50.
          Time per epoch = wallclock_train_s / num_epochs_aggregated (epoch 0 excluded by default).
          Peak memory = max(torch.cuda.memory_reserved()) per rank.
          Weak-scaling efficiency = (throughput(N) / throughput(1)) × (1/N) × 100%. Reported step-time and
          throughput metrics exclude epoch 0 (compile / first-epoch dataloader warm-up).
        </Para>

        <Section n={3} title="Results" />

        <H2 style={{ fontFamily: "inherit" }}>3.1. Single-GPU Baseline</H2>
        <Para>
          This section characterizes the per-device cost of training GeoTransolver Surface and GeoTransolver
          Volume on DrivAerML before multi-GPU scaling effects are introduced.
        </Para>
        <Para>
          This is the canonical single-GPU baseline used as the reference for all multi-GPU scaling efficiency
          calculations. Reported step-time and throughput metrics exclude epoch 0 (compile and first-epoch
          dataloader warm-up); aggregates pool epochs 1–4.
        </Para>
        <Para>
          Figure below shows throughput (samples/s, P50) and wall-clock time per epoch for GeoTransolver Surface
          and GeoTransolver Volume on a single B200 GPU with data read from Lustre, swept across subsampling
          levels.
        </Para>
        <Figure n={1} src={IMG_LUSTRE_G1}
          caption="Single GPU Baseline — Throughput vs. Subsampling (B200, Lustre, g = 1). Left panel: aggregate throughput (samples/s, P50). Right panel: wall-clock time per epoch (s). Green = GeoTransolver Surface; teal = GeoTransolver Volume."
        />
        <Para>
          Table below summarizes throughput, epoch duration, median step latency (P50, epochs 1–4), and peak
          reserved GPU memory as a function of subsampling resolution for both model variants under Lustre at
          g = 1 (batch_size = 1, 5 training epochs).
        </Para>
        <Table
          headers={["Sampling", "Surface thr", "Surface t/ep", "Surface step", "Surface mem", "Volume thr", "Volume t/ep", "Volume step", "Volume mem"]}
          rows={lustreG1}
          columnAlign={["left","right","right","right","right","right","right","right","right"]}
          striped
        />
        <TC n={1}>Lustre g = 1, single-GPU baseline. Surface = GeoTransolver Surface; Volume = GeoTransolver Volume. thr = samples/s; t/ep = s/epoch; step = P50 (s); mem = peak reserved VRAM (GB).</TC>

        <Sub id="3.1.2" title="NVMe Storage Tier" />
        <Para>
          {{PHASE3_NVME_PARA}}
        </Para>
        <Figure n={2} src={IMG_NVME_G16}
          caption="{{FIG2_CAPTION}}"
        />
        <Table
          headers={["Sampling", "Surface thr", "Surface t/ep", "Surface step", "Volume thr", "Volume t/ep", "Volume step"]}
          rows={nvmeG16}
          columnAlign={["left","right","right","right","right","right","right"]}
          striped
        />
        <TC n={2}>NVMe g=16 aggregate cluster throughput. Per-rank memory matches Table 1.</TC>

        <Sub id="3.1.3" title="Storage Tier Comparison" />
        <Figure n={3} src={IMG_STORAGE_CMP}
          caption="Lustre g=1 vs Lustre g=16 vs NVMe g=16. Rows: GeoTransolver Surface (top), GeoTransolver Volume (bottom). Columns: throughput and time per epoch."
        />
        <Table
          headers={["Config @ 100k", "Surface thr", "Surface t/ep", "Surface step", "Volume thr", "Volume t/ep", "Volume step"]}
          rows={compare100k}
          columnAlign={["left","right","right","right","right","right","right"]}
          striped
        />
        <TC n={3}>Cross-tier @ sub = 100k. NVMe vs Lustre g=16: +90% GeoTransolver Surface throughput, −47% step time.</TC>

        <Sub id="3.1.4" title="Memory Efficiency (Single GPU)" />
        <Figure n={4} src={IMG_SINGLE_GPU_MEM}
          caption="Peak reserved VRAM vs subsampling for GeoTransolver Surface and GeoTransolver Volume. Dashed line = 192 GB B200 HBM3e ceiling."
        />
        <Table headers={["Subsampling Resolution", "Surface mem", "Surface headroom", "Volume mem", "Volume headroom"]} rows={memRows}
          columnAlign={["left","right","right","right","right"]} striped
        />
        <TC n={4}>Per-rank peak memory. ⓘ = memcheck batch (g=4). GeoTransolver Volume @ 300k: 115.79 GB (40% headroom on 192 GB B200).</TC>

        <Section n={4} title="Results — Multi-GPU Scaling" />
        <Para>
          Aggregate throughput rises through g = 4 on Lustre, then flattens at g ≥ 16. NVMe stage-in restores healthy
          scaling through g = 32 for both GeoTransolver Surface and GeoTransolver Volume.
        </Para>

        <Figure n={5} src={IMG_THROUGHPUT}
          caption="Aggregate throughput (samples/s, P50) vs GPU count. Top row: Volume; bottom row: Surface. Columns: N=50k, 100k, 200k. Green = NVMe; gray = Lustre. GPUs: 1–64."
        />
        <Figure n={6} src={IMG_EFFICIENCY}
          caption="Parallel efficiency η(N) = Φ(N)/(N·Φ(1)). Dashed = 100% ideal weak scaling. Same layout as Figure 5."
        />
        <Table
          headers={["GPUs", "Surface thr", "Surface eff", "Volume thr", "Volume eff"]}
          rows={eff100kLustre}
          columnAlign={["left","right","right","right","right"]}
          striped
        />
        <TC n={5}>Efficiency @ sub = 100k, Lustre. g ≥ 16 drops below 70% — I/O bound on shared Lustre MDS.</TC>

        <Sub id="4.1" title="NVMe vs Lustre — Data-Tier Dividend @ g = 16" />
        <Figure n={7} src={IMG_NVME_16}
          caption="GeoTransolver Surface and GeoTransolver Volume throughput at 16 GPUs: Lustre vs NVMe by sampling resolution. Labels show NVMe gain over Lustre."
        />
        <Table
          headers={["Sampling", "Lustre Surface", "NVMe Surface", "Gain"]}
          rows={nvmeGain16Surface}
          columnAlign={["left","right","right","right"]}
          striped
        />
        <TC n={6}>GeoTransolver Surface @ g = 16. GeoTransolver Volume gains +80% to +131% across the matrix.</TC>

        <Sub id="4.2" title="Peak Aggregate Throughput" />
        <Table
          headers={["Rank", "Model", "GPUs", "Storage", "Sampling", "Throughput (samples/s)"]}
          rows={peakRuns}
          columnAlign={["left","left","right","left","right","right"]}
          striped
        />

        <Section n={5} title="Memory Efficiency (All Runs)" />
        <Para>
          Peak reserved GPU memory vs subsampling on B200 (192 GB HBM3e), per-rank at g = 4. Figure 8
          plots Phase 1 (10k–200k) plus memcheck at 300k; 250k and 400k are table-only (memRows). Memory scales
          linearly (R² ≈ 0.999) through 115.79 GB on GeoTransolver Volume @ 300k without OOM. Volume @ 400k
          reached 154 GB (80% utilization). Extrapolated OOM: ~640k (Surface) · ~500k (Volume).
        </Para>
        <Figure n={8} src={IMG_MEMORY}
          caption="Peak GPU memory vs subsampling resolution (1 node, 4 B200 GPUs). Phase 1 sweep plus 300k memcheck; 250k and 400k omitted from chart (see memRows table). Green = Surface, teal = Volume."
        />

        <Section n={6} title="GPU Platform Comparison" />
        <Table
          headers={["Category", "GeoTransolver Surface", "GeoTransolver Volume"]}
          rows={[
            ["Single-GPU @ 100k (Lustre)", "3.7 samples/s", "1.6 samples/s"],
            ["Peak memory @ 200k", "60 GB (69% headroom)", "78 GB (60% headroom)"],
            ["Peak memory @ 300k ⓘ", "91 GB (53% headroom)", "116 GB (40% headroom)"],
            ["Best multi-GPU", "212 samples/s (g=64 NVMe)", "64 samples/s (g=64 Lustre)"],
            ["Lustre scaling", "<70% eff @ g≥16", "<70% eff @ g≥16"],
            ["NVMe scaling", "Healthy through g=32", "Healthy through g=32"],
          ]}
          striped
        />
        <Callout tone="warning" title="RTX 6000 Pro — pending (Milestone 1)">
          Replay the same matrix on RTX 6000 Pro for workstation-class positioning. Compare single-GPU throughput,
          peak memory, and g = 16 NVMe scaling against B200 baselines in this report.
        </Callout>

        <Section n={7} title="Conclusions and Milestone 1 Next Steps" />
        <Para>
          GeoTransolver Surface on DrivAerML sustains 3.7–5.3 samples/s on B200. GeoTransolver Volume is ~2× slower
          and more memory-intensive. Use NVMe stage-in for production runs ≥ 16 GPUs. Lustre-only multi-node training
          is I/O-bound past 4 GPUs without staging.
        </Para>
        <Table
          headers={["Item", "Status", "Maps to Milestone 1"]}
          rows={[
            ["NVMe g=1 Phase 3", "{{PHASE3_STATUS}}", "Complete I/O vs end-to-end single-GPU NVMe baseline"],
            ["Memcheck2 (400k/500k Volume)", "{{MEMCHECK2_STATUS}}", "Empirical OOM ceiling confirmation"],
            ["RTX 6000 Pro matrix", "Not run", "Workstation vs datacenter GPU SKU comparison"],
            ["Longer g=64 runs", "Needed", "Statistical rigor KPI — stable P50/P95 at scale"],
            ["Batch-size sweep", "Blocked", "Original memory-vs-batch-size ask — recipe change required"],
            ["BallQuery / I/O ablation", "Not run", "Per Mohammad — optimization knob impact study"],
            ["Inference PBR", "Not run", "Separate Milestone 1 ask — validation-pass latencies"],
          ]}
          striped
        />

        <Divider />
        <Text tone="tertiary" size="small">
          CAE Benchmarking · Milestone 1 · results/_scaling_snapshot/ · plot_scaling_snapshot.py · build_canvas_report.py
        </Text>
      </Stack>
    </div>
  );
}
'''


def _load_summaries(results_root: Path) -> list[dict]:
    rows: list[dict] = []
    for path in results_root.rglob("benchmark_summary.json"):
        if "_smoketest" in str(path):
            continue
        rows.append(json.loads(path.read_text(encoding="utf-8")))
    return rows


def _pct_delta(new: float, old: float) -> str:
    if old <= 0:
        return "—"
    pct = (new - old) / old * 100
    sign = "+" if pct >= 0 else ""
    return f"{sign}{pct:.0f}%"


def _find_summary(summaries: list[dict], **keys) -> dict | None:
    for row in summaries:
        if all(row.get(k) == v for k, v in keys.items()):
            return row
    return None


def _build_live_substitutions(results_root: Path | None) -> dict[str, str]:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    subs: dict[str, str] = {
        "{{RUN_COUNT}}": "?",
        "{{LAST_UPDATED}}": now,
        "{{PHASE3_NVME_PARA}}": (
            "Phase 3 g = 1 NVMe runs are pending. Figure 2 uses g = 16 NVMe until the full "
            "Phase 3 sweep completes."
        ),
        "{{FIG2_CAPTION}}": (
            "NVMe staged baseline at g=16 (Phase 3 g=1 NVMe pending). Aggregate throughput and "
            "time per epoch for GeoTransolver Surface and GeoTransolver Volume."
        ),
        "{{PHASE3_STATUS}}": "Not run",
        "{{MEMCHECK2_STATUS}}": "Queued",
    }
    if results_root is None or not results_root.is_dir():
        return subs

    summaries = _load_summaries(results_root)
    subs["{{RUN_COUNT}}"] = str(len(summaries))

    nvme_g1 = [s for s in summaries if s.get("storage") == "nvme" and s.get("num_gpus") == 1]
    phase3_total = 16
    phase3_done = len(nvme_g1)
    if phase3_done == 0:
        subs["{{PHASE3_STATUS}}"] = "Not run"
    elif phase3_done >= phase3_total:
        subs["{{PHASE3_STATUS}}"] = "Complete"
    else:
        subs["{{PHASE3_STATUS}}"] = f"In progress ({phase3_done}/{phase3_total})"

    memcheck2 = list((results_root / "_memcheck2").rglob("benchmark_summary.json"))
    subs["{{MEMCHECK2_STATUS}}"] = (
        f"Complete ({len(memcheck2)} runs)" if memcheck2 else "Queued"
    )

    lines: list[str] = []
    if phase3_done == 0:
        lines.append("Phase 3 g = 1 NVMe runs are pending.")
    else:
        lines.append(
            f"Phase 3 g = 1 NVMe is in progress ({phase3_done}/{phase3_total} complete)."
        )
    for model, label in [
        ("geotransolver_surface", "GeoTransolver Surface"),
        ("geotransolver_volume", "GeoTransolver Volume"),
    ]:
        lustre = _find_summary(
            summaries, model=model, storage="lustre", num_gpus=1, sampling_resolution=200_000
        )
        nvme = _find_summary(
            summaries, model=model, storage="nvme", num_gpus=1, sampling_resolution=200_000
        )
        if lustre and nvme:
            l_thr = lustre["throughput_samples_per_sec_p50"]
            n_thr = nvme["throughput_samples_per_sec_p50"]
            l_ep = lustre["wallclock_train_s"] / num_epochs_for_aggregate(lustre)
            n_ep = nvme["wallclock_train_s"] / num_epochs_for_aggregate(nvme)
            ep_delta = (n_ep - l_ep) / l_ep * 100 if l_ep else 0
            ep_sign = "+" if ep_delta >= 0 else ""
            lines.append(
                f"{label} @ sub = 200k: {_pct_delta(n_thr, l_thr)} throughput "
                f"({n_thr:.2f} vs {l_thr:.2f} samples/s), {ep_sign}{ep_delta:.0f}% epoch time "
                f"({n_ep:.1f} vs {l_ep:.1f} s)."
            )
    if phase3_done < phase3_total:
        lines.append("Figure 2 uses g = 16 NVMe until the full Phase 3 sweep completes.")
    subs["{{PHASE3_NVME_PARA}}"] = " ".join(lines)

    if phase3_done == 0:
        subs["{{FIG2_CAPTION}}"] = (
            "NVMe staged baseline at g=16 (Phase 3 g=1 NVMe pending). Aggregate throughput and "
            "time per epoch for GeoTransolver Surface and GeoTransolver Volume."
        )
    elif phase3_done >= phase3_total:
        subs["{{FIG2_CAPTION}}"] = (
            "Single GPU Baseline — Throughput vs. Subsampling (B200, NVMe, g = 1). "
            "Aggregate throughput and time per epoch for GeoTransolver Surface and GeoTransolver Volume."
        )
    else:
        subs["{{FIG2_CAPTION}}"] = (
            f"NVMe staged baseline at g=16 (Phase 3 g=1 sweep in progress — "
            f"{phase3_done}/{phase3_total} complete). Aggregate throughput and time per epoch "
            f"for GeoTransolver Surface and GeoTransolver Volume."
        )
    return subs


def _apply_substitutions(text: str, subs: dict[str, str]) -> str:
    for key, value in subs.items():
        text = text.replace(key, value)
    return text


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=Path("results/_scaling_snapshot"),
        help="Directory containing PNG plots",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=canvas_path("e2e-training-performance.canvas.tsx"),
        help="Output .canvas.tsx path (default: benchmarks/canvases/ or PHYSICSNEMO_CAE_CANVAS_DIR)",
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("results"),
        help="Results root for live run-count / Phase 3 stats (relative to recipe root)",
    )
    args = parser.parse_args()

    recipe_root = Path(__file__).resolve().parent.parent
    results_root = args.results if args.results.is_absolute() else recipe_root / args.results
    subs = _build_live_substitutions(results_root)

    parts = [IMPORTS]
    for const_name, filename in PLOT_FILES.items():
        path = args.plots_dir if args.plots_dir.is_absolute() else recipe_root / args.plots_dir
        path = path / filename
        if not path.is_file():
            raise SystemExit(f"missing plot: {path}")
        b64 = base64.b64encode(path.read_bytes()).decode("ascii")
        parts.append(f'const {const_name} = "data:image/png;base64,{b64}";')
        print(f"  embedded {filename}")
    parts.append(_apply_substitutions(BODY.strip(), subs))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(parts) + "\n")
    print(f"Wrote {args.out} ({args.out.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
