#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Rebuild milestone1-full-report.canvas.tsx — all four Milestone 1 performance asks."""

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

PLOT_FILES = {
    "IMG_THROUGHPUT": "01_throughput_vs_gpus.png",
    "IMG_EFFICIENCY": "02_efficiency_vs_gpus.png",
    "IMG_NVME_16": "03_nvme_vs_lustre_16gpu.png",
    "IMG_MEMORY": "04_memory_vs_sampling.png",
    "IMG_LUSTRE_G1": "05_single_gpu_throughput_epochtime_lustre.png",
    "IMG_SINGLE_GPU_MEM": "06_single_gpu_memory.png",
    "IMG_NVME_G1": "07_single_gpu_throughput_epochtime_nvme.png",
    "IMG_STORAGE_CMP": "08_lustre_vs_nvme_storage_compare.png",
}

IMPORTS = """import {
  BarChart, Callout, Divider, Grid, H1, H2, H3, LineChart, PieChart,
  Pill, Row, Stack, Stat, Table, Text, TodoList, useHostTheme,
} from "cursor/canvas";
"""

BODY = r'''
const GT_SURFACE = "GeoTransolver Surface";
const GT_VOLUME = "GeoTransolver Volume";

const askStatus = [
  ["1", "Training perf vs PyTorch — optimized layer kernels", "Complete", "success"],
  ["2", "ETL+ datapipe — PNM.mesh vs VTK", "Pending", "warning"],
  ["3", "End-to-end model training — single- and multi-GPU", "Complete (B200)", "success"],
  ["4", "End-to-end model inference — latency and memory", "Complete (B200)", "success"],
];

const layerSpeedups = [
  ["Ball Query (fwd)", "44.0×"],
  ["PointToGrid (fwd)", "191.9×"],
  ["GridToPoint (fwd)", "10.9×"],
  ["MeshGreenGauss (fwd)", "6.3×"],
  ["Ball Query (bwd)", "10.1×"],
  ["PointToGrid (bwd)", "61.4×"],
  ["MeshLSQ (fwd)", "1.9×"],
  ["RectilinearGrid (fwd)", "1.3×"],
];

const layerTable = [
  ["Ball Query (Radius Search)", "8192→4096 pts", "0.886 ms", "39.0 ms", "44.0×"],
  ["PointToGridInterpolation", "g=32³, n=512", "0.186 ms", "35.7 ms", "191.9×"],
  ["GridToPointInterpolation", "g=32³, n=512", "0.194 ms", "2.12 ms", "10.9×"],
  ["MeshGreenGaussGradient", "2D tri 36×36", "0.283 ms", "1.79 ms", "6.3×"],
  ["RadiusSearch backward", "4096→2048 pts", "1.03 ms", "10.4 ms", "10.1×"],
  ["PointToGridInterp backward", "g=32³, n=512", "0.963 ms", "59.1 ms", "61.4×"],
  ["MeshLSQGradient", "1024 pts, k=16", "0.367 ms", "0.685 ms", "1.9×"],
  ["RectilinearGridGradient", "3D 96³", "1.02 ms", "1.30 ms", "1.3×"],
];

const lustreG1 = [
  ["10,000", "6.34", "78.8", "0.158", "3.6", "2.91", "163.0", "0.344", "4.7"],
  ["50,000", "4.74", "95.3", "0.211", "17.7", "2.26", "193.6", "0.442", "24.6"],
  ["100,000", "4.76", "97.4", "0.210", "35.0", "1.72", "251.3", "0.583", "48.3"],
  ["200,000", "3.77", "120.7", "0.265", "69.4", "1.12", "362.8", "0.893", "95.7"],
];

const scaling200k = [
  ["1", "1.46", "0.683", "1.00", "95.6"],
  ["4", "5.04", "0.794", "3.44", "77.6"],
  ["8", "10.08", "0.793", "6.89", "77.6"],
  ["16", "19.05", "0.840", "13.01", "77.6"],
  ["32", "37.26", "0.859", "25.45", "77.6"],
  ["64", "72.73", "0.880", "49.67", "77.6"],
  ["96", "93.83", "1.023", "64.08", "77.6"],
  ["128", "102.29", "1.251", "69.85", "77.6"],
];

const nvmeGain16 = [
  ["Surface 10k", "51", "95", "+86%"],
  ["Surface 50k", "50", "83", "+67%"],
  ["Surface 100k", "39", "75", "+90%"],
  ["Surface 200k", "31", "59", "+92%"],
  ["Volume 10k", "23", "53", "+131%"],
  ["Volume 50k", "24", "43", "+80%"],
  ["Volume 100k", "17", "31", "+87%"],
  ["Volume 200k", "10", "19", "+84%"],
];

const inferenceTable = [
  ["50,000", "166", "187", "202", "45", "121", "24.4"],
  ["100,000", "266", "289", "305", "74", "192", "48.2"],
  ["200,000", "488", "536", "566", "97", "391", "95.6"],
  ["300,000", "727", "770", "792", "121", "605", "144.3"],
];

const gpuCompare = [
  ["Single-GPU train @ 100k (Volume)", "1.72 samples/s", "—", "—"],
  ["Peak train memory @ 200k (Volume)", "95.7 GB", "—", "—"],
  ["Inference P50 @ 200k (Volume)", "488 ms", "—", "—"],
  ["Multi-GPU @ g=16 NVMe (Volume, 200k)", "19.05 samples/s", "—", "—"],
  ["VRAM ceiling", "192 GB", "TBD", "TBD"],
];

const nextSteps = [
  { id: "p0-plots", label: "Generate P0 plots (layer speedup, CUDA breakdown, inference)", status: "pending" as const },
  { id: "datapipe", label: "Run VTK vs PNM.mesh datapipe ablation (Ask 2)", status: "pending" as const },
  { id: "rtx", label: "Replay matrix on RTX 6000 Pro and RTX 5080", status: "pending" as const },
  { id: "memcheck", label: "Memcheck2 @ 500k — confirm OOM ceiling", status: "pending" as const },
  { id: "g64", label: "Longer g=64+ runs (20+ epochs) for stable P95/P99", status: "pending" as const },
];

function Figure({ n, src, caption }: { n: string; src: string; caption: string }) {
  const theme = useHostTheme();
  return (
    <Stack gap={8} style={{ marginBottom: 20 }}>
      <div style={{ border: `1px solid ${theme.stroke.tertiary}`, borderRadius: 4, overflow: "hidden" }}>
        <img src={src} alt={`Figure ${n}`} style={{ width: "100%", display: "block" }} />
      </div>
      <Text tone="secondary" size="small" style={{ textAlign: "center", lineHeight: 1.6 }}>
        <Text weight="semibold" as="span">Figure {n}. </Text>{caption}
      </Text>
    </Stack>
  );
}

function ChartCaption({ children }: { children: string }) {
  return (
    <Text tone="secondary" size="small" style={{ lineHeight: 1.6, marginTop: 4 }}>
      {children}
    </Text>
  );
}

function Para({ children }: { children: any }) {
  return <Text style={{ lineHeight: 1.75 }}>{children}</Text>;
}

function Section({ n, title }: { n: number; title: string }) {
  return <H2 style={{ fontFamily: "inherit", marginTop: 12 }}>{n}. {title}</H2>;
}

function Sub({ title }: { title: string }) {
  return <H3 style={{ fontFamily: "inherit", fontSize: 15 }}>{title}</H3>;
}

export default function Milestone1FullPerformanceReport() {
  const theme = useHostTheme();
  return (
    <div style={{ maxWidth: 820, margin: "0 auto", padding: "48px 32px 64px", fontFamily: "Georgia, serif" }}>
      <Stack gap={28}>
        <Stack gap={12} style={{ borderBottom: `1px solid ${theme.stroke.tertiary}`, paddingBottom: 24 }}>
          <Text tone="secondary" size="small">CAE Benchmarking · Milestone 1 — Internal Perf Benchmark</Text>
          <H1 style={{ fontFamily: "inherit" }}>
            Milestone 1 Performance Report: GeoTransolver × DrivAerML
          </H1>
          <Text tone="secondary">
            NVIDIA B200 (HSG) · June 1, 2026 · {{RUN_COUNT}} training runs · All four Milestone 1 asks
          </Text>
        </Stack>

        <Grid columns={3} gap={12}>
          <Stat value="44×" label="Ball Query kernel speedup vs PyTorch (B200)" tone="success" />
          <Stat value="1.27×" label="End-to-end model speedup (Amdahl, Ball Query @ 22%)" tone="success" />
          <Stat value="+67–131%" label="NVMe stage-in gain @ g=16" tone="success" />
          <Stat value="212" label="Peak train throughput samples/s (Surface, g=64)" />
          <Stat value="488 ms" label="Inference P50 @ 200k points (Volume)" />
          <Stat value="Pending" label="RTX 6000 Pro / 5080 baselines" tone="warning" />
        </Grid>

        <Sub title="Milestone 1 Ask Status" />
        <Table
          headers={["Ask", "Topic", "Status"]}
          rows={askStatus.map(([a, t, s]) => [a, t, s])}
          striped
        />
        <Row gap={8} style={{ flexWrap: "wrap" }}>
          {askStatus.map(([a, , s, tone]) => (
            <Pill key={a} tone={tone as any} size="small">Ask {a}: {s}</Pill>
          ))}
        </Row>

        <Section n={1} title="Training Performance vs PyTorch — Optimized Layer Kernels" />
        <Para>
          PhysicsNeMo replaces CAE-dominant kernels (ball query, scatter/gather interpolation, mesh gradients)
          with NVIDIA Warp implementations using spatial hashing and BVH acceleration. ASV micro-benchmarks on B200
          compare PhysicsNeMo (Warp) against native PyTorch baselines.
        </Para>

        <Sub title="1.1 Per-Layer Speedup (B200, 1 GPU)" />
        <BarChart
          categories={layerSpeedups.map(([l]) => l)}
          series={[{ name: "Speedup vs PyTorch", data: [44.0, 191.9, 10.9, 6.3, 10.1, 61.4, 1.9, 1.3] }]}
          horizontal
          height={320}
          valueSuffix="×"
        />
        <ChartCaption>
          Source: ASV micro-benchmarks · B200 · 1 GPU · representative case per layer family.
          Log-scale span omitted — values range 1.3× to 191.9×.
        </ChartCaption>

        <Table
          headers={["Layer", "Input", "PNM (Warp)", "PyTorch", "Speedup"]}
          rows={layerTable}
          columnAlign={["left", "left", "right", "right", "right"]}
          striped
        />

        <Sub title="1.2 Model-Level Translation — CUDA Step Breakdown" />
        <Grid columns={2} gap={16}>
          <Stack gap={8}>
            <PieChart
              donut
              size={220}
              data={[
                { label: "Ball Query (Warp)", value: 21.6, tone: "warning" },
                { label: "Optimizer (Muon)", value: 5.4 },
                { label: "GEMM / MatMul", value: 2.6 },
                { label: "BVH Query (Warp)", value: 1.8 },
                { label: "Attention", value: 1.6 },
                { label: "Other", value: 67.0, tone: "neutral" },
              ]}
            />
            <ChartCaption>
              CUDA time share · GeoTransolver Volume training step · 200k points · DrivAerML · 1× B200.
              Ball Query = 21.6% of step time.
            </ChartCaption>
          </Stack>
          <Callout tone="info" title="Amdahl worked example">
            At 44× Ball Query kernel speedup and f = 22% of step time, end-to-end model speedup ≈ 1.27×.
            The 67% "Other" bucket caps theoretical ceiling — layer speedups alone cannot 10× the model.
          </Callout>
        </Grid>

        <Section n={2} title="Optimized ETL+ Datapipe Implementation" />
        <Callout tone="warning" title="Ask 2 — runs pending">
          Compare DrivAerML loaded via PhysicsNeMo mesh datapipe (`.pdmsh` DomainMesh tensors) vs a VTK/VTP
          baseline that parses mesh topology on every epoch. When data lands, report I/O time and data loading
          time separately per sample.
        </Callout>
        <Para>
          The measured training path uses MeshDataset + MeshReader on pre-curated `.pdmsh` files. VTK-based
          pipelines shift parse/transform cost into every training step — the NVMe dividend in §3.5 is partly
          a datapipe and metadata story.
        </Para>

        <Section n={3} title="End-to-End Model Training Performance" />
        <Para>
          GeoTransolver Surface and Volume on DrivAerML, batch_size = 1 per GPU (weak scaling). First five
          steps per epoch excluded from timing. Phase 3 NVMe single-GPU baseline complete.
        </Para>

        <Sub title="3.1 Single-GPU Baseline" />
        <Figure n="3.1" src={IMG_LUSTRE_G1}
          caption="Throughput (samples/s, P50) and time per epoch on single B200, Lustre, g=1. Green = Surface; teal = Volume."
        />
        <Figure n="3.2" src={IMG_NVME_G1}
          caption="Single-GPU NVMe baseline (g=1). Data staged via stage_data.sh."
        />
        <Table
          headers={["Sampling", "Surf thr", "Surf t/ep", "Surf step", "Surf mem", "Vol thr", "Vol t/ep", "Vol step", "Vol mem"]}
          rows={lustreG1}
          columnAlign={["left","right","right","right","right","right","right","right","right"]}
          striped
        />

        <Sub title="3.2 Memory Efficiency" />
        <Grid columns={2} gap={12}>
          <Figure n="3.3" src={IMG_MEMORY}
            caption="Peak GPU memory vs subsampling (g=4, per-rank). Linear fit R² ≈ 0.999."
          />
          <Figure n="3.4" src={IMG_SINGLE_GPU_MEM}
            caption="Single-GPU peak memory vs subsampling. Dashed = 192 GB B200 ceiling."
          />
        </Grid>

        <Sub title="3.3 Multi-GPU Scaling" />
        <Figure n="3.5" src={IMG_THROUGHPUT}
          caption="Aggregate throughput vs GPU count. Green = NVMe, gray = Lustre, dashed = ideal weak scaling."
        />
        <Figure n="3.6" src={IMG_EFFICIENCY}
          caption="Weak scaling efficiency (%). Dotted = 70% gate. Open markers = ≤10 timing samples."
        />

        <Sub title="3.4 GeoTransolver Volume @ 200k — Extended NVMe Sweep" />
        <LineChart
          categories={scaling200k.map(([g]) => g)}
          series={[
            { name: "Throughput (samples/s)", data: scaling200k.map(([, t]) => parseFloat(t)), tone: "success" },
            { name: "Step time P50 (×10 s)", data: scaling200k.map(([, , s]) => parseFloat(s) * 10), tone: "info" },
          ]}
          height={240}
        />
        <ChartCaption>
          GeoTransolver Volume · 200k subsample · NVMe · B200. Step time series scaled ×10 for shared axis.
          Throughput plateaus past g=64; step time jumps at g=96 (1.023 s).
        </ChartCaption>
        <Table
          headers={["GPUs", "Throughput (s/s)", "Step P50 (s)", "Speedup", "Peak mem (GB)"]}
          rows={scaling200k}
          columnAlign={["right","right","right","right","right"]}
          striped
        />

        <Sub title="3.5 NVMe vs Lustre — Data-Tier Dividend" />
        <Grid columns={2} gap={12}>
          <Figure n="3.7" src={IMG_NVME_16}
            caption="Throughput at g=16: Lustre vs NVMe by sampling resolution."
          />
          <Figure n="3.8" src={IMG_STORAGE_CMP}
            caption="Lustre g=1 vs Lustre g=16 vs NVMe g=16 across subsampling levels."
          />
        </Grid>
        <Table
          headers={["Config @ g=16", "Lustre", "NVMe", "Gain"]}
          rows={nvmeGain16}
          columnAlign={["left","right","right","right"]}
          striped
        />

        <Section n={4} title="End-to-End Model Inference Performance" />
        <Para>
          Latencies from validation pass (single forward pass, torch.no_grad(), batch_size = 1). Excludes
          service-layer overhead. GeoTransolver Volume on B200.
        </Para>

        <Sub title="4.1 Latency vs Subsample Size" />
        <LineChart
          categories={inferenceTable.map(([s]) => s)}
          series={[
            { name: "P50 (ms)", data: inferenceTable.map(([, p50]) => parseInt(p50, 10)) },
            { name: "P95 (ms)", data: inferenceTable.map(([, , p95]) => parseInt(p95, 10)), tone: "info" },
            { name: "P99 (ms)", data: inferenceTable.map(([, , , p99]) => parseInt(p99, 10)), tone: "warning" },
          ]}
          height={240}
          valueSuffix=" ms"
        />
        <ChartCaption>
          Overall inference latency · GeoTransolver Volume · B200 · batch_size = 1.
          P99 is ~22% above P50 at 200k (566 vs 488 ms).
        </ChartCaption>

        <Sub title="4.2 I/O + Preprocess vs Model Inference" />
        <BarChart
          categories={inferenceTable.map(([s]) => s)}
          series={[
            { name: "I/O + Preprocess (ms)", data: inferenceTable.map(([, , , , io]) => parseInt(io, 10)), tone: "info" },
            { name: "Model inference (ms)", data: inferenceTable.map(([, , , , , model]) => parseInt(model, 10)), tone: "success" },
          ]}
          stacked
          height={220}
          valueSuffix=" ms"
        />
        <ChartCaption>
          Stacked latency breakdown · subsample points per sample. Model compute dominates at scale;
          I/O fraction shrinks from 27% (50k) to 17% (300k).
        </ChartCaption>

        <Table
          headers={["Input size", "P50", "P95", "P99", "I/O+prep", "Model", "Peak mem (GB)"]}
          rows={inferenceTable}
          columnAlign={["right","right","right","right","right","right","right"]}
          striped
        />
        <Callout tone="warning" title="OOM @ 500k">
          500k points exceeds 192 GB B200 HBM3e. 300k at 144.3 GB leaves headroom for batching experiments.
        </Callout>

        <Section n={5} title="GPU Platform Comparison" />
        <Table
          headers={["Metric", "B200 (measured)", "RTX 6000 Pro", "RTX 5080"]}
          rows={gpuCompare}
          striped
        />
        <Callout tone="warning" title="RTX SKUs pending">
          Replay training and inference matrix on RTX 6000 Pro (datacenter workstation) and RTX 5080
          (desktop) for hardware positioning collateral.
        </Callout>

        <Section n={6} title="Conclusions and Next Steps" />
        <Grid columns={2} gap={16}>
          <Stack gap={8}>
            <Text weight="semibold">Layers vs PyTorch</Text>
            <Para>
              Warp kernels: 1.3×–192× speedup. Ball Query at 44× → ~1.27× model speedup (22% of step).
              PointToGrid at 192× is largest micro-benchmark but lower step-profile weight.
            </Para>
            <Text weight="semibold">Training</Text>
            <Para>
              NVMe stage-in for g≥16 (+67–131%). Production sweet spot: Volume @ 200k, g=64, NVMe —
              72.7 samples/s, 0.88 s/step, 77.6 GB.
            </Para>
            <Text weight="semibold">Inference</Text>
            <Para>
              488 ms P50 @ 200k on B200 — suitable for offline batch. Monitor P99 (566 ms) for interactive use.
            </Para>
          </Stack>
          <TodoList title="Milestone 1 next steps" items={nextSteps} />
        </Grid>

        <Divider />
        <Text tone="tertiary" size="small">
          Source: results/_scaling_snapshot/END_TO_END_TRAINING_PERFORMANCE_REPORT.md ·
          plot_scaling_snapshot.py · build_full_report_canvas.py · updated {{LAST_UPDATED}}
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
        default=canvas_path("milestone1-full-report.canvas.tsx"),
        help="Output .canvas.tsx path (default: benchmarks/canvases/ or PHYSICSNEMO_CAE_CANVAS_DIR)",
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("results"),
        help="Results root for run count",
    )
    args = parser.parse_args()

    recipe_root = Path(__file__).resolve().parent.parent
    results_root = args.results if args.results.is_absolute() else recipe_root / args.results
    summaries = _load_summaries(results_root) if results_root.is_dir() else []
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    body = BODY.replace("{{RUN_COUNT}}", str(len(summaries) or 110)).replace(
        "{{LAST_UPDATED}}", now
    )

    parts = [IMPORTS]
    plots_path = args.plots_dir if args.plots_dir.is_absolute() else recipe_root / args.plots_dir
    for const_name, filename in PLOT_FILES.items():
        path = plots_path / filename
        if not path.is_file():
            raise SystemExit(f"missing plot: {path}")
        b64 = base64.b64encode(path.read_bytes()).decode("ascii")
        parts.append(f'const {const_name} = "data:image/png;base64,{b64}";')
        print(f"  embedded {filename}")

    parts.append(body.strip())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(parts) + "\n")
    print(f"Wrote {args.out} ({args.out.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
