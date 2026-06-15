#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Rebuild e2e-full-report.canvas.tsx — full Milestone 1 report with all four asks."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import sys
from pathlib import Path

_BENCHMARKS_ROOT = Path(__file__).resolve().parents[1]
if str(_BENCHMARKS_ROOT) not in sys.path:
    sys.path.insert(0, str(_BENCHMARKS_ROOT))
from canvas.build_canvas_report import (
    IMPORTS as TRAINING_IMPORTS,
    PLOT_FILES,
    _apply_substitutions,
    _find_summary,
    _load_summaries,
    _pct_delta,
    num_epochs_for_aggregate,
)
from paths import canvas_path
from plots.inference_metrics import collect_inference_points, js_table_rows
from plots.plot_inference import OUT_FILES as INFERENCE_PLOT_FILES

IMPORTS = TRAINING_IMPORTS

DEFAULT_CANVAS = canvas_path("e2e-full-report.canvas.tsx")

PHASE3_SUBS = (10_000, 50_000, 100_000, 200_000)
PHASE3_GPUS = (1, 4)
MODELS = ("geotransolver_surface", "geotransolver_volume")

BODY = r'''
const GT_SURFACE = "GeoTransolver Surface";
const GT_VOLUME = "GeoTransolver Volume";

const layerSpeedups = [
  ["Ball Query (Radius Search)", "8192→4096, r=0.1, m=32", "0.886 ms", "39.0 ms", "44.0×"],
  ["PointToGridInterpolation", "3D Gaussian, 32³, n=512", "0.186 ms", "35.7 ms", "191.9×"],
  ["GridToPointInterpolation", "3D smooth₂, 32³, n=512", "0.194 ms", "2.12 ms", "10.9×"],
  ["MeshGreenGaussGradient", "2D tri 36×36, scalar", "0.283 ms", "1.79 ms", "6.3×"],
  ["RadiusSearch backward", "4096→2048 pts", "1.03 ms", "10.4 ms", "10.1×"],
  ["PointToGridInterp backward", "3D Gaussian, 32³", "0.963 ms", "59.1 ms", "61.4×"],
  ["MeshLSQGradient", "1024 pts, k=16", "0.367 ms", "0.685 ms", "1.9×"],
  ["RectilinearGridGradient", "3D 96³, d=1", "1.02 ms", "1.30 ms", "1.3×"],
];

const cudaBreakdown = [
  ["Ball Query – RadiusSearch (Warp)", "21.6%"],
  ["Optimizer (Muon)", "5.4%"],
  ["GEMM / MatMul", "2.6%"],
  ["BVH Query (Warp)", "1.8%"],
  ["Attention", "1.6%"],
  ["Other (elementwise, runtime, casts)", "67.0%"],
];

const scaling200kExtended = [
  ["1", "1.46", "0.683", "1.00"],
  ["4", "5.04", "0.794", "3.44"],
  ["16", "19.05", "0.840", "13.01"],
  ["32", "37.26", "0.859", "25.45"],
  ["64", "72.73", "0.880", "49.67"],
  ["96", "93.83", "1.023", "64.08"],
  ["128", "102.29", "1.251", "69.85"],
];

const lustreG1 = [
{{LUSTRE_G1_ROWS}}
];

const memRowsG4 = [
{{MEM_ROWS_G4}}
];

const volumeNvmeScaling = [
{{VOLUME_NVME_SCALING_ROWS}}
];

const nvmeGain16 = [
  ["10,000", "51", "95", "+86%"],
  ["50,000", "50", "83", "+67%"],
  ["100,000", "39", "75", "+90%"],
  ["200,000", "31", "59", "+92%"],
];

const inferenceSurface = [
{{INFERENCE_SURFACE_ROWS}}
];

const inferenceVolume = [
{{INFERENCE_VOLUME_ROWS}}
];

const meshIoBenchmark = [
  ["ShiftSUV", "4.1 GiB", "1.5 GiB", "2.7×", "58.5 s", "1.7 s", "35×"],
  ["DrivAerML", "46.3 GiB", "5.8 GiB", "7.9×", "412.6 s", "4.7 s", "88×"],
  ["HiLiftAeroML", "99.9 GiB", "18.7 GiB", "5.4×", "119.4 s †", "12.6 s", "9.5×"],
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
  return <Text style={{ lineHeight: 1.8, maxWidth: 820 }}>{children}</Text>;
}

function Tech({ title, children }: { title: string; children: any }) {
  const theme = useHostTheme();
  return (
    <Stack gap={6} style={{
      padding: "14px 16px",
      borderLeft: `3px solid ${theme.stroke.secondary}`,
      background: theme.fill.quaternary,
      borderRadius: 2,
    }}>
      <Text weight="semibold" size="small">{title}</Text>
      <Text style={{ lineHeight: 1.8, fontSize: 14 }}>{children}</Text>
    </Stack>
  );
}

function TC({ n, children }: { n: number; children: any }) {
  return (
    <Text tone="secondary" size="small" style={{ marginTop: 8, lineHeight: 1.6 }}>
      <Text weight="semibold" as="span">Table {n}. </Text>{children}
    </Text>
  );
}

function Section({ n, title }: { n: number; title: string }) {
  return <H2 style={{ fontFamily: "inherit", marginTop: 16 }}>{n}. {title}</H2>;
}

function Sub({ id, title }: { id: string; title: string }) {
  return <H3 style={{ fontFamily: "inherit", fontSize: 15, marginTop: 8 }}>{id} {title}</H3>;
}

export default function CAEBenchmarkMilestone1FullReport() {
  return (
    <div style={{ maxWidth: 820, margin: "0 auto", padding: "48px 32px 64px", fontFamily: "Georgia, serif" }}>
      <Stack gap={28}>
        <Stack gap={12}>
          <Text tone="secondary" size="small">CAE Benchmarking · Milestone 1 · Technical Analysis</Text>
          <H1 style={{ fontFamily: "inherit" }}>
            Performance Characterization of GeoTransolver on DrivAerML (NVIDIA B200)
          </H1>
          <Text tone="secondary">
            {{RUN_COUNT}} benchmark runs · HSG cluster · updated {{LAST_UPDATED}} ·
            companion markdown: END_TO_END_TRAINING_PERFORMANCE_REPORT.md
          </Text>
        </Stack>

        <Grid columns={3} gap={12}>
          <Stat value="44×" label="RadiusSearch kernel speedup (Warp vs PyTorch)" tone="success" />
          <Stat value="1.27×" label="Predicted end-to-end speedup (Amdahl, f=0.216)" />
          <Stat value="R²≈0.999" label="Linear VRAM fit vs subsample count" />
          <Stat value="67–131%" label="NVMe throughput gain @ g=16 vs Lustre" tone="success" />
          <Stat value="488 ms" label="Inference P50 @ 200k points (Volume)" />
          <Stat value="η&lt;70%" label="Lustre weak-scaling efficiency @ g≥16" tone="warning" />
        </Grid>

        <Section n={0} title="Experimental Apparatus and Measurement Protocol" />
        <Para>
          We benchmark two variants of the GeoTransolver architecture — surface (boundary mesh /
          point-cloud modality) and volume (volumetric point cloud) — trained on the DrivAerML
          external-aerodynamics dataset. All measurements use NVIDIA B200 accelerators (192 GB HBM3e
          per device) on the HSG cluster. Training employs data-parallel replication with a fixed
          per-rank batch size of one CFD sample (batch_size = 1), so increasing GPU count adds
          concurrent samples rather than enlarging the local mini-batch (weak scaling).
        </Para>
        <Tech title="Metric definitions">
          Step time (P50): median wall-clock duration of a training iteration (forward + backward +
          optimizer), pooling epochs 1–4 only (epoch 0 excluded). Throughput: Φ = B·N / t_P50 samples/s, where B is per-rank
          batch size and N is GPU count. Weak-scaling efficiency: η(N) = Φ(N) / (N·Φ(1)), with η=100%
          indicating ideal linear aggregate throughput. Peak memory: max(torch.cuda.memory_reserved())
          per rank over aggregated train steps. Inference latencies are extracted from the validation
          forward pass under torch.no_grad() at batch_size = 1.
        </Tech>
        <Para>
          Storage is evaluated at two tiers: Lustre (network parallel filesystem, shared metadata
          server) and NVMe (node-local stage-in via cp -a prior to training). Subsampling resolution
          (10k–400k points per sample) is the operative axis for memory and compute scaling because
          the recipe does not support batch_size &gt; 1.
        </Para>

        <Section n={1} title="Kernel-Level Performance: PhysicsNeMo (Warp) vs Native PyTorch" />
        <Sub id="1.1" title="Micro-benchmark methodology" />
        <Para>
          Layer timings are measured with ASV (Airspeed Velocity) on FunctionSpec classes that expose
          dual implementations: a PhysicsNeMo Warp kernel and a reference PyTorch baseline. Each entry
          in Table 1 is a representative forward or backward pass at fixed tensor geometry on a single
          B200 GPU. Speedup S = t_PyTorch / t_Warp.
        </Para>
        <Table
          headers={["Layer", "Configuration", "Warp (ms)", "PyTorch (ms)", "Speedup"]}
          rows={layerSpeedups}
          columnAlign={["left", "left", "right", "right", "right"]}
          striped
        />
        <TC n={1}>ASV micro-benchmarks, B200, 1 GPU. Appendix I (markdown) will report variance across point counts.</TC>

        <BarChart
          categories={["BallQ fwd", "Pt2Grid fwd", "Grid2Pt fwd", "MGauss fwd", "BallQ bwd", "Pt2Grid bwd", "LSQ fwd", "RectGrad"]}
          series={[{ name: "Speedup vs PyTorch (×)", data: [44.0, 191.9, 10.9, 6.3, 10.1, 61.4, 1.9, 1.3], tone: "success" }]}
          horizontal
          height={280}
          valueSuffix="×"
        />
        <Text tone="secondary" size="small">
          Figure A — Kernel speedup ratio S = t_PyTorch / t_Warp. Source: ASV, B200, June 2026.
        </Text>

        <Sub id="1.2" title="Mechanistic interpretation" />
        <Tech title="Why Warp wins on irregular geometry ops">
          Ball Query (radius search) and PointToGridInterpolation are irregular, sparse gather/scatter
          operations over unstructured point sets. Naive PyTorch implementations materialize full
          N×M distance matrices or use Python-level loops, incurring O(N·M) work and poor memory
          locality. PhysicsNeMo Warp kernels employ spatial hashing and bounded-volume hierarchies
          (BVH) to reduce neighbor queries to approximately O(N·k) with small constant k, and execute
          as structure-aware GPU kernels with coalesced memory access. PointToGrid scatter at 191.9×
          reflects the extreme cost of atomic scatter in pure PyTorch versus a fused Warp reduction path.
          RectilinearGridGradient (1.3×) and MeshLSQGradient (1.9×) operate on structured or
          low-cardinality stencils where PyTorch is already competitive — diminishing returns are expected.
        </Tech>

        <Sub id="1.3" title="End-to-end attribution via Amdahl's law" />
        <Para>
          Micro-benchmark speedups do not translate one-to-one to model speedups. A torch.profiler
          trace of GeoTransolver Volume at 200k subsample (DrivAerML, 1× B200) decomposes CUDA time
          by kernel category:
        </Para>
        <Grid columns={2} gap={16}>
          <PieChart
            donut
            size={210}
            data={[
              { label: "Ball Query (Warp)", value: 21.6, tone: "warning" },
              { label: "Optimizer", value: 5.4 },
              { label: "GEMM", value: 2.6 },
              { label: "BVH Query", value: 1.8 },
              { label: "Attention", value: 1.6 },
              { label: "Other", value: 67.0, tone: "neutral" },
            ]}
          />
          <Stack gap={8}>
            <Table headers={["CUDA category", "% step time"]} rows={cudaBreakdown} striped />
            <Tech title="Amdahl bound for Ball Query">
              Let f = 0.216 be the serial fraction attributable to RadiusSearch and k = 44 the measured
              kernel speedup. End-to-end speedup S_model = 1 / (1 − f + f/k) ≈ 1.27×. The 67% "Other"
              bucket (elementwise ops, framework runtime, dtype casts, non-Warp kernels) caps the
              maximum achievable model speedup from kernel replacement alone. Compound optimization of
              all Warp categories (Ball Query + BVH + interpolation + gradients) yields a larger but still
              bounded gain; use profile_and_attribute.py --mode estimate for full attribution.
            </Tech>
          </Stack>
        </Grid>

        <Section n={2} title="ETL and Datapipe — Mesh I/O (Ask 2)" />
        <Para>
          The production recipe ingests DrivAerML through PhysicsNeMo's MeshDataset pipeline: MeshReader
          loads pre-serialized DomainMesh tensors from .pdmsh files (topology, fields, and metadata
          pre-computed offline). A conventional PyTorch CAE workflow typically reads raw VTK/VTP,
          parses mesh connectivity in Python, and assembles torch tensors on every __getitem__ call —
          amortizing ETL cost across every epoch rather than once at curation time.
        </Para>
        <Para>
          Cold-disk mesh I/O (Peter Sharpe mesh_benchmarking) quantifies deserialize cost before
          in-training dataloader ablation: 3 trials, page-cache eviction, interior + boundary per sample.
          PhysicsNeMo-Mesh memmap loads are 9×–88× faster than VTK; disk use is 2.7×–7.9× smaller.
          HSG replicated ShiftSUV (58.1 s vs 1.6 s PMSH, Jun 2026).
        </Para>
        <Figure n={0} src={IMG_MESH_IO_REF}
          caption="Fig 2.0 — VTU vs PhysicsNeMo-Mesh: disk size and cold-disk load time (published reference, Apr 2026)."
        />
        <Table
          headers={["Dataset", "VTU", "PMSH", "Smaller", "VTU load", "PMSH load", "Faster"]}
          rows={meshIoBenchmark}
          columnAlign={["left","right","right","right","right","right","right"]}
          striped
        />
        <TC n={1}>† HiLiftAeroML VTU physics-fields-only: 112.2 s. Source: Peter Sharpe README; HSG job 3077314 (ShiftSUV only).</TC>
        <Callout tone="info" title="Still pending">
          Table 2.1 — VTK vs PNM.mesh inside the training dataloader (ms/sample) — and full HSG replication
          for DrivAerML/HiLift (blocked on .pdmsh file permissions on Lustre). NVMe training dividend in §3.6
          (+67–131% at g=16) is partly this I/O story at scale.
        </Callout>

        <Section n={3} title="End-to-End Training Performance" />
        <Sub id="3.1" title="Single-device compute characterization (g = 1, Lustre)" />
        <Para>
          At single-GPU, throughput is limited by per-sample forward/backward cost on unstructured
          point clouds. GeoTransolver Surface processes boundary representations with lower active
          point counts and shallower operator graphs than Volume; consequently Surface sustains
          2–4× higher sample throughput at every subsampling level.
        </Para>
        <Figure n={1} src={IMG_LUSTRE_G1}
          caption="Single-GPU throughput Φ (samples/s, P50) and epoch wall time vs subsampling. Lustre, B200, batch_size=1."
        />
        <Table
          headers={["Subsample N", "Φ_surf", "t_ep surf", "t_step surf", "M_surf", "Φ_vol", "t_ep vol", "t_step vol", "M_vol"]}
          rows={lustreG1}
          columnAlign={["left","right","right","right","right","right","right","right","right"]}
          striped
        />
        <TC n={2}>Live Lustre g=1 summaries. Φ = samples/s; t_ep = s/epoch; t_step = P50 (s); M = peak VRAM (GB).</TC>
        <Tech title="Subsampling sensitivity">
          From N=10k→200k: Surface throughput drops 41% (6.34→3.77 s⁻¹) while Volume drops 62%
          (2.91→1.12 s⁻¹). Step latency grows super-linearly in N because neighbor queries, attention
          over local patches, and activation memory traffic all scale with active point count. At N=100k
          (representative operating point): Φ_surf=4.76 s⁻¹, Φ_vol=1.72 s⁻¹.
        </Tech>

        <Sub id="3.2" title="Storage-tier effect at single GPU (NVMe stage-in)" />
        <Para>{{PHASE3_NVME_PARA}}</Para>
        <Figure n={2} src={IMG_NVME_G16} caption="{{FIG2_CAPTION}}" />
        <Tech title="Interpretation">
          NVMe stage-in removes network-filesystem read latency from the critical path. At g=1 the
          throughput gain (+20–42% across the sweep) arises entirely from reduced step time — no
          additional parallelism is introduced. Peak VRAM is identical between Lustre and NVMe for the
          same (model, N) because storage tier affects I/O only, not activation footprint.
        </Tech>

        <Sub id="3.3" title="GPU memory scaling law" />
        <Para>
          Peak reserved VRAM scales linearly with subsample count N over 30× dynamic range (R² ≈ 0.999):
        </Para>
        <Tech title="Empirical memory models (g=4 per-rank peaks)">
          M_surf(N) ≈ 0.300 · (N/1000) + 0.6 GB · M_vol(N) ≈ 0.383 · (N/1000) + 0.77 GB.
          Volume carries ~28% higher memory coefficient due to volumetric field tensors and deeper
          intermediate activations. Memcheck confirms linearity through 115.79 GB (Volume @ 300k) and
          154.21 GB (Volume @ 400k, 80% of 192 GB ceiling). Extrapolated OOM thresholds: N≈640k
          (Surface), N≈500k (Volume).
        </Tech>
        <Figure n={3} src={IMG_MEMORY}
          caption="Peak per-rank VRAM vs N (g=4). Phase 1 sweep + 300k memcheck anchor."
        />
        <Figure n={4} src={IMG_SINGLE_GPU_MEM}
          caption="Single-GPU (g=1) memory envelope vs 192 GB HBM3e ceiling."
        />
        <Table headers={["N", "M_surf", "Headroom", "M_vol", "Headroom"]} rows={memRowsG4} striped />

        <Sub id="3.4" title="Multi-GPU weak scaling" />
        <Para>
          We tested GeoTransolver Volume and Surface on DrivAerML at 50k, 100k, and 200k subsampling,
          scaling from 1 to 64 B200 GPUs. Each GPU trains one sample per step. Data was read either
          from Lustre (shared network storage) or from NVMe (copied to local disk first). The first
          training epoch is excluded from all numbers — it includes one-time warmup like JIT compile
          and CUDA startup that would skew the results.
        </Para>
        <Figure n={5} src={IMG_THROUGHPUT}
          caption="Training throughput (samples/s) vs GPU count. Top: Volume. Bottom: Surface. Green = NVMe, gray = Lustre."
        />
        <Para>
          Throughput is how many training samples the whole cluster completes per second. NVMe wins
          at every GPU count, and the gap widens past 16 GPUs. Volume on NVMe scales well — e.g.
          50k subsampling goes from 3.5 samples/s on 1 GPU to 161 on 64. The standout regression is
          Surface @ 50k: throughput peaks at 32 GPUs (~155 samples/s) then drops at 64 (~125
          samples/s). Surface steps are short, so GPUs finish quickly but still wait on network sync
          across nodes. That fixed wait time hurts more when compute is light. Volume steps are
          heavier, so communication is a smaller fraction of total time.
        </Para>
        <Figure n={6} src={IMG_EFFICIENCY}
          caption="Parallel efficiency vs GPU count. 100% dashed line = perfect linear scaling."
        />
        <Para>
          Efficiency asks: if 1 GPU gives X samples/s, did N GPUs give N×X? Volume on NVMe holds
          72–78% at 64 GPUs — good, not perfect. Lustre drops to 31–58%. Surface @ 50k–100k falls
          to ~29% at 64 GPUs: you added 64× hardware but only got ~19× throughput. That is the same
          Surface regression from the throughput plot, seen from a different angle.
        </Para>
        <Para>
          The table below lists GeoTransolver Volume on NVMe in detail. Each row is one GPU count at
          a fixed subsampling level (50k, 100k, or 200k points per sample). Speedup compares to the
          1-GPU run at the same subsampling.
        </Para>
        <Table
          headers={["N pts", "GPUs", "Φ (s/s)", "t_step (s)", "Speedup"]}
          rows={volumeNvmeScaling}
          columnAlign={["right","right","right","right","right"]}
          striped
        />
        <TC n={3}>GeoTransolver Volume · NVMe · extended sweep through g=128 (figures focus on g≤64).</TC>

        <Sub id="3.5" title="Extended scaling — Volume @ 200k (reported sweep to g=128)" />
        <Para>
          An extended NVMe sweep documents scaling beyond the Phase 1 matrix. Throughput rises
          approximately linearly through g=64, then plateaus/regresses as inter-node communication
          dominates:
        </Para>
        <LineChart
          categories={scaling200kExtended.map(([g]) => g)}
          series={[
            { name: "Throughput Φ (samples/s)", data: scaling200kExtended.map(([, p]) => parseFloat(p)), tone: "success" },
            { name: "Step time t_P50 (×10 s)", data: scaling200kExtended.map(([, , t]) => parseFloat(t) * 10), tone: "info" },
          ]}
          height={220}
        />
        <Text tone="secondary" size="small">
          Figure B — Volume @ 200k, NVMe, B200. Step time series scaled ×10 for shared axis visibility.
        </Text>
        <Table headers={["GPUs", "Φ (s/s)", "t_step (s)", "Speedup vs g=1"]} rows={scaling200kExtended} striped />
        <Tech title="g=96 regression">
          At g=64→96 (50k subsample): Φ drops 161→146 s⁻¹ while t_step jumps 0.397→0.657 s. This is
          consistent with NCCL collective latency and straggler effects exceeding the marginal compute
          gain. Production recommendation: g=64, N=200k, NVMe — Φ=72.7 s⁻¹, t_step=0.880 s, M=77.6 GB.
        </Tech>

        <Sub id="3.6" title="NVMe vs Lustre — isolating the I/O bottleneck" />
        <Figure n={7} src={IMG_NVME_16} caption="Φ at g=16: Lustre vs NVMe. Percent = NVMe gain." />
        <Figure n={8} src={IMG_STORAGE_CMP}
          caption="Cross-tier comparison: Lustre g=1, Lustre g=16, NVMe g=16."
        />
        <Table headers={["Subsample", "Φ Lustre", "Φ NVMe", "Δ"]} rows={nvmeGain16} striped />
        <Tech title="Causal inference">
          Comparing Lustre vs NVMe at fixed g=16 isolates the storage hypothesis: model weights,
          optimizer state, learning-rate schedule, and NCCL topology are identical; only the dataset
          mount point differs. A +67–131% throughput swing with zero code changes rejects compute
          saturation and communication cost as dominant explanations at g=16 and confirms I/O starvation
          on Lustre. Stage-in cost (cp -a of DrivAerML tree) amortizes within 1–2 epochs.
        </Tech>

        <Section n={4} title="Inference Latency Characterization" />
        <Sub id="4.1" title="Measurement scope" />
        <Para>
          Validation-pass wall times (val_step in metrics.jsonl, torch.no_grad(), batch_size = 1, g = 1,
          Lustre, B200). Covers subsampling <strong>10k–300k</strong> (Surface) and <strong>10k–400k</strong>
          (Volume) from completed matrix runs. Excludes serving overhead.
        </Para>
        <Figure n={9} src={IMG_INFER_LAT}
          caption="Validation-step latency vs subsampling — Surface and Volume. P50/P95/P99 from benchmark summaries."
        />
        <Figure n={12} src={IMG_INFER_BOX_NVME}
          caption="Validation-step latency distribution (NVMe, g=1): Volume (left) and Surface (right). Box = P25–P75; whiskers = P5–P95; diamond = P99. Subsamples 10k–300k."
        />
        <Figure n={13} src={IMG_INFER_BOX_LUSTRE}
          caption="Same as Fig 12 on Lustre — both models through 300k subsample in completed matrix."
        />
        <Figure n={10} src={IMG_INFER_BREAK}
          caption="P50 breakdown: min validation step + (P50 − min). Dataloader/forward not separately instrumented in matrix runs."
        />
        <Figure n={11} src={IMG_INFER_MEM}
          caption="Peak reserved VRAM during same runs (training peak; proxy for inference memory envelope)."
        />
        <Sub id="4.2" title="GeoTransolver Surface" />
        <Table
          headers={["Subsampling", "P50", "P95", "P99", "Min", "P50−min", "Peak GB"]}
          rows={inferenceSurface}
          columnAlign={["left","right","right","right","right","right","right"]}
          striped
        />
        <Sub id="4.3" title="GeoTransolver Volume" />
        <Para>
          On NVMe (g=1), Volume median latency scales from <strong>127 ms @ 10k</strong> to
          <strong>727 ms @ 300k</strong> with tight IQRs. Surface stays flat below
          <strong>80 ms P50</strong> through 200k (no NVMe 300k run in matrix).
        </Para>
        <Table
          headers={["Subsampling", "P50", "P95", "P99", "Min", "P50−min", "Peak GB"]}
          rows={inferenceVolume}
          columnAlign={["left","right","right","right","right","right","right"]}
          striped
        />
        <Tech title="Cross-model comparison @ 200k">
          Surface P50 <strong>135 ms</strong> vs Volume <strong>690 ms</strong> (~5.1×). Volume @ 400k:
          P50 <strong>1.20 s</strong>, peak <strong>162 GB</strong> (84% of 192 GB). Inference throughput
          ≈ 1000/P50: Surface <strong>7.4 s⁻¹</strong>, Volume <strong>1.45 s⁻¹</strong> @ 200k.
        </Tech>

        <Section n={5} title="Synthesis and Operating Recommendations" />
        <Grid columns={2} gap={16}>
          <Stack gap={10}>
            <Text weight="semibold">Kernel layer (Ask 1)</Text>
            <Para>
              Warp delivers 1.3×–192× speedup on geometry-heavy ops; end-to-end model gain is bounded
              by Amdahl's law (~1.27× from Ball Query alone). Marketing should cite profiler-attributed
              speedups, not peak ASV micro-benchmarks in isolation.
            </Para>
            <Text weight="semibold">Training (Ask 3)</Text>
            <Para>
              Stage to NVMe for g≥16. Use Volume @ 200k, g=64, NVMe as production point
              (Φ=72.7 s⁻¹, M=77.6 GB). Avoid g&gt;64 unless aggregate throughput, not per-step latency,
              is the objective.
            </Para>
          </Stack>
          <Stack gap={10}>
            <Text weight="semibold">Inference (Ask 4)</Text>
            <Para>
              Volume P50 690 ms @ 200k (~1.45 inf/s); Surface 135 ms @ 200k. RTX 6000 Pro / 5080 baselines
              pending.
            </Para>
            <Text weight="semibold">Outstanding (Ask 2, platform)</Text>
            <Para>
              VTK vs PNM.mesh datapipe ablation and RTX SKU replay remain open. Memcheck2 @ N=500k will
              confirm empirical Volume OOM ceiling.
            </Para>
          </Stack>
        </Grid>

        <Divider />
        <Text tone="tertiary" size="small">
          Regenerate: python benchmarks/build_full_canvas_report.py ·
          python benchmarks/plot_scaling_snapshot.py · {{RUN_COUNT}} runs · {{LAST_UPDATED}}
        </Text>
      </Stack>
    </div>
  );
}
'''


def _js_row(cells: list) -> str:
    return "  [" + ", ".join(json.dumps(str(c)) for c in cells) + "]"


def _lustre_g1_rows(summaries: list[dict]) -> str:
    rows: list[str] = []
    for sub in PHASE3_SUBS:
        cells = [f"{sub:,}"]
        for model in MODELS:
            s = _find_summary(
                summaries,
                model=model,
                storage="lustre",
                num_gpus=1,
                sampling_resolution=sub,
            )
            if s is None:
                cells.extend(["—", "—", "—", "—"])
            else:
                ep = s["wallclock_train_s"] / num_epochs_for_aggregate(s)
                cells.extend(
                    [
                        f"{s['throughput_samples_per_sec_p50']:.2f}",
                        f"{ep:.1f}",
                        f"{s['train']['p50']:.3f}",
                        f"{s['memory']['peak_gb']:.1f}",
                    ]
                )
        rows.append(_js_row(cells))
    return ",\n".join(rows)


def _mem_rows_g4(summaries: list[dict]) -> str:
    hbm = 192.0
    subs = (10_000, 50_000, 100_000, 200_000, 250_000, 300_000, 400_000)
    rows: list[str] = []
    for sub in subs:
        cells = [f"{sub:,}"]
        for model in MODELS:
            s = _find_summary(
                summaries,
                model=model,
                storage="lustre",
                num_gpus=4,
                sampling_resolution=sub,
            )
            if s is None and sub in (250_000, 300_000, 400_000):
                s = next(
                    (
                        x
                        for x in summaries
                        if x.get("model") == model
                        and x.get("num_gpus") == 4
                        and x.get("sampling_resolution") == sub
                    ),
                    None,
                )
            if s is None:
                cells.extend(["—", "—"])
            else:
                peak = s["memory"]["peak_gb"]
                head = f"{(hbm - peak) / hbm * 100:.0f}%"
                cells.extend([f"{peak:.2f}", head])
        mark = ""
        if sub == 250_000:
            mark = " ⓘ"
        elif sub == 400_000:
            mark = " ⓘⓘ"
        rows.append(_js_row([cells[0] + mark] + cells[1:]))
    return ",\n".join(rows)


def _volume_nvme_scaling_rows(summaries: list[dict]) -> str:
    vol_nvme = [
        s
        for s in summaries
        if s.get("model") == "geotransolver_volume" and s.get("storage") == "nvme"
    ]
    vol_nvme.sort(key=lambda s: (s["sampling_resolution"], s["num_gpus"]))
    baselines: dict[int, float] = {}
    for s in vol_nvme:
        if s["num_gpus"] == 1:
            baselines[s["sampling_resolution"]] = s["throughput_samples_per_sec_p50"]
    rows: list[str] = []
    for s in vol_nvme:
        sub = s["sampling_resolution"]
        g = s["num_gpus"]
        thr = s["throughput_samples_per_sec_p50"]
        step = s["train"]["p50"]
        base = baselines.get(sub)
        speedup = f"{thr / base:.2f}" if base else "—"
        rows.append(
            _js_row(
                [
                    f"{sub:,}",
                    str(g),
                    f"{thr:.2f}",
                    f"{step:.3f}",
                    speedup,
                    f"{s['memory']['peak_gb']:.1f}",
                ]
            )
        )
    if not rows:
        rows.append(_js_row(["—", "—", "—", "—", "—", "—"]))
    return ",\n".join(rows)


def _phase3_status(summaries: list[dict]) -> tuple[str, str, str]:
    keys = {
        (s["model"], s["num_gpus"], s["sampling_resolution"])
        for s in summaries
        if s.get("storage") == "nvme"
        and s.get("num_gpus") in PHASE3_GPUS
        and s.get("sampling_resolution") in PHASE3_SUBS
        and s.get("model") in MODELS
    }
    total = len(MODELS) * len(PHASE3_GPUS) * len(PHASE3_SUBS)
    done = len(keys)
    if done >= total:
        status = "Complete (16/16)"
    elif done == 0:
        status = "Not run"
    else:
        status = f"In progress ({done}/{total})"
    return status, str(done), str(total)


def _build_substitutions(results_root: Path | None) -> dict[str, str]:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    subs: dict[str, str] = {
        "{{RUN_COUNT}}": "?",
        "{{LAST_UPDATED}}": now,
        "{{LUSTRE_G1_ROWS}}": _js_row(["—"] * 9),
        "{{MEM_ROWS_G4}}": _js_row(["—"] * 5),
        "{{VOLUME_NVME_SCALING_ROWS}}": _js_row(["—"] * 6),
        "{{PHASE3_NVME_PARA}}": "Results pending.",
        "{{FIG2_CAPTION}}": "NVMe baseline throughput and epoch time.",
        "{{PHASE3_STATUS}}": "Not run",
        "{{MEMCHECK2_STATUS}}": "Queued",
        "{{P0_PLOTS_STATUS}}": "Inference 23–25 done; layer 09–10 pending",
        "{{VOL_100K_THR}}": "—",
        "{{VOL_200K_MEM}}": "—",
        "{{VOL_16_200K_THR}}": "—",
        "{{INFERENCE_SURFACE_ROWS}}": _js_row(["—"] * 7),
        "{{INFERENCE_VOLUME_ROWS}}": _js_row(["—"] * 7),
    }
    if results_root is None or not results_root.is_dir():
        return subs

    summaries = _load_summaries(results_root)
    infer_pts = collect_inference_points(summaries)
    subs["{{INFERENCE_SURFACE_ROWS}}"] = js_table_rows(infer_pts, "geotransolver_surface")
    subs["{{INFERENCE_VOLUME_ROWS}}"] = js_table_rows(infer_pts, "geotransolver_volume")
    subs["{{RUN_COUNT}}"] = str(len(summaries))
    subs["{{LUSTRE_G1_ROWS}}"] = _lustre_g1_rows(summaries)
    subs["{{MEM_ROWS_G4}}"] = _mem_rows_g4(summaries)
    subs["{{VOLUME_NVME_SCALING_ROWS}}"] = _volume_nvme_scaling_rows(summaries)

    phase3_status, done_s, total_s = _phase3_status(summaries)
    subs["{{PHASE3_STATUS}}"] = phase3_status

    memcheck2 = list((results_root / "_memcheck2").rglob("benchmark_summary.json"))
    subs["{{MEMCHECK2_STATUS}}"] = (
        f"Partial ({len(memcheck2)}/4)" if memcheck2 else "Queued"
    )

    nvme_g1 = [
        s
        for s in summaries
        if s.get("storage") == "nvme" and s.get("num_gpus") == 1
    ]
    lines: list[str] = []
    if int(done_s) >= int(total_s):
        lines.append(
            f"Phase 3 NVMe matrix is complete ({done_s}/{total_s} cells). "
            "Figure 2 shows g=1 NVMe when the sweep is complete."
        )
    else:
        lines.append(f"Phase 3 NVMe: {done_s}/{total_s} cells complete.")
    for model, label in [
        ("geotransolver_surface", "GeoTransolver Surface"),
        ("geotransolver_volume", "GeoTransolver Volume"),
    ]:
        lustre = _find_summary(
            summaries, model=model, storage="lustre", num_gpus=1, sampling_resolution=100_000
        )
        nvme = _find_summary(
            summaries, model=model, storage="nvme", num_gpus=1, sampling_resolution=100_000
        )
        if lustre and nvme:
            l_thr = lustre["throughput_samples_per_sec_p50"]
            n_thr = nvme["throughput_samples_per_sec_p50"]
            lines.append(
                f"{label} @ 100k: {_pct_delta(n_thr, l_thr)} throughput "
                f"({n_thr:.2f} vs {l_thr:.2f} samples/s)."
            )
    subs["{{PHASE3_NVME_PARA}}"] = " ".join(lines)

    if int(done_s) >= int(total_s) and len(nvme_g1) >= 8:
        subs["{{FIG2_CAPTION}}"] = (
            "Single-GPU NVMe baseline — throughput and epoch time (B200, g=1, Phase 3 complete)."
        )
    else:
        subs["{{FIG2_CAPTION}}"] = (
            f"NVMe throughput and epoch time (g=1 NVMe Phase 3: {done_s}/{total_s}; "
            "may show g=16 fallback until complete)."
        )

    vol_100 = _find_summary(
        summaries,
        model="geotransolver_volume",
        storage="lustre",
        num_gpus=1,
        sampling_resolution=100_000,
    )
    vol_200 = _find_summary(
        summaries,
        model="geotransolver_volume",
        storage="lustre",
        num_gpus=1,
        sampling_resolution=200_000,
    )
    vol_16 = _find_summary(
        summaries,
        model="geotransolver_volume",
        storage="nvme",
        num_gpus=16,
        sampling_resolution=200_000,
    )
    if vol_100:
        subs["{{VOL_100K_THR}}"] = f"{vol_100['throughput_samples_per_sec_p50']:.2f} samples/s"
    if vol_200:
        subs["{{VOL_200K_MEM}}"] = f"{vol_200['memory']['peak_gb']:.1f} GB"
    if vol_16:
        subs["{{VOL_16_200K_THR}}"] = f"{vol_16['throughput_samples_per_sec_p50']:.2f} samples/s"

    return subs


INFERENCE_IMG_CONST = {
    "IMG_INFER_LAT": INFERENCE_PLOT_FILES["latency"],
    "IMG_INFER_BREAK": INFERENCE_PLOT_FILES["breakdown"],
    "IMG_INFER_MEM": INFERENCE_PLOT_FILES["memory"],
    "IMG_INFER_BOX_NVME": INFERENCE_PLOT_FILES["inference_box_nvme"],
    "IMG_INFER_BOX_LUSTRE": INFERENCE_PLOT_FILES["inference_box_lustre"],
    "IMG_MESH_IO_REF": "29_mesh_io_vtu_vs_pmsh_reference.png",
}


def _embed_plots(plots_dir: Path, recipe_root: Path) -> list[str]:
    import base64

    parts: list[str] = []
    root = plots_dir if plots_dir.is_absolute() else recipe_root / plots_dir
    embed_map = {**PLOT_FILES, **INFERENCE_IMG_CONST}
    for const_name, filename in embed_map.items():
        path = root / filename
        if not path.is_file():
            raise SystemExit(f"missing plot: {path}")
        b64 = base64.b64encode(path.read_bytes()).decode("ascii")
        parts.append(f'const {const_name} = "data:image/png;base64,{b64}";')
        print(f"  embedded {filename}")
    return parts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=Path("results/_scaling_snapshot"),
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_CANVAS)
    parser.add_argument("--results", type=Path, default=Path("results"))
    args = parser.parse_args()

    recipe_root = Path(__file__).resolve().parent.parent
    results_root = args.results if args.results.is_absolute() else recipe_root / args.results
    subs = _build_substitutions(results_root)

    parts = [IMPORTS, *_embed_plots(args.plots_dir, recipe_root)]
    parts.append(_apply_substitutions(BODY.strip(), subs))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(parts) + "\n", encoding="utf-8")
    print(f"Wrote {args.out} ({args.out.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
