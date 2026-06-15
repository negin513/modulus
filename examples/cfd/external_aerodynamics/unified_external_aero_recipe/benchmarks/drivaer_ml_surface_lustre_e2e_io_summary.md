# GeoTransolver Surface × DrivAerML surface — Lustre E2E + I/O Summary

**Platform:** B200 · batch=1/GPU · 5 epochs · epoch 0 excluded · storage=Lustre

---

## 1. Single-GPU end-to-end (g=1 Lustre)

Training cost grows with subsampling; throughput falls as point count rises.

| Sub | Throughput | Step (P50) | Time/epoch | Peak VRAM | Val step (P50) |
|-----|------------|------------|------------|-----------|----------------|
| 10k | 6.38 s/s | 0.157 s | 89.8 s | 3.6 GB | 0.082 s |
| 50k | 4.76 s/s | 0.210 s | 108.0 s | 17.7 GB | 0.119 s |
| 100k | 4.79 s/s | 0.209 s | 109.9 s | 35.0 GB | 0.120 s |
| **200k** | **3.80 s/s** | **0.263 s** | **134.8 s** | **69.4 GB** | **0.136 s** |
| 250k | 3.51 s/s | 0.285 s | 137.9 s | 86.8 GB | 0.146 s |
| 300k | 3.18 s/s | 0.314 s | 152.9 s | 103.8 GB | 0.156 s |

**Takeaways (g=1):**

- Throughput drops **41%** from 10k→200k (6.38 → 3.80 s/s).
- Step time rises modestly at g=1 (0.16 → 0.26 s) — compute/memory dominate, not I/O.
- VRAM scales ~linearly with subsampling (3.6 → 69.4 GB @ 200k).

---

## 2. Multi-GPU on Lustre — I/O becomes visible

Per-GPU step time **inflates** as GPU count rises → shared Lustre contention.

### @ 100k subsampling

| GPUs | Aggregate thr | Weak-scaling eff | Step (P50) |
|------|---------------|------------------|------------|
| 1 | 4.8 s/s | 100% | 0.209 s |
| 4 | 15.2 s/s | 79% | 0.263 s |
| 16 | 42.4 s/s | **55%** | 0.378 s |
| 32 | 84.5 s/s | **55%** | 0.379 s |
| 64 | 100.1 s/s | **33%** | 0.640 s |

### @ 200k subsampling

| GPUs | Aggregate thr | Weak-scaling eff | Step (P50) |
|------|---------------|------------------|------------|
| 1 | 3.8 s/s | 100% | 0.263 s |
| 4 | 11.8 s/s | 78% | 0.339 s |
| 16 | 32.9 s/s | **54%** | 0.487 s |
| 32 | 51.5 s/s | **42%** | 0.621 s |
| 64 | 155.0 s/s | 64% † | 0.413 s |

† g=64 runs have very few aggregated steps — treat as directional, not production-grade.

**I/O signal:** g=1→g=16 @ 200k, step time **0.263 → 0.487 s (+85%)** while aggregate throughput is only **8.7×** vs ideal **16×**. Lustre is I/O-bound past ~4 GPUs.

---

## 3. Lustre vs NVMe @ g=16 (I/O dividend)

Staging DrivAerML to node-local NVMe restores throughput by cutting per-step stall.

| Sub | Lustre thr | NVMe thr | NVMe gain | Lustre step | NVMe step |
|-----|------------|----------|-----------|-------------|-----------|
| 10k | 51.9 s/s | 97.3 s/s | +88% | 0.309 s | 0.164 s |
| 50k | 50.5 s/s | 84.0 s/s | +66% | 0.317 s | 0.191 s |
| 100k | 42.4 s/s | 75.3 s/s | +78% | 0.378 s | 0.212 s |
| 200k | 32.9 s/s | 60.2 s/s | +83% | 0.487 s | 0.266 s |

At g=16, NVMe roughly **halves** per-GPU step time vs Lustre across subsampling levels.

---

## 4. Dataset I/O (DrivAerML mesh load — not subsampling-specific)

Cold load of one DrivAerML sample on Lustre (HSG measured, Jun 2026):

| Format | On-disk | Load time |
|--------|---------|-----------|
| VTK/VTU | 46.3 GiB | **412 s** |
| PhysicsNeMo `.pdmsh` | 6.5 GiB | **3.1 s** |

**~135× faster load**, **7.9× smaller** on disk vs VTK. Training uses the pre-curated `.pdmsh` datapipe; remaining I/O pain in multi-GPU Lustre runs is **runtime dataloader / metadata access**, not one-time mesh parse.

---

## 5. Bottom line

| Regime | Behavior |
|--------|----------|
| **g=1 Lustre** | Clean E2E baseline; subsampling drives step time & VRAM |
| **g=4 Lustre** | ~78–79% weak-scaling — still healthy |
| **g≥16 Lustre** | I/O-bound; step time inflates, efficiency <70% |
| **g≥16 NVMe** | +66–88% throughput vs Lustre; step times near g=1 |

**Production guidance:** g=1–4 on Lustre is fine for benchmarking subsampling. For **≥16 GPUs**, stage to NVMe.
