# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Collect validation-pass inference latency from benchmark_summary.json files."""

from __future__ import annotations

import sys
from pathlib import Path

_BENCHMARKS_ROOT = Path(__file__).resolve().parents[1]
if str(_BENCHMARKS_ROOT) not in sys.path:
    sys.path.insert(0, str(_BENCHMARKS_ROOT))
import json
from dataclasses import dataclass
from pathlib import Path

from ingest.parse_metrics_timing import parse_log
from ingest.summarize_run import DEFAULT_EXCLUDE_EPOCHS, _percentile

INFERENCE_SUBS = (10_000, 50_000, 100_000, 200_000, 300_000, 400_000, 500_000)
INFERENCE_BOX_SUBS = (10_000, 50_000, 100_000, 200_000, 300_000)
INFERENCE_THROUGHPUT_SUBS = (50_000, 100_000, 200_000, 300_000)
VOLUME_BOX_SUBS = INFERENCE_BOX_SUBS  # backward compat

MODEL_SURFACE = "geotransolver_surface"
MODEL_VOLUME = "geotransolver_volume"

MODEL_LABELS = {
    MODEL_SURFACE: "GeoTransolver Surface",
    MODEL_VOLUME: "GeoTransolver Volume",
}


@dataclass(frozen=True)
class InferencePoint:
    model: str
    model_label: str
    subsampling: int
    num_gpus: int
    storage: str
    p50_ms: float
    p95_ms: float
    p99_ms: float
    min_ms: float
    train_p50_ms: float
    peak_gb: float
    val_estimated: bool
    n_val_steps: int

    @property
    def overhead_ms(self) -> float:
        """P50 minus fastest observed validation step (dataloader variance + tail)."""
        return max(0.0, self.p50_ms - self.min_ms)

    @property
    def forward_lower_bound_ms(self) -> float:
        """Fastest validation step — lower bound on forward + minimal stall."""
        return self.min_ms


@dataclass(frozen=True)
class ValLatencyBox:
    """Validation-step latency percentiles (ms) for box-plot rendering."""

    subsampling: int
    p5_ms: float
    p25_ms: float
    p50_ms: float
    p75_ms: float
    p95_ms: float
    p99_ms: float
    n_steps: int


@dataclass(frozen=True)
class ValThroughputBox:
    """Validation-step throughput percentiles (samples/s) for box-plot rendering."""

    subsampling: int
    p5: float
    p25: float
    p50: float
    p75: float
    p95: float
    n_steps: int


def _summary_metrics_path(summary: dict) -> Path | None:
    metrics_path = Path(summary.get("metrics_path", ""))
    if metrics_path.is_file():
        return metrics_path
    return None


def collect_val_step_times_ms(
    results_root: Path,
    *,
    model: str,
    subsampling: int,
    num_gpus: int = 1,
    storage: str = "nvme",
    exclude_epochs: tuple[int, ...] = DEFAULT_EXCLUDE_EPOCHS,
) -> list[float]:
    """Per validation-step wall times (ms), epoch 0 excluded by default."""
    summaries = load_summaries(results_root)
    summary = _find_summary(
        summaries,
        model=model,
        subsampling=subsampling,
        num_gpus=num_gpus,
        storage=storage,
    )
    if summary is None:
        return []
    metrics_path = _summary_metrics_path(summary)
    if metrics_path is None:
        local = (
            results_root
            / model
            / summary["dataset"]
            / f"gpus_{num_gpus}"
            / storage
            / f"sub_{subsampling}"
            / "runs"
        )
        for run_dir in local.glob("*/metrics.jsonl"):
            metrics_path = run_dir
            break
    if metrics_path is None or not metrics_path.is_file():
        return []

    data = parse_log(metrics_path)
    exclude = set(exclude_epochs)
    times_s: list[float] = []
    for ep in data.epochs:
        if ep.epoch in exclude:
            continue
        times_s.extend(ep.val_step_times)
    return [t * 1000.0 for t in times_s]


def val_latency_box(
    samples_ms: list[float], *, subsampling: int
) -> ValLatencyBox | None:
    if not samples_ms:
        return None
    sorted_s = sorted(samples_ms)
    return ValLatencyBox(
        subsampling=subsampling,
        p5_ms=_percentile(sorted_s, 0.05),
        p25_ms=_percentile(sorted_s, 0.25),
        p50_ms=_percentile(sorted_s, 0.50),
        p75_ms=_percentile(sorted_s, 0.75),
        p95_ms=_percentile(sorted_s, 0.95),
        p99_ms=_percentile(sorted_s, 0.99),
        n_steps=len(samples_ms),
    )


def collect_val_latency_boxes(
    results_root: Path,
    *,
    model: str = MODEL_VOLUME,
    num_gpus: int = 1,
    storage: str = "nvme",
    subs: tuple[int, ...] = INFERENCE_BOX_SUBS,
) -> list[ValLatencyBox]:
    boxes: list[ValLatencyBox] = []
    for sub in subs:
        samples = collect_val_step_times_ms(
            results_root,
            model=model,
            subsampling=sub,
            num_gpus=num_gpus,
            storage=storage,
        )
        box = val_latency_box(samples, subsampling=sub)
        if box is not None:
            boxes.append(box)
    return boxes


def val_throughput_box(
    samples_ms: list[float], *, subsampling: int
) -> ValThroughputBox | None:
    """Per-step throughput = 1 / step_time; batch_size = 1 → samples/s."""
    if not samples_ms:
        return None
    thr = sorted(1000.0 / t for t in samples_ms if t > 0)
    if not thr:
        return None
    return ValThroughputBox(
        subsampling=subsampling,
        p5=_percentile(thr, 0.05),
        p25=_percentile(thr, 0.25),
        p50=_percentile(thr, 0.50),
        p75=_percentile(thr, 0.75),
        p95=_percentile(thr, 0.95),
        n_steps=len(thr),
    )


def collect_val_throughput_boxes(
    results_root: Path,
    *,
    model: str = MODEL_VOLUME,
    num_gpus: int = 1,
    storage: str = "nvme",
    subs: tuple[int, ...] = INFERENCE_THROUGHPUT_SUBS,
) -> list[ValThroughputBox]:
    boxes: list[ValThroughputBox] = []
    for sub in subs:
        samples = collect_val_step_times_ms(
            results_root,
            model=model,
            subsampling=sub,
            num_gpus=num_gpus,
            storage=storage,
        )
        box = val_throughput_box(samples, subsampling=sub)
        if box is not None:
            boxes.append(box)
    return boxes


def _find_summary(
    summaries: list[dict],
    *,
    model: str,
    subsampling: int,
    num_gpus: int,
    storage: str,
) -> dict | None:
    for row in summaries:
        if (
            row.get("model") == model
            and row.get("sampling_resolution") == subsampling
            and row.get("num_gpus") == num_gpus
            and row.get("storage") == storage
        ):
            return row
    return None


def load_summaries(results_root: Path) -> list[dict]:
    rows: list[dict] = []
    for path in results_root.rglob("benchmark_summary.json"):
        if "_smoketest" in str(path):
            continue
        rows.append(json.loads(path.read_text(encoding="utf-8")))
    return rows


def collect_inference_points(
    summaries: list[dict],
    *,
    num_gpus: int = 1,
    storage: str = "lustre",
    subs: tuple[int, ...] = INFERENCE_SUBS,
) -> list[InferencePoint]:
    """Validation-step latencies for Surface and Volume at each subsampling."""
    points: list[InferencePoint] = []
    for model in (MODEL_SURFACE, MODEL_VOLUME):
        for sub in subs:
            summary = _find_summary(
                summaries,
                model=model,
                subsampling=sub,
                num_gpus=num_gpus,
                storage=storage,
            )
            if summary is None:
                continue
            val = summary["val"]
            if not val.get("n"):
                continue
            points.append(
                InferencePoint(
                    model=model,
                    model_label=MODEL_LABELS[model],
                    subsampling=sub,
                    num_gpus=num_gpus,
                    storage=storage,
                    p50_ms=val["p50"] * 1000.0,
                    p95_ms=val["p95"] * 1000.0,
                    p99_ms=val["p99"] * 1000.0,
                    min_ms=val["min"] * 1000.0,
                    train_p50_ms=summary["train"]["p50"] * 1000.0,
                    peak_gb=summary["memory"]["peak_gb"],
                    val_estimated=bool(summary.get("val_estimated")),
                    n_val_steps=int(val["n"]),
                )
            )
    return points


def points_for_model(
    points: list[InferencePoint], model: str
) -> list[InferencePoint]:
    return sorted(
        [p for p in points if p.model == model],
        key=lambda p: p.subsampling,
    )


def js_table_rows(points: list[InferencePoint], model: str) -> str:
    """Comma-separated JS row literals for canvas Table rows."""
    rows: list[str] = []
    for p in points_for_model(points, model):
        rows.append(
            "  ["
            + ", ".join(
                json.dumps(x)
                for x in (
                    f"{p.subsampling:,}",
                    f"{p.p50_ms:.0f}",
                    f"{p.p95_ms:.0f}",
                    f"{p.p99_ms:.0f}",
                    f"{p.min_ms:.0f}",
                    f"{p.overhead_ms:.0f}",
                    f"{p.peak_gb:.1f}",
                )
            )
            + "]"
        )
    return ",\n".join(rows) if rows else '  ["—", "—", "—", "—", "—", "—", "—"]'


def format_markdown_table(points: list[InferencePoint], model: str) -> str:
    """Markdown table for one model."""
    model_pts = points_for_model(points, model)
    if not model_pts:
        return "_No data._\n"
    lines = [
        "| Subsampling | P50 (ms) | P95 (ms) | P99 (ms) | "
        "Min step (ms) | Peak VRAM (GB) | Val steps |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for p in model_pts:
        est = " ⓔ" if p.val_estimated else ""
        lines.append(
            f"| **{p.subsampling:,}**{est} | {p.p50_ms:.0f} | {p.p95_ms:.0f} | "
            f"{p.p99_ms:.0f} | {p.min_ms:.0f} | {p.peak_gb:.1f} | {p.n_val_steps} |"
        )
    return "\n".join(lines) + "\n"
