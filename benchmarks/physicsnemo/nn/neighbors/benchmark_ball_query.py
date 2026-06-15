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

"""Benchmark the accelerated Warp ball-query layer against the PyTorch baseline.

GeoTransolver's ``GeometricFeatureProcessor`` uses ``BQWarp``
(``physicsnemo.nn.module.ball_query``), which calls
``radius_search(points, queries, radius, max_points, return_points=True)``.
``BQWarp`` always dispatches to the accelerated Warp backend, so this script
calls ``radius_search`` directly with ``implementation="warp"`` vs
``implementation="torch"`` -- the same call signature ``BQWarp`` issues
internally -- to isolate the kernel difference (Warp spatial-hash grid vs the
PyTorch ``cdist`` brute-force search).

Query and target point clouds are random subsamples of real DrivAerML surface
meshes, normalized to a unit bounding box so the search ``radius`` is physically
meaningful. Only the forward pass is timed.

Example
-------
    python benchmark-ball-query.py --plot
    python benchmark-ball-query.py --query-sizes 4096 16384 --target-sizes 65536 131072
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import threading
import time
from pathlib import Path

DEFAULT_DATA_DIR = (
    "/lustre/fsw/portfolios/coreai/projects/coreai_modulus_cae/datasets/"
    "PhysicsNeMo-DrivaerML"
)
# Kept deliberately small: the PyTorch (cdist) backend materializes a dense
# (P x Q) distance matrix, so memory scales as O(P*Q) and large sizes OOM.
DEFAULT_SIZES = [512, 1024, 2048, 4096, 8192, 16384]


def _recover_cuda_after_failure(is_cuda: bool) -> None:
    """Best-effort CUDA reset after a backend failure mid-sweep."""
    import torch

    if not is_cuda:
        return
    torch.cuda.empty_cache()
    try:
        torch.cuda.synchronize()
    except RuntimeError:
        pass


def _is_backend_failure(exc: BaseException) -> bool:
    """Return True when a backend failure should be recorded as NaN, not raised."""
    import torch

    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    if isinstance(exc, getattr(torch, "AcceleratorError", ())):
        return True
    if isinstance(exc, RuntimeError):
        msg = str(exc).lower()
        return "out of memory" in msg or "cuda error" in msg
    return False


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the ball-query benchmark."""
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark the accelerated Warp ball-query backend against the "
            "PyTorch baseline on DrivAerML surface meshes."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=DEFAULT_DATA_DIR,
        help="Root directory of the PhysicsNeMo-DrivaerML dataset.",
    )
    parser.add_argument(
        "--num-meshes",
        type=int,
        default=3,
        help="Number of DrivAerML surface meshes to average timings over.",
    )
    parser.add_argument(
        "--query-sizes",
        type=int,
        nargs="+",
        default=DEFAULT_SIZES,
        help="Query point counts (Q) to sweep.",
    )
    parser.add_argument(
        "--target-sizes",
        type=int,
        nargs="+",
        default=DEFAULT_SIZES,
        help="Target point counts (P) to sweep.",
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=0.05,
        help="Search radius (in unit-cube-normalized coordinates).",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=32,
        help="Maximum neighbors per query (neighbors_in_radius).",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Number of warmup iterations (untimed) per backend/size.",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=20,
        help="Number of timed iterations per backend/size.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run the benchmark on.",
    )
    parser.add_argument(
        "--jsonl",
        type=str,
        default="ball_query_benchmark.jsonl",
        help="Path to write the results JSONL (one JSON object per line).",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Save a forward-latency bar chart PNG (fixed target points).",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Regenerate plot PNGs from an existing --jsonl file (no benchmark run).",
    )
    parser.add_argument(
        "--plot-target-pts",
        type=int,
        default=None,
        help=(
            "Target point count (P) to use for the bar chart; defaults to the "
            "largest target size in the sweep."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for point subsampling.",
    )
    return parser.parse_args()


def load_surface_point_clouds(
    data_dir: str | Path,
    num_meshes: int,
    device: torch.device,
) -> list[torch.Tensor]:
    """Load and unit-cube-normalize DrivAerML surface mesh point clouds.

    Globs ``run_*/drivaer_*.stl.pmsh`` (excluding the ``_single_solid``
    variant), loads each via :meth:`physicsnemo.mesh.Mesh.load`, extracts the
    vertex coordinates, and min-max normalizes each cloud into ``[0, 1]^3`` so
    the search radius is meaningful and comparable across meshes.

    Args:
        data_dir: Root directory of the DrivAerML dataset.
        num_meshes: Maximum number of meshes to load.
        device: Device to place the point clouds on.

    Returns:
        List of point-cloud tensors, each of shape ``(N, 3)``.

    Raises:
        FileNotFoundError: If no matching surface meshes are found.
    """
    import torch

    from physicsnemo.mesh import Mesh

    data_dir = Path(data_dir)
    # Sort for deterministic selection; skip the single-solid variant so each
    # run contributes exactly one (full) surface mesh.
    candidates = sorted(
        p
        for p in data_dir.glob("run_*/drivaer_*.stl.pmsh")
        if "_single_solid" not in p.name
    )
    if not candidates:
        raise FileNotFoundError(
            f"No 'run_*/drivaer_*.stl.pmsh' surface meshes found under {data_dir}. "
            f"Pass --data-dir to point at the PhysicsNeMo-DrivaerML root."
        )

    clouds: list[torch.Tensor] = []
    for path in candidates[:num_meshes]:
        mesh = Mesh.load(path, device=device)
        # Force the requested device explicitly: Mesh.load may leave the
        # memory-mapped tensors on CPU, which would silently run the whole
        # benchmark on CPU.
        points = mesh.points.to(device=device, dtype=torch.float32)
        # Min-max normalize into the unit cube. Guard against degenerate axes.
        mins = points.amin(dim=0, keepdim=True)
        maxs = points.amax(dim=0, keepdim=True)
        span = (maxs - mins).clamp_min(1e-8)
        points = (points - mins) / span
        clouds.append(points.contiguous())
        print(f"  loaded {path.parent.name}/{path.name}: {points.shape[0]} points")

    return clouds


def subsample(points: torch.Tensor, n: int, generator: torch.Generator) -> torch.Tensor:
    """Randomly subsample ``n`` points and add a batch dimension.

    If ``n`` exceeds the available point count, points are sampled with
    replacement so the requested size is always satisfied.

    Args:
        points: Source point cloud of shape ``(N, 3)``.
        n: Number of points to sample.
        generator: CPU RNG for reproducible sampling. Indices are generated on
            CPU and moved to the points' device, so this works regardless of
            where ``points`` live.

    Returns:
        Tensor of shape ``(1, n, 3)`` matching the ``radius_search`` batched path.
    """
    import torch

    num_available = points.shape[0]
    if n <= num_available:
        idx = torch.randperm(num_available, generator=generator)[:n]
    else:
        idx = torch.randint(num_available, (n,), generator=generator)
    return points[idx.to(points.device)].unsqueeze(0).contiguous()


def time_backend(
    points: torch.Tensor,
    queries: torch.Tensor,
    radius: float,
    max_points: int,
    implementation: str,
    warmup: int,
    iters: int,
) -> float:
    """Time the forward pass of a ``radius_search`` backend.

    Runs ``warmup`` untimed iterations (covering Warp JIT / grid-build and
    cuDNN autotuning) before measuring ``iters`` timed iterations. On CUDA the
    timing uses ``torch.cuda.Event`` for accurate device-side latency; on CPU
    it falls back to ``time.perf_counter``. Failures (e.g. the PyTorch
    ``cdist`` backend running out of memory at large sizes) are caught and
    reported as ``nan`` so the sweep can continue.

    Args:
        points: Target/reference points of shape ``(1, P, 3)``.
        queries: Query points of shape ``(1, Q, 3)``.
        radius: Search radius.
        max_points: Maximum neighbors per query.
        implementation: Backend name, ``"warp"`` or ``"torch"``.
        warmup: Number of untimed warmup iterations.
        iters: Number of timed iterations.

    Returns:
        Mean forward-pass latency in milliseconds, or ``nan`` on failure.
    """
    import torch

    from physicsnemo.nn.functional import radius_search

    is_cuda = points.is_cuda

    def _run() -> None:
        radius_search(
            points,
            queries,
            radius,
            max_points=max_points,
            return_points=True,
            implementation=implementation,
        )

    try:
        for _ in range(warmup):
            _run()

        if is_cuda:
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iters):
                _run()
            end.record()
            torch.cuda.synchronize()
            return start.elapsed_time(end) / iters

        # CPU path: wall-clock timing in milliseconds.
        start_t = time.perf_counter()
        for _ in range(iters):
            _run()
        return (time.perf_counter() - start_t) * 1e3 / iters
    except torch.cuda.OutOfMemoryError:
        _recover_cuda_after_failure(is_cuda)
        return math.nan
    except Exception as exc:
        if _is_backend_failure(exc):
            _recover_cuda_after_failure(is_cuda)
            return math.nan
        raise


def init_nvml(device: torch.device):
    """Initialize NVML and return a device handle, or ``None`` if unavailable.

    NVML reports *total* device memory usage (including allocations made by
    Warp, which live outside the PyTorch caching allocator and are therefore
    invisible to ``torch.cuda`` memory stats). This makes it the right tool for
    a fair memory comparison between the two backends.

    Args:
        device: The torch device the benchmark runs on.

    Returns:
        An opaque NVML handle, or ``None`` when NVML/pynvml is unavailable or
        the device is not CUDA.
    """
    import torch

    if device.type != "cuda":
        return None
    try:
        import pynvml
    except ImportError:
        print("  (pynvml not available -- skipping NVML memory capture)")
        return None
    try:
        pynvml.nvmlInit()
        index = device.index if device.index is not None else torch.cuda.current_device()
        return pynvml.nvmlDeviceGetHandleByIndex(index)
    except Exception as exc:  # noqa: BLE001 - NVML errors are opaque
        print(f"  (NVML init failed: {exc} -- skipping memory capture)")
        return None


def _nvml_used_bytes(handle) -> int:
    """Return current total used device memory in bytes via NVML."""
    import pynvml

    return int(pynvml.nvmlDeviceGetMemoryInfo(handle).used)


def _warp_device_str(device: torch.device) -> str:
    """Return a Warp-style device string (e.g. ``cuda:0``) for a torch device."""
    return f"cuda:{device.index}" if device.index is not None else "cuda:0"


@contextlib.contextmanager
def _warp_mempool_disabled(device: torch.device):
    """Temporarily disable Warp's CUDA memory pool within the context.

    Warp pools device allocations, so once memory is requested during warmup it
    is reused (not returned to the driver) on subsequent calls. That makes
    Warp's per-call usage invisible to a device-total tool like NVML. Disabling
    the pool routes allocations straight through ``cudaMalloc`` / ``cudaFree``
    so NVML can observe them. This is a no-op (yields unchanged) when Warp or
    memory-pool support is unavailable.

    Note:
        This is used only in the NVML memory pass, never during latency timing,
        so the (slower) non-pooled path does not affect reported speeds.
    """
    wp = None
    dev_str = None
    enabled = False
    try:
        import warp as wp  # noqa: PLC0415

        dev_str = _warp_device_str(device)
        enabled = wp.is_mempool_supported(dev_str) and wp.is_mempool_enabled(dev_str)
    except Exception:  # noqa: BLE001 - any failure -> just skip toggling
        enabled = False

    if not enabled:
        yield
        return

    wp.set_mempool_enabled(dev_str, False)
    try:
        yield
    finally:
        wp.set_mempool_enabled(dev_str, True)


def _run_loop(
    points: torch.Tensor,
    queries: torch.Tensor,
    radius: float,
    max_points: int,
    implementation: str,
    iters: int,
) -> None:
    """Run ``radius_search`` ``iters`` times, discarding outputs."""
    from physicsnemo.nn.functional import radius_search

    for _ in range(iters):
        out = radius_search(
            points,
            queries,
            radius,
            max_points=max_points,
            return_points=True,
            implementation=implementation,
        )
        del out


def measure_nvml_mem(
    points: torch.Tensor,
    queries: torch.Tensor,
    radius: float,
    max_points: int,
    implementation: str,
    handle,
    iters: int,
) -> float:
    """Measure peak device-total memory growth (MiB) during a call via NVML.

    Drains the PyTorch cache, records a clean baseline of total used device
    memory, then repeatedly runs ``radius_search`` while a background thread
    polls NVML for the device-wide high-water mark. For the Warp backend the
    memory pool is disabled (:func:`_warp_mempool_disabled`) so its allocations
    reach the driver and become visible to NVML.

    NVML reports *device-total* usage (coarse, a few MiB granularity), so very
    small footprints can still read as ~0.

    Returns:
        Peak memory delta in MiB, or ``nan`` when unavailable or on OOM.
    """
    import torch

    if handle is None or not points.is_cuda:
        return math.nan

    mempool_ctx = (
        _warp_mempool_disabled(points.device)
        if implementation == "warp"
        else contextlib.nullcontext()
    )

    with mempool_ctx:
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        baseline = _nvml_used_bytes(handle)

        peak = baseline
        stop = threading.Event()

        def _sampler() -> None:
            nonlocal peak
            while not stop.is_set():
                used = _nvml_used_bytes(handle)
                if used > peak:
                    peak = used

        sampler = threading.Thread(target=_sampler, daemon=True)
        sampler.start()
        try:
            # Repeat so the (sub-millisecond) transient peak is more likely to
            # be caught by the Python-level NVML sampler between launches.
            _run_loop(
                points, queries, radius, max_points, implementation, max(iters, 10)
            )
            torch.cuda.synchronize()
            final = _nvml_used_bytes(handle)
            if final > peak:
                peak = final
            return max(peak - baseline, 0) / (1024.0**2)
        except torch.cuda.OutOfMemoryError:
            _recover_cuda_after_failure(True)
            return math.nan
        except Exception as exc:
            if _is_backend_failure(exc):
                _recover_cuda_after_failure(True)
                return math.nan
            raise
        finally:
            stop.set()
            sampler.join()


def measure_native_mem(
    points: torch.Tensor,
    queries: torch.Tensor,
    radius: float,
    max_points: int,
    implementation: str,
    iters: int,
) -> float:
    """Measure peak working-set memory (MiB) using each backend's own accounting.

    This is exact (byte-level) and immune to allocator-pool reuse, unlike the
    device-total NVML reading:

    - **torch**: ``torch.cuda.reset_peak_memory_stats`` +
      ``torch.cuda.max_memory_allocated`` captures the dense ``(P x Q)``
      distance matrix precisely.
    - **warp**: a background thread samples
      ``warp.get_mempool_used_mem_current`` (live pool usage, which spikes for
      the transient spatial-hash grid and neighbor buffers, then drops), so the
      peak is captured regardless of pool reuse.

    Returns:
        Peak memory delta in MiB, or ``nan`` when unavailable or on OOM.
    """
    import torch

    if not points.is_cuda:
        return math.nan

    device = points.device

    if implementation == "torch":
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize()
            base = torch.cuda.memory_allocated(device)
            _run_loop(points, queries, radius, max_points, "torch", max(iters, 1))
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated(device)
            return max(peak - base, 0) / (1024.0**2)
        except torch.cuda.OutOfMemoryError:
            _recover_cuda_after_failure(True)
            return math.nan
        except Exception as exc:
            if _is_backend_failure(exc):
                _recover_cuda_after_failure(True)
                return math.nan
            raise

    # Warp backend: sample live mempool usage for the transient high-water mark.
    try:
        import warp as wp  # noqa: PLC0415

        dev_str = _warp_device_str(device)
        if not (wp.is_mempool_supported(dev_str)):
            return math.nan
        wp.synchronize_device(dev_str)
        base = wp.get_mempool_used_mem_current(dev_str)
        peak = base
        stop = threading.Event()

        def _sampler() -> None:
            nonlocal peak
            while not stop.is_set():
                used = wp.get_mempool_used_mem_current(dev_str)
                if used > peak:
                    peak = used

        sampler = threading.Thread(target=_sampler, daemon=True)
        sampler.start()
        try:
            _run_loop(points, queries, radius, max_points, "warp", max(iters, 10))
            wp.synchronize_device(dev_str)
            used = wp.get_mempool_used_mem_current(dev_str)
            if used > peak:
                peak = used
            return max(peak - base, 0) / (1024.0**2)
        finally:
            stop.set()
            sampler.join()
    except Exception:  # noqa: BLE001 - Warp accounting is best-effort
        return math.nan


TABLE_HEADERS = [
    "query_pts",
    "target_pts",
    "radius",
    "max_points",
    "warp_ms",
    "torch_ms",
    "speedup",
    "warp_mem_mb",
    "torch_mem_mb",
    "mem_ratio",
    "warp_nvml_mb",
    "torch_nvml_mb",
]


# Columns whose NaN means "ran out of memory" rather than "not measured".
_LATENCY_COLS = {"warp_ms", "torch_ms", "speedup"}


def _fmt_cell(header: str, value: object) -> str:
    """Format a single table cell, NaN-aware per column semantics."""
    if isinstance(value, float):
        if math.isnan(value):
            return "OOM" if header in _LATENCY_COLS else "n/a"
        return f"{value:.3f}"
    return str(value)


def format_table(rows: list[dict]) -> str:
    """Render benchmark result rows as an aligned text table."""
    table = [TABLE_HEADERS]
    for row in rows:
        table.append([_fmt_cell(h, row[h]) for h in TABLE_HEADERS])

    widths = [max(len(r[c]) for r in table) for c in range(len(TABLE_HEADERS))]
    lines = []
    for r_idx, row in enumerate(table):
        lines.append("  ".join(cell.rjust(widths[c]) for c, cell in enumerate(row)))
        if r_idx == 0:
            lines.append("  ".join("-" * w for w in widths))
    return "\n".join(lines)


def write_jsonl(rows: list[dict], path: str | Path) -> None:
    """Write benchmark result rows to a JSONL file (one JSON object per line)."""
    with open(path, "w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


TORCH_CDIST_FOOTNOTE = (
    "PyTorch backend unavailable: torch.cdist hits a CUDA kernel launch "
    "limit when Q > 32,768 (max grid dimension 65,535); this is not an "
    "OOM error."
)
FAILED_MARKER_FONTSIZE = 28
FAILED_FOOTNOTE_MARKER_FONTSIZE = 16
NVIDIA_GREEN = "#76B900"
TORCH_GREY = "#9E9E9E"
FAILED_MARKER = "X"


def _fmt_int(value: int) -> str:
    """Format an integer with thousands separators."""

    return f"{value:,}"


def _query_tick_label(query_pts: int, torch_failed: bool) -> str:
    """Format an x-axis tick label for query point count."""

    del torch_failed
    return _fmt_int(query_pts)


def _select_series(
    rows: list[dict], target_pts: int | None
) -> tuple[int, list[dict]]:
    """Return ``(target_pts, rows)`` filtered to one target-point sweep."""

    if target_pts is None:
        target_pts = max(row["target_pts"] for row in rows)

    series = sorted(
        (r for r in rows if r["target_pts"] == target_pts),
        key=lambda r: r["query_pts"],
    )
    if not series:
        raise ValueError(
            f"No rows with target_pts={target_pts}; "
            f"available targets: {sorted({r['target_pts'] for r in rows})}"
        )
    return target_pts, series


def _configure_matplotlib():
    """Return matplotlib.pyplot after applying shared bold styling."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.weight": "bold",
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
            "mathtext.default": "bf",
            "font.size": 11,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )
    return plt


def _style_axes(ax) -> None:
    """Apply bold tick labels to an axes."""

    ax.tick_params(axis="both", labelsize=10, width=1.2)
    for label in (*ax.get_xticklabels(), *ax.get_yticklabels()):
        label.set_fontweight("bold")


def _comma_axis_formatter():
    """Build a tick formatter that renders large numbers with commas."""

    from matplotlib.ticker import FuncFormatter

    def _fmt(value: float, _pos: int) -> str:
        if value <= 0:
            return ""
        if float(value).is_integer():
            return _fmt_int(int(value))
        return f"{value:,.1f}"

    return FuncFormatter(_fmt)


def _draw_comparison_bars(
    ax,
    query_sizes: list[int],
    warp_vals: list[float],
    torch_vals: list[float],
    torch_failed: list[bool],
    *,
    ylabel: str,
    title: str,
    log_y: bool,
    ratio_vals: list[float] | None = None,
    ratio_label: str = "x",
) -> None:
    """Draw grouped PhysicsNeMo vs PyTorch bars with optional ratio labels."""

    import numpy as np

    torch_plot = [v if not math.isnan(v) else 0.0 for v in torch_vals]
    x = np.arange(len(query_sizes))
    width = 0.36

    ax.bar(
        x - width / 2,
        warp_vals,
        width,
        label="PhysicsNeMo",
        color=NVIDIA_GREEN,
        edgecolor="white",
        linewidth=0.5,
    )
    ax.bar(
        x + width / 2,
        torch_plot,
        width,
        label="PyTorch",
        color=TORCH_GREY,
        edgecolor="white",
        linewidth=0.5,
    )

    if log_y:
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(_comma_axis_formatter())

    ax.set_xticks(x)
    ax.set_xticklabels(
        [
            _query_tick_label(q, failed)
            for q, failed in zip(query_sizes, torch_failed, strict=True)
        ],
        fontweight="bold",
    )
    ax.set_xlabel("query points (Q)", fontweight="bold")
    ax.set_ylabel(ylabel, fontweight="bold")
    ax.set_title(title, fontweight="bold")
    legend = ax.legend(loc="upper left", prop={"weight": "bold", "size": 10})
    for text in legend.get_texts():
        text.set_fontweight("bold")
    ax.grid(True, which="both", axis="y", alpha=0.3)
    _style_axes(ax)

    finite_vals = [v for v in (*warp_vals, *torch_vals) if not math.isnan(v)]
    ymax = max(finite_vals) if finite_vals else 1.0
    ratios = ratio_vals or [math.nan] * len(query_sizes)

    for i, (w, t, ratio, failed) in enumerate(
        zip(warp_vals, torch_vals, ratios, torch_failed, strict=True)
    ):
        if failed:
            ax.text(
                i + width / 2,
                ymax * 0.12,
                FAILED_MARKER,
                ha="center",
                va="bottom",
                fontsize=FAILED_MARKER_FONTSIZE,
                color=TORCH_GREY,
                fontweight="bold",
            )
            continue
        if math.isnan(ratio):
            continue
        label = f"{ratio:.0f}{ratio_label}" if ratio >= 10 else f"{ratio:.1f}{ratio_label}"
        top = max(v for v in (w, t) if not math.isnan(v))
        ax.text(
            i,
            top * 1.15,
            label,
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    bottom = min(v for v in warp_vals if not math.isnan(v)) if warp_vals else 0.5
    ax.set_ylim(bottom=max(bottom * 0.5, 0.5 if log_y else 0), top=ymax * 2.5)


def _save_comparison_figure(fig, path: str | Path, *, footnote: str | None) -> None:
    """Save a comparison figure, optionally reserving space for a footnote."""

    if footnote:
        fig.text(
            0.02,
            0.012,
            FAILED_MARKER,
            ha="left",
            va="bottom",
            fontsize=FAILED_FOOTNOTE_MARKER_FONTSIZE,
            fontweight="bold",
        )
        fig.text(
            0.045,
            0.015,
            footnote,
            ha="left",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )
        fig.tight_layout(rect=(0, 0.08, 1, 1))
    else:
        fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"\nSaved plot to {path}")


def make_plot(rows: list[dict], path: str | Path, target_pts: int | None = None) -> None:
    """Save a grouped bar chart of forward latency vs query points at fixed target size."""
    plt = _configure_matplotlib()

    target_pts, series = _select_series(rows, target_pts)
    query_sizes = [r["query_pts"] for r in series]
    torch_ms = [r["torch_ms"] for r in series]
    torch_failed = [math.isnan(t) for t in torch_ms]

    fig, ax = plt.subplots(figsize=(12, 6.5))
    _draw_comparison_bars(
        ax,
        query_sizes,
        [r["warp_ms"] for r in series],
        torch_ms,
        torch_failed,
        ylabel="forward latency (ms)",
        title=(
            "Forward-pass latency: Warp vs PyTorch "
            f"(target points = {_fmt_int(target_pts)})"
        ),
        log_y=True,
        ratio_vals=[r["speedup"] for r in series],
    )
    footnote = TORCH_CDIST_FOOTNOTE if any(torch_failed) else None
    _save_comparison_figure(fig, path, footnote=footnote)


def make_mem_plot(
    rows: list[dict], path: str | Path, target_pts: int | None = None
) -> None:
    """Save a grouped bar chart of native peak memory vs query points."""

    plt = _configure_matplotlib()

    target_pts, series = _select_series(rows, target_pts)
    query_sizes = [r["query_pts"] for r in series]
    torch_mem = [r["torch_mem_mb"] for r in series]
    torch_failed = [math.isnan(t) for t in torch_mem]

    fig, ax = plt.subplots(figsize=(12, 6.5))
    _draw_comparison_bars(
        ax,
        query_sizes,
        [r["warp_mem_mb"] for r in series],
        torch_mem,
        torch_failed,
        ylabel="peak memory (MB)",
        title=(
            "Peak memory: Warp vs PyTorch "
            f"(target points = {_fmt_int(target_pts)})"
        ),
        log_y=True,
        ratio_vals=[r["mem_ratio"] for r in series],
    )
    footnote = TORCH_CDIST_FOOTNOTE if any(torch_failed) else None
    _save_comparison_figure(fig, path, footnote=footnote)


def main() -> None:
    """Run the ball-query backend benchmark sweep and report results."""
    args = parse_args()

    if args.plot_only:
        jsonl_path = Path(args.jsonl)
        rows = [json.loads(line) for line in jsonl_path.read_text().splitlines() if line.strip()]
        plot_path = jsonl_path.with_suffix(".png")
        make_plot(rows, plot_path, target_pts=args.plot_target_pts)
        mem_path = plot_path.with_name(f"{plot_path.stem}_mem.png")
        make_mem_plot(rows, mem_path, target_pts=args.plot_target_pts)
        return

    import torch

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA device requested but torch.cuda.is_available() is False."
        )

    print(f"Loading up to {args.num_meshes} surface meshes from {args.data_dir} ...")
    clouds = load_surface_point_clouds(args.data_dir, args.num_meshes, device)
    print(f"Loaded {len(clouds)} mesh(es).\n")

    nvml_handle = init_nvml(device)
    # CPU generator: subsample() generates indices on CPU and moves them to the
    # points' device, so the generator stays device-independent.
    generator = torch.Generator()

    rows: list[dict] = []
    for q_size in args.query_sizes:
        for p_size in args.target_sizes:
            warp_times: list[float] = []
            torch_times: list[float] = []
            warp_mems: list[float] = []
            torch_mems: list[float] = []
            warp_nvml: list[float] = []
            torch_nvml: list[float] = []
            for mesh_idx, cloud in enumerate(clouds):
                # Re-seed per (size, mesh) so subsamples are reproducible and
                # both backends see identical inputs.
                generator.manual_seed(args.seed + mesh_idx)
                points = subsample(cloud, p_size, generator)
                queries = subsample(cloud, q_size, generator)

                warp_times.append(
                    time_backend(
                        points,
                        queries,
                        args.radius,
                        args.max_points,
                        "warp",
                        args.warmup,
                        args.iters,
                    )
                )
                torch_times.append(
                    time_backend(
                        points,
                        queries,
                        args.radius,
                        args.max_points,
                        "torch",
                        args.warmup,
                        args.iters,
                    )
                )
                # Memory is measured in a separate pass so the queries and cache
                # flushes do not pollute the latency timings above. Native
                # accounting is exact per backend; NVML gives device-total.
                warp_mems.append(
                    measure_native_mem(
                        points, queries, args.radius, args.max_points, "warp", args.iters
                    )
                )
                torch_mems.append(
                    measure_native_mem(
                        points,
                        queries,
                        args.radius,
                        args.max_points,
                        "torch",
                        args.iters,
                    )
                )
                warp_nvml.append(
                    measure_nvml_mem(
                        points,
                        queries,
                        args.radius,
                        args.max_points,
                        "warp",
                        nvml_handle,
                        args.iters,
                    )
                )
                torch_nvml.append(
                    measure_nvml_mem(
                        points,
                        queries,
                        args.radius,
                        args.max_points,
                        "torch",
                        nvml_handle,
                        args.iters,
                    )
                )

            warp_ms = _nanmean(warp_times)
            torch_ms = _nanmean(torch_times)
            warp_mem = _nanmean(warp_mems)
            torch_mem = _nanmean(torch_mems)
            warp_nvml_mb = _nanmean(warp_nvml)
            torch_nvml_mb = _nanmean(torch_nvml)
            speedup = _safe_ratio(torch_ms, warp_ms)
            mem_ratio = _safe_ratio(torch_mem, warp_mem)
            row = {
                "query_pts": q_size,
                "target_pts": p_size,
                "radius": args.radius,
                "max_points": args.max_points,
                "num_meshes": len(clouds),
                "iters": args.iters,
                "device": str(device),
                "warp_ms": warp_ms,
                "torch_ms": torch_ms,
                "speedup": speedup,
                "warp_mem_mb": warp_mem,
                "torch_mem_mb": torch_mem,
                "mem_ratio": mem_ratio,
                "warp_nvml_mb": warp_nvml_mb,
                "torch_nvml_mb": torch_nvml_mb,
            }
            rows.append(row)
            print(
                f"Q={q_size:>7} P={p_size:>7}  "
                f"warp={_fmt_ms(warp_ms)}ms torch={_fmt_ms(torch_ms)}ms "
                f"speedup={_fmt_ms(speedup)}x  "
                f"warp_mem={_fmt_mem(warp_mem)}MB torch_mem={_fmt_mem(torch_mem)}MB "
                f"(nvml warp={_fmt_mem(warp_nvml_mb)}MB torch={_fmt_mem(torch_nvml_mb)}MB)"
            )

    print("\n" + format_table(rows))

    write_jsonl(rows, args.jsonl)
    print(f"\nWrote JSONL to {args.jsonl}")

    if args.plot:
        plot_path = Path(args.jsonl).with_suffix(".png")
        make_plot(rows, plot_path, target_pts=args.plot_target_pts)
        mem_path = plot_path.with_name(f"{plot_path.stem}_mem.png")
        make_mem_plot(rows, mem_path, target_pts=args.plot_target_pts)


def _nanmean(values: list[float]) -> float:
    """Mean of a list ignoring NaNs; returns NaN if all values are NaN."""
    finite = [v for v in values if not math.isnan(v)]
    if not finite:
        return math.nan
    return sum(finite) / len(finite)


def _safe_ratio(numerator: float, denominator: float) -> float:
    """Return ``numerator / denominator``, or NaN if either is NaN/zero."""
    if math.isnan(numerator) or math.isnan(denominator) or denominator <= 0:
        return math.nan
    return numerator / denominator


def _fmt_ms(value: float) -> str:
    """Format a latency/speedup value, showing 'OOM' for NaN."""
    return "OOM" if math.isnan(value) else f"{value:.3f}"


def _fmt_mem(value: float) -> str:
    """Format a memory value, showing 'n/a' for NaN (unmeasured)."""
    return "n/a" if math.isnan(value) else f"{value:.3f}"


if __name__ == "__main__":
    main()
