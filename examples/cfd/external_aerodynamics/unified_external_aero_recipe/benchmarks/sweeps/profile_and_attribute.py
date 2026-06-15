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

"""Profile a training step, attribute time to each PhysicsNeMo FunctionSpec,
and combine with per-op ASV micro-benchmarks into a single end-to-end
``X×`` speedup number for PhysicsNeMo vs a PyTorch-only baseline.

Two modes
---------
``--mode estimate``
    Profile one step with default dispatch (fast impls). Multiply each
    measured fast-impl time by its ASV speedup factor to estimate the
    equivalent PyTorch-only step time, then report
    ``T_torch_equiv / T_fast`` as the end-to-end speedup. Single profile
    pass; cheap; depends on ASV results being present.

``--mode measure``
    Time the same step three ways: (1) torch baseline, no profiler;
    (2) default dispatch (Warp), no profiler — headline speedup uses
    these two wall times; (3) default dispatch with profiler for per-op
    breakdown only. No ASV needed for the headline number, but ASV is
    still parsed (when available) to enrich the per-op table.

Two step sources
----------------
``--step synthetic``
    Build a zero-arg step that walks a curated list of FunctionSpec
    classes and dispatches each one using its own ``make_inputs_forward``
    benchmark case. Self-contained; useful for smoke-testing the
    instrumentation without needing a real dataset or training loop.

``--step pkg.mod:fn``
    Import ``pkg.mod`` and call ``fn()`` (zero-arg) to obtain another
    zero-arg callable that runs one training step. Use this to plug
    the script into the real recipe (e.g. wrap ``forward_pass`` + loss
    backward in a closure and expose it).

Outputs
-------
``<output>.json``  Structured report with breakdown.
``<output>.md``    Human-readable markdown table for sharing.
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

_BENCHMARKS_ROOT = Path(__file__).resolve().parents[1]
if str(_BENCHMARKS_ROOT) not in sys.path:
    sys.path.insert(0, str(_BENCHMARKS_ROOT))
import importlib
import json
import math
import statistics
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

import torch

from physicsnemo.core.function_spec import FunctionSpec

# ASV labels every event we care about with this prefix (see instrument_dispatch).
PROFILER_TAG_PREFIX = "physicsnemo::FunctionSpec/"

# Default head-to-head specs (each has an optimized impl AND a `torch` baseline).
# Imported lazily inside _resolve_default_specs() to keep top-level imports thin.
_DEFAULT_SPEC_DOTTED_NAMES: tuple[str, ...] = (
    "physicsnemo.nn.functional.neighbors.KNN",
    "physicsnemo.nn.functional.neighbors.RadiusSearch",
    "physicsnemo.nn.functional.interpolation.GridToPointInterpolation",
    "physicsnemo.nn.functional.interpolation.PointToGridInterpolation",
    "physicsnemo.nn.functional.derivatives.UniformGridGradient",
    "physicsnemo.nn.functional.derivatives.RectilinearGridGradient",
    "physicsnemo.nn.functional.derivatives.MeshLSQGradient",
    "physicsnemo.nn.functional.derivatives.MeshGreenGaussGradient",
)


@dataclass
class CallRecord:
    """Profiler-attributed accumulator for one (spec, impl) pair."""

    spec: str
    impl: str
    count: int = 0
    cuda_time_s: float = 0.0
    cpu_time_s: float = 0.0

    @property
    def total_time_s(self) -> float:
        # CPU + CUDA captures both sync overhead and kernel time.
        return self.cuda_time_s + self.cpu_time_s


# ---------------------------------------------------------------------------
# FunctionSpec.dispatch instrumentation
# ---------------------------------------------------------------------------


def _dispatch_descriptor():
    """Return the raw classmethod descriptor for FunctionSpec.dispatch."""

    return FunctionSpec.__dict__["dispatch"]


def _resolve_impl_name(cls: type[FunctionSpec], kwargs: dict[str, Any]) -> str:
    """Mirror dispatch's impl-selection logic so we can label calls accurately."""

    explicit = kwargs.get("implementation")
    impls = cls._get_impls()
    if explicit is not None and explicit in impls:
        return explicit
    available = [impl for impl in impls.values() if impl.available]
    if not available:
        return "unknown"
    return sorted(available, key=lambda impl: impl.rank)[0].name


@contextmanager
def instrument_dispatch() -> Iterator[None]:
    """Wrap FunctionSpec.dispatch with torch.profiler.record_function tags.

    The tag format is ``physicsnemo::FunctionSpec/<ClassName>/<impl_name>``
    so the profiler trace can be regrouped by (spec, impl) after the run.
    Restoration is unconditional so an exception in the step doesn't leave
    the class permanently monkey-patched.
    """

    original_desc = _dispatch_descriptor()
    original_func = original_desc.__func__

    def _patched(cls, *args, **kwargs):
        impl_name = _resolve_impl_name(cls, kwargs)
        tag = f"{PROFILER_TAG_PREFIX}{cls.__name__}/{impl_name}"
        with torch.profiler.record_function(tag):
            return original_func(cls, *args, **kwargs)

    FunctionSpec.dispatch = classmethod(_patched)
    try:
        yield
    finally:
        FunctionSpec.dispatch = original_desc


@contextmanager
def force_baseline_impls() -> Iterator[None]:
    """Patch dispatch so every functional runs its baseline (torch) impl.

    Used by ``--mode measure`` to A/B the same step. Falls back to original
    dispatch for any spec without an available baseline (the "non-functional"
    path of the model stays unchanged).
    """

    original_desc = _dispatch_descriptor()
    original_func = original_desc.__func__

    def _patched(cls, *args, **kwargs):
        impls = cls._get_impls()
        baseline = next((impl for impl in impls.values() if impl.baseline), None)
        target = baseline if (baseline is not None and baseline.available) else None
        if target is None:
            torch_impl = impls.get("torch")
            if torch_impl is not None and torch_impl.available:
                target = torch_impl
        if target is None:
            return original_func(cls, *args, **kwargs)
        kwargs.pop("implementation", None)
        # Re-enter the profiler tag so attribution stays consistent.
        tag = f"{PROFILER_TAG_PREFIX}{cls.__name__}/{target.name}"
        with torch.profiler.record_function(tag):
            return target.func(*args, **kwargs)

    FunctionSpec.dispatch = classmethod(_patched)
    try:
        yield
    finally:
        FunctionSpec.dispatch = original_desc


# ---------------------------------------------------------------------------
# Step profiling
# ---------------------------------------------------------------------------


def _profile_activities(device: torch.device) -> list[torch.profiler.ProfilerActivity]:
    """Return the profiler activities appropriate for this device."""

    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    return activities


def _records_from_profiler(prof: torch.profiler.profile) -> dict[tuple[str, str], CallRecord]:
    """Collect (spec, impl) -> CallRecord aggregates from a profiler trace."""

    records: dict[tuple[str, str], CallRecord] = {}
    for event in prof.key_averages():
        name = event.key
        if not name.startswith(PROFILER_TAG_PREFIX):
            continue
        # Tag layout: "<prefix><SpecName>/<impl>". Split only on first '/'.
        spec_impl = name[len(PROFILER_TAG_PREFIX) :]
        if "/" not in spec_impl:
            continue
        spec, impl = spec_impl.split("/", 1)
        rec = records.setdefault((spec, impl), CallRecord(spec=spec, impl=impl))
        rec.count += int(event.count)
        # torch.profiler returns microseconds; convert to seconds once here.
        # PyTorch 2.9+ renamed cuda_time_total -> device_time_total on FunctionEventAvg.
        cuda_us = getattr(event, "cuda_time_total", None)
        if cuda_us is None:
            cuda_us = getattr(event, "device_time_total", 0.0)
        rec.cuda_time_s += float(cuda_us or 0.0) * 1e-6
        rec.cpu_time_s += float(event.cpu_time_total or 0.0) * 1e-6
    return records


def run_profiled_step(
    step_fn: Callable[[], None],
    *,
    device: torch.device,
    n_warmup: int,
    n_record: int,
) -> tuple[float, dict[tuple[str, str], CallRecord]]:
    """Run the step ``n_warmup + n_record`` times; return per-step wall time + per-op records."""

    # Warm up CUDA caches / autotune outside the profile window.
    for _ in range(n_warmup):
        step_fn()
    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    with instrument_dispatch(), torch.profiler.profile(
        activities=_profile_activities(device), record_shapes=False
    ) as prof:
        for _ in range(n_record):
            step_fn()
        if device.type == "cuda":
            torch.cuda.synchronize()
    wall = (time.perf_counter() - start) / max(1, n_record)

    records = _records_from_profiler(prof)
    # Aggregate counts/times were summed across n_record reps; normalize.
    for rec in records.values():
        rec.cuda_time_s /= n_record
        rec.cpu_time_s /= n_record
        rec.count = max(1, rec.count // n_record)
    return wall, records


def run_step_no_profiler(
    step_fn: Callable[[], None],
    *,
    device: torch.device,
    n_warmup: int,
    n_record: int,
) -> float:
    """Run a step ``n_warmup + n_record`` times; return median per-step wall time.

    No profiler overhead — used in ``--mode measure`` so the baseline/optimized
    timings are not skewed by profiling instrumentation.
    """

    for _ in range(n_warmup):
        step_fn()
    if device.type == "cuda":
        torch.cuda.synchronize()

    samples: list[float] = []
    for _ in range(n_record):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        step_fn()
        if device.type == "cuda":
            torch.cuda.synchronize()
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples)


# ---------------------------------------------------------------------------
# Step sources
# ---------------------------------------------------------------------------


def _resolve_default_specs() -> tuple[type[FunctionSpec], ...]:
    """Import the curated head-to-head FunctionSpec classes by dotted name."""

    resolved: list[type[FunctionSpec]] = []
    for dotted in _DEFAULT_SPEC_DOTTED_NAMES:
        module, name = dotted.rsplit(".", 1)
        spec = getattr(importlib.import_module(module), name)
        resolved.append(spec)
    return tuple(resolved)


def _spec_first_case(
    spec: type[FunctionSpec], *, device: torch.device, case_index: int
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Materialize one forward benchmark case for ``spec`` at ``case_index``."""

    cases = list(spec.make_inputs_forward(device=device))
    if not cases:
        raise RuntimeError(f"{spec.__name__} has no forward benchmark cases")
    idx = min(case_index, len(cases) - 1)
    _, args, kwargs = cases[idx]
    return args, kwargs


def build_synthetic_step(
    device: torch.device,
    *,
    specs: Iterable[type[FunctionSpec]] | None = None,
    case_index: int = 1,
) -> Callable[[], None]:
    """Build a zero-arg step that dispatches each spec once per call.

    Reuses each FunctionSpec's own ``make_inputs_forward`` generator so the
    inputs are guaranteed to be valid and representative. ``case_index``
    picks which preset to use (0=smallest case, larger indices = larger
    workloads); defaults to 1 (medium) to give the profiler signal without
    being slow on small GPUs.
    """

    specs = tuple(specs) if specs is not None else _resolve_default_specs()
    prepared: list[tuple[type[FunctionSpec], tuple[Any, ...], dict[str, Any]]] = []
    for spec in specs:
        try:
            args, kwargs = _spec_first_case(spec, device=device, case_index=case_index)
        except Exception as exc:  # noqa: BLE001 -- skip specs whose inputs blow up
            print(f"[synthetic] skipping {spec.__name__}: {exc}")
            continue
        prepared.append((spec, args, kwargs))

    if not prepared:
        raise RuntimeError("No FunctionSpec inputs prepared for synthetic step")

    def step() -> None:
        for spec, args, kwargs in prepared:
            spec.dispatch(*args, **kwargs)

    return step


def load_callable_step(spec: str) -> Callable[[], None]:
    """Resolve ``pkg.mod:factory_fn`` and call ``factory_fn()`` for the step."""

    if ":" not in spec:
        raise ValueError("--step must be 'synthetic' or 'pkg.mod:factory_fn'")
    mod_path, fn_name = spec.split(":", 1)
    module = importlib.import_module(mod_path)
    factory = getattr(module, fn_name)
    step_fn = factory()
    if not callable(step_fn):
        raise TypeError(f"{spec} did not return a callable")
    return step_fn


# ---------------------------------------------------------------------------
# ASV result parsing
# ---------------------------------------------------------------------------


def _walk_dicts(value: Any) -> Iterator[dict[str, Any]]:
    """Yield every dict node inside arbitrarily-nested JSON-like containers."""

    if isinstance(value, dict):
        yield value
        for nested in value.values():
            yield from _walk_dicts(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _walk_dicts(nested)


def _latest_asv_result_file(results_dir: Path) -> Path | None:
    """Return the newest ASV result JSON (excluding metadata files)."""

    if not results_dir.exists():
        return None
    candidates = [
        p
        for p in results_dir.rglob("*.json")
        if p.name not in {"benchmarks.json", "machine.json"}
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _extract_benchmark_entry(payload: dict) -> Any | None:
    """Locate the ``FunctionalBenchmarks.time_functional`` entry in an ASV JSON."""

    suffix = "FunctionalBenchmarks.time_functional"
    for mapping in _walk_dicts(payload):
        for key, value in mapping.items():
            if isinstance(key, str) and suffix in key:
                return value
    return None


def _entry_values_and_labels(entry: Any) -> tuple[list[Any], list[str]]:
    """Normalize ASV's two storage shapes into (values, labels)."""

    if isinstance(entry, dict):
        values = entry.get("result") or entry.get("results") or []
        params = entry.get("params") or [[]]
        labels = params[0] if isinstance(params, list) and params else []
    else:
        values = entry[0] if entry else []
        labels = entry[1] if len(entry) > 1 else []
        if labels and isinstance(labels[0], list):
            labels = labels[0]
    return list(values), [str(lbl) for lbl in labels]


def parse_asv_speedups(asv_results_dir: Path) -> dict[str, tuple[str, float]]:
    """Return ``{spec_name -> (fastest_non_torch_impl, speedup_vs_torch)}``.

    Speedup is computed as ``geom_mean(torch_times) / geom_mean(fast_times)``
    across all forward-phase cases for that spec. Geometric mean is robust
    to extreme outliers across small/medium/large benchmark cases.
    Returns an empty dict when no ASV results exist yet.
    """

    result_file = _latest_asv_result_file(asv_results_dir)
    if result_file is None:
        return {}

    payload = json.loads(result_file.read_text())
    entry = _extract_benchmark_entry(payload)
    if entry is None:
        return {}

    values, labels = _entry_values_and_labels(entry)
    per_spec_impl_times: dict[tuple[str, str], list[float]] = defaultdict(list)
    for label, val in zip(labels, values):
        # ASV stores None for skipped/failed cases.
        if val is None:
            continue
        try:
            parsed = ast.literal_eval(label)
        except (ValueError, SyntaxError):
            continue
        if not isinstance(parsed, tuple) or len(parsed) != 4:
            continue
        phase, spec_name, impl_name, _case_idx = parsed
        if phase != "forward":
            continue
        try:
            per_spec_impl_times[(str(spec_name), str(impl_name))].append(float(val))
        except (TypeError, ValueError):
            continue

    # Aggregate per (spec, impl), then compute speedups vs torch.
    spec_to_impl_times: dict[str, dict[str, float]] = defaultdict(dict)
    for (spec_name, impl_name), times in per_spec_impl_times.items():
        if not times:
            continue
        # Geom mean is undefined for zero/negative; clamp away nonsense.
        clean = [t for t in times if t > 0.0]
        if not clean:
            continue
        spec_to_impl_times[spec_name][impl_name] = statistics.geometric_mean(clean)

    speedups: dict[str, tuple[str, float]] = {}
    for spec_name, impl_times in spec_to_impl_times.items():
        if "torch" not in impl_times:
            continue
        torch_time = impl_times["torch"]
        candidates = {
            name: t for name, t in impl_times.items() if name != "torch" and t > 0
        }
        if not candidates:
            continue
        fastest_impl = min(candidates, key=candidates.get)
        speedups[spec_name] = (fastest_impl, torch_time / candidates[fastest_impl])
    return speedups


# ---------------------------------------------------------------------------
# Amdahl combiner
# ---------------------------------------------------------------------------


@dataclass
class StepReport:
    """Per-step timing + breakdown that drives the markdown/JSON report."""

    fast_total_s: float
    baseline_total_s: float
    nonfunc_time_s: float
    breakdown: list[dict[str, Any]]

    @property
    def overall_speedup(self) -> float:
        return (
            self.baseline_total_s / self.fast_total_s
            if self.fast_total_s > 0
            else float("nan")
        )


def estimate_speedup(
    *,
    fast_wall_s: float,
    records: dict[tuple[str, str], CallRecord],
    spec_speedups: dict[str, tuple[str, float]],
) -> StepReport:
    """Amdahl-combine measured per-op times with ASV speedups.

    For each spec ``i`` whose fast impl ran in the profile:
      ``baseline_i = t_fast_i * s_i``  (s_i from ASV; 1.0 if missing)
    The remaining "non-functional" time is left unchanged.
    """

    breakdown: list[dict[str, Any]] = []
    func_total_fast = 0.0
    func_total_baseline = 0.0
    for (spec_name, impl_name), rec in records.items():
        t_fast = rec.total_time_s
        if impl_name == "torch":
            speedup = 1.0
            fastest_impl = "torch"
        else:
            entry = spec_speedups.get(spec_name)
            fastest_impl, speedup = entry if entry else (impl_name, 1.0)
        t_baseline = t_fast * speedup
        func_total_fast += t_fast
        func_total_baseline += t_baseline
        breakdown.append(
            {
                "spec": spec_name,
                "impl_used": impl_name,
                "asv_fastest_impl": fastest_impl,
                "calls": rec.count,
                "fast_time_s": t_fast,
                "asv_speedup": speedup,
                "baseline_time_s": t_baseline,
            }
        )

    # Non-functional time = total step wall - time inside profiled functionals.
    # Clamp to >= 0 so jitter / overlap doesn't push us into negative territory.
    nonfunc_time = max(0.0, fast_wall_s - func_total_fast)
    return StepReport(
        fast_total_s=fast_wall_s,
        baseline_total_s=nonfunc_time + func_total_baseline,
        nonfunc_time_s=nonfunc_time,
        breakdown=sorted(breakdown, key=lambda e: -e["fast_time_s"]),
    )


def measure_speedup(
    *,
    fast_wall_s: float,
    baseline_wall_s: float,
    records: dict[tuple[str, str], CallRecord],
    spec_speedups: dict[str, tuple[str, float]],
) -> StepReport:
    """Use the directly-measured baseline wall time; reuse the fast breakdown."""

    breakdown: list[dict[str, Any]] = []
    func_total_fast = 0.0
    for (spec_name, impl_name), rec in records.items():
        t_fast = rec.total_time_s
        entry = spec_speedups.get(spec_name)
        fastest_impl, asv_speedup = entry if entry else (impl_name, 1.0)
        func_total_fast += t_fast
        breakdown.append(
            {
                "spec": spec_name,
                "impl_used": impl_name,
                "asv_fastest_impl": fastest_impl,
                "calls": rec.count,
                "fast_time_s": t_fast,
                "asv_speedup": asv_speedup,
                "baseline_time_s": t_fast * asv_speedup,
            }
        )
    return StepReport(
        fast_total_s=fast_wall_s,
        baseline_total_s=baseline_wall_s,
        nonfunc_time_s=max(0.0, fast_wall_s - func_total_fast),
        breakdown=sorted(breakdown, key=lambda e: -e["fast_time_s"]),
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def write_reports(
    *,
    mode: str,
    report: StepReport,
    spec_speedups: dict[str, tuple[str, float]],
    output_stem: Path,
) -> tuple[Path, Path]:
    """Write a JSON blob and a markdown summary; return their paths."""

    json_path = output_stem.with_suffix(".json")
    md_path = output_stem.with_suffix(".md")

    json_payload = {
        "mode": mode,
        "overall_speedup": report.overall_speedup,
        "fast_total_s": report.fast_total_s,
        "baseline_total_s": report.baseline_total_s,
        "nonfunc_time_s": report.nonfunc_time_s,
        "breakdown": report.breakdown,
        "asv_spec_speedups": {
            spec: {"impl": impl, "speedup": s}
            for spec, (impl, s) in spec_speedups.items()
        },
    }
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(json_payload, indent=2))

    lines: list[str] = []
    lines.append(f"# PhysicsNeMo end-to-end speedup vs PyTorch (mode=`{mode}`)")
    lines.append("")
    lines.append(f"**Headline:** {report.overall_speedup:.2f}× faster end-to-end.")
    lines.append("")
    lines.append(
        f"- Profiled step time (PNM, default dispatch): "
        f"{report.fast_total_s * 1e3:.2f} ms"
    )
    lines.append(
        f"- Equivalent PyTorch-only step time: "
        f"{report.baseline_total_s * 1e3:.2f} ms"
    )
    lines.append(
        f"- Non-functional (model + dataloader + comm) time: "
        f"{report.nonfunc_time_s * 1e3:.2f} ms "
        f"({100.0 * report.nonfunc_time_s / max(report.fast_total_s, 1e-9):.1f}% of step)"
    )
    lines.append("")
    lines.append("## Per-functional breakdown")
    lines.append("")
    lines.append(
        "| FunctionSpec | impl run | calls | fast (ms) | ASV speedup | baseline (ms) |"
    )
    lines.append("|---|---|---:|---:|---:|---:|")
    for entry in report.breakdown:
        lines.append(
            f"| {entry['spec']} | {entry['impl_used']} | {entry['calls']} "
            f"| {entry['fast_time_s'] * 1e3:.3f} | {entry['asv_speedup']:.2f}× "
            f"| {entry['baseline_time_s'] * 1e3:.3f} |"
        )
    lines.append("")
    if spec_speedups:
        lines.append("## ASV per-op speedups (forward, geom-mean across cases)")
        lines.append("")
        lines.append("| FunctionSpec | fastest impl | speedup vs torch |")
        lines.append("|---|---|---:|")
        for spec, (impl, s) in sorted(
            spec_speedups.items(), key=lambda kv: -kv[1][1]
        ):
            lines.append(f"| {spec} | {impl} | {s:.2f}× |")
    else:
        lines.append("> _No ASV results were found; per-op speedups defaulted to 1.0._")
    lines.append("")
    md_path.write_text("\n".join(lines))
    return json_path, md_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Profile one training step, attribute time to each "
        "PhysicsNeMo FunctionSpec, combine with ASV micro-benchmarks, "
        "and report an end-to-end speedup vs PyTorch baseline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=("estimate", "measure"),
        default="estimate",
        help="estimate: 1 profile run + ASV. measure: 2 runs (baseline forced + default).",
    )
    parser.add_argument(
        "--step",
        default="synthetic",
        help="'synthetic' (default) or 'pkg.mod:factory_fn' returning a zero-arg step.",
    )
    parser.add_argument(
        "--asv-results",
        type=Path,
        default=Path(".asv/results"),
        help="ASV results directory (latest *.json is used).",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for the step (cuda/cpu).",
    )
    parser.add_argument(
        "--n-warmup",
        type=int,
        default=3,
        help="Warm-up iterations before profiling/timing.",
    )
    parser.add_argument(
        "--n-record",
        type=int,
        default=5,
        help="Recorded iterations (averaged for wall time, summed for per-op time).",
    )
    parser.add_argument(
        "--synthetic-case-index",
        type=int,
        default=1,
        help="Which make_inputs_forward case to use for synthetic step (0=smallest).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("profile_attribute_report"),
        help="Output stem; '.json' and '.md' will be appended.",
    )
    return parser


def main() -> int:
    args = _build_arg_parser().parse_args()
    device = torch.device(args.device)

    # 1. Load whatever ASV results exist so we can enrich the breakdown.
    spec_speedups = parse_asv_speedups(args.asv_results)
    asv_file = _latest_asv_result_file(args.asv_results)
    if asv_file is None:
        print(f"[asv] No results found under {args.asv_results}; per-op speedups will be 1.0×.")
    else:
        print(
            f"[asv] Using {asv_file}; found speedups for "
            f"{len(spec_speedups)} FunctionSpec(s)."
        )

    # 2. Resolve the step source.
    if args.step == "synthetic":
        print("[step] Building synthetic step from FunctionSpec.make_inputs_forward")
        step_fn = build_synthetic_step(
            device=device, case_index=args.synthetic_case_index
        )
    else:
        print(f"[step] Loading callable step from {args.step}")
        step_fn = load_callable_step(args.step)

    # 3. Run the chosen mode.
    if args.mode == "estimate":
        fast_wall, records = run_profiled_step(
            step_fn, device=device, n_warmup=args.n_warmup, n_record=args.n_record
        )
        report = estimate_speedup(
            fast_wall_s=fast_wall, records=records, spec_speedups=spec_speedups
        )
    else:  # measure: torch baseline + default dispatch (both without profiler).
        print("[measure] Pass 1/3: forcing torch baseline impls (no profiler).")
        with force_baseline_impls():
            baseline_wall = run_step_no_profiler(
                step_fn,
                device=device,
                n_warmup=args.n_warmup,
                n_record=args.n_record,
            )
        print(
            f"[measure] baseline wall (median over {args.n_record}): "
            f"{baseline_wall * 1e3:.2f} ms"
        )
        print("[measure] Pass 2/3: default dispatch (no profiler).")
        fast_wall = run_step_no_profiler(
            step_fn,
            device=device,
            n_warmup=args.n_warmup,
            n_record=args.n_record,
        )
        print(
            f"[measure] fast wall (median over {args.n_record}): "
            f"{fast_wall * 1e3:.2f} ms"
        )
        print("[measure] Pass 3/3: default dispatch with profiler (breakdown only).")
        _, records = run_profiled_step(
            step_fn,
            device=device,
            n_warmup=args.n_warmup,
            n_record=args.n_record,
        )
        report = measure_speedup(
            fast_wall_s=fast_wall,
            baseline_wall_s=baseline_wall,
            records=records,
            spec_speedups=spec_speedups,
        )

    # 4. Emit reports.
    print()
    print("=== Summary ===")
    print(f"mode             : {args.mode}")
    print(f"fast step time   : {report.fast_total_s * 1e3:.2f} ms")
    print(f"baseline step    : {report.baseline_total_s * 1e3:.2f} ms")
    print(f"overall speedup  : {report.overall_speedup:.2f}×")
    json_path, md_path = write_reports(
        mode=args.mode,
        report=report,
        spec_speedups=spec_speedups,
        output_stem=args.output,
    )
    print(f"report (md)      : {md_path}")
    print(f"report (json)    : {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
