#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Factory for a real GeoTransolver train step (forward + backward + optimizer).

Used by ``profile_and_attribute.py`` via::

    --step sweeps.recipe_train_step:make_step

Environment variables (set by ``sweep_pytorch_measure.py`` or the sbatch job):

* ``PROFILE_SUBSAMPLING`` — ``sampling_resolution`` override (default ``200000``)
* ``PROFILE_MODEL``       — Hydra model template (default ``geotransolver_surface``)
* ``PROFILE_DATASET``     — Hydra dataset name (default ``shift_suv_estate_surface``)
* ``PROFILE_COMPILE``     — ``true``/``false`` (default ``true``, matches production)
* ``DATASET_PATH_SHIFT_SUV`` — required for ShiftSUV (default dataset)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Callable

import hydra
import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf
from torch.amp import GradScaler

RECIPE_ROOT = Path(__file__).resolve().parent.parent
_SRC = RECIPE_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from loss import LossCalculator  # noqa: E402
from metrics import DEFAULT_METRICS, MetricCalculator  # noqa: E402
from physicsnemo.distributed import DistributedManager  # noqa: E402
from train import _recursive_to_device, _resolve_dict, build_dataloaders, forward_pass  # noqa: E402
from utils import FieldType, build_muon_optimizer, field_dim, set_seed  # noqa: E402


def _load_dataset_targets(dataset_name: str) -> dict[str, FieldType]:
    dataset_path = RECIPE_ROOT / "datasets" / f"{dataset_name}.yaml"
    targets = OmegaConf.to_container(OmegaConf.load(dataset_path).targets, resolve=True)
    if not isinstance(targets, dict) or not targets:
        raise ValueError(f"{dataset_name} has no targets block")
    return targets


def _compose_train_cfg(
    *,
    model: str,
    dataset: str,
    subsampling: int,
    compile_model: bool,
) -> DictConfig:
    targets = _load_dataset_targets(dataset)
    out_dim = sum(field_dim(ftype) for ftype in targets.values())
    overrides = [
        f"model={model}",
        f"dataset={dataset}",
        f"+out_dim={out_dim}",
        f"sampling_resolution={subsampling}",
        f"compile={'true' if compile_model else 'false'}",
        "training.num_epochs=1",
    ]
    with initialize_config_dir(
        config_dir=str(RECIPE_ROOT / "conf"),
        version_base=None,
    ):
        return compose(config_name="train", overrides=overrides)


def _build_train_step(cfg: DictConfig) -> Callable[[], None]:
    if not DistributedManager.is_initialized():
        DistributedManager.initialize()
    dist_manager = DistributedManager()
    device = dist_manager.device

    seed = cfg.training.get("seed", 42)
    set_seed(seed, rank=dist_manager.rank)

    train_loader, _val_loader, _normalizer, dataset_info = build_dataloaders(cfg)
    target_config: dict[str, FieldType] = dataset_info["targets"]

    model = hydra.utils.instantiate(cfg.model, _convert_="partial")
    model.to(device)
    if cfg.compile:
        model = torch.compile(model)

    optimizer = build_muon_optimizer(model, cfg, compile_optimizer=cfg.compile)
    scheduler = hydra.utils.instantiate(cfg.training.scheduler, optimizer=optimizer)

    precision = cfg.precision
    scaler = GradScaler() if precision == "float16" else None
    output_type = cfg.get("output_type")
    if output_type is None:
        raise ValueError("cfg.output_type is required")

    metrics_cfg = OmegaConf.select(cfg, "metrics", default=None)
    metrics_list = (
        list(DEFAULT_METRICS)
        if metrics_cfg is None
        else OmegaConf.to_container(metrics_cfg, resolve=True)
    )
    field_weights = _resolve_dict(cfg, "training.field_weights")
    metric_calculator = MetricCalculator(
        target_config=target_config,
        metrics=metrics_list,
    )
    loss_calculator = LossCalculator(
        target_config=target_config,
        loss_type=cfg.training.get("loss_type", "huber"),
        field_weights=field_weights,
    )

    loader_iter: dict[str, Any] = {"it": iter(train_loader)}

    def _next_batch() -> dict[str, Any]:
        try:
            return next(loader_iter["it"])
        except StopIteration:
            train_loader.set_epoch(0)
            loader_iter["it"] = iter(train_loader)
            return next(loader_iter["it"])

    model.train()

    def step() -> None:
        batch = _recursive_to_device(_next_batch(), device)
        loss, _losses, _metrics = forward_pass(
            batch,
            model,
            precision,
            loss_calculator,
            metric_calculator,
            output_type=output_type,
            target_config=target_config,
        )
        optimizer.zero_grad()
        if precision == "float16" and scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        if cfg.training.get("scheduler_update_mode", "epoch") == "step":
            scheduler.step()

    return step


def make_step() -> Callable[[], None]:
    """Return a zero-arg callable that runs one full training step."""

    subsampling = int(os.environ.get("PROFILE_SUBSAMPLING", "200000"))
    model = os.environ.get("PROFILE_MODEL", "geotransolver_surface")
    dataset = os.environ.get("PROFILE_DATASET", "shift_suv_estate_surface")
    compile_model = os.environ.get("PROFILE_COMPILE", "true").lower() in (
        "1",
        "true",
        "yes",
    )
    cfg = _compose_train_cfg(
        model=model,
        dataset=dataset,
        subsampling=subsampling,
        compile_model=compile_model,
    )
    return _build_train_step(cfg)
