# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Portable default paths for the CAE benchmark package."""

from __future__ import annotations

import os
from pathlib import Path

BENCHMARKS_DIR = Path(__file__).resolve().parent
RECIPE_ROOT = BENCHMARKS_DIR.parent


def canvas_dir() -> Path:
    """Directory for generated ``.canvas.tsx`` files.

    Override with ``PHYSICSNEMO_CAE_CANVAS_DIR`` (e.g. a Cursor project canvases folder).
    Default: ``benchmarks/canvases/`` under this recipe.
    """
    override = os.environ.get("PHYSICSNEMO_CAE_CANVAS_DIR")
    if override:
        return Path(override)
    return BENCHMARKS_DIR / "canvases"


def canvas_path(filename: str) -> Path:
    return canvas_dir() / filename
