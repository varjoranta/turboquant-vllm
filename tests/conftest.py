# SPDX-License-Identifier: MIT
"""Shared pytest fixtures + collection hooks.

Skips the MLX test modules on runners where the `mlx` wheel isn't available
(Linux CI). MLX only publishes macOS arm64 wheels, so importing any of the
``test_mlx_*`` / ``bench_mlx_*`` modules would fail collection on Linux.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_HAS_MLX = importlib.util.find_spec("mlx") is not None

if not _HAS_MLX:
    _here = Path(__file__).parent
    collect_ignore = sorted(
        str(p.relative_to(_here))
        for p in list(_here.glob("*mlx*.py")) + [_here / "smoke_metal_kernel.py"]
        if p.exists() and p.name != "conftest.py"
    )
