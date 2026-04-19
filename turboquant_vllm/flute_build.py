"""Build the vendored FLUTE CUDA extension.

Compiles csrc/flute/*.cpp and *.cu into a shared library loadable as
``turboquant_vllm.flute._C``. FLUTE is Guo et al., EMNLP Findings 2024
(arXiv:2407.10960) — a LUT-based GEMM engine for low-bit quantized LLMs;
upstream at github.com/HanGuo97/flute. Vendored under Apache-2.0 with
the namespace renamed from ``flute::`` → ``turboquant_flute::`` to avoid
conflicts with any externally installed flute-kernel.

CUTLASS is expected at one of:
  1. The directory pointed to by ``$TURBOQUANT_CUTLASS_PATH``
  2. ``third_party/cutlass/`` in the turboquant-vllm repo root
  3. ``/workspace/cutlass/`` (FLUTE upstream default; legacy fallback)

Usage:
    python -m turboquant_vllm.flute_build
"""

from __future__ import annotations

import os
from pathlib import Path

_pkg_dir = Path(__file__).resolve().parent
_repo_root = _pkg_dir.parent
FLUTE_CSRC_DIR = _repo_root / "csrc" / "flute"


def _resolve_cutlass_path() -> Path:
    env = os.environ.get("TURBOQUANT_CUTLASS_PATH")
    candidates = []
    if env:
        candidates.append(Path(env))
    candidates.extend([
        _repo_root / "third_party" / "cutlass",
        Path("/workspace/cutlass"),
    ])
    for p in candidates:
        if (p / "include" / "cute" / "tensor.hpp").exists():
            return p
    raise FileNotFoundError(
        "CUTLASS headers not found (need cute/tensor.hpp). "
        f"Searched: {', '.join(str(p) for p in candidates)}. "
        "Set TURBOQUANT_CUTLASS_PATH or clone CUTLASS v3.9.2 "
        "into third_party/cutlass:\n"
        "    git clone --depth 1 --branch v3.9.2 "
        "https://github.com/NVIDIA/cutlass.git third_party/cutlass"
    )


def _cuda_version_tuple() -> tuple[int, int]:
    import torch

    v = getattr(torch.version, "cuda", None) or "0.0"
    try:
        major, minor = v.split(".")[:2]
        return (int(major), int(minor))
    except ValueError:
        return (0, 0)


def build():
    """JIT-compile the FLUTE extension. Returns the loaded module.

    Note: this is a slow compile (~5-15 min first time) because of
    CUTLASS template instantiation. Cached in torch's extension cache
    afterward (see ``torch.utils.cpp_extension._get_build_directory``).
    """
    from torch.utils.cpp_extension import load

    if not FLUTE_CSRC_DIR.exists():
        raise FileNotFoundError(
            f"Vendored FLUTE sources not found at {FLUTE_CSRC_DIR}. "
            "Install turboquant-plus-vllm from git source (not a minimal wheel)."
        )

    cutlass_path = _resolve_cutlass_path()

    sources = sorted(
        str(p)
        for p in FLUTE_CSRC_DIR.iterdir()
        if p.suffix in (".cu", ".cpp")
    )
    if not sources:
        raise FileNotFoundError(f"No .cu or .cpp files in {FLUTE_CSRC_DIR}")

    cuda_major, cuda_minor = _cuda_version_tuple()
    extra_cuda_cflags = [
        "-O3",
        "--use_fast_math",
        "-std=c++17",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        # FLUTE targets Ampere/Ada (mma.sync + cp.async). Hopper (SM90)
        # and Blackwell (SM100/120) need a separate WGMMA/TMA path; until
        # that's ported, don't emit code for SM90+ or FLUTE's template
        # asserts fail on Hopper-specific cute instantiations.
        "-gencode=arch=compute_80,code=sm_80",   # A100
        "-gencode=arch=compute_86,code=sm_86",   # A10/A40/RTX 3090
        "-gencode=arch=compute_89,code=sm_89",   # L40S, RTX 4090
    ]

    include_paths = [
        str(FLUTE_CSRC_DIR),
        str(cutlass_path / "include"),
        str(cutlass_path / "tools" / "util" / "include"),
    ]

    # Load under the name `turboquant_flute_C` so torch caches the build
    # separately from our main extension.
    module = load(
        name="turboquant_flute_C",
        sources=sources,
        extra_cuda_cflags=extra_cuda_cflags,
        extra_cflags=["-O3", "-std=c++17"],
        extra_include_paths=include_paths,
        verbose=True,
    )
    return module


if __name__ == "__main__":
    mod = build()
    print(f"Built successfully: {mod}")
