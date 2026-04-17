"""Build the TurboQuant+ CUDA extension.

Usage:
    python -m turboquant_vllm.build

This compiles csrc/turbo_quant.cu and csrc/torch_bindings.cpp into
a shared library that can be loaded as a PyTorch extension.
"""

from pathlib import Path

# csrc/ is either a sibling of turboquant_vllm/ (dev) or a sibling package (installed)
_pkg_dir = Path(__file__).resolve().parent
CSRC_DIR = _pkg_dir.parent / "csrc"
if not (CSRC_DIR / "turbo_quant.cu").exists():
    # Installed as package — csrc is a sibling package in site-packages
    CSRC_DIR = _pkg_dir.parent / "csrc"
if not (CSRC_DIR / "turbo_quant.cu").exists():
    raise FileNotFoundError(
        f"Cannot find csrc/turbo_quant.cu. Searched: {_pkg_dir.parent / 'csrc'}. "
        "Install from source (git clone) to get CUDA kernels, or use PyTorch fallback."
    )


def _cuda_version_tuple():
    """Return (major, minor) for the CUDA toolkit torch was built against.

    Matches the toolchain `torch.utils.cpp_extension.load()` will invoke,
    so it's the right version to gate gencode flags on.
    """
    import torch

    v = getattr(torch.version, "cuda", None) or "0.0"
    try:
        major, minor = v.split(".")[:2]
        return (int(major), int(minor))
    except ValueError:
        return (0, 0)


def build():
    """JIT-compile the CUDA extension. Returns the loaded module."""
    from torch.utils.cpp_extension import load

    sources = [
        str(CSRC_DIR / "turbo_quant.cu"),
        str(CSRC_DIR / "tq_weight_dequant.cu"),
        str(CSRC_DIR / "tq_weight_gemv_bs1.cu"),
        str(CSRC_DIR / "torch_bindings.cpp"),
    ]

    # sm_121 (GB10 / DGX Spark) needs CUDA >= 12.9. Older toolchains fail hard.
    cuda_major, cuda_minor = _cuda_version_tuple()
    extra_cuda_cflags = [
        "-O3",
        "--use_fast_math",
        "-gencode=arch=compute_75,code=sm_75",  # T4, RTX 2080
        "-gencode=arch=compute_80,code=sm_80",  # A100
        "-gencode=arch=compute_86,code=sm_86",  # A10, A40, RTX 3090, RTX A6000
        "-gencode=arch=compute_89,code=sm_89",  # L40S, RTX 4090
        "-gencode=arch=compute_90,code=sm_90",  # H100, H200
        "-gencode=arch=compute_120,code=sm_120",  # Blackwell consumer (RTX 50xx)
    ]
    if (cuda_major, cuda_minor) >= (12, 9):
        extra_cuda_cflags.append(
            "-gencode=arch=compute_121,code=sm_121",  # GB10 (DGX Spark)
        )

    module = load(
        name="turbo_quant_cuda",
        sources=sources,
        extra_cuda_cflags=extra_cuda_cflags,
        extra_include_paths=[str(CSRC_DIR)],
        verbose=True,
    )
    return module


if __name__ == "__main__":
    mod = build()
    print(f"Built successfully: {mod}")
    print(f"Available functions: {[x for x in dir(mod) if not x.startswith('_')]}")
