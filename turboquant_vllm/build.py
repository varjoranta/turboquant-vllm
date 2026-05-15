"""Build the TurboQuant+ CUDA extension.

Usage:
    python -m turboquant_vllm.build

This compiles csrc/turbo_quant.cu and csrc/torch_bindings.cpp into
a shared library that can be loaded as a PyTorch extension.
"""

import logging
import os
import shutil
from importlib import util as importlib_util
from pathlib import Path

logger = logging.getLogger(__name__)

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

PREBUILT_DIR = _pkg_dir / "_native"
PREBUILT_BASENAME = "turbo_quant_cuda"


def _cuda_version_tuple():
    """Return (major, minor) for the CUDA toolkit torch was built against.

    Matches the toolchain `torch.utils.cpp_extension.load()` will invoke,
    so it's the right version to gate gencode flags on.
    """
    import torch

    v = getattr(torch.version, "cuda", None) or "0.0"
    parts = v.split(".")
    try:
        major = int(parts[0])
    except (TypeError, ValueError):
        return (0, 0)
    try:
        minor = int(parts[1]) if len(parts) > 1 else 0
    except (TypeError, ValueError):
        minor = 0
    return (major, minor)


def _detect_local_arches() -> list[str]:
    """Return sorted local CUDA SM targets like ``["86"]``.

    Compiling for every historical architecture makes the first JIT build
    expensive in both time and host RAM. For runtime JIT we default to the
    local machine's visible GPUs instead. Users can override this with the
    standard ``TORCH_CUDA_ARCH_LIST`` env var or ``TQ_CUDA_ARCH_LIST``.
    """
    import torch

    if not torch.cuda.is_available():
        return []
    arches = {
        f"{major}{minor}"
        for idx in range(torch.cuda.device_count())
        for major, minor in [torch.cuda.get_device_capability(idx)]
    }
    return sorted(arches, key=int)


def _gencode_flags() -> list[str]:
    """Build a compact gencode list for the current runtime host."""
    cuda_major, cuda_minor = _cuda_version_tuple()

    override = os.environ.get("TQ_CUDA_ARCH_LIST") or os.environ.get("TORCH_CUDA_ARCH_LIST")
    if override:
        arch_tokens = [tok.strip() for tok in override.replace(";", ",").split(",") if tok.strip()]
        flags: list[str] = []
        for token in arch_tokens:
            token = token.replace(".", "")
            if not token.isdigit():
                continue
            arch_num = int(token)
            if arch_num == 121 and (cuda_major, cuda_minor) < (12, 9):
                continue
            flags.append(f"-gencode=arch=compute_{arch_num},code=sm_{arch_num}")
        if flags:
            return flags

    local_arches = _detect_local_arches()
    if local_arches:
        flags = []
        for arch in local_arches:
            arch_num = int(arch)
            if arch_num == 121 and (cuda_major, cuda_minor) < (12, 9):
                continue
            flags.append(f"-gencode=arch=compute_{arch_num},code=sm_{arch_num}")
        if flags:
            return flags

    # Fallback for environments where no GPU is visible during build.
    flags = [
        "-gencode=arch=compute_80,code=sm_80",
        "-gencode=arch=compute_86,code=sm_86",
        "-gencode=arch=compute_89,code=sm_89",
        "-gencode=arch=compute_90,code=sm_90",
    ]
    if (cuda_major, cuda_minor) >= (12, 9):
        flags.append("-gencode=arch=compute_121,code=sm_121")
    return flags


def _candidate_prebuilt_paths() -> list[Path]:
    """Return candidate prebuilt extension paths in priority order."""
    candidates: list[Path] = []
    explicit = os.environ.get("TQ_CUDA_PREBUILT_PATH")
    if explicit:
        candidates.append(Path(explicit))

    for directory in (PREBUILT_DIR, _pkg_dir):
        if directory.exists():
            candidates.extend(sorted(directory.glob(f"{PREBUILT_BASENAME}*.so")))

    return candidates


def _load_module_from_path(path: Path):
    spec = importlib_util.spec_from_file_location(PREBUILT_BASENAME, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create import spec for {path}")
    module = importlib_util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_prebuilt_module():
    """Load a prebuilt extension bundled into the package/image."""
    for candidate in _candidate_prebuilt_paths():
        if not candidate.is_file():
            continue
        try:
            module = _load_module_from_path(candidate)
            logger.warning("Loaded prebuilt TurboQuant CUDA extension from %s", candidate)
            return module
        except Exception as exc:
            logger.warning("Failed to load prebuilt TurboQuant CUDA extension from %s: %s", candidate, exc)
    return None


def _bundle_module(module) -> Path:
    """Copy the compiled extension into the package for runtime reuse."""
    PREBUILT_DIR.mkdir(parents=True, exist_ok=True)
    source = Path(module.__file__).resolve()
    target = PREBUILT_DIR / source.name
    if source != target:
        shutil.copy2(source, target)
    logger.warning("Bundled TurboQuant CUDA extension to %s", target)
    return target


def build():
    """JIT-compile the CUDA extension. Returns the loaded module."""
    from torch.utils.cpp_extension import load

    prebuilt = _load_prebuilt_module()
    if prebuilt is not None:
        return prebuilt

    # Runtime JIT compilation can otherwise fan out to many ninja workers
    # and transiently consume tens of GiB of host RAM. Keep the default
    # conservative; power users can override via MAX_JOBS.
    os.environ.setdefault("MAX_JOBS", os.environ.get("TQ_CUDA_MAX_JOBS", "1"))
    gencode_flags = _gencode_flags()

    logger.warning(
        "TurboQuant CUDA build config: MAX_JOBS=%s TQ_CUDA_MAX_JOBS=%s "
        "TQ_CUDA_ARCH_LIST=%s TORCH_CUDA_ARCH_LIST=%s final_gencode=%s",
        os.environ.get("MAX_JOBS"),
        os.environ.get("TQ_CUDA_MAX_JOBS"),
        os.environ.get("TQ_CUDA_ARCH_LIST"),
        os.environ.get("TORCH_CUDA_ARCH_LIST"),
        " ".join(gencode_flags),
    )

    sources = [
        str(CSRC_DIR / "turbo_quant.cu"),
        str(CSRC_DIR / "tq_weight_dequant.cu"),
        str(CSRC_DIR / "tq_weight_gemv_bs1.cu"),
        str(CSRC_DIR / "torch_bindings.cpp"),
    ]

    extra_cuda_cflags = [
        "-O3",
        "--use_fast_math",
        *gencode_flags,
    ]

    module = load(
        name="turbo_quant_cuda",
        sources=sources,
        extra_cuda_cflags=extra_cuda_cflags,
        extra_include_paths=[str(CSRC_DIR)],
        verbose=True,
    )
    if os.environ.get("TQ_CUDA_BUNDLE", "0") == "1":
        _bundle_module(module)
    return module


if __name__ == "__main__":
    os.environ.setdefault("TQ_CUDA_BUNDLE", "1")
    mod = build()
    print(f"Built successfully: {mod}")
    print(f"Available functions: {[x for x in dir(mod) if not x.startswith('_')]}")
