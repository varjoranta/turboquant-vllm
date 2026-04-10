"""Python interface to TurboQuant+ CUDA kernels.

Handles:
- JIT compilation of the CUDA extension (first call only)
- Codebook computation (Lloyd's algorithm, matching turboquant_plus)
- WHT rotation sign generation (deterministic from seed)
- Asymmetric K/V support (different bit widths for K and V)
- QJL initialization for K cache inner product preservation

Usage:
    from turboquant_vllm.ops import TurboQuantOps

    # Symmetric: same bit width for K and V
    ops = TurboQuantOps(head_dim=128, k_bits=4, v_bits=4, seed=42)

    # Asymmetric: K gets more precision (dominates attention quality)
    ops = TurboQuantOps(head_dim=128, k_bits=4, v_bits=3, seed=42)

    # With QJL for K cache (full Algorithm 2)
    ops = TurboQuantOps(head_dim=128, k_bits=4, v_bits=3, seed=42, use_qjl=True)
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Allows static analysis (ruff, pyright) to resolve `torch.Tensor`
    # annotations without forcing a top-level runtime import — keeping
    # ops.py importable on codebook / CLI paths that don't need GPU.
    import torch  # noqa: F401

logger = logging.getLogger(__name__)

_cuda_module = None


def _get_torch():
    """Lazy import torch — not needed for codebook computation."""
    import torch
    return torch


def _get_cuda_module():
    """Lazy-load the CUDA extension via JIT compilation."""
    global _cuda_module
    if _cuda_module is None:
        from turboquant_vllm.build import build
        _cuda_module = build()
        logger.info("TurboQuant+ CUDA extension loaded")
    return _cuda_module


def _lloyds_centroids(bit_width: int, dim: int) -> tuple[list[float], list[float]]:
    """Compute optimal centroids for N(0, 1/d) via Lloyd's algorithm.

    Returns (centroids, boundaries). Matches turboquant_plus.codebook exactly.
    """
    from scipy import stats

    n = 1 << bit_width
    sigma = 1.0 / math.sqrt(dim)

    if bit_width == 1:
        c = math.sqrt(2.0 / (math.pi * dim))
        return [-c, c], [0.0]
    if bit_width == 2:
        s = math.sqrt(dim)
        centroids = [-1.51 / s, -0.453 / s, 0.453 / s, 1.51 / s]
        return centroids, [(centroids[i] + centroids[i + 1]) / 2 for i in range(3)]

    boundaries = list(stats.norm.ppf([i / n for i in range(1, n)], scale=sigma))
    centroids = [0.0] * n

    def cond_exp(a: float, b: float) -> float:
        a_s = a / sigma if math.isfinite(a) else a
        b_s = b / sigma if math.isfinite(b) else b
        if not math.isfinite(a_s):
            prob = stats.norm.cdf(b_s)
        elif not math.isfinite(b_s):
            prob = stats.norm.sf(a_s)
        else:
            prob = stats.norm.cdf(b_s) - stats.norm.cdf(a_s)
        if prob < 1e-15:
            return ((a if math.isfinite(a) else 0) + (b if math.isfinite(b) else 0)) / 2
        return sigma * (stats.norm.pdf(a_s) - stats.norm.pdf(b_s)) / prob

    for _ in range(100):
        centroids[0] = cond_exp(-math.inf, boundaries[0])
        for i in range(1, n - 1):
            centroids[i] = cond_exp(boundaries[i - 1], boundaries[i])
        centroids[-1] = cond_exp(boundaries[-1], math.inf)
        boundaries = [(centroids[i] + centroids[i + 1]) / 2 for i in range(n - 1)]

    centroids.sort()
    return centroids, boundaries


def _generate_wht_signs(dim: int, seed: int) -> tuple["torch.Tensor", "torch.Tensor"]:
    """Deterministic WHT rotation signs. Matches turboquant_plus.rotation."""
    torch = _get_torch()
    gen = torch.Generator().manual_seed(seed)
    signs1 = (torch.randint(0, 2, (dim,), generator=gen) * 2 - 1).float()
    signs2 = (torch.randint(0, 2, (dim,), generator=gen) * 2 - 1).float()
    return signs1, signs2


def _generate_qjl_matrix(dim: int, seed: int) -> "torch.Tensor":
    """QJL random projection matrix. Matches turboquant_plus.qjl.QJL."""
    torch = _get_torch()
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(dim, dim, generator=gen)


def _packed_dim(bit_width: int, head_dim: int) -> int:
    """Bytes per vector in packed cache for a given bit width."""
    if bit_width == 4:
        return head_dim // 2
    if bit_width == 2:
        return head_dim // 4
    return head_dim  # 3-bit: 1 byte per index (no sub-byte packing)


class TurboQuantOps:
    """High-level interface to TurboQuant+ CUDA operations.

    Supports asymmetric K/V bit widths and optional QJL for K cache.

    For the full TurboQuant+ Algorithm 2 on K cache:
      K total bits = k_bits = (k_bits - 1) PolarQuant + 1 QJL
      Requires use_qjl=True. The CUDA init sets K PolarQuant to (k_bits - 1) bits.

    Without QJL (use_qjl=False):
      K uses PolarQuant at full k_bits. Simpler, slightly worse inner product
      preservation but +0.23% PPL at 4-bit is acceptable for most use cases.
    """

    def __init__(
        self,
        head_dim: int,
        k_bits: int = 4,
        v_bits: int = 4,
        seed: int = 42,
        device: str = "cuda",
        use_qjl: bool = False,
    ):
        assert head_dim > 0 and (head_dim & (head_dim - 1)) == 0, "head_dim must be power of 2"
        assert 2 <= k_bits <= 4 and 2 <= v_bits <= 4

        self.head_dim = head_dim
        self.k_bits = k_bits
        self.v_bits = v_bits
        self.seed = seed
        self.device = device
        self.use_qjl = use_qjl

        # PolarQuant bit widths:
        # With QJL: K PolarQuant uses k_bits - 1, QJL adds 1 bit
        # Without QJL: K PolarQuant uses full k_bits
        k_pq_bits = (k_bits - 1) if use_qjl else k_bits
        v_pq_bits = v_bits

        # Compute codebooks
        k_centroids, k_boundaries = _lloyds_centroids(k_pq_bits, head_dim)
        v_centroids, v_boundaries = _lloyds_centroids(v_pq_bits, head_dim)

        self._k_centroids = torch.tensor(k_centroids, dtype=torch.float32, device=device)
        self._k_boundaries = torch.tensor(k_boundaries, dtype=torch.float32, device=device)
        self._v_centroids = torch.tensor(v_centroids, dtype=torch.float32, device=device)
        self._v_boundaries = torch.tensor(v_boundaries, dtype=torch.float32, device=device)

        # Separate WHT rotation signs for K and V (different seeds → independent rotations)
        k_signs1, k_signs2 = _generate_wht_signs(head_dim, seed)
        v_signs1, v_signs2 = _generate_wht_signs(head_dim, seed + 500)

        # Packed dimensions (may differ for asymmetric)
        self.k_packed_dim = _packed_dim(k_pq_bits, head_dim)
        self.v_packed_dim = _packed_dim(v_pq_bits, head_dim)

        # Initialize CUDA constant memory — separate K and V
        cuda = _get_cuda_module()
        cuda.init_k(
            self._k_centroids, self._k_boundaries,
            k_signs1.to(device), k_signs2.to(device),
            head_dim, k_pq_bits,
        )
        cuda.init_v(
            self._v_centroids, self._v_boundaries,
            v_signs1.to(device), v_signs2.to(device),
            head_dim, v_pq_bits,
        )

        # QJL for K cache (Algorithm 2: PolarQuant residual → sign bits)
        if use_qjl:
            qjl_mat = _generate_qjl_matrix(head_dim, seed + 1000).to(device)
            cuda.init_qjl(qjl_mat)
            self._qjl_matrix = qjl_mat

        stats = self.memory_stats()
        mode = f"K={k_bits}b({'PQ' + str(k_pq_bits) + '+QJL1' if use_qjl else 'PQ' + str(k_pq_bits)}) V={v_bits}b(PQ{v_pq_bits})"
        logger.info(
            "TurboQuant+ initialized: head_dim=%d %s → %.1fx compression",
            head_dim, mode, stats["compression_ratio"],
        )

    def quantize(self, vectors: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize fp16 vectors (PolarQuant only). Returns (indices, norms)."""
        n = vectors.shape[0]
        indices = torch.empty(n, self.head_dim, dtype=torch.uint8, device=self.device)
        norms = torch.empty(n, dtype=torch.float32, device=self.device)
        _get_cuda_module().quantize(vectors.half(), indices, norms)
        return indices, norms

    def dequantize(self, indices: torch.Tensor, norms: torch.Tensor) -> torch.Tensor:
        """Dequantize indices + norms back to fp16 vectors (PolarQuant only)."""
        n = indices.shape[0]
        output = torch.empty(n, self.head_dim, dtype=torch.float16, device=self.device)
        _get_cuda_module().dequantize(indices, norms, output)
        return output

    def reshape_and_cache(
        self,
        key: torch.Tensor, value: torch.Tensor,
        key_cache: torch.Tensor, value_cache: torch.Tensor,
        k_norms: torch.Tensor, v_norms: torch.Tensor,
        slot_mapping: torch.Tensor,
    ):
        """Fused quantize + pack into paged cache. Drop-in for vLLM."""
        _get_cuda_module().reshape_and_cache(
            key.half(), value.half(),
            key_cache, value_cache, k_norms, v_norms, slot_mapping,
        )

    def dequant_paged_cache(
        self,
        cache: torch.Tensor, norms: torch.Tensor,
        output: torch.Tensor, block_table: torch.Tensor,
        seq_len: int,
    ):
        """Dequantize from paged cache to contiguous buffer."""
        _get_cuda_module().dequant_paged_cache(
            cache, norms, output, block_table, seq_len,
        )

    def allocate_cache(
        self, num_blocks: int, block_size: int, num_kv_heads: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Allocate paged cache tensors with correct packed dimensions.

        Returns (key_cache, value_cache, k_norms, v_norms).
        K and V caches may have different packed_dim if asymmetric.
        """
        key_cache = torch.zeros(
            num_blocks, block_size, num_kv_heads, self.k_packed_dim,
            dtype=torch.uint8, device=self.device,
        )
        value_cache = torch.zeros(
            num_blocks, block_size, num_kv_heads, self.v_packed_dim,
            dtype=torch.uint8, device=self.device,
        )
        norms_shape = (num_blocks, block_size, num_kv_heads)
        k_norms = torch.zeros(norms_shape, dtype=torch.float32, device=self.device)
        v_norms = torch.zeros(norms_shape, dtype=torch.float32, device=self.device)
        return key_cache, value_cache, k_norms, v_norms

    def memory_stats(self) -> dict:
        """Per-token, per-head memory usage."""
        k_bytes = self.k_packed_dim + 4  # packed indices + fp32 norm
        v_bytes = self.v_packed_dim + 4
        if self.use_qjl:
            k_bytes += self.head_dim // 8 + 4  # QJL sign bits + residual norm
        fp16_bytes = self.head_dim * 2

        return {
            "k_bytes_per_token": k_bytes,
            "v_bytes_per_token": v_bytes,
            "bytes_per_token": k_bytes + v_bytes,
            "fp16_bytes_per_token": fp16_bytes * 2,
            "compression_ratio": (fp16_bytes * 2) / (k_bytes + v_bytes),
            "k_bits": self.k_bits,
            "v_bits": self.v_bits,
            "use_qjl": self.use_qjl,
        }
