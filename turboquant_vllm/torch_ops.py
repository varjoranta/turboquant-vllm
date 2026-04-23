"""TurboQuant+ operations in PyTorch (GPU-compatible).

Implements the full TurboQuant+ algorithm from turboquant_plus:
- K cache: TurboQuant (Algorithm 2) = PolarQuant (b-1 bits) + QJL (1 bit)
- V cache: PolarQuant MSE-only (Algorithm 1) at full b bits

Key difference from turbo-quant-lite (our PyPI package):
- turbo-quant-lite: PolarQuant only, Beta-distribution codebook at d=512, for embeddings
- turboquant_plus: PolarQuant + QJL, Gaussian codebook at N(0,1/d), for KV cache

Uses fast Walsh-Hadamard rotation (O(d log d)) instead of dense rotation (O(d²)).
"""

import math
import torch
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Codebook: Lloyd-Max optimal centroids for N(0, 1/d)
# ---------------------------------------------------------------------------


def _gaussian_conditional_expectation(sigma: float, a: float, b: float) -> float:
    """E[X | a < X < b] for X ~ N(0, sigma²)."""
    from scipy import stats as sp_stats

    a_std = a / sigma if math.isfinite(a) else a
    b_std = b / sigma if math.isfinite(b) else b

    if not math.isfinite(a_std):
        prob = sp_stats.norm.cdf(b_std)
    elif not math.isfinite(b_std):
        prob = sp_stats.norm.sf(a_std)
    else:
        prob = sp_stats.norm.cdf(b_std) - sp_stats.norm.cdf(a_std)

    if prob < 1e-15:
        if math.isfinite(a) and not math.isfinite(b):
            return a + sigma
        elif not math.isfinite(a) and math.isfinite(b):
            return b - sigma
        return (a + b) / 2.0

    pdf_diff = sp_stats.norm.pdf(a_std) - sp_stats.norm.pdf(b_std)
    return sigma * pdf_diff / prob


def _lloyds_gaussian(n_centroids: int, sigma: float, n_iter: int = 100) -> list[float]:
    """Lloyd's algorithm for optimal scalar quantization of N(0, sigma²)."""
    from scipy import stats as sp_stats

    boundaries = list(sp_stats.norm.ppf([i / n_centroids for i in range(1, n_centroids)], scale=sigma))
    centroids = [0.0] * n_centroids

    for _ in range(n_iter):
        centroids[0] = _gaussian_conditional_expectation(sigma, -math.inf, boundaries[0])
        for i in range(1, n_centroids - 1):
            centroids[i] = _gaussian_conditional_expectation(sigma, boundaries[i - 1], boundaries[i])
        centroids[-1] = _gaussian_conditional_expectation(sigma, boundaries[-1], math.inf)
        boundaries = [(centroids[i] + centroids[i + 1]) / 2 for i in range(n_centroids - 1)]

    return sorted(centroids)


def optimal_centroids(bit_width: int, dim: int) -> list[float]:
    """Compute optimal centroids for post-rotation coordinates ~ N(0, 1/d)."""
    n = 1 << bit_width
    if bit_width == 1:
        c = math.sqrt(2.0 / (math.pi * dim))
        return [-c, c]
    if bit_width == 2:
        s = math.sqrt(dim)
        return [-1.51 / s, -0.453 / s, 0.453 / s, 1.51 / s]
    return _lloyds_gaussian(n, sigma=1.0 / math.sqrt(dim))


# ---------------------------------------------------------------------------
# Fast Walsh-Hadamard Transform (O(d log d) rotation)
# ---------------------------------------------------------------------------


def _fast_wht_batch(x: torch.Tensor) -> torch.Tensor:
    """Batched fast Walsh-Hadamard transform. x shape: (batch, n) where n is power of 2."""
    n = x.shape[1]
    h = 1
    while h < n:
        # Butterfly: pairs at distance h
        x_view = x.view(x.shape[0], n // (h * 2), 2, h)
        a = x_view[:, :, 0, :].clone()
        b = x_view[:, :, 1, :].clone()
        x_view[:, :, 0, :] = a + b
        x_view[:, :, 1, :] = a - b
        h *= 2
    return x / math.sqrt(n)


def _fast_wht_batch_blocked(x: torch.Tensor, block_size: int) -> torch.Tensor:
    """Batched fast WHT applied independently to each block of width ``block_size``.

    Equivalent to ``block_diag(H, H, ..., H) @ x`` where each ``H`` is a
    ``block_size × block_size`` Walsh-Hadamard matrix. Used for partial-rotary
    models (MiniMax M2.5/M2.7, Qwen3.6-A3B, etc.) so the RoPE-rotated prefix
    and the content-only suffix are kept under independent rotations and don't
    mix inside a group.

    ``x.shape[1]`` must be a multiple of ``block_size``; ``block_size`` must
    be a power of two.
    """
    n = x.shape[1]
    assert n % block_size == 0, f"dim {n} not divisible by block_size {block_size}"
    assert block_size & (block_size - 1) == 0, f"block_size {block_size} must be a power of two"
    num_blocks = n // block_size
    x_blocks = x.reshape(x.shape[0] * num_blocks, block_size)
    x_blocks = _fast_wht_batch(x_blocks)
    return x_blocks.reshape(x.shape[0], n)


def _next_power_of_2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


# ---------------------------------------------------------------------------
# Rotation backends: WHT (original) and Planar (RotorQuant-inspired)
# ---------------------------------------------------------------------------


def _planar_rotate(x: torch.Tensor, cos_sin: torch.Tensor) -> torch.Tensor:
    """Apply 2D Givens rotation to pairs of elements. O(d), 4 FMAs per pair.

    x: (batch, dim)
    cos_sin: (n_pairs, 2) as [cos θ, sin θ]
    Returns: (batch, dim) rotated
    """
    n_pairs = cos_sin.shape[0]
    cos_t = cos_sin[:, 0]  # (n_pairs,)
    sin_t = cos_sin[:, 1]
    v0 = x[:, 0::2][:, :n_pairs]  # even indices
    v1 = x[:, 1::2][:, :n_pairs]  # odd indices
    r0 = cos_t * v0 - sin_t * v1
    r1 = sin_t * v0 + cos_t * v1
    out = x.clone()
    out[:, 0::2][:, :n_pairs] = r0
    out[:, 1::2][:, :n_pairs] = r1
    return out


def _planar_rotate_inverse(x: torch.Tensor, cos_sin: torch.Tensor) -> torch.Tensor:
    """Inverse 2D Givens rotation (negate sin). O(d), 4 FMAs per pair."""
    n_pairs = cos_sin.shape[0]
    cos_t = cos_sin[:, 0]
    sin_t = cos_sin[:, 1]
    v0 = x[:, 0::2][:, :n_pairs]
    v1 = x[:, 1::2][:, :n_pairs]
    r0 = cos_t * v0 + sin_t * v1
    r1 = -sin_t * v0 + cos_t * v1
    out = x.clone()
    out[:, 0::2][:, :n_pairs] = r0
    out[:, 1::2][:, :n_pairs] = r1
    return out


# ---------------------------------------------------------------------------
# PolarQuant: rotation + optimal scalar quantization
# ---------------------------------------------------------------------------


class PolarQuantTorch:
    """PolarQuant — rotation + Gaussian codebook.

    Supports two rotation modes:
    - 'wht' (default): D2 @ H @ D1 structured random rotation. O(d log d).
      Original TurboQuant algorithm, full-rank decorrelation.
    - 'planar': 2D Givens rotation per pair of elements. O(d), 4 FMAs per pair.
      Inspired by RotorQuant/PlanarQuant. Faster, fewer parameters (128 vs 16K),
      comparable or better PPL for KV cache.
    """

    def __init__(
        self,
        dim: int,
        bit_width: int,
        seed: int = 42,
        device: str = "cuda",
        rotation: str = "wht",
        rotary_dim: "int | None" = None,
    ):
        self.dim = dim
        self.bit_width = bit_width
        dev = torch.device(device)
        if dev.type == "cuda" and dev.index is None:
            dev = torch.device("cuda", torch.cuda.current_device())
        self.device = dev
        self.rotation_mode = rotation

        gen = torch.Generator(device="cpu").manual_seed(seed)

        if rotation == "planar":
            # Random Givens angles: one (cos, sin) pair per 2 elements
            n_pairs = (dim + 1) // 2
            angles = torch.rand(n_pairs, generator=gen) * 2 * math.pi
            self.cos_sin = torch.stack([angles.cos(), angles.sin()], dim=1).float().to(self.device)
            # Dummy signs for compatibility (not used in planar mode)
            self.signs1 = torch.ones(dim, device=self.device)
            self.signs2 = torch.ones(dim, device=self.device)
            self.padded_dim = dim
            self.rotary_dim = None
        else:
            # WHT rotation: D2 @ H @ D1
            self.padded_dim = _next_power_of_2(dim)
            self.signs1 = (torch.randint(0, 2, (self.padded_dim,), generator=gen) * 2 - 1).float().to(self.device)
            self.signs2 = (torch.randint(0, 2, (self.padded_dim,), generator=gen) * 2 - 1).float().to(self.device)
            self.cos_sin = None
            if rotary_dim is not None and 0 < rotary_dim < dim:
                assert rotary_dim & (rotary_dim - 1) == 0, f"rotary_dim {rotary_dim} must be a power of two"
                assert self.padded_dim % rotary_dim == 0, (
                    f"rotary_dim {rotary_dim} must divide padded_dim {self.padded_dim}"
                )
                self.rotary_dim = rotary_dim
            else:
                self.rotary_dim = None

        # Codebook: optimal centroids for N(0, 1/dim)
        centroids_list = optimal_centroids(bit_width, dim)
        self.centroids = torch.tensor(centroids_list, dtype=torch.float32, device=self.device)
        self.boundaries = (self.centroids[:-1] + self.centroids[1:]) / 2

    def _apply_wht(self, padded: torch.Tensor) -> torch.Tensor:
        """Full-width or block-diagonal WHT based on self.rotary_dim."""
        if self.rotary_dim is None:
            return _fast_wht_batch(padded)
        return _fast_wht_batch_blocked(padded, block_size=self.rotary_dim)

    def _rotate(self, x: torch.Tensor) -> torch.Tensor:
        """Apply forward rotation. x shape: (batch, dim)."""
        if self.rotation_mode == "planar":
            return _planar_rotate(x, self.cos_sin)

        batch = x.shape[0]
        if self.padded_dim > self.dim:
            padded = torch.zeros(batch, self.padded_dim, device=x.device, dtype=x.dtype)
            padded[:, : self.dim] = x
        else:
            padded = x.clone()
        padded *= self.signs1.unsqueeze(0)
        padded = self._apply_wht(padded)
        padded *= self.signs2.unsqueeze(0)
        return padded[:, : self.dim]

    def _rotate_inverse(self, y: torch.Tensor) -> torch.Tensor:
        """Apply inverse rotation."""
        if self.rotation_mode == "planar":
            return _planar_rotate_inverse(y, self.cos_sin)

        batch = y.shape[0]
        if self.padded_dim > self.dim:
            padded = torch.zeros(batch, self.padded_dim, device=y.device, dtype=y.dtype)
            padded[:, : self.dim] = y
        else:
            padded = y.clone()
        padded *= self.signs2.unsqueeze(0)
        padded = self._apply_wht(padded)
        padded *= self.signs1.unsqueeze(0)
        return padded[:, : self.dim]

    def quantize(self, x: torch.Tensor, norm_correction: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize vectors. x: (batch, dim) or (dim,). Returns (indices, norms).

        Args:
            norm_correction: If True, store original_norm / reconstruction_norm
                instead of raw original_norm. This corrects the magnitude error
                introduced by quantization, improving PPL. Based on MidasMining's
                finding adopted in vLLM PR #38479.
        """
        squeeze = x.dim() == 1
        if squeeze:
            x = x.unsqueeze(0)
        x = x.to(device=self.device, dtype=torch.float32)

        norms = torch.linalg.norm(x, dim=1)
        safe_norms = torch.where(norms > 0, norms, torch.ones_like(norms))
        x_unit = x / safe_norms.unsqueeze(1)

        y = self._rotate(x_unit)
        indices = torch.searchsorted(self.boundaries, y.contiguous())

        if norm_correction:
            # Compute reconstruction norm to correct magnitude error
            y_hat = self.centroids[indices]
            x_hat_unit = self._rotate_inverse(y_hat)
            recon_norm = torch.linalg.norm(x_hat_unit, dim=1)
            safe_recon = torch.where(recon_norm > 0, recon_norm, torch.ones_like(recon_norm))
            norms = norms / safe_recon

        if squeeze:
            return indices.squeeze(0), norms.squeeze(0)
        return indices, norms

    def dequantize(self, indices: torch.Tensor, norms: torch.Tensor) -> torch.Tensor:
        """Dequantize. indices: (batch, dim) or (dim,). Returns reconstructed vectors.

        Works with both raw norms and norm-corrected norms (from quantize(norm_correction=True)).
        """
        squeeze = indices.dim() == 1
        if squeeze:
            indices = indices.unsqueeze(0)
            norms = norms.unsqueeze(0)
        indices = indices.to(device=self.device)
        norms = norms.to(device=self.device, dtype=torch.float32)

        y_hat = self.centroids[indices]
        x_hat_unit = self._rotate_inverse(y_hat)
        x_hat = x_hat_unit * norms.unsqueeze(1)

        if squeeze:
            return x_hat.squeeze(0)
        return x_hat

    def quantize_and_residual(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize and return (indices, norms, residual). Used by TurboQuant stage 2."""
        squeeze = x.dim() == 1
        if squeeze:
            x = x.unsqueeze(0)
        x = x.to(device=self.device, dtype=torch.float32)

        indices, norms = self.quantize(x)
        x_hat = self.dequantize(indices, norms)
        residual = x - x_hat

        if squeeze:
            return indices.squeeze(0), norms.squeeze(0), residual.squeeze(0)
        return indices, norms, residual


# ---------------------------------------------------------------------------
# QJL: 1-bit quantization via random projection + sign
# ---------------------------------------------------------------------------


class QJLTorch:
    """Quantized Johnson-Lindenstrauss 1-bit quantizer.

    Matches turboquant_plus.qjl.QJL but uses a seeded projection
    that doesn't store the full d×d matrix — instead generates
    projection rows on-the-fly from the seed for memory efficiency.

    For production CUDA: the projection can be done via a random
    number generator seeded per-row, avoiding O(d²) storage.
    For this PyTorch prototype, we store the full matrix.
    """

    QJL_CONST = math.sqrt(math.pi / 2)

    def __init__(self, dim: int, seed: int = 1042, device: str = "cuda"):
        self.dim = dim
        self.device = device
        gen = torch.Generator(device="cpu").manual_seed(seed)
        self.S = torch.randn(dim, dim, generator=gen, dtype=torch.float32).to(device)

    def quantize(self, r: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize residual. r: (batch, dim) or (dim,). Returns (signs, norms)."""
        squeeze = r.dim() == 1
        if squeeze:
            r = r.unsqueeze(0)
        r = r.float()

        norms = torch.linalg.norm(r, dim=1)
        projected = r @ self.S.T  # (batch, dim)
        signs = torch.sign(projected).to(torch.int8)
        signs[signs == 0] = 1

        if squeeze:
            return signs.squeeze(0), norms.squeeze(0)
        return signs, norms

    def dequantize(self, signs: torch.Tensor, norms: torch.Tensor) -> torch.Tensor:
        """Dequantize signs to approximate residual."""
        squeeze = signs.dim() == 1
        if squeeze:
            signs = signs.unsqueeze(0)
            norms = norms.unsqueeze(0)

        signs_f = signs.float()
        reconstructed = signs_f @ self.S  # (batch, dim) — S^T @ signs
        scale = self.QJL_CONST / self.dim * norms
        reconstructed *= scale.unsqueeze(1)

        if squeeze:
            return reconstructed.squeeze(0)
        return reconstructed


# ---------------------------------------------------------------------------
# Full TurboQuant (Algorithm 2): PolarQuant + QJL
# ---------------------------------------------------------------------------


@dataclass
class CompressedKV:
    """Compressed K or V cache for one layer+head."""

    # PolarQuant part
    indices: torch.Tensor  # (seq_len, head_dim) int64
    norms: torch.Tensor  # (seq_len,) float32
    # QJL part (only for K cache, None for V)
    qjl_signs: torch.Tensor | None = None  # (seq_len, head_dim) int8
    qjl_norms: torch.Tensor | None = None  # (seq_len,) float32


class KVCacheCompressorTorch:
    """Full TurboQuant+ KV cache compression on GPU.

    K cache: TurboQuant = PolarQuant(b-1 bits) + QJL(1 bit)
      → inner product preservation for attention scores (Q @ K^T)

    V cache: PolarQuant MSE-only(b bits)
      → MSE preservation for value reconstruction (attn_weights @ V)

    This matches turboquant_plus.kv_cache.KVCacheCompressor but runs on GPU.
    """

    def __init__(
        self,
        head_dim: int,
        k_bits: int = 4,
        v_bits: int = 4,
        seed: int = 42,
        device: str = "cuda",
        use_cuda: bool = False,
        norm_correction: bool = True,
        use_qjl: bool = False,
        rotation: str = "wht",
    ):
        """
        Args:
            norm_correction: Store original_norm/reconstruction_norm instead
                of raw norm. Corrects magnitude error from quantization.
                Default True (matches vLLM PR #38479 presets).
            use_qjl: Use QJL residual correction for K cache. Default False
                (QJL hurts quality per TheTom's turbo4-resurrection research).
                When False, all k_bits go to PolarQuant centroids.
            rotation: 'wht' (Walsh-Hadamard, O(d log d)) or 'planar'
                (2D Givens, O(d)). Planar is faster with comparable quality.
        """
        self.head_dim = head_dim
        self.k_bits = k_bits
        self.v_bits = v_bits
        self.device = device
        self.norm_correction = norm_correction
        self.use_qjl = use_qjl
        self.rotation = rotation
        self._cuda_mod = None

        # The CUDA store kernel does not currently apply norm_correction —
        # it returns raw norms. Silently falling through would degrade
        # reconstruction quality, so force the PyTorch path when norm
        # correction is requested. Log once so users know.
        if use_cuda and norm_correction:
            import logging as _logging

            _logging.getLogger(__name__).warning(
                "TurboQuant+: norm_correction=True is incompatible with the "
                "CUDA KV compression kernel; falling back to the PyTorch path. "
                "Set norm_correction=False to use CUDA."
            )
            use_cuda = False

        self.use_cuda = use_cuda

        if use_cuda and rotation == "wht":
            # CUDA kernels only support WHT rotation currently
            self._init_cuda(head_dim, k_bits, v_bits, seed, device)

        if use_qjl and k_bits >= 2:
            self.k_polar = PolarQuantTorch(head_dim, k_bits - 1, seed=seed, device=device, rotation=rotation)
            self.k_qjl = QJLTorch(head_dim, seed=seed + 1000, device=device)
        else:
            self.k_polar = PolarQuantTorch(head_dim, k_bits, seed=seed, device=device, rotation=rotation)
            self.k_qjl = None

        self.v_polar = PolarQuantTorch(head_dim, v_bits, seed=seed + 500, device=device, rotation=rotation)

    def _init_cuda(self, head_dim, k_bits, v_bits, seed, device):
        """Initialize CUDA kernels with matching codebooks and rotations."""
        from turboquant_vllm.build import build

        self._cuda_mod = build()

        k_pq_bits = (k_bits - 1 if k_bits >= 2 else 1) if self.use_qjl else k_bits
        v_pq_bits = v_bits

        # Compute codebooks (reuse the same Lloyd's algorithm)
        k_pq = PolarQuantTorch(head_dim, k_pq_bits, seed=seed, device=device)
        v_pq = PolarQuantTorch(head_dim, v_pq_bits, seed=seed + 500, device=device)

        self._cuda_mod.init_k(
            k_pq.centroids.cpu(),
            k_pq.boundaries.cpu(),
            k_pq.signs1.cpu().float(),
            k_pq.signs2.cpu().float(),
            head_dim,
            k_pq_bits,
        )
        self._cuda_mod.init_v(
            v_pq.centroids.cpu(),
            v_pq.boundaries.cpu(),
            v_pq.signs1.cpu().float(),
            v_pq.signs2.cpu().float(),
            head_dim,
            v_pq_bits,
        )

    def compress_k(self, k: torch.Tensor) -> CompressedKV:
        """Compress key vectors. k: (num_tokens, head_dim).

        CUDA fast path is only used when norm_correction=False and
        use_qjl=False. The __init__ guard forces use_cuda off when
        norm_correction is requested, so this path only fires for
        plain raw-norm quantization.
        """
        if self.use_cuda and self._cuda_mod is not None and not self.use_qjl:
            n = k.shape[0]
            indices = torch.zeros(n, self.head_dim, dtype=torch.uint8, device=self.device)
            norms = torch.zeros(n, dtype=torch.float32, device=self.device)
            self._cuda_mod.quantize(k.half(), indices, norms)
            return CompressedKV(indices=indices.to(torch.int64), norms=norms)

        if self.k_qjl is not None:
            # Legacy QJL path: PolarQuant + QJL residual correction
            indices, norms, residual = self.k_polar.quantize_and_residual(k)
            qjl_signs, qjl_norms = self.k_qjl.quantize(residual)
            return CompressedKV(indices=indices, norms=norms, qjl_signs=qjl_signs, qjl_norms=qjl_norms)

        # Default: PolarQuant only, all bits for centroids
        indices, norms = self.k_polar.quantize(k, norm_correction=self.norm_correction)
        return CompressedKV(indices=indices, norms=norms)

    def compress_v(self, v: torch.Tensor) -> CompressedKV:
        """Compress value vectors. v: (num_tokens, head_dim). MSE-only, no QJL."""
        indices, norms = self.v_polar.quantize(v, norm_correction=self.norm_correction)
        return CompressedKV(indices=indices, norms=norms)

    def decompress_k(self, compressed: CompressedKV) -> torch.Tensor:
        """Decompress key vectors to fp32."""
        if self.use_cuda and self._cuda_mod is not None and self.k_qjl is None:
            n = compressed.indices.shape[0]
            x_hat = torch.zeros(n, self.head_dim, dtype=torch.float16, device=self.device)
            self._cuda_mod.dequantize(compressed.indices.to(torch.uint8), compressed.norms, x_hat)
            return x_hat.float()

        x_mse = self.k_polar.dequantize(compressed.indices, compressed.norms)
        if self.k_qjl is not None and compressed.qjl_signs is not None:
            x_qjl = self.k_qjl.dequantize(compressed.qjl_signs, compressed.qjl_norms)
            return x_mse + x_qjl
        return x_mse

    def decompress_v(self, compressed: CompressedKV) -> torch.Tensor:
        """Decompress value vectors to fp32."""
        return self.v_polar.dequantize(compressed.indices, compressed.norms)

    def memory_stats(self) -> dict:
        """Storage per token per head in bytes."""
        if self.use_qjl:
            # K: (k_bits-1) PolarQuant + 1 QJL + 2 norms
            k_bits_per_token = self.head_dim * self.k_bits + 64
        else:
            # K: k_bits PolarQuant + 1 norm
            k_bits_per_token = self.head_dim * self.k_bits + 32
        # V: v_bits PolarQuant + 1 norm
        v_bits_per_token = self.head_dim * self.v_bits + 32
        # FP16 baseline
        fp16_bits = self.head_dim * 16

        k_bytes = k_bits_per_token / 8
        v_bytes = v_bits_per_token / 8
        fp16_bytes = fp16_bits / 8

        return {
            "k_bytes_per_token": k_bytes,
            "v_bytes_per_token": v_bytes,
            "total_bytes_per_token": k_bytes + v_bytes,
            "fp16_bytes_per_token": fp16_bytes * 2,
            "compression_ratio": (fp16_bytes * 2) / (k_bytes + v_bytes),
        }
