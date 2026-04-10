"""TurboQuant KV-cache configuration, rotation matrices, and Lloyd-Max centroids.

Ported from the varjoranta/vllm turboquant-integration branch so this package
works as a standalone vLLM plugin without requiring a vLLM fork.
"""

import math
from dataclasses import dataclass
from functools import lru_cache

import torch


# ---------------------------------------------------------------------------
# Lloyd-Max optimal quantizer
# ---------------------------------------------------------------------------

def _gaussian_pdf(x: float, sigma2: float) -> float:
    return (1.0 / math.sqrt(2 * math.pi * sigma2)) * math.exp(-x * x / (2 * sigma2))


def solve_lloyd_max(
    d: int,
    bits: int,
    max_iter: int = 200,
    tol: float = 1e-10,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Solve Lloyd-Max optimal quantizer for N(0, 1/d) distribution."""
    from scipy import integrate

    n_levels = 2 ** bits
    sigma2 = 1.0 / d
    sigma = math.sqrt(sigma2)

    def pdf(x):
        return _gaussian_pdf(x, sigma2)

    lo, hi = -3.5 * sigma, 3.5 * sigma
    centroids = [lo + (hi - lo) * (i + 0.5) / n_levels for i in range(n_levels)]

    for _ in range(max_iter):
        boundaries = [(centroids[i] + centroids[i + 1]) / 2.0 for i in range(n_levels - 1)]
        edges = [lo * 3] + boundaries + [hi * 3]
        new_centroids = []
        for i in range(n_levels):
            a, b = edges[i], edges[i + 1]
            num, _ = integrate.quad(lambda x: x * pdf(x), a, b)
            den, _ = integrate.quad(pdf, a, b)
            new_centroids.append(num / den if den > 1e-15 else centroids[i])
        if max(abs(new_centroids[i] - centroids[i]) for i in range(n_levels)) < tol:
            break
        centroids = new_centroids

    boundaries = [(centroids[i] + centroids[i + 1]) / 2.0 for i in range(n_levels - 1)]
    return (
        torch.tensor(centroids, dtype=torch.float32),
        torch.tensor(boundaries, dtype=torch.float32),
    )


@lru_cache(maxsize=32)
def get_centroids(d: int, bits: int) -> torch.Tensor:
    """Get precomputed Lloyd-Max centroids (cached)."""
    centroids, _ = solve_lloyd_max(d, bits)
    return centroids


@lru_cache(maxsize=32)
def get_boundaries(d: int, bits: int) -> torch.Tensor:
    """Get precomputed Lloyd-Max boundaries (cached)."""
    _, boundaries = solve_lloyd_max(d, bits)
    return boundaries


# ---------------------------------------------------------------------------
# Random matrix generation
# ---------------------------------------------------------------------------

def generate_rotation_matrix(
    d: int, seed: int, device: torch.device = torch.device("cpu")
) -> torch.Tensor:
    """Generate Haar-distributed random orthogonal matrix via QR decomposition."""
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    G = torch.randn(d, d, generator=gen, device="cpu", dtype=torch.float32)
    Q, R = torch.linalg.qr(G)
    diag_sign = torch.sign(torch.diag(R))
    diag_sign[diag_sign == 0] = 1.0
    Q = Q * diag_sign.unsqueeze(0)
    return Q.to(device)


def generate_qjl_matrix(
    d: int, seed: int, device: torch.device = torch.device("cpu")
) -> torch.Tensor:
    """Generate i.i.d. N(0,1) projection matrix for QJL."""
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    S = torch.randn(d, d, generator=gen, device="cpu", dtype=torch.float32)
    return S.to(device)


# ---------------------------------------------------------------------------
# TurboQuantConfig
# ---------------------------------------------------------------------------

import os


@dataclass
class TurboQuantConfig:
    """Configuration for TurboQuant KV-cache quantization.

    Args:
        head_dim: Attention head dimension.
        total_bits: Total bits per key coordinate (3 or 4).
        value_quant_bits: Bits per value coordinate.
            8 = FP8 E4M3 (default, near-lossless).
            4 = 4-bit uniform (higher compression).
            2 = 2-bit uniform (aggressive).
        no_qjl: If True (default), skip QJL sign correction.
        asymmetric: If True, K and V use different bit widths.
        v_total_bits: Total bits for V when asymmetric=True.
        seed: Base seed for deterministic random matrix generation.
    """
    head_dim: int = 128
    total_bits: int = 3
    value_quant_bits: int = 8
    no_qjl: bool = True
    asymmetric: bool = False
    v_total_bits: int = 3
    seed: int = 42

    @property
    def mse_bits(self) -> int:
        if self.no_qjl:
            return self.total_bits
        return max(self.total_bits - 1, 1)

    @property
    def n_centroids(self) -> int:
        return 2 ** self.mse_bits

    @property
    def key_packed_size(self) -> int:
        """Packed bytes for one compressed KEY vector.

        Packing scheme (must match native_backend._store_kv/_decode_attention_python):
          4-bit: nibble pack → head_dim // 2 bytes  (2 indices per byte, 4 bits each)
          3-bit: true 3-bit pack → 3 * head_dim // 8 bytes  (8 indices into 3 bytes, requires head_dim % 8 == 0)
          2-bit: 4-per-byte → head_dim // 4 bytes
          other: tight bit pack → ceil(head_dim * mse_bits / 8)
        """
        d = self.head_dim
        b = self.mse_bits
        if b == 4 and d % 2 == 0:
            mse_bytes = d // 2   # nibble packing
        elif b == 3 and d % 8 == 0:
            mse_bytes = 3 * d // 8  # true 3-bit: 8 indices → 3 bytes
        elif b == 2 and d % 4 == 0:
            mse_bytes = d // 4   # 2-per-byte
        else:
            mse_bytes = math.ceil(d * b / 8)  # tight, cross-byte

        if self.no_qjl:
            return mse_bytes + 2  # indices + vec_norm
        qjl_bytes = math.ceil(self.head_dim / 8)
        return mse_bytes + qjl_bytes + 4  # + vec_norm + res_norm

    @property
    def effective_value_quant_bits(self) -> int:
        return self.value_quant_bits

    @property
    def value_fp8(self) -> bool:
        return self.effective_value_quant_bits == 8

    @property
    def value_packed_size(self) -> int:
        """Packed bytes for one VALUE vector."""
        if self.value_fp8:
            return self.head_dim
        data_bytes = math.ceil(self.head_dim * self.value_quant_bits / 8)
        return data_bytes + 4  # +2 scale(fp16) +2 zero(fp16)

    @property
    def slot_size(self) -> int:
        """Total packed bytes per head per position."""
        return self.key_packed_size + self.value_packed_size

    @property
    def padded_slot_size(self) -> int:
        """Slot size rounded up to next power of 2."""
        raw = self.slot_size
        s = 1
        while s < raw:
            s <<= 1
        return s

    @staticmethod
    def from_cache_dtype(cache_dtype: str, head_dim: int,
                         value_quant_bits: int = 8) -> "TurboQuantConfig":
        """Create config from cache dtype string: tq3, tq4, tq_k4v3."""
        vqb_env = os.environ.get("TQ_VALUE_BITS")
        if vqb_env is not None:
            value_quant_bits = int(vqb_env)
        no_qjl = os.environ.get("TQ_NO_QJL", "1") == "1"

        if cache_dtype == "tq3":
            return TurboQuantConfig(head_dim=head_dim, total_bits=3,
                                    value_quant_bits=value_quant_bits, no_qjl=no_qjl)
        elif cache_dtype == "tq4":
            return TurboQuantConfig(head_dim=head_dim, total_bits=4,
                                    value_quant_bits=value_quant_bits, no_qjl=no_qjl)
        elif cache_dtype == "tq_k4v3":
            return TurboQuantConfig(head_dim=head_dim, total_bits=4,
                                    asymmetric=True, v_total_bits=3,
                                    value_quant_bits=value_quant_bits, no_qjl=no_qjl)
        else:
            raise ValueError(f"Unknown TurboQuant cache dtype: {cache_dtype}")
