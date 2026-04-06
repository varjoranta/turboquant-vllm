"""Triton fused dequant-GEMM kernel for TurboQuant weight compression.

Fuses weight decompression (unpack → codebook → rotation → scale) with
matrix multiplication into a single kernel. Eliminates the intermediate
decompressed weight buffer and the second global memory round-trip.

Design: pre-compute the rotation matrix W_rot = D1 @ H @ D2 / sqrt(n) once
(128×128 = 64 KB). In the kernel, dequant becomes:
  centroid_vec = centroids[packed_indices]  # codebook lookup
  w_deq = centroid_vec @ W_rot^T           # rotation via matmul
  w_deq *= norm                            # scale
Then fuse with the outer GEMM: acc += a_tile @ w_deq^T.

Compared to separate dequant + cuBLAS GEMM:
- No intermediate FP16 buffer (saves 32 MB for 4096×4096)
- No second global memory read (saves bandwidth)

Usage:
    from turboquant_vllm.triton_ops import tq_fused_gemm
    output = tq_fused_gemm(x, packed_weight, norms, signs1, signs2,
                           centroids, group_size=128, bits=4)
"""

import torch

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


if HAS_TRITON:

    @triton.jit
    def _tq_fused_gemm_kernel(
        # Activation: (M, K) row-major
        a_ptr,
        stride_am, stride_ak,
        # Compressed weight: packed (N, K_packed) uint8, norms (N, n_groups) float32
        packed_ptr, norms_ptr,
        stride_packed_n, stride_packed_k,
        stride_norms_n, stride_norms_g,
        # Pre-computed rotation matrix: (GROUP_SIZE, GROUP_SIZE) float32
        w_rot_ptr,
        # Centroids: (n_centroids,) float32
        centroids_ptr,
        # Output: (M, N)
        c_ptr,
        stride_cm, stride_cn,
        # Bias: (N,) or None
        bias_ptr,
        # Dimensions
        M, N, K,
        n_groups,
        # Constexprs
        GROUP_SIZE: tl.constexpr,
        BITS: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """Fused TQ dequant-GEMM.

        Per output tile (BLOCK_M × BLOCK_N), iterates over K in steps of
        GROUP_SIZE. Each step: unpack → codebook lookup → rotate → scale →
        accumulate GEMM.
        """
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, GROUP_SIZE)

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        # Load rotation matrix once: (GROUP_SIZE, GROUP_SIZE) — fits in L1
        rot_offs = offs_k[:, None] * GROUP_SIZE + offs_k[None, :]
        w_rot = tl.load(w_rot_ptr + rot_offs)

        for g in range(n_groups):
            k_start = g * GROUP_SIZE

            # Load activation tile: (BLOCK_M, GROUP_SIZE)
            a_offs = offs_m[:, None] * stride_am + (k_start + offs_k[None, :]) * stride_ak
            a_mask = (offs_m[:, None] < M) & ((k_start + offs_k[None, :]) < K)
            a_tile = tl.load(a_ptr + a_offs, mask=a_mask, other=0.0).to(tl.float32)

            # Unpack indices: (BLOCK_N, GROUP_SIZE)
            # packed_weight layout: (N * n_groups, packed_per_group)
            # Row for output n, group g = n * n_groups + g
            packed_row = offs_n * n_groups + g  # (BLOCK_N,)

            if BITS == 4:
                byte_idx = offs_k // 2
                is_hi = (offs_k % 2).to(tl.int32)
                packed_elem_offs = (packed_row[:, None] * stride_packed_n
                                   + byte_idx[None, :] * stride_packed_k)
                packed_bytes = tl.load(packed_ptr + packed_elem_offs,
                                       mask=offs_n[:, None] < N, other=0).to(tl.int32)
                indices = tl.where(is_hi[None, :] > 0,
                                   (packed_bytes >> 4) & 0xF,
                                   packed_bytes & 0xF)
            elif BITS == 2:
                byte_idx = offs_k // 4
                shift = (offs_k % 4).to(tl.int32) * 2
                packed_elem_offs = (packed_row[:, None] * stride_packed_n
                                   + byte_idx[None, :] * stride_packed_k)
                packed_bytes = tl.load(packed_ptr + packed_elem_offs,
                                       mask=offs_n[:, None] < N, other=0).to(tl.int32)
                indices = (packed_bytes >> shift[None, :]) & 0x3
            else:
                # 3-bit or other: 1 byte per index
                packed_elem_offs = (packed_row[:, None] * stride_packed_n
                                   + offs_k[None, :] * stride_packed_k)
                indices = tl.load(packed_ptr + packed_elem_offs,
                                  mask=offs_n[:, None] < N, other=0).to(tl.int32)

            # Codebook lookup: (BLOCK_N, GROUP_SIZE) float
            centroid_vec = tl.load(centroids_ptr + indices)

            # Rotate: centroid_vec @ W_rot → (BLOCK_N, GROUP_SIZE)
            # W_rot[i,j] encodes the inverse rotation: row i is the rotated basis vector i
            w_deq = tl.dot(centroid_vec, w_rot)

            # Scale by per-group norm
            norm_offs = offs_n * stride_norms_n + g * stride_norms_g
            norms = tl.load(norms_ptr + norm_offs, mask=offs_n < N, other=0.0)
            w_deq = w_deq * norms[:, None]

            # Accumulate GEMM: (BLOCK_M, GROUP_SIZE) @ (GROUP_SIZE, BLOCK_N)
            acc += tl.dot(a_tile, tl.trans(w_deq))

        # Bias
        if bias_ptr:
            bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
            acc += bias[None, :]

        # Store
        c_offs = offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        tl.store(c_ptr + c_offs, acc.to(c_ptr.type.element_ty), mask=c_mask)


def _build_rotation_matrix(signs1: torch.Tensor, signs2: torch.Tensor,
                           group_size: int) -> torch.Tensor:
    """Pre-compute W_rot = D1 @ H @ D2 / sqrt(n).

    Computed once per (signs1, signs2, group_size) and cached.
    128×128 = 64 KB — trivial memory cost.
    """
    from turboquant_vllm.torch_ops import _fast_wht_batch

    n = group_size
    eye = torch.eye(n, device=signs1.device, dtype=torch.float32)

    # Inverse rotation: D2 → H → D1
    rotated = eye * signs2.unsqueeze(0)
    rotated = _fast_wht_batch(rotated)
    rotated = rotated * signs1.unsqueeze(0)

    return rotated


def tq_fused_gemm(
    x: torch.Tensor,
    packed_weight: torch.Tensor,
    norms: torch.Tensor,
    signs1: torch.Tensor,
    signs2: torch.Tensor,
    centroids: torch.Tensor,
    group_size: int = 128,
    bits: int = 4,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fused TurboQuant dequant + GEMM.

    Args:
        x: Input activation (M, K).
        packed_weight: Packed indices (N, K_packed), uint8.
        norms: Per-group norms (N, n_groups), float32.
        signs1, signs2: WHT sign vectors (group_size,).
        centroids: Codebook (n_centroids,).
        group_size: Group size (power of 2, default 128).
        bits: Bit width (2, 3, or 4).
        bias: Optional bias (N,).

    Returns:
        (M, N) tensor in x.dtype.
    """
    if not HAS_TRITON:
        raise ImportError("Triton required for fused GEMM")

    # Handle 3D input (batch, seq, hidden) from transformers
    orig_shape = x.shape
    if x.dim() > 2:
        x = x.reshape(-1, x.shape[-1])

    M, K = x.shape
    N = norms.shape[0]
    n_groups = norms.shape[1]  # use norms shape as authority (may differ from K // group_size due to padding)

    if K % group_size != 0 or K // group_size != n_groups:
        # Fall back to non-fused path for layers with padding mismatch
        raise ValueError(f"K={K} not aligned with group_size={group_size}, n_groups={n_groups}")

    # Cache rotation matrix (keyed by tensor identity, not data_ptr which
    # can be reused after deallocation)
    cache_key = (id(signs1), id(signs2), group_size)
    if not hasattr(tq_fused_gemm, '_rot_cache'):
        tq_fused_gemm._rot_cache = {}
    if cache_key not in tq_fused_gemm._rot_cache:
        tq_fused_gemm._rot_cache[cache_key] = _build_rotation_matrix(
            signs1, signs2, group_size).contiguous()
    w_rot = tq_fused_gemm._rot_cache[cache_key]

    output = torch.empty(M, N, dtype=x.dtype, device=x.device)

    # Tile sizes limited by shared memory: rotation matrix (GROUP_SIZE^2 * 4 bytes)
    # plus tile data must fit in ~164 KB (A100). For group_size=128: rotation = 64 KB,
    # leaving ~100 KB for tiles → BLOCK_M, BLOCK_N ≤ 32.
    max_block = 32 if group_size >= 128 else 64
    BLOCK_M = min(max_block, triton.next_power_of_2(M))
    BLOCK_N = min(max_block, triton.next_power_of_2(N))

    if not packed_weight.is_contiguous():
        packed_weight = packed_weight.contiguous()
    if not norms.is_contiguous():
        norms = norms.contiguous()

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    _tq_fused_gemm_kernel[grid](
        x, x.stride(0), x.stride(1),
        packed_weight, norms,
        packed_weight.stride(0), packed_weight.stride(1),
        norms.stride(0), norms.stride(1),
        w_rot,
        centroids,
        output, output.stride(0), output.stride(1),
        bias,
        M, N, K, n_groups,
        GROUP_SIZE=group_size,
        BITS=bits,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )

    # Restore original batch dimensions
    if len(orig_shape) > 2:
        output = output.reshape(*orig_shape[:-1], N)

    return output
