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
            elif BITS == 3:
                # 3-bit sub-byte: 8 indices per 3 bytes
                group_of_8 = offs_k // 8
                pos_in_8 = offs_k % 8
                bit_off_in_3 = pos_in_8 * 3
                first_byte = bit_off_in_3 // 8
                bit_in_byte = (bit_off_in_3 % 8).to(tl.int32)
                crosses = bit_in_byte > 5

                byte_idx0 = group_of_8 * 3 + first_byte
                byte_idx1 = byte_idx0 + 1

                ptrs0 = packed_row[:, None] * stride_packed_n + byte_idx0[None, :] * stride_packed_k
                b0 = tl.load(packed_ptr + ptrs0, mask=offs_n[:, None] < N, other=0).to(tl.int32)
                ptrs1 = packed_row[:, None] * stride_packed_n + byte_idx1[None, :] * stride_packed_k
                b1 = tl.load(packed_ptr + ptrs1, mask=offs_n[:, None] < N, other=0).to(tl.int32)

                single = (b0 >> bit_in_byte[None, :]) & 0x7
                cross = ((b0 >> bit_in_byte[None, :]) | (b1 << (8 - bit_in_byte[None, :]))) & 0x7
                indices = tl.where(crosses[None, :], cross, single)
            elif BITS == 2:
                byte_idx = offs_k // 4
                shift = (offs_k % 4).to(tl.int32) * 2
                packed_elem_offs = (packed_row[:, None] * stride_packed_n
                                   + byte_idx[None, :] * stride_packed_k)
                packed_bytes = tl.load(packed_ptr + packed_elem_offs,
                                       mask=offs_n[:, None] < N, other=0).to(tl.int32)
                indices = (packed_bytes >> shift[None, :]) & 0x3
            else:
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
    """Pre-compute inverse rotation matrix W_rot = H @ D2 @ D1 / sqrt(n).

    Row i is the rotated basis vector i. Used by:
    - dequant-GEMM: centroid_vec @ W_rot (weight-side inverse rotation)
    - FWHT-on-input: x @ W_rot.T (input-side forward rotation, transposed)
    - learned_rotation: initial R for Cayley optimization

    Computed once per (signs1, signs2, group_size) and cached.
    128x128 = 64 KB -- trivial memory cost.
    """
    from turboquant_vllm.torch_ops import _fast_wht_batch

    n = group_size
    eye = torch.eye(n, device=signs1.device, dtype=torch.float32)

    # Apply inverse rotation to identity rows: D2 -> H -> scale by D1
    rotated = eye * signs2.unsqueeze(0)      # right-multiply by D2
    rotated = _fast_wht_batch(rotated)       # H @ D2
    rotated = rotated * signs1.unsqueeze(0)  # column-wise D1: H @ D2 @ D1

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

    # Shared rotation matrix cache (also used by rotate_input / FWHT-on-input)
    w_rot = _get_cached_rotation_matrix(signs1, signs2, group_size)

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


# ---------------------------------------------------------------------------
# FWHT-on-input path: rotate input once, fused codebook dot product
# ---------------------------------------------------------------------------
# Instead of decompressing weights (unpack -> codebook -> inverse WHT -> matmul),
# apply the WHT rotation to the INPUT vector and do the dot product against
# quantized codes directly. The key identity:
#
#   x @ w = x @ (norm * D1 @ H @ D2 @ centroids[codes])
#         = norm * dot(D2 @ H @ D1 @ x, centroids[codes])
#
# The rotation moves from N weight rows to 1 input vector.

_rotation_matrix_cache: dict[tuple, torch.Tensor] = {}


def _get_cached_rotation_matrix(signs1: torch.Tensor, signs2: torch.Tensor,
                                group_size: int) -> torch.Tensor:
    """Get cached rotation matrix for (signs1, signs2, group_size).

    Keyed by tensor identity (id) since sign vectors are long-lived module
    attributes. Returns W_rot such that W_rot.T applies the forward rotation.
    """
    key = (id(signs1), id(signs2), group_size)
    if key not in _rotation_matrix_cache:
        _rotation_matrix_cache[key] = _build_rotation_matrix(
            signs1, signs2, group_size).contiguous()
    return _rotation_matrix_cache[key]


def _tensor_fingerprint(x: torch.Tensor) -> torch.Tensor:
    """Cheap content fingerprint: first 4 + last 4 elements of flattened tensor."""
    flat = x.detach().flatten()
    n = min(4, flat.numel())
    return torch.cat([flat[:n], flat[-n:]]).cpu()


class FWHTInputCache:
    """Cache rotated input across Q/K/V projections sharing the same hidden state.

    Saves ~67% of FWHT calls in a standard attention block.
    Uses pointer + shape + content fingerprint to detect stale entries
    from PyTorch memory reuse across forward passes.
    """

    def __init__(self):
        self._ptr: int = -1
        self._storage_ptr: int = -1
        self._shape: tuple = ()
        self._fingerprint: torch.Tensor | None = None
        self._result: torch.Tensor | None = None

    def get(self, x: torch.Tensor) -> torch.Tensor | None:
        if (x.data_ptr() == self._ptr
                and x.untyped_storage().data_ptr() == self._storage_ptr
                and x.shape == self._shape):
            # Verify content hasn't changed (catches memory reuse across forward passes).
            # Cheap: compare first + last few elements instead of full tensor.
            if self._fingerprint is not None:
                fp = _tensor_fingerprint(x)
                if not torch.equal(fp, self._fingerprint):
                    return None
            return self._result
        return None

    def put(self, x: torch.Tensor, result: torch.Tensor):
        self._ptr = x.data_ptr()
        self._storage_ptr = x.untyped_storage().data_ptr()
        self._shape = x.shape
        self._fingerprint = _tensor_fingerprint(x)
        self._result = result

    def clear(self):
        self._ptr = -1
        self._storage_ptr = -1
        self._shape = ()
        self._fingerprint = None
        result = self._result
        self._result = None
        del result


# Default cache instance. For multi-model setups, pass a per-model
# FWHTInputCache to tq_fwht_input_gemm(..., cache=my_cache) instead.
_fwht_input_cache = FWHTInputCache()


def rotate_input(
    x: torch.Tensor,
    signs1: torch.Tensor,
    signs2: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    """Apply forward rotation (D2 @ H @ D1) to each group of the input.

    Returns (batch, in_f_padded). Uses matmul with the cached rotation matrix
    (1 kernel launch per group) instead of butterfly WHT (log2 launches).

    This is the transpose of _build_rotation_matrix, which encodes the inverse
    rotation. The identity: x @ W_rot.T = D2 @ H @ D1 @ x.
    """
    batch = x.shape[0]
    K = x.shape[1]
    padded_K = ((K + group_size - 1) // group_size) * group_size

    if padded_K > K:
        x = torch.nn.functional.pad(x, (0, padded_K - K))

    # W_rot encodes inverse rotation; transpose gives forward rotation
    w_rot = _get_cached_rotation_matrix(signs1, signs2, group_size)

    x_grouped = x.reshape(-1, group_size)
    x_grouped = torch.matmul(x_grouped, w_rot.T)

    return x_grouped.reshape(batch, padded_K)


if HAS_TRITON:

    @triton.autotune(
        configs=[
            triton.Config({'BLOCK_M': 1, 'BLOCK_N': 128}, num_warps=4, num_stages=2),
            triton.Config({'BLOCK_M': 1, 'BLOCK_N': 256}, num_warps=8, num_stages=2),
            triton.Config({'BLOCK_M': 4, 'BLOCK_N': 128}, num_warps=4, num_stages=2),
            triton.Config({'BLOCK_M': 8, 'BLOCK_N': 64}, num_warps=4, num_stages=2),
            triton.Config({'BLOCK_M': 8, 'BLOCK_N': 128}, num_warps=4, num_stages=2),
            triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64}, num_warps=4, num_stages=2),
            triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128}, num_warps=8, num_stages=2),
            triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64}, num_warps=4, num_stages=2),
            triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128}, num_warps=8, num_stages=2),
            triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=2),
        ],
        key=['batch_size', 'out_f', 'in_f_padded'],
    )
    @triton.jit
    def _polar_fused_gemm_kernel(
        # Rotated input: (batch, in_f_padded), float32
        x_rot_ptr, stride_xm, stride_xk,
        # Packed codes: (out_f, packed_cols), uint8
        codes_ptr, stride_cn, stride_ck,
        # Per-group norms: (out_f, n_groups), float32
        norms_ptr, stride_nn, stride_ng,
        # Pre-scaled centroids: (n_centroids,), float32
        ct_ptr,
        # Output: (batch, out_f)
        out_ptr, stride_om, stride_on,
        # Bias: (out_f,) or None
        bias_ptr,
        # Dims
        batch_size, out_f, in_f_padded, n_groups,
        # Constexprs
        BLOCK_K: tl.constexpr,    # = group_size = 128
        BITS: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """Fused codebook-weighted dot product.

        For each (batch_tile, output_tile), iterates over groups:
          load codes -> centroid lookup -> scale by norm -> dot with x_rot -> accumulate.
        No weight decompression needed.
        """
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_m = offs_m < batch_size
        mask_n = offs_n < out_f

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for g in range(n_groups):
            offs_k = tl.arange(0, BLOCK_K)

            x_ptrs = offs_m[:, None] * stride_xm + (g * BLOCK_K + offs_k)[None, :] * stride_xk
            x_tile = tl.load(x_rot_ptr + x_ptrs, mask=mask_m[:, None], other=0.0)

            # packed_weight is (out_f * n_groups, packed_cols_per_group):
            # row for output n, group g = n * n_groups + g
            packed_row = offs_n * n_groups + g
            if BITS == 4:
                byte_idx = offs_k // 2
                is_hi = offs_k % 2
                code_ptrs = packed_row[:, None] * stride_cn + byte_idx[None, :] * stride_ck
                packed = tl.load(codes_ptr + code_ptrs, mask=mask_n[:, None], other=0).to(tl.int32)
                codes = tl.where(is_hi[None, :] > 0, (packed >> 4) & 0xF, packed & 0xF)
            elif BITS == 3:
                # 3-bit sub-byte: 8 indices per 3 bytes (24 bits).
                # For position k: bit_offset = (k%8)*3, spans 1-2 bytes.
                group_of_8 = offs_k // 8
                pos_in_8 = offs_k % 8
                bit_off_in_3 = pos_in_8 * 3  # 0,3,6,9,12,15,18,21
                first_byte = bit_off_in_3 // 8  # 0,0,0,1,1,1,2,2
                bit_in_byte = (bit_off_in_3 % 8).to(tl.int32)
                crosses = bit_in_byte > 5  # True for pos 2 (bit=6) and 5 (bit=7)

                byte_idx0 = group_of_8 * 3 + first_byte
                byte_idx1 = byte_idx0 + 1

                ptrs0 = packed_row[:, None] * stride_cn + byte_idx0[None, :] * stride_ck
                b0 = tl.load(codes_ptr + ptrs0, mask=mask_n[:, None], other=0).to(tl.int32)
                ptrs1 = packed_row[:, None] * stride_cn + byte_idx1[None, :] * stride_ck
                b1 = tl.load(codes_ptr + ptrs1, mask=mask_n[:, None], other=0).to(tl.int32)

                single = (b0 >> bit_in_byte[None, :]) & 0x7
                cross = ((b0 >> bit_in_byte[None, :]) | (b1 << (8 - bit_in_byte[None, :]))) & 0x7
                codes = tl.where(crosses[None, :], cross, single)
            elif BITS == 2:
                byte_idx = offs_k // 4
                shift = (offs_k % 4).to(tl.int32) * 2
                code_ptrs = packed_row[:, None] * stride_cn + byte_idx[None, :] * stride_ck
                packed = tl.load(codes_ptr + code_ptrs, mask=mask_n[:, None], other=0).to(tl.int32)
                codes = (packed >> shift[None, :]) & 0x3
            else:
                code_ptrs = packed_row[:, None] * stride_cn + offs_k[None, :] * stride_ck
                codes = tl.load(codes_ptr + code_ptrs, mask=mask_n[:, None], other=0).to(tl.int32)

            # Centroid lookup: (BLOCK_N, BLOCK_K)
            values = tl.load(ct_ptr + codes)

            # Per-group norm: (BLOCK_N,)
            norm_ptrs = offs_n * stride_nn + g * stride_ng
            norms = tl.load(norms_ptr + norm_ptrs, mask=mask_n, other=0.0)

            # Scale centroids by norm
            values = values * norms[:, None]

            # Dot product: (BLOCK_M, BLOCK_N) += (BLOCK_M, BLOCK_K) @ (BLOCK_K, BLOCK_N)
            acc += tl.dot(x_tile, tl.trans(values))

        # Bias
        if bias_ptr:
            bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0)
            acc += bias[None, :]

        # Store
        out_ptrs = offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
        out_mask = mask_m[:, None] & mask_n[None, :]
        tl.store(out_ptr + out_ptrs, acc.to(out_ptr.type.element_ty), mask=out_mask)


def tq_fwht_input_gemm(
    x: torch.Tensor,
    packed_weight: torch.Tensor,
    norms: torch.Tensor,
    signs1: torch.Tensor,
    signs2: torch.Tensor,
    centroids: torch.Tensor,
    group_size: int = 128,
    bits: int = 4,
    bias: torch.Tensor | None = None,
    cache: FWHTInputCache | None = None,
) -> torch.Tensor:
    """FWHT-on-input fused GEMM. Rotates input once, then codebook dot product.

    2x+ faster than dequant-GEMM because:
    - FWHT applied to input (1 vector) not weights (N rows)
    - No intermediate decompressed weight buffer
    - FWHT cacheable across Q/K/V projections (67% cache hit rate)

    Args:
        cache: Optional per-model FWHTInputCache. Defaults to the global
               singleton. Pass a dedicated instance for multi-model or
               multi-threaded inference to avoid stale cache hits.
    """
    if not HAS_TRITON:
        raise ImportError("Triton required")

    orig_shape = x.shape
    if x.dim() > 2:
        x = x.reshape(-1, x.shape[-1])

    M, K = x.shape
    N = norms.shape[0]
    n_groups = norms.shape[1]

    padded_K = n_groups * group_size
    if K != padded_K:
        raise ValueError(f"K={K} not aligned with group_size={group_size}, expected {padded_K}")

    input_cache = cache if cache is not None else _fwht_input_cache
    cached = input_cache.get(x)
    if cached is not None:
        x_rot = cached
    else:
        x_rot = rotate_input(x.float(), signs1, signs2, group_size)
        input_cache.put(x, x_rot)

    output = torch.empty(M, N, dtype=x.dtype, device=x.device)

    BLOCK_K = group_size
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_M']),
        triton.cdiv(N, META['BLOCK_N']),
    )

    _polar_fused_gemm_kernel[grid](
        x_rot, x_rot.stride(0), x_rot.stride(1),
        packed_weight, packed_weight.stride(0), packed_weight.stride(1),
        norms, norms.stride(0), norms.stride(1),
        centroids,
        output, output.stride(0), output.stride(1),
        bias,
        M, N, padded_K, n_groups,
        BLOCK_K=BLOCK_K, BITS=bits,
    )

    if len(orig_shape) > 2:
        output = output.reshape(*orig_shape[:-1], N)

    return output
