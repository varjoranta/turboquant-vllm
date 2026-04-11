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
        stride_am,
        stride_ak,
        # Compressed weight: packed (N, K_packed) uint8, norms (N, n_groups) float32
        packed_ptr,
        norms_ptr,
        stride_packed_n,
        stride_packed_k,
        stride_norms_n,
        stride_norms_g,
        # Pre-computed rotation matrix: (GROUP_SIZE, GROUP_SIZE) float32
        w_rot_ptr,
        # Centroids: (n_centroids,) float32
        centroids_ptr,
        # Output: (M, N)
        c_ptr,
        stride_cm,
        stride_cn,
        # Bias: (N,) or None
        bias_ptr,
        # Dimensions
        M,
        N,
        K,
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
                packed_elem_offs = packed_row[:, None] * stride_packed_n + byte_idx[None, :] * stride_packed_k
                packed_bytes = tl.load(packed_ptr + packed_elem_offs, mask=offs_n[:, None] < N, other=0).to(tl.int32)
                indices = tl.where(is_hi[None, :] > 0, (packed_bytes >> 4) & 0xF, packed_bytes & 0xF)
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
                packed_elem_offs = packed_row[:, None] * stride_packed_n + byte_idx[None, :] * stride_packed_k
                packed_bytes = tl.load(packed_ptr + packed_elem_offs, mask=offs_n[:, None] < N, other=0).to(tl.int32)
                indices = (packed_bytes >> shift[None, :]) & 0x3
            else:
                packed_elem_offs = packed_row[:, None] * stride_packed_n + offs_k[None, :] * stride_packed_k
                indices = tl.load(packed_ptr + packed_elem_offs, mask=offs_n[:, None] < N, other=0).to(tl.int32)

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


def _build_rotation_matrix(signs1: torch.Tensor, signs2: torch.Tensor, group_size: int) -> torch.Tensor:
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
    rotated = eye * signs2.unsqueeze(0)  # right-multiply by D2
    rotated = _fast_wht_batch(rotated)  # H @ D2
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

    # Handle both 2D (M, K) and 3D (B, L, K) input uniformly. vLLM 0.19
    # fullgraph dynamo can't handle a runtime-conditional reshape here:
    # it traces BOTH branches of `if x.dim() > 2`, and the else branch
    # (where x is assumed 2D) crashes on the downstream `M, K = x.shape`
    # when the actual input is 3D. Unconditional reshape is a no-op for
    # already-2D input and gives dynamo a single straight-line trace.
    orig_shape = x.shape
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
        x,
        x.stride(0),
        x.stride(1),
        packed_weight,
        norms,
        packed_weight.stride(0),
        packed_weight.stride(1),
        norms.stride(0),
        norms.stride(1),
        w_rot,
        centroids,
        output,
        output.stride(0),
        output.stride(1),
        bias,
        M,
        N,
        K,
        n_groups,
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

# Process-wide cache of pre-computed rotation matrices, keyed by
# ``(id(signs1), id(signs2), group_size)``. A single model share of sign
# vectors across all its TurboQuantWrapper instances collapses this to
# roughly one entry per (bit_width, group_size) pair.
#
# **Capture-safety invariant (load-bearing)**: this cache must be
# populated before any forward pass that might run under CUDA graph
# capture. Cache misses trigger a ``torch.eye`` allocation plus a
# butterfly WHT inside ``_build_rotation_matrix`` — both are capture-safe
# via torch's caching allocator, but the extra allocations and Python-
# side dict writes end up baked into the replayed graph structure and
# the capture phase wall-clock. ``TurboQuantWrapper.__init__`` and
# ``vllm_quant.TurboQuantLinearMethod.process_weights_after_loading``
# both call ``_get_cached_rotation_matrix`` eagerly at construction /
# post-load time so that by the time vLLM's warmup fires, every entry
# this process needs is already in the dict. Do not remove those eager
# calls without replacing them with an equivalent guarantee.
#
# **Device-migration note**: keyed on ``id(signs1)``, which changes
# when ``nn.Module.to(device)`` replaces a registered buffer with a new
# tensor. After a device migration the old cache entry becomes orphaned
# (no impact on correctness — the next call rebuilds on the new device)
# but leaks its memory footprint until process exit. Acceptable for
# vLLM's once-per-process device placement; if a future caller hot-
# swaps devices, migrate the eager refresh into the ``.to()`` hook.
_rotation_matrix_cache: dict[tuple, torch.Tensor] = {}


def _get_cached_rotation_matrix(signs1: torch.Tensor, signs2: torch.Tensor, group_size: int) -> torch.Tensor:
    """Get cached rotation matrix for (signs1, signs2, group_size).

    Keyed by tensor identity (id) since sign vectors are long-lived module
    attributes. Returns W_rot such that W_rot.T applies the forward rotation.

    See the module-level comment on ``_rotation_matrix_cache`` for the
    capture-safety invariant this function participates in.
    """
    key = (id(signs1), id(signs2), group_size)
    if key not in _rotation_matrix_cache:
        _rotation_matrix_cache[key] = _build_rotation_matrix(signs1, signs2, group_size).contiguous()
    return _rotation_matrix_cache[key]


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

    # Autotune key deliberately excludes `batch_size`. Under vLLM 0.19
    # piecewise CUDA graph capture, each of the ~51 capture batch sizes
    # would otherwise trigger its own 10-config autotune run per Linear
    # layer — that is where the 10-25 minute capture-time stall came
    # from on the first TQ3 benchmark runs (see archive branch
    # archive/fwht-input-cache-unmerged for the full history).
    #
    # Keying only on the weight-shape components makes autotune run
    # once per distinct (out_f, in_f_padded) pair — roughly 3-5×
    # per decoder layer instead of 150+. Measured on A100 80GB:
    #
    #   Qwen2.5-0.5B TQ3 startup: 310 s -> 217 s (incl. ~250 s -> ~16 s
    #     on the piecewise CUDA graph capture phase alone, a ~16×
    #     speedup on that phase).
    #   Qwen3-8B TQ3 startup:    1060 s -> 382 s (~2.8× overall; the
    #     remaining 382 s is dominated by model download + weight load
    #     + CUDA ext build, not capture).
    #
    # Tradeoff: the selected config is frozen at the first batch size
    # the kernel sees. For decoder-dominated workloads (small batch,
    # BLOCK_M=1 optimal) this is fine because the initial capture runs
    # at batch=1. For prefill-heavy workloads a coarser bucketed-key
    # pass may be worth revisiting.
    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_M": 1, "BLOCK_N": 128}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_M": 1, "BLOCK_N": 256}, num_warps=8, num_stages=2),
            triton.Config({"BLOCK_M": 4, "BLOCK_N": 128}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_M": 8, "BLOCK_N": 64}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_M": 8, "BLOCK_N": 128}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_M": 16, "BLOCK_N": 64}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_M": 16, "BLOCK_N": 128}, num_warps=8, num_stages=2),
            triton.Config({"BLOCK_M": 32, "BLOCK_N": 64}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_M": 32, "BLOCK_N": 128}, num_warps=8, num_stages=2),
            triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=2),
        ],
        key=["out_f", "in_f_padded"],
    )
    @triton.jit
    def _polar_fused_gemm_kernel(
        # Rotated input: (batch, in_f_padded), float32
        x_rot_ptr,
        stride_xm,
        stride_xk,
        # Packed codes: (out_f, packed_cols), uint8
        codes_ptr,
        stride_cn,
        stride_ck,
        # Per-group norms: (out_f, n_groups), float32
        norms_ptr,
        stride_nn,
        stride_ng,
        # Pre-scaled centroids: (n_centroids,), float32
        ct_ptr,
        # Output: (batch, out_f)
        out_ptr,
        stride_om,
        stride_on,
        # Bias: (out_f,) or None
        bias_ptr,
        # Dims
        batch_size,
        out_f,
        in_f_padded,
        n_groups,
        # Constexprs
        BLOCK_K: tl.constexpr,  # = group_size = 128
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
) -> torch.Tensor:
    """FWHT-on-input fused GEMM. Rotates input once, then codebook dot product.

    Faster than dequant-GEMM on wide output matmuls because:
    - FWHT applied to input (1 vector) not weights (N rows)
    - No intermediate decompressed weight buffer

    An earlier iteration of this module kept an FWHTInputCache that
    reused the rotated input across Q/K/V projections. A capture-safe
    version of that cache was tried and is preserved on the branch
    ``archive/fwht-input-cache-unmerged``. It is NOT wired in here
    because modern vLLM model implementations (Qwen3, Llama 3, Gemma
    families, most transformers ≥2024) use a **fused** ``qkv_proj``
    Linear that produces [Q|K|V] in one call. The cache's "same x
    across three calls" premise only holds on architectures with
    separate ``q_proj``/``k_proj``/``v_proj`` Linears — a shrinking
    minority — so it is zero-benefit on today's benchmarks. Revisit
    if such a model ever becomes a target.
    """
    if not HAS_TRITON:
        raise ImportError("Triton required")

    # Unconditional reshape for dynamo-fullgraph-compatibility — see
    # tq_fused_gemm above for the full explanation.
    orig_shape = x.shape
    x = x.reshape(-1, x.shape[-1])

    M, K = x.shape
    N = norms.shape[0]
    n_groups = norms.shape[1]

    padded_K = n_groups * group_size
    if K != padded_K:
        raise ValueError(f"K={K} not aligned with group_size={group_size}, expected {padded_K}")

    x_rot = rotate_input(x.float(), signs1, signs2, group_size)

    output = torch.empty(M, N, dtype=x.dtype, device=x.device)

    BLOCK_K = group_size
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]),
        triton.cdiv(N, META["BLOCK_N"]),
    )

    _polar_fused_gemm_kernel[grid](
        x_rot,
        x_rot.stride(0),
        x_rot.stride(1),
        packed_weight,
        packed_weight.stride(0),
        packed_weight.stride(1),
        norms,
        norms.stride(0),
        norms.stride(1),
        centroids,
        output,
        output.stride(0),
        output.stride(1),
        bias,
        M,
        N,
        padded_K,
        n_groups,
        BLOCK_K=BLOCK_K,
        BITS=bits,
    )

    if len(orig_shape) > 2:
        output = output.reshape(*orig_shape[:-1], N)

    return output


# vLLM 0.19 fullgraph-AOT-compiles TurboQuantWrapper.forward. The launcher
# functions above contain Python-side bookkeeping (id()-keyed rotation
# matrix cache, Triton kernel launches, runtime shape padding) that dynamo
# cannot trace. `allow_in_graph` is not enough here because dynamo still
# runs the wrapped function on
# FakeTensors during trace for shape inference, and Triton's kernel
# launcher calls .data_ptr() on the fakes — PyTorch itself directs us to
# the right fix in the resulting error message:
#
#   "Cannot access data pointer of Tensor (e.g. FakeTensor, ...). If
#    you're using torch.compile/export/fx, it is likely that we are
#    erroneously tracing into a custom kernel. To fix this, please wrap
#    the custom kernel into an opaque custom op."
#
# So we register both launchers as `torch.library.custom_op` with a
# FakeTensor (meta) implementation that returns an empty output of the
# correct shape/dtype. At trace time, dynamo uses the fake impl and never
# sees inside the real function. At runtime the real Python body runs as
# usual, including all its Triton kernel launches and host-side caches.
#
# We keep the original functions bound to module-level names so tests and
# eager-mode callers still see the same callables, but we rebind them to
# the custom-op wrappers after registration. Old imports by reference
# (`from ... import tq_fused_gemm`) done after module load pick up the
# wrapper automatically.
try:
    _tq_fused_gemm_impl = tq_fused_gemm
    _tq_fwht_input_gemm_impl = tq_fwht_input_gemm

    @torch.library.custom_op(
        "turboquant::tq_fused_gemm", mutates_args=(), device_types=("cuda",)
    )
    def _tq_fused_gemm_op(
        x: torch.Tensor,
        packed_weight: torch.Tensor,
        norms: torch.Tensor,
        signs1: torch.Tensor,
        signs2: torch.Tensor,
        centroids: torch.Tensor,
        group_size: int,
        bits: int,
        bias: torch.Tensor | None,
    ) -> torch.Tensor:
        return _tq_fused_gemm_impl(
            x, packed_weight, norms, signs1, signs2, centroids,
            group_size=group_size, bits=bits, bias=bias,
        )

    @_tq_fused_gemm_op.register_fake
    def _(x, packed_weight, norms, signs1, signs2, centroids, group_size, bits, bias):
        N = norms.shape[0]
        return x.new_empty((*x.shape[:-1], N))

    @torch.library.custom_op(
        "turboquant::tq_fwht_input_gemm", mutates_args=(), device_types=("cuda",)
    )
    def _tq_fwht_input_gemm_op(
        x: torch.Tensor,
        packed_weight: torch.Tensor,
        norms: torch.Tensor,
        signs1: torch.Tensor,
        signs2: torch.Tensor,
        centroids: torch.Tensor,
        group_size: int,
        bits: int,
        bias: torch.Tensor | None,
    ) -> torch.Tensor:
        return _tq_fwht_input_gemm_impl(
            x, packed_weight, norms, signs1, signs2, centroids,
            group_size=group_size, bits=bits, bias=bias,
        )

    @_tq_fwht_input_gemm_op.register_fake
    def _(x, packed_weight, norms, signs1, signs2, centroids, group_size, bits, bias):
        N = norms.shape[0]
        return x.new_empty((*x.shape[:-1], N))

    # Public callables that dynamo treats as opaque ops. Keyword args are
    # not supported across the custom_op boundary (schema is positional),
    # so we wrap to restore the original keyword-friendly call signature.
    def tq_fused_gemm(  # type: ignore[no-redef]
        x, packed_weight, norms, signs1, signs2, centroids,
        group_size=128, bits=4, bias=None,
    ):
        return torch.ops.turboquant.tq_fused_gemm(
            x, packed_weight, norms, signs1, signs2, centroids,
            group_size, bits, bias,
        )

    def tq_fwht_input_gemm(  # type: ignore[no-redef]
        x, packed_weight, norms, signs1, signs2, centroids,
        group_size=128, bits=4, bias=None,
    ):
        return torch.ops.turboquant.tq_fwht_input_gemm(
            x, packed_weight, norms, signs1, signs2, centroids,
            group_size, bits, bias,
        )

except (AttributeError, RuntimeError):
    # torch < 2.4 or torch.library.custom_op unavailable — fall through to
    # the plain Python functions. Eager mode still works; fullgraph compile
    # will re-surface the underlying tracing error.
    pass


