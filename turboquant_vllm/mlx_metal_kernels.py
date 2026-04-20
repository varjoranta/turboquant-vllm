# SPDX-License-Identifier: MIT
"""Custom Metal kernels for HIGGS-scalar 3-bit MLX inference.

The hot kernel is a bs=1 GEMV that fuses 3-bit unpack + codebook lookup +
norm scaling + matmul into one pass. Layout follows the CUDA design from
``vllm-1/.../turboquant_gemv/kernel.cu`` (warp-per-output-channel) adapted
to Metal's 32-wide SIMD-group + threadgroup-memory codebook stage.

Inputs (per call):
    x_rot:    (K,)               half   — pre-rotated activation (FWHT applied)
    packed:   (OC * n_groups, 48) uint8 — TQ3 sub-byte 3-bit pack
    norms:    (OC, n_groups)     half   — per-group L2 scale
    codebook: (8,)               half   — HIGGS Lloyd-Max grid

Output:
    out:      (OC,)              half

Pack format (must match weight_quant.pack_indices for bits=3):
    48 bytes per 128-value group, 16 × 3-byte triplets with cross-byte
    indices at positions 2 and 5.
"""

from __future__ import annotations

from functools import lru_cache
import mlx.core as mx


_GEMV_SOURCE = """
    // Stage 8-entry HIGGS codebook into threadgroup memory once per group.
    threadgroup half cb_lut[8];
    uint tid = thread_position_in_threadgroup.x;
    if (tid < 8u) {
        cb_lut[tid] = codebook[tid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint oc       = thread_position_in_grid.y;
    uint lane     = thread_position_in_grid.x;
    uint OC_v     = OC[0];
    uint n_groups_v = n_groups[0];

    if (oc >= OC_v) return;

    float psum = 0.0f;

    for (uint g = lane; g < n_groups_v; g += 32u) {
        device const uint8_t* grp = packed + (oc * n_groups_v + g) * 48u;
        float norm = float(norms[oc * n_groups_v + g]);
        device const half* x_chunk = x_rot + g * 128u;

        // 4 chunks per group, 12 bytes per chunk → 32 indices per chunk.
        for (uint c = 0u; c < 4u; ++c) {
            device const uint8_t* chunk = grp + c * 12u;
            device const half* xc = x_chunk + c * 32u;

            for (uint t = 0u; t < 4u; ++t) {
                uint b0 = (uint)chunk[t * 3u + 0u];
                uint b1 = (uint)chunk[t * 3u + 1u];
                uint b2 = (uint)chunk[t * 3u + 2u];

                uint i0 =  b0                  & 0x7u;
                uint i1 = (b0 >> 3)            & 0x7u;
                uint i2 = ((b0 >> 6) | (b1 << 2)) & 0x7u;
                uint i3 = (b1 >> 1)            & 0x7u;
                uint i4 = (b1 >> 4)            & 0x7u;
                uint i5 = ((b1 >> 7) | (b2 << 1)) & 0x7u;
                uint i6 = (b2 >> 2)            & 0x7u;
                uint i7 = (b2 >> 5)            & 0x7u;

                uint base = t * 8u;
                psum += float(cb_lut[i0]) * norm * float(xc[base + 0u]);
                psum += float(cb_lut[i1]) * norm * float(xc[base + 1u]);
                psum += float(cb_lut[i2]) * norm * float(xc[base + 2u]);
                psum += float(cb_lut[i3]) * norm * float(xc[base + 3u]);
                psum += float(cb_lut[i4]) * norm * float(xc[base + 4u]);
                psum += float(cb_lut[i5]) * norm * float(xc[base + 5u]);
                psum += float(cb_lut[i6]) * norm * float(xc[base + 6u]);
                psum += float(cb_lut[i7]) * norm * float(xc[base + 7u]);
            }
        }
    }

    psum = simd_sum(psum);
    if (lane == 0u) {
        out[oc] = (half)psum;
    }
"""


@lru_cache(maxsize=1)
def _gemv_kernel():
    return mx.fast.metal_kernel(
        name="tq3_gemv_bs1_mlx",
        input_names=["x_rot", "packed", "norms", "codebook", "OC", "n_groups"],
        output_names=["out"],
        source=_GEMV_SOURCE,
    )


# MoE batched variant: same x_rot, but per-expert (packed, norms).
# Grid z dim selects the active expert slot. Output shape (K_active, OC).
_GEMV_BATCHED_SOURCE = """
    threadgroup half cb_lut[8];
    uint tid = thread_position_in_threadgroup.x;
    if (tid < 8u) {
        cb_lut[tid] = codebook[tid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint oc          = thread_position_in_grid.y;
    uint k_active    = thread_position_in_grid.z;
    uint lane        = thread_position_in_grid.x;
    uint OC_v        = OC[0];
    uint n_groups_v  = n_groups[0];
    uint K_active_v  = K_active[0];

    if (oc >= OC_v || k_active >= K_active_v) return;

    // Per-expert offsets into packed and norms.
    device const uint8_t* packed_e = packed + k_active * OC_v * n_groups_v * 48u;
    device const half*    norms_e  = norms  + k_active * OC_v * n_groups_v;

    float psum = 0.0f;

    for (uint g = lane; g < n_groups_v; g += 32u) {
        device const uint8_t* grp = packed_e + (oc * n_groups_v + g) * 48u;
        float norm = float(norms_e[oc * n_groups_v + g]);
        device const half* x_chunk = x_rot + g * 128u;

        for (uint c = 0u; c < 4u; ++c) {
            device const uint8_t* chunk = grp + c * 12u;
            device const half* xc = x_chunk + c * 32u;
            for (uint t = 0u; t < 4u; ++t) {
                uint b0 = (uint)chunk[t * 3u + 0u];
                uint b1 = (uint)chunk[t * 3u + 1u];
                uint b2 = (uint)chunk[t * 3u + 2u];
                uint i0 =  b0                  & 0x7u;
                uint i1 = (b0 >> 3)            & 0x7u;
                uint i2 = ((b0 >> 6) | (b1 << 2)) & 0x7u;
                uint i3 = (b1 >> 1)            & 0x7u;
                uint i4 = (b1 >> 4)            & 0x7u;
                uint i5 = ((b1 >> 7) | (b2 << 1)) & 0x7u;
                uint i6 = (b2 >> 2)            & 0x7u;
                uint i7 = (b2 >> 5)            & 0x7u;
                uint base = t * 8u;
                psum += float(cb_lut[i0]) * norm * float(xc[base + 0u]);
                psum += float(cb_lut[i1]) * norm * float(xc[base + 1u]);
                psum += float(cb_lut[i2]) * norm * float(xc[base + 2u]);
                psum += float(cb_lut[i3]) * norm * float(xc[base + 3u]);
                psum += float(cb_lut[i4]) * norm * float(xc[base + 4u]);
                psum += float(cb_lut[i5]) * norm * float(xc[base + 5u]);
                psum += float(cb_lut[i6]) * norm * float(xc[base + 6u]);
                psum += float(cb_lut[i7]) * norm * float(xc[base + 7u]);
            }
        }
    }

    psum = simd_sum(psum);
    if (lane == 0u) {
        out[k_active * OC_v + oc] = (half)psum;
    }
"""


@lru_cache(maxsize=1)
def _gemv_batched_kernel():
    return mx.fast.metal_kernel(
        name="tq3_gemv_bs1_batched_mlx",
        input_names=["x_rot", "packed", "norms", "codebook", "OC", "n_groups", "K_active"],
        output_names=["out"],
        source=_GEMV_BATCHED_SOURCE,
    )


# Per-expert-x batched variant: x_rot shape (K_active, K). Used by MoE
# down_proj where each active expert sees a different activation
# (gate-up output multiplied by gating).
_GEMV_BATCHED_PER_X_SOURCE = """
    threadgroup half cb_lut[8];
    uint tid = thread_position_in_threadgroup.x;
    if (tid < 8u) {
        cb_lut[tid] = codebook[tid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint oc          = thread_position_in_grid.y;
    uint k_active    = thread_position_in_grid.z;
    uint lane        = thread_position_in_grid.x;
    uint OC_v        = OC[0];
    uint n_groups_v  = n_groups[0];
    uint K_active_v  = K_active[0];
    uint K_v         = n_groups_v * 128u;

    if (oc >= OC_v || k_active >= K_active_v) return;

    device const uint8_t* packed_e = packed + k_active * OC_v * n_groups_v * 48u;
    device const half*    norms_e  = norms  + k_active * OC_v * n_groups_v;
    device const half*    x_e      = x_rot  + k_active * K_v;

    float psum = 0.0f;

    for (uint g = lane; g < n_groups_v; g += 32u) {
        device const uint8_t* grp = packed_e + (oc * n_groups_v + g) * 48u;
        float norm = float(norms_e[oc * n_groups_v + g]);
        device const half* x_chunk = x_e + g * 128u;

        for (uint c = 0u; c < 4u; ++c) {
            device const uint8_t* chunk = grp + c * 12u;
            device const half* xc = x_chunk + c * 32u;
            for (uint t = 0u; t < 4u; ++t) {
                uint b0 = (uint)chunk[t * 3u + 0u];
                uint b1 = (uint)chunk[t * 3u + 1u];
                uint b2 = (uint)chunk[t * 3u + 2u];
                uint i0 =  b0                  & 0x7u;
                uint i1 = (b0 >> 3)            & 0x7u;
                uint i2 = ((b0 >> 6) | (b1 << 2)) & 0x7u;
                uint i3 = (b1 >> 1)            & 0x7u;
                uint i4 = (b1 >> 4)            & 0x7u;
                uint i5 = ((b1 >> 7) | (b2 << 1)) & 0x7u;
                uint i6 = (b2 >> 2)            & 0x7u;
                uint i7 = (b2 >> 5)            & 0x7u;
                uint base = t * 8u;
                psum += float(cb_lut[i0]) * norm * float(xc[base + 0u]);
                psum += float(cb_lut[i1]) * norm * float(xc[base + 1u]);
                psum += float(cb_lut[i2]) * norm * float(xc[base + 2u]);
                psum += float(cb_lut[i3]) * norm * float(xc[base + 3u]);
                psum += float(cb_lut[i4]) * norm * float(xc[base + 4u]);
                psum += float(cb_lut[i5]) * norm * float(xc[base + 5u]);
                psum += float(cb_lut[i6]) * norm * float(xc[base + 6u]);
                psum += float(cb_lut[i7]) * norm * float(xc[base + 7u]);
            }
        }
    }

    psum = simd_sum(psum);
    if (lane == 0u) {
        out[k_active * OC_v + oc] = (half)psum;
    }
"""


@lru_cache(maxsize=1)
def _gemv_batched_per_x_kernel():
    return mx.fast.metal_kernel(
        name="tq3_gemv_bs1_batched_per_x_mlx",
        input_names=["x_rot", "packed", "norms", "codebook", "OC", "n_groups", "K_active"],
        output_names=["out"],
        source=_GEMV_BATCHED_PER_X_SOURCE,
    )


def tq3_gemv_bs1_batched_per_x_mlx(
    x_rot: mx.array,
    packed: mx.array,
    norms: mx.array,
    codebook: mx.array,
) -> mx.array:
    """bs=1 GEMV for K_active experts, each with its own x_rot.

    Args:
        x_rot:    (K_active, K) float16
        packed:   (K_active, OC * n_groups, 48) uint8
        norms:    (K_active, OC, n_groups) float16
        codebook: (8,) float16

    Returns:
        (K_active, OC) float16
    """
    assert x_rot.dtype == mx.float16 and x_rot.ndim == 2
    assert packed.dtype == mx.uint8 and packed.ndim == 3 and packed.shape[2] == 48
    assert norms.dtype == mx.float16 and norms.ndim == 3
    assert codebook.dtype == mx.float16 and codebook.size == 8

    K_active = x_rot.shape[0]
    K = x_rot.shape[1]
    OC = norms.shape[1]
    n_groups = norms.shape[2]
    assert K == n_groups * 128
    assert norms.shape[0] == K_active
    assert packed.shape[0] == K_active and packed.shape[1] == OC * n_groups

    OC_arg = mx.array([OC], dtype=mx.uint32)
    ng_arg = mx.array([n_groups], dtype=mx.uint32)
    ka_arg = mx.array([K_active], dtype=mx.uint32)

    kernel = _gemv_batched_per_x_kernel()
    out = kernel(
        inputs=[x_rot, packed, norms, codebook, OC_arg, ng_arg, ka_arg],
        grid=(32, OC, K_active),
        threadgroup=(32, 1, 1),
        output_shapes=[(K_active, OC)],
        output_dtypes=[mx.float16],
    )[0]
    return out


def tq3_gemv_bs1_batched_mlx(
    x_rot: mx.array,
    packed: mx.array,
    norms: mx.array,
    codebook: mx.array,
) -> mx.array:
    """bs=1 GEMV for K_active experts, same x_rot.

    Args:
        x_rot:    (K,) float16 — same activation broadcast to all experts
        packed:   (K_active, OC * n_groups, 48) uint8
        norms:    (K_active, OC, n_groups) float16
        codebook: (8,) float16

    Returns:
        (K_active, OC) float16
    """
    assert x_rot.dtype == mx.float16
    assert packed.dtype == mx.uint8 and packed.ndim == 3 and packed.shape[2] == 48
    assert norms.dtype == mx.float16 and norms.ndim == 3
    assert codebook.dtype == mx.float16 and codebook.size == 8

    K_active = norms.shape[0]
    OC = norms.shape[1]
    n_groups = norms.shape[2]
    assert x_rot.size == n_groups * 128
    assert packed.shape[0] == K_active and packed.shape[1] == OC * n_groups

    OC_arg = mx.array([OC], dtype=mx.uint32)
    ng_arg = mx.array([n_groups], dtype=mx.uint32)
    ka_arg = mx.array([K_active], dtype=mx.uint32)

    kernel = _gemv_batched_kernel()
    out = kernel(
        inputs=[x_rot, packed, norms, codebook, OC_arg, ng_arg, ka_arg],
        grid=(32, OC, K_active),
        threadgroup=(32, 1, 1),
        output_shapes=[(K_active, OC)],
        output_dtypes=[mx.float16],
    )[0]
    return out


def tq3_gemv_bs1_mlx(
    x_rot: mx.array,
    packed: mx.array,
    norms: mx.array,
    codebook: mx.array,
) -> mx.array:
    """bs=1 GEMV via custom Metal kernel.

    Args:
        x_rot:    (K,) float16, K = n_groups * 128, FWHT-rotated
        packed:   (OC * n_groups, 48) uint8
        norms:    (OC, n_groups) float16
        codebook: (8,) float16

    Returns:
        (OC,) float16
    """
    assert x_rot.dtype == mx.float16, f"x_rot must be float16, got {x_rot.dtype}"
    assert packed.dtype == mx.uint8 and packed.ndim == 2 and packed.shape[1] == 48
    assert norms.dtype == mx.float16 and norms.ndim == 2
    assert codebook.dtype == mx.float16 and codebook.size == 8

    OC = norms.shape[0]
    n_groups = norms.shape[1]
    assert x_rot.size == n_groups * 128, f"K={x_rot.size} != n_groups*128={n_groups*128}"
    assert packed.shape[0] == OC * n_groups, (
        f"packed.shape[0]={packed.shape[0]} != OC*n_groups={OC * n_groups}"
    )

    OC_arg = mx.array([OC], dtype=mx.uint32)
    ng_arg = mx.array([n_groups], dtype=mx.uint32)

    kernel = _gemv_kernel()
    out = kernel(
        inputs=[x_rot, packed, norms, codebook, OC_arg, ng_arg],
        grid=(32, OC, 1),         # one SIMD-group per output channel
        threadgroup=(32, 1, 1),   # one SIMD-group per threadgroup
        output_shapes=[(OC,)],
        output_dtypes=[mx.float16],
    )[0]
    return out


# ---------------------------------------------------------------------------
# TQ4 bs=1 GEMV kernels (4-bit packing: 2 values per byte, 64 bytes/group)
# ---------------------------------------------------------------------------
#
# Same warp-per-output-channel design as the TQ3 kernels. Differences:
#   - 16-entry codebook (4 bits) staged to threadgroup memory
#   - 64 bytes per 128-element group (2 indices/byte nibble layout)
#   - No cross-byte unpack complexity (vs TQ3's 8-in-3-byte packing)
#   - Chunked as 4 x 16 bytes = 64 bytes per group, each chunk supplies
#     32 indices that cover 32 x activation values
# Kernel bodies are intentionally kept close to the TQ3 versions so a
# future unified TQN kernel (with bits as a template constant) is a
# small diff away.


_GEMV4_SOURCE = """
    threadgroup half cb_lut[16];
    uint tid = thread_position_in_threadgroup.x;
    if (tid < 16u) {
        cb_lut[tid] = codebook[tid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint oc         = thread_position_in_grid.y;
    uint lane       = thread_position_in_grid.x;
    uint OC_v       = OC[0];
    uint n_groups_v = n_groups[0];

    if (oc >= OC_v) return;

    float psum = 0.0f;

    for (uint g = lane; g < n_groups_v; g += 32u) {
        device const uint8_t* grp = packed + (oc * n_groups_v + g) * 64u;
        float norm = float(norms[oc * n_groups_v + g]);
        device const half* x_chunk = x_rot + g * 128u;

        // 4 chunks × 16 bytes each. Each byte gives 2 indices (lo, hi).
        for (uint c = 0u; c < 4u; ++c) {
            device const uint8_t* chunk = grp + c * 16u;
            device const half* xc = x_chunk + c * 32u;
            for (uint b = 0u; b < 16u; ++b) {
                uint byte = (uint)chunk[b];
                uint i_lo =  byte        & 0xFu;
                uint i_hi = (byte >> 4)  & 0xFu;
                uint base = b * 2u;
                psum += float(cb_lut[i_lo]) * norm * float(xc[base + 0u]);
                psum += float(cb_lut[i_hi]) * norm * float(xc[base + 1u]);
            }
        }
    }

    psum = simd_sum(psum);
    if (lane == 0u) {
        out[oc] = (half)psum;
    }
"""


@lru_cache(maxsize=1)
def _gemv4_kernel():
    return mx.fast.metal_kernel(
        name="tq4_gemv_bs1_mlx",
        input_names=["x_rot", "packed", "norms", "codebook", "OC", "n_groups"],
        output_names=["out"],
        source=_GEMV4_SOURCE,
    )


def tq4_gemv_bs1_mlx(
    x_rot: mx.array,
    packed: mx.array,
    norms: mx.array,
    codebook: mx.array,
) -> mx.array:
    """bs=1 GEMV for TQ4 Linear. Same interface as tq3_gemv_bs1_mlx except
    packed.shape[1] == 64 (not 48) and codebook.size == 16 (not 8)."""
    assert x_rot.dtype == mx.float16, f"x_rot must be float16, got {x_rot.dtype}"
    assert packed.dtype == mx.uint8 and packed.ndim == 2 and packed.shape[1] == 64
    assert norms.dtype == mx.float16 and norms.ndim == 2
    assert codebook.dtype == mx.float16 and codebook.size == 16

    OC = norms.shape[0]
    n_groups = norms.shape[1]
    assert x_rot.size == n_groups * 128

    OC_arg = mx.array([OC], dtype=mx.uint32)
    ng_arg = mx.array([n_groups], dtype=mx.uint32)
    kernel = _gemv4_kernel()
    out = kernel(
        inputs=[x_rot, packed, norms, codebook, OC_arg, ng_arg],
        grid=(32, OC, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(OC,)],
        output_dtypes=[mx.float16],
    )[0]
    return out


_GEMV4_BATCHED_SOURCE = """
    threadgroup half cb_lut[16];
    uint tid = thread_position_in_threadgroup.x;
    if (tid < 16u) {
        cb_lut[tid] = codebook[tid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint oc         = thread_position_in_grid.y;
    uint k_active   = thread_position_in_grid.z;
    uint lane       = thread_position_in_grid.x;
    uint OC_v       = OC[0];
    uint n_groups_v = n_groups[0];
    uint K_active_v = K_active[0];

    if (oc >= OC_v || k_active >= K_active_v) return;

    device const uint8_t* packed_e = packed + k_active * OC_v * n_groups_v * 64u;
    device const half*    norms_e  = norms  + k_active * OC_v * n_groups_v;

    float psum = 0.0f;

    for (uint g = lane; g < n_groups_v; g += 32u) {
        device const uint8_t* grp = packed_e + (oc * n_groups_v + g) * 64u;
        float norm = float(norms_e[oc * n_groups_v + g]);
        device const half* x_chunk = x_rot + g * 128u;

        for (uint c = 0u; c < 4u; ++c) {
            device const uint8_t* chunk = grp + c * 16u;
            device const half* xc = x_chunk + c * 32u;
            for (uint b = 0u; b < 16u; ++b) {
                uint byte = (uint)chunk[b];
                uint i_lo =  byte        & 0xFu;
                uint i_hi = (byte >> 4)  & 0xFu;
                uint base = b * 2u;
                psum += float(cb_lut[i_lo]) * norm * float(xc[base + 0u]);
                psum += float(cb_lut[i_hi]) * norm * float(xc[base + 1u]);
            }
        }
    }

    psum = simd_sum(psum);
    if (lane == 0u) {
        out[k_active * OC_v + oc] = (half)psum;
    }
"""


@lru_cache(maxsize=1)
def _gemv4_batched_kernel():
    return mx.fast.metal_kernel(
        name="tq4_gemv_bs1_batched_mlx",
        input_names=["x_rot", "packed", "norms", "codebook", "OC", "n_groups", "K_active"],
        output_names=["out"],
        source=_GEMV4_BATCHED_SOURCE,
    )


def tq4_gemv_bs1_batched_mlx(
    x_rot: mx.array,
    packed: mx.array,
    norms: mx.array,
    codebook: mx.array,
) -> mx.array:
    """bs=1 GEMV for K_active experts sharing the same x_rot, TQ4 layout."""
    assert x_rot.dtype == mx.float16
    assert packed.dtype == mx.uint8 and packed.ndim == 3 and packed.shape[2] == 64
    assert norms.dtype == mx.float16 and norms.ndim == 3
    assert codebook.dtype == mx.float16 and codebook.size == 16

    K_active = norms.shape[0]
    OC = norms.shape[1]
    n_groups = norms.shape[2]
    assert x_rot.size == n_groups * 128
    assert packed.shape[0] == K_active and packed.shape[1] == OC * n_groups

    OC_arg = mx.array([OC], dtype=mx.uint32)
    ng_arg = mx.array([n_groups], dtype=mx.uint32)
    ka_arg = mx.array([K_active], dtype=mx.uint32)
    kernel = _gemv4_batched_kernel()
    out = kernel(
        inputs=[x_rot, packed, norms, codebook, OC_arg, ng_arg, ka_arg],
        grid=(32, OC, K_active),
        threadgroup=(32, 1, 1),
        output_shapes=[(K_active, OC)],
        output_dtypes=[mx.float16],
    )[0]
    return out


_GEMV4_BATCHED_PER_X_SOURCE = """
    threadgroup half cb_lut[16];
    uint tid = thread_position_in_threadgroup.x;
    if (tid < 16u) {
        cb_lut[tid] = codebook[tid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint oc         = thread_position_in_grid.y;
    uint k_active   = thread_position_in_grid.z;
    uint lane       = thread_position_in_grid.x;
    uint OC_v       = OC[0];
    uint n_groups_v = n_groups[0];
    uint K_active_v = K_active[0];
    uint K_v        = n_groups_v * 128u;

    if (oc >= OC_v || k_active >= K_active_v) return;

    device const uint8_t* packed_e = packed + k_active * OC_v * n_groups_v * 64u;
    device const half*    norms_e  = norms  + k_active * OC_v * n_groups_v;
    device const half*    x_e      = x_rot  + k_active * K_v;

    float psum = 0.0f;

    for (uint g = lane; g < n_groups_v; g += 32u) {
        device const uint8_t* grp = packed_e + (oc * n_groups_v + g) * 64u;
        float norm = float(norms_e[oc * n_groups_v + g]);
        device const half* x_chunk = x_e + g * 128u;

        for (uint c = 0u; c < 4u; ++c) {
            device const uint8_t* chunk = grp + c * 16u;
            device const half* xc = x_chunk + c * 32u;
            for (uint b = 0u; b < 16u; ++b) {
                uint byte = (uint)chunk[b];
                uint i_lo =  byte        & 0xFu;
                uint i_hi = (byte >> 4)  & 0xFu;
                uint base = b * 2u;
                psum += float(cb_lut[i_lo]) * norm * float(xc[base + 0u]);
                psum += float(cb_lut[i_hi]) * norm * float(xc[base + 1u]);
            }
        }
    }

    psum = simd_sum(psum);
    if (lane == 0u) {
        out[k_active * OC_v + oc] = (half)psum;
    }
"""


@lru_cache(maxsize=1)
def _gemv4_batched_per_x_kernel():
    return mx.fast.metal_kernel(
        name="tq4_gemv_bs1_batched_per_x_mlx",
        input_names=["x_rot", "packed", "norms", "codebook", "OC", "n_groups", "K_active"],
        output_names=["out"],
        source=_GEMV4_BATCHED_PER_X_SOURCE,
    )


def tq4_gemv_bs1_batched_per_x_mlx(
    x_rot: mx.array,
    packed: mx.array,
    norms: mx.array,
    codebook: mx.array,
) -> mx.array:
    """bs=1 GEMV for K_active experts with per-expert x, TQ4 layout."""
    assert x_rot.dtype == mx.float16 and x_rot.ndim == 2
    assert packed.dtype == mx.uint8 and packed.ndim == 3 and packed.shape[2] == 64
    assert norms.dtype == mx.float16 and norms.ndim == 3
    assert codebook.dtype == mx.float16 and codebook.size == 16

    K_active = x_rot.shape[0]
    K = x_rot.shape[1]
    OC = norms.shape[1]
    n_groups = norms.shape[2]
    assert K == n_groups * 128

    OC_arg = mx.array([OC], dtype=mx.uint32)
    ng_arg = mx.array([n_groups], dtype=mx.uint32)
    ka_arg = mx.array([K_active], dtype=mx.uint32)
    kernel = _gemv4_batched_per_x_kernel()
    out = kernel(
        inputs=[x_rot, packed, norms, codebook, OC_arg, ng_arg, ka_arg],
        grid=(32, OC, K_active),
        threadgroup=(32, 1, 1),
        output_shapes=[(K_active, OC)],
        output_dtypes=[mx.float16],
    )[0]
    return out
