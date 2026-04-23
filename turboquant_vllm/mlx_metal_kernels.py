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


# v2: register-pattern rewrite matching MLX's `qmv_fast_impl` — codebook in
# per-thread registers (was threadgroup memory), results_per_simdgroup=4,
# block_size=512. Details in .plans/tq3-metal-kernel-register-rewrite.md.
# Requires OC % 8 == 0 (true for all quantized tensors in Qwen3.6-35B-A3B).
_GEMV_V2_SOURCE = """
    // --- Threadgroup / simdgroup positions
    uint tid_y      = threadgroup_position_in_grid.y;   // row-group index [0, OC/8)
    uint simd_gid   = simdgroup_index_in_threadgroup;   // 0 or 1
    uint simd_lid   = thread_index_in_simdgroup;        // 0..31

    // --- Compile-time constants (match MLX qmv_fast_impl for bits=3)
    constexpr int pack_factor           = 8;
    constexpr int bytes_per_pack        = 3;
    constexpr int packs_per_thread      = 2;
    constexpr int values_per_thread     = 16;   // packs_per_thread * pack_factor
    constexpr int num_simdgroups        = 2;
    constexpr int results_per_simdgroup = 4;
    constexpr int block_size            = 512;  // values_per_thread * 32
    constexpr int group_size            = 128;

    uint OC_v       = OC[0];
    uint n_groups_v = n_groups[0];
    uint K_v        = n_groups_v * group_size;
    uint in_vec_size_w = K_v * (uint)bytes_per_pack / (uint)pack_factor;  // = K*3/8

    // --- Output row this simdgroup's 4 results start at
    uint out_row = tid_y * (num_simdgroups * results_per_simdgroup)
                 + simd_gid * results_per_simdgroup;  // +0 for simd 0, +4 for simd 1

    // --- Codebook in thread-local array (Metal will place in registers
    // when indices fit; spills to local memory for runtime-index fallback).
    // Ternary cascade alternative benchmarked slower on M4 Pro (~2× overhead).
    float cb_reg[8];
    for (int i = 0; i < 8; i++) cb_reg[i] = float(codebook[i]);

    // --- Thread-local x and per-row accumulators
    float x_thread[values_per_thread];
    float result[results_per_simdgroup];
    for (int r = 0; r < results_per_simdgroup; r++) result[r] = 0.0f;

    // --- Base pointers for this thread
    // packed:  (OC, K*3/8) row-major, thread reads 6 bytes per iter at offset simd_lid*6
    // norms:   (OC, n_groups) row-major, thread reads norm for its group (simd_lid/8)
    // x_rot:   (K,), thread reads 16 values starting at simd_lid*16
    device const uint8_t* ws = packed + out_row * in_vec_size_w + simd_lid * (uint)(packs_per_thread * bytes_per_pack);
    device const half*    sp = norms  + out_row * n_groups_v + (simd_lid / 8);
    device const half*    xp = x_rot  + simd_lid * (uint)values_per_thread;

    // --- Main loop: iterate K in blocks of block_size=512 weights = 4 groups
    uint n_iters = K_v / (uint)block_size;
    for (uint it = 0; it < n_iters; it++) {
        // Load 16 x values into registers
        for (int i = 0; i < values_per_thread; i++) x_thread[i] = float(xp[i]);

        // 4 output rows: each reads 6 bytes from its row and accumulates
        for (int row = 0; row < results_per_simdgroup; row++) {
            device const uint8_t* wl = ws + row * in_vec_size_w;
            float norm_val = float(sp[row * n_groups_v]);

            float accum = 0.0f;
            for (int p = 0; p < packs_per_thread; p++) {
                uint b0 = (uint)wl[p * 3 + 0];
                uint b1 = (uint)wl[p * 3 + 1];
                uint b2 = (uint)wl[p * 3 + 2];
                uint i0 =  b0                      & 0x7u;
                uint i1 = (b0 >> 3)                & 0x7u;
                uint i2 = ((b0 >> 6) | (b1 << 2))  & 0x7u;
                uint i3 = (b1 >> 1)                & 0x7u;
                uint i4 = (b1 >> 4)                & 0x7u;
                uint i5 = ((b1 >> 7) | (b2 << 1))  & 0x7u;
                uint i6 = (b2 >> 2)                & 0x7u;
                uint i7 = (b2 >> 5)                & 0x7u;

                int base = p * 8;
                accum += cb_reg[i0] * x_thread[base + 0];
                accum += cb_reg[i1] * x_thread[base + 1];
                accum += cb_reg[i2] * x_thread[base + 2];
                accum += cb_reg[i3] * x_thread[base + 3];
                accum += cb_reg[i4] * x_thread[base + 4];
                accum += cb_reg[i5] * x_thread[base + 5];
                accum += cb_reg[i6] * x_thread[base + 6];
                accum += cb_reg[i7] * x_thread[base + 7];
            }
            result[row] += norm_val * accum;
        }

        // Advance pointers to next block
        ws += (uint)block_size * (uint)bytes_per_pack / (uint)pack_factor;  // +192
        sp += (uint)block_size / (uint)group_size;                          // +4
        xp += (uint)block_size;                                             // +512
    }

    // --- simd_sum reduction + output (first lane per simdgroup writes)
    for (int row = 0; row < results_per_simdgroup; row++) {
        float r = simd_sum(result[row]);
        if (simd_lid == 0u) {
            out[out_row + row] = (half)r;
        }
    }
"""


@lru_cache(maxsize=1)
def _gemv_v2_kernel():
    return mx.fast.metal_kernel(
        name="tq3_gemv_bs1_mlx_v2",
        input_names=["x_rot", "packed", "norms", "codebook", "OC", "n_groups"],
        output_names=["out"],
        source=_GEMV_V2_SOURCE,
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


# Fused-gather MoE variant: takes the full per-expert packed tensor +
# the active-expert index array, gathers inside the kernel instead of
# requiring the caller to do mx.take() first. Saves ~400 µs sync-cost
# per MoE switch call on Qwen3.6-35B-A3B (3 calls × 40 layers = ~50 ms
# sync-cost shaved per decode step; real wall save ~15-25 ms after
# MLX's async pipelining reclaims part of it).
_GEMV_MOE_FUSED_SOURCE = """
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

    // Gather inside the kernel: look up this slot's expert id, then
    // offset into the full (num_experts, OC*n_groups, 48) tensor.
    uint expert_id = indices[k_active];
    device const uint8_t* packed_e = packed + expert_id * OC_v * n_groups_v * 48u;
    device const half*    norms_e  = norms  + expert_id * OC_v * n_groups_v;

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
def _gemv_moe_fused_kernel():
    return mx.fast.metal_kernel(
        name="tq3_gemv_bs1_moe_fused_mlx",
        input_names=["x_rot", "packed", "norms", "codebook", "indices", "OC", "n_groups", "K_active"],
        output_names=["out"],
        source=_GEMV_MOE_FUSED_SOURCE,
    )


def tq3_gemv_bs1_moe_fused_mlx(
    x_rot: mx.array,
    packed_per_expert: mx.array,  # (num_experts, OC*n_groups, 48) uint8
    norms: mx.array,  # (num_experts, OC, n_groups) or (num_experts, OC*n_groups) half
    codebook: mx.array,
    indices: mx.array,  # (K_active,) uint32
) -> mx.array:
    """TQ3 MoE GEMV that gathers active experts inside the kernel.

    Replaces the two-step ``mx.take(packed) + mx.take(norms) + batched_gemv``
    with a single kernel launch that indexes directly into the per-expert
    stacks, saving ~400 µs sync-measured per MoE switch call on
    Qwen3.6-35B-A3B. Returns shape ``(K_active, OC)`` float16.
    """
    assert x_rot.dtype == mx.float16
    assert packed_per_expert.dtype == mx.uint8 and packed_per_expert.ndim == 3
    assert packed_per_expert.shape[-1] == 48
    assert norms.dtype == mx.float16
    assert codebook.dtype == mx.float16 and codebook.size == 8
    assert indices.dtype == mx.uint32 and indices.ndim == 1

    num_experts = packed_per_expert.shape[0]
    # OC * n_groups inferred from second dim; n_groups from x_rot size
    K_active = indices.size
    n_groups = x_rot.size // 128
    OC_flat = packed_per_expert.shape[1]
    OC = OC_flat // n_groups
    assert OC * n_groups == OC_flat, f"shape mismatch: {OC_flat} / {n_groups}"

    # Normalize norms to flat (num_experts, OC*n_groups) for kernel
    norms_flat = norms.reshape(num_experts, OC_flat) if norms.ndim == 3 else norms

    OC_arg = mx.array([OC], dtype=mx.uint32)
    ng_arg = mx.array([n_groups], dtype=mx.uint32)
    ka_arg = mx.array([K_active], dtype=mx.uint32)

    kernel = _gemv_moe_fused_kernel()
    out = kernel(
        inputs=[x_rot, packed_per_expert, norms_flat, codebook, indices, OC_arg, ng_arg, ka_arg],
        grid=(32, OC, K_active),
        threadgroup=(32, 1, 1),
        output_shapes=[(K_active * OC,)],
        output_dtypes=[mx.float16],
    )[0]
    return out.reshape(K_active, OC)


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
    assert x_rot.size == n_groups * 128, f"K={x_rot.size} != n_groups*128={n_groups * 128}"
    assert packed.shape[0] == OC * n_groups, f"packed.shape[0]={packed.shape[0]} != OC*n_groups={OC * n_groups}"

    OC_arg = mx.array([OC], dtype=mx.uint32)
    ng_arg = mx.array([n_groups], dtype=mx.uint32)

    kernel = _gemv_kernel()
    out = kernel(
        inputs=[x_rot, packed, norms, codebook, OC_arg, ng_arg],
        grid=(32, OC, 1),  # one SIMD-group per output channel
        threadgroup=(32, 1, 1),  # one SIMD-group per threadgroup
        output_shapes=[(OC,)],
        output_dtypes=[mx.float16],
    )[0]
    return out


def tq3_gemv_bs1_mlx_v2(
    x_rot: mx.array,
    packed: mx.array,
    norms: mx.array,
    codebook: mx.array,
) -> mx.array:
    """Register-pattern bs=1 GEMV — port of MLX's qmv_fast_impl shape.

    Requires ``OC % 8 == 0`` (8 outputs per threadgroup, 4 per simdgroup).
    Takes the same row-major packed layout ``(OC * n_groups, 48)`` that
    the v1 kernel uses — repacking happens implicitly via pointer math
    (each output row of 48*n_groups bytes is contiguous, which is what
    the v2 kernel walks).
    """
    assert x_rot.dtype == mx.float16
    assert packed.dtype == mx.uint8 and packed.ndim == 2 and packed.shape[1] == 48
    assert norms.dtype == mx.float16 and norms.ndim == 2
    assert codebook.dtype == mx.float16 and codebook.size == 8

    OC = norms.shape[0]
    n_groups = norms.shape[1]
    assert OC % 8 == 0, f"v2 kernel requires OC % 8 == 0, got OC={OC}"
    assert x_rot.size == n_groups * 128
    assert packed.shape[0] == OC * n_groups

    OC_arg = mx.array([OC], dtype=mx.uint32)
    ng_arg = mx.array([n_groups], dtype=mx.uint32)

    kernel = _gemv_v2_kernel()
    # Dispatch: threadgroup has 2 simdgroups (64 threads), each computes 4
    # output rows. So we need OC/8 threadgroups.
    threads_per_tg = 64
    n_tgs_y = OC // 8
    out = kernel(
        inputs=[x_rot, packed, norms, codebook, OC_arg, ng_arg],
        grid=(threads_per_tg, n_tgs_y, 1),
        threadgroup=(threads_per_tg, 1, 1),
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
