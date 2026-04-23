// SPDX-License-Identifier: MIT
// TurboQuant fused weight dequantization kernel.
//
// Decompresses TQ-quantized weights (packed uint8 indices + per-group norms)
// to full-precision output in a single kernel launch. Inverse WHT rotation
// is performed in shared memory using the butterfly pattern.
//
// Supports 2-bit (4 centroids), 3-bit (8 centroids), and 4-bit (16 centroids).

#pragma once
#include <torch/extension.h>

// Fused dequant: packed_weight + norms + codebook → full weight matrix.
// Output shape: (out_dim, in_dim) in float16 or float32.
//
// packed_weight: (out_dim * n_groups, packed_group_bytes) uint8
//   4-bit: packed_group_bytes = group_size / 2
//   2-bit: packed_group_bytes = group_size / 4
//   3-bit: packed_group_bytes = group_size (no sub-byte packing)
//
// norms: (out_dim, n_groups) float32
// signs1, signs2: (group_size,) float32 -- random sign vectors for WHT
// centroids: (n_centroids,) float32
// `block_size` selects the WHT width the butterfly unrolls within each group.
// Defaults to group_size (full-width WHT). A smaller block_size < group_size
// runs an independent WHT per sub-block of block_size elements (block-
// diagonal WHT), needed by partial-rotary models (Qwen3.6-35B-A3B,
// MiniMax M2.5/M2.7). block_size must be a power of two that divides
// group_size.
void tq_weight_dequant(
    torch::Tensor packed_weight,
    torch::Tensor norms,
    torch::Tensor signs1,
    torch::Tensor signs2,
    torch::Tensor centroids,
    torch::Tensor output,
    int64_t group_size,
    int64_t bits,
    int64_t out_dim,
    int64_t in_dim,
    int64_t block_size);

// Batch dequant for MoE expert weights.
// Same as above but operates on 3D tensors:
//   packed_weight: (n_experts * out_dim * n_groups, packed_group_bytes)
//   norms: (n_experts * out_dim, n_groups)
//   output: (n_experts, out_dim, in_dim)
void tq_weight_dequant_3d(
    torch::Tensor packed_weight,
    torch::Tensor norms,
    torch::Tensor signs1,
    torch::Tensor signs2,
    torch::Tensor centroids,
    torch::Tensor output,
    int64_t group_size,
    int64_t bits,
    int64_t n_experts,
    int64_t out_dim,
    int64_t in_dim,
    int64_t block_size);

// Sparse batch dequant for MoE expert weights — decompresses only the
// experts listed in `active_expert_ids`. Single kernel launch; safe for
// CUDA graph capture when the grid size (n_active × out_dim, n_groups)
// is static across graph replays (bs=1 decode with fixed top_k).
//
//   packed_weight: (n_experts * out_dim * n_groups, packed_group_bytes)
//   norms: (n_experts * out_dim, n_groups)
//   active_expert_ids: (n_active,) int32, duplicates ok (idempotent writes),
//                      out-of-range values cause their blocks to no-op
//   output: (n_experts, out_dim, in_dim) — only active slots are written;
//           other slots are left untouched (caller's responsibility to
//           ensure they're not read downstream)
void tq_weight_dequant_sparse_3d(
    torch::Tensor packed_weight,
    torch::Tensor norms,
    torch::Tensor signs1,
    torch::Tensor signs2,
    torch::Tensor centroids,
    torch::Tensor active_expert_ids,
    torch::Tensor output,
    int64_t group_size,
    int64_t bits,
    int64_t n_experts,
    int64_t out_dim,
    int64_t in_dim,
    int64_t block_size);
