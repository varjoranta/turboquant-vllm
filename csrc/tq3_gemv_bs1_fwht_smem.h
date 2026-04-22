// SPDX-License-Identifier: MIT
// TurboQuant bs=1 GEMV with FWHT fused in shared memory.
//
// Replaces the two-kernel pipeline `wht_on_input + tq3_gemv_bs1` with a
// single kernel that dequantizes weights into SMEM, applies the inverse
// randomized WHT (signs2 -> H -> signs1) in-place, then dot-products the
// original-space activation. Motivated by ITQ3_S's Algorithm 2 but adapted
// for our randomized WHT (two sign vectors instead of plain FWHT).
//
// Launch config: one block per output channel, 128 threads per block (one
// thread per group element). sm_80+.

#pragma once
#include <torch/extension.h>

// x:        (K,) bf16 — ORIGINAL-SPACE activation (not rotated)
// packed:   (OC * n_groups, 48) uint8 — 3-bit pack
// norms:    (OC, n_groups) bf16
// codebook: (8,) bf16
// signs1:   (128,) bf16 — ±1, randomized-WHT sign vector (pre-WHT on input)
// signs2:   (128,) bf16 — ±1, randomized-WHT sign vector (post-WHT on input)
// Returns: (OC,) bf16.
torch::Tensor tq3_gemv_bs1_fwht_smem(
    torch::Tensor x,
    torch::Tensor packed,
    torch::Tensor norms,
    torch::Tensor codebook,
    torch::Tensor signs1,
    torch::Tensor signs2);
