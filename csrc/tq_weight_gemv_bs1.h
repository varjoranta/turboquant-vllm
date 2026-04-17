// SPDX-License-Identifier: MIT
// TurboQuant bs=1 GEMV for 3-bit weight quantization.
//
// Warp-per-output-channel CUDA kernel intended for decode-time single-token
// latency. Replaces the Triton GEMV path when M=1, which otherwise pads
// tensor-core tiles up to M=16 and wastes throughput.
//
// Requires sm_80+ (bf16 + warp shuffle).

#pragma once
#include <torch/extension.h>

// bs=1 GEMV.
//   x_rot:    (K,) bf16 — pre-rotated activation
//   packed:   (OC * n_groups, 48) uint8 — sub-byte 3-bit pack
//   norms:    (OC, n_groups) bf16
//   codebook: (8,) bf16
// Returns: (OC,) bf16.
torch::Tensor tq3_gemv_bs1(
    torch::Tensor x_rot,
    torch::Tensor packed,
    torch::Tensor norms,
    torch::Tensor codebook);
