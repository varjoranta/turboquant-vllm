// SPDX-License-Identifier: MIT
// TurboQuant bs=1 GEMV kernel.
//
// Pack format must match weight_quant.pack_indices/unpack_indices for bits=3:
// 48 bytes per 128-value group, 16 × 3-byte triplets with cross-byte
// positions 2 and 5.

#include "tq_weight_gemv_bs1.h"

#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace {

constexpr int GROUP_SIZE = 128;
constexpr int BYTES_PER_GROUP = 48;          // GROUP_SIZE * 3 / 8
constexpr int INDICES_PER_CHUNK = 32;
constexpr int BYTES_PER_CHUNK = 12;          // INDICES_PER_CHUNK * 3 / 8
constexpr int CHUNKS_PER_GROUP = 4;
constexpr unsigned FULL_MASK = 0xFFFFFFFFu;

__device__ __forceinline__ void decode_32_indices(
    const uint8_t* __restrict__ chunk_12_bytes,
    int32_t out[INDICES_PER_CHUNK])
{
    uint32_t u0, u1, u2;
    memcpy(&u0, chunk_12_bytes + 0, 4);
    memcpy(&u1, chunk_12_bytes + 4, 4);
    memcpy(&u2, chunk_12_bytes + 8, 4);
    const uint32_t b[12] = {
        (u0 >>  0) & 0xFF, (u0 >>  8) & 0xFF, (u0 >> 16) & 0xFF, (u0 >> 24) & 0xFF,
        (u1 >>  0) & 0xFF, (u1 >>  8) & 0xFF, (u1 >> 16) & 0xFF, (u1 >> 24) & 0xFF,
        (u2 >>  0) & 0xFF, (u2 >>  8) & 0xFF, (u2 >> 16) & 0xFF, (u2 >> 24) & 0xFF,
    };
    #pragma unroll
    for (int t = 0; t < 4; ++t) {
        const uint32_t b0 = b[t * 3 + 0];
        const uint32_t b1 = b[t * 3 + 1];
        const uint32_t b2 = b[t * 3 + 2];
        out[t * 8 + 0] = int32_t( b0                  & 0x7);
        out[t * 8 + 1] = int32_t((b0 >> 3)            & 0x7);
        out[t * 8 + 2] = int32_t(((b0 >> 6) | (b1 << 2)) & 0x7);
        out[t * 8 + 3] = int32_t((b1 >> 1)            & 0x7);
        out[t * 8 + 4] = int32_t((b1 >> 4)            & 0x7);
        out[t * 8 + 5] = int32_t(((b1 >> 7) | (b2 << 1)) & 0x7);
        out[t * 8 + 6] = int32_t((b2 >> 2)            & 0x7);
        out[t * 8 + 7] = int32_t((b2 >> 5)            & 0x7);
    }
}

__global__ void tq3_gemv_bs1_kernel(
    const __nv_bfloat16* __restrict__ x_rot,
    const uint8_t*       __restrict__ packed,
    const __nv_bfloat16* __restrict__ norms,
    const __nv_bfloat16* __restrict__ codebook,
    __nv_bfloat16*       __restrict__ out,
    int K, int OC, int n_groups)
{
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
    const int oc  = blockIdx.x;
    const int tid = threadIdx.x;
    if (oc >= OC) return;

    float cb[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        cb[i] = __bfloat162float(codebook[i]);
    }

    float psum = 0.0f;

    for (int g = tid; g < n_groups; g += 32) {
        const uint8_t* grp = packed + (oc * n_groups + g) * BYTES_PER_GROUP;
        const float norm = __bfloat162float(norms[oc * n_groups + g]);

        #pragma unroll
        for (int c = 0; c < CHUNKS_PER_GROUP; ++c) {
            int32_t idx[INDICES_PER_CHUNK];
            decode_32_indices(grp + c * BYTES_PER_CHUNK, idx);
            const __nv_bfloat16* x_chunk =
                x_rot + g * GROUP_SIZE + c * INDICES_PER_CHUNK;

            #pragma unroll
            for (int i = 0; i < INDICES_PER_CHUNK; ++i) {
                const float w = cb[idx[i]] * norm;
                const float x = __bfloat162float(x_chunk[i]);
                psum += w * x;
            }
        }
    }

    // Block is one warp, so FULL_MASK is correct.
    psum += __shfl_xor_sync(FULL_MASK, psum, 16);
    psum += __shfl_xor_sync(FULL_MASK, psum,  8);
    psum += __shfl_xor_sync(FULL_MASK, psum,  4);
    psum += __shfl_xor_sync(FULL_MASK, psum,  2);
    psum += __shfl_xor_sync(FULL_MASK, psum,  1);

    if (tid == 0) {
        out[oc] = __float2bfloat16(psum);
    }
#endif  // __CUDA_ARCH__ >= 800
}

}  // anonymous namespace

torch::Tensor tq3_gemv_bs1(
    torch::Tensor x_rot,
    torch::Tensor packed,
    torch::Tensor norms,
    torch::Tensor codebook)
{
    TORCH_CHECK(x_rot.is_cuda() && packed.is_cuda() && norms.is_cuda() && codebook.is_cuda(),
                "all inputs must be CUDA");
    TORCH_CHECK(x_rot.dtype() == torch::kBFloat16, "x_rot must be bf16");
    TORCH_CHECK(packed.dtype() == torch::kUInt8, "packed must be uint8");
    TORCH_CHECK(norms.dtype() == torch::kBFloat16, "norms must be bf16");
    TORCH_CHECK(codebook.dtype() == torch::kBFloat16, "codebook must be bf16");
    TORCH_CHECK(packed.is_contiguous() && norms.is_contiguous() && codebook.is_contiguous(),
                "packed, norms, codebook must be contiguous");
    TORCH_CHECK(codebook.numel() == 8, "codebook must have 8 entries");

    const int K = x_rot.numel();
    const int OC = norms.size(0);
    const int n_groups = norms.size(1);
    TORCH_CHECK(K == n_groups * GROUP_SIZE, "K must equal n_groups * 128");
    TORCH_CHECK(packed.dim() == 2 && packed.size(0) == OC * n_groups
                && packed.size(1) == BYTES_PER_GROUP,
                "packed shape must be (OC * n_groups, 48)");

    auto out = torch::empty({OC},
        torch::TensorOptions().dtype(torch::kBFloat16).device(x_rot.device()));

    dim3 grid(OC);
    dim3 block(32);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream(
        x_rot.device().index()).stream();
    tq3_gemv_bs1_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x_rot.data_ptr()),
        packed.data_ptr<uint8_t>(),
        reinterpret_cast<const __nv_bfloat16*>(norms.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(codebook.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
        K, OC, n_groups);
    return out;
}
