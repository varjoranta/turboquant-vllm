// SPDX-License-Identifier: MIT
// Fused bs=1 GEMV with inverse randomized WHT applied to decoded weights
// in shared memory. Kernel signature intentionally parallels
// tq_weight_gemv_bs1.cu so the A/B harness can swap them directly.

#include "tq3_gemv_bs1_fwht_smem.h"

#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace {

constexpr int GROUP_SIZE = 128;             // one thread per position
constexpr int BYTES_PER_GROUP = 48;         // 128 * 3 / 8
constexpr int WHT_STAGES = 7;               // log2(128)
constexpr float INV_SQRT_N = 0.08838834764831844f;  // 1 / sqrt(128)
constexpr unsigned FULL_MASK = 0xFFFFFFFFu;

// Decode position `tid`'s 3-bit index from a 48-byte group buffer.
// 3 bits at bit offset 3*tid; reads at most 2 adjacent bytes.
__device__ __forceinline__ uint32_t decode_position(
    const uint8_t* __restrict__ grp, int tid)
{
    const int bit_off = 3 * tid;
    const int byte_off = bit_off >> 3;
    const int shift = bit_off & 7;
    const uint32_t b0 = grp[byte_off];
    const uint32_t b1 = (byte_off + 1 < BYTES_PER_GROUP) ? grp[byte_off + 1] : 0u;
    return ((b0 | (b1 << 8)) >> shift) & 0x7u;
}

// In-place 128-lane butterfly Walsh-Hadamard on `w_smem`, using tid as lane.
// Caller must hold 128 threads in the block; caller is responsible for the
// __syncthreads() before entry so all writers have landed.
__device__ __forceinline__ void fwht_128_inplace(float* __restrict__ w_smem, int tid)
{
    #pragma unroll
    for (int stage = 0; stage < WHT_STAGES; ++stage) {
        const int h = 1 << stage;
        __syncthreads();
        const float a = w_smem[tid];
        const float b = w_smem[tid ^ h];
        __syncthreads();
        w_smem[tid] = (tid & h) ? (b - a) : (a + b);
    }
    __syncthreads();
}

__global__ void tq3_gemv_bs1_fwht_smem_kernel(
    const __nv_bfloat16* __restrict__ x,
    const uint8_t*       __restrict__ packed,
    const __nv_bfloat16* __restrict__ norms,
    const __nv_bfloat16* __restrict__ codebook,
    const __nv_bfloat16* __restrict__ signs1,
    const __nv_bfloat16* __restrict__ signs2,
    __nv_bfloat16*       __restrict__ out,
    int K, int OC, int n_groups)
{
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
    const int oc  = blockIdx.x;
    const int tid = threadIdx.x;
    if (oc >= OC) return;

    __shared__ float w_smem[GROUP_SIZE];
    __shared__ float warp_reduce[4];  // 128 threads = 4 warps

    // Load codebook into registers (tiny, shared by all groups).
    float cb[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        cb[i] = __bfloat162float(codebook[i]);
    }
    const float s1_lane = __bfloat162float(signs1[tid]);
    const float s2_lane = __bfloat162float(signs2[tid]);

    float psum = 0.0f;

    for (int g = 0; g < n_groups; ++g) {
        const uint8_t* grp = packed + (oc * n_groups + g) * BYTES_PER_GROUP;
        const float norm = __bfloat162float(norms[oc * n_groups + g]);

        // 1. Per-thread decode + dequant in rotated/quantized space. Multiply
        //    by signs2 here (inverse rotate: signs2 applied before WHT).
        const uint32_t idx = decode_position(grp, tid);
        w_smem[tid] = cb[idx] * norm * s2_lane;

        // 2. In-place FWHT across the 128 elements of this group.
        fwht_128_inplace(w_smem, tid);

        // 3. Apply signs1 and the 1/sqrt(N) unitarity scale; now in
        //    original-space weight. Dot with original-space x[g*128 + tid].
        const float w_orig = w_smem[tid] * s1_lane * INV_SQRT_N;
        const float x_val = __bfloat162float(x[g * GROUP_SIZE + tid]);
        psum = fmaf(w_orig, x_val, psum);
    }

    // Reduce 128 partial sums to one output via warp shuffles + SMEM.
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        psum += __shfl_xor_sync(FULL_MASK, psum, off);
    }
    if (lane_id == 0) warp_reduce[warp_id] = psum;
    __syncthreads();

    if (warp_id == 0) {
        float total = (lane_id < 4) ? warp_reduce[lane_id] : 0.0f;
        total += __shfl_xor_sync(FULL_MASK, total, 2);
        total += __shfl_xor_sync(FULL_MASK, total, 1);
        if (lane_id == 0) {
            out[oc] = __float2bfloat16(total);
        }
    }
#endif  // __CUDA_ARCH__ >= 800
}

}  // anonymous namespace

torch::Tensor tq3_gemv_bs1_fwht_smem(
    torch::Tensor x,
    torch::Tensor packed,
    torch::Tensor norms,
    torch::Tensor codebook,
    torch::Tensor signs1,
    torch::Tensor signs2)
{
    TORCH_CHECK(x.is_cuda() && packed.is_cuda() && norms.is_cuda()
                && codebook.is_cuda() && signs1.is_cuda() && signs2.is_cuda(),
                "all inputs must be CUDA");
    TORCH_CHECK(x.dtype() == torch::kBFloat16, "x must be bf16");
    TORCH_CHECK(packed.dtype() == torch::kUInt8, "packed must be uint8");
    TORCH_CHECK(norms.dtype() == torch::kBFloat16, "norms must be bf16");
    TORCH_CHECK(codebook.dtype() == torch::kBFloat16, "codebook must be bf16");
    TORCH_CHECK(signs1.dtype() == torch::kBFloat16 && signs2.dtype() == torch::kBFloat16,
                "signs must be bf16");
    TORCH_CHECK(packed.is_contiguous() && norms.is_contiguous() && codebook.is_contiguous()
                && signs1.is_contiguous() && signs2.is_contiguous(),
                "packed/norms/codebook/signs must be contiguous");
    TORCH_CHECK(codebook.numel() == 8, "codebook must have 8 entries");
    TORCH_CHECK(signs1.numel() == GROUP_SIZE && signs2.numel() == GROUP_SIZE,
                "signs1/signs2 must each have 128 entries");

    const int K = x.numel();
    const int OC = norms.size(0);
    const int n_groups = norms.size(1);
    TORCH_CHECK(K == n_groups * GROUP_SIZE, "K must equal n_groups * 128");
    TORCH_CHECK(packed.dim() == 2 && packed.size(0) == OC * n_groups
                && packed.size(1) == BYTES_PER_GROUP,
                "packed shape must be (OC * n_groups, 48)");

    auto out = torch::empty({OC},
        torch::TensorOptions().dtype(torch::kBFloat16).device(x.device()));

    dim3 grid(OC);
    dim3 block(GROUP_SIZE);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream(
        x.device().index()).stream();
    tq3_gemv_bs1_fwht_smem_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
        packed.data_ptr<uint8_t>(),
        reinterpret_cast<const __nv_bfloat16*>(norms.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(codebook.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(signs1.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(signs2.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
        K, OC, n_groups);
    return out;
}
