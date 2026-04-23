// SPDX-License-Identifier: MIT
// TurboQuant fused weight dequantization kernel.
//
// Decompresses group-quantized weights in a single kernel launch:
//   1. Unpack nibble/byte indices from uint8
//   2. Codebook lookup
//   3. Apply sign vector D2, inverse WHT butterfly, apply sign vector D1
//   4. Rescale by per-group L2 norm
//   5. Write to output (fp32 or fp16)
//
// One thread block per (row, group). Block size = group_size threads.

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <type_traits>

// Constant memory for codebook and sign vectors.
// Cached across kernel launches — only re-uploaded when config changes.
__constant__ float c_centroids[16];
__constant__ float c_signs1[256];
__constant__ float c_signs2[256];

// Track last-uploaded config to skip redundant cudaMemcpyToSymbol
static const float* s_last_centroids = nullptr;
static const float* s_last_signs1 = nullptr;
static const float* s_last_signs2 = nullptr;

static inline void maybe_upload_constants(
    const float* cptr, const float* s1ptr, const float* s2ptr,
    int64_t bits, int64_t group_size, cudaStream_t stream
) {
    if (cptr != s_last_centroids) {
        cudaMemcpyToSymbolAsync(c_centroids, cptr, (1 << bits) * sizeof(float),
                                0, cudaMemcpyDeviceToDevice, stream);
        s_last_centroids = cptr;
    }
    if (s1ptr != s_last_signs1) {
        cudaMemcpyToSymbolAsync(c_signs1, s1ptr, group_size * sizeof(float),
                                0, cudaMemcpyDeviceToDevice, stream);
        s_last_signs1 = s1ptr;
    }
    if (s2ptr != s_last_signs2) {
        cudaMemcpyToSymbolAsync(c_signs2, s2ptr, group_size * sizeof(float),
                                0, cudaMemcpyDeviceToDevice, stream);
        s_last_signs2 = s2ptr;
    }
}

static inline int packed_group_bytes_for(int64_t bits, int64_t group_size) {
    if (bits == 4) return group_size / 2;
    if (bits == 3) return (group_size / 8) * 3;
    if (bits == 2) return group_size / 4;
    return group_size;
}


// BLOCK_SIZE < GROUP_SIZE runs the butterfly as an independent WHT per
// BLOCK_SIZE sub-block within each group (block-diagonal WHT, used by
// partial-rotary models).

template <typename OutputT, int GROUP_SIZE, int BITS, int BLOCK_SIZE = GROUP_SIZE>
__global__ void tq_weight_dequant_kernel(
    const uint8_t* __restrict__ packed_weight,
    const float*   __restrict__ norms,
    OutputT*       __restrict__ output,
    int n_groups,
    int in_dim,
    int packed_group_bytes
) {
    const int row = blockIdx.x;
    const int group = blockIdx.y;
    const int tid = threadIdx.x;

    if (tid >= GROUP_SIZE) return;

    extern __shared__ float smem[];

    const int flat_group = row * n_groups + group;
    const uint8_t* group_ptr = packed_weight + flat_group * packed_group_bytes;

    int index;
    if constexpr (BITS == 4) {
        uint8_t byte = group_ptr[tid / 2];
        index = (tid & 1) ? (byte >> 4) & 0xF : byte & 0xF;
    } else if constexpr (BITS == 3) {
        // 3-bit sub-byte packing: 8 indices per 3 bytes. Offsets per position
        // in a triplet are fixed; see pack_indices for the write side.
        int group_of_8 = tid / 8;
        int pos_in_group = tid % 8;
        const uint8_t* base = group_ptr + group_of_8 * 3;
        uint8_t b0 = base[0], b1 = base[1], b2 = base[2];
        switch (pos_in_group) {
            case 0: index = b0 & 0x7; break;
            case 1: index = (b0 >> 3) & 0x7; break;
            case 2: index = ((b0 >> 6) | (b1 << 2)) & 0x7; break;
            case 3: index = (b1 >> 1) & 0x7; break;
            case 4: index = (b1 >> 4) & 0x7; break;
            case 5: index = ((b1 >> 7) | (b2 << 1)) & 0x7; break;
            case 6: index = (b2 >> 2) & 0x7; break;
            case 7: index = (b2 >> 5) & 0x7; break;
            default: index = 0;
        }
    } else if constexpr (BITS == 2) {
        uint8_t byte = group_ptr[tid / 4];
        index = (byte >> ((tid & 3) * 2)) & 0x3;
    } else {
        index = group_ptr[tid];
    }

    float val = c_centroids[index] * c_signs2[tid];

    // Butterfly runs log2(BLOCK_SIZE) stages. For BLOCK_SIZE < GROUP_SIZE, the
    // butterfly confines itself to each sub-block automatically because every
    // stage h < BLOCK_SIZE means tid ^ h stays in the same sub-block. Warp-
    // shuffle stages are fine up to min(BLOCK_SIZE, 32); cross-warp stages
    // only apply when BLOCK_SIZE > 32.
    constexpr int LOG2_WARP = 5;
    constexpr int NUM_STAGES = (BLOCK_SIZE == 256) ? 8
                             : (BLOCK_SIZE == 128) ? 7
                             : (BLOCK_SIZE == 64)  ? 6
                             : (BLOCK_SIZE == 32)  ? 5
                             : (BLOCK_SIZE == 16)  ? 4
                             : 0;

    #pragma unroll
    for (int stage = 0; stage < LOG2_WARP && stage < NUM_STAGES; stage++) {
        int h = 1 << stage;
        float other = __shfl_xor_sync(0xFFFFFFFF, val, h);
        val = (tid & h) ? (other - val) : (val + other);
    }

    if constexpr (NUM_STAGES > LOG2_WARP) {
        smem[tid] = val;
        __syncthreads();

        #pragma unroll
        for (int stage = LOG2_WARP; stage < NUM_STAGES; stage++) {
            int h = 1 << stage;
            int pair = ((tid % (2 * h)) < h) ? (tid + h) : (tid - h);
            float a = smem[tid];
            float b = smem[pair];
            __syncthreads();
            smem[tid] = ((tid % (2 * h)) < h) ? (a + b) : (b - a);
            __syncthreads();
        }
        val = smem[tid];
    }

    val = val * rsqrtf((float)BLOCK_SIZE) * c_signs1[tid]
        * norms[row * n_groups + group];

    int col = group * GROUP_SIZE + tid;
    if (col < in_dim) {
        if constexpr (std::is_same_v<OutputT, __half>)
            output[row * in_dim + col] = __float2half(val);
        else if constexpr (std::is_same_v<OutputT, __nv_bfloat16>)
            output[row * in_dim + col] = __float2bfloat16(val);
        else
            output[row * in_dim + col] = val;
    }
}


// ---------------------------------------------------------------------------
// Launch wrapper
// ---------------------------------------------------------------------------

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
    int64_t block_size
) {
    TORCH_CHECK(packed_weight.is_cuda(), "packed_weight must be CUDA");
    TORCH_CHECK(norms.is_cuda(), "norms must be CUDA");
    TORCH_CHECK(output.is_cuda(), "output must be CUDA");
    TORCH_CHECK(group_size == 64 || group_size == 128 || group_size == 256,
                "group_size must be 64, 128, or 256");
    TORCH_CHECK(bits >= 2 && bits <= 4, "bits must be 2-4");
    TORCH_CHECK(block_size > 0 && block_size <= group_size
                && (block_size & (block_size - 1)) == 0,
                "block_size must be a power of two ≤ group_size");
    TORCH_CHECK(group_size % block_size == 0,
                "group_size must be a multiple of block_size");

    int n_groups = (in_dim + group_size - 1) / group_size;
    int packed_group_bytes = packed_group_bytes_for(bits, group_size);

    // PyTorch's current CUDA stream so launches are captured by CUDA graphs
    // (vLLM piecewise capture, torch.compile reduce-overhead).
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream(
        packed_weight.device().index()).stream();

    maybe_upload_constants(
        centroids.data_ptr<float>(),
        signs1.data_ptr<float>(),
        signs2.data_ptr<float>(),
        bits, group_size, stream);

    dim3 grid(out_dim, n_groups);
    dim3 block(group_size);
    size_t smem_bytes = group_size * sizeof(float);

    const auto out_dtype = output.scalar_type();

    #define LAUNCH(OutT, GS, B, BS)                                                      \
        tq_weight_dequant_kernel<OutT, GS, B, BS><<<grid, block, smem_bytes, stream>>>(  \
            packed_weight.data_ptr<uint8_t>(), norms.data_ptr<float>(),                  \
            reinterpret_cast<OutT*>(output.data_ptr()),                                  \
            n_groups, (int)in_dim, packed_group_bytes);

    #define DISPATCH(GS, B, BS)                                                          \
        if      (out_dtype == at::ScalarType::Half)     LAUNCH(__half, GS, B, BS)        \
        else if (out_dtype == at::ScalarType::BFloat16) LAUNCH(__nv_bfloat16, GS, B, BS) \
        else                                            LAUNCH(float, GS, B, BS)

    if (block_size == group_size) {
        if      (group_size == 64  && bits == 2) DISPATCH(64, 2, 64)
        else if (group_size == 64  && bits == 3) DISPATCH(64, 3, 64)
        else if (group_size == 64  && bits == 4) DISPATCH(64, 4, 64)
        else if (group_size == 128 && bits == 2) DISPATCH(128, 2, 128)
        else if (group_size == 128 && bits == 3) DISPATCH(128, 3, 128)
        else if (group_size == 128 && bits == 4) DISPATCH(128, 4, 128)
        else if (group_size == 256 && bits == 2) DISPATCH(256, 2, 256)
        else if (group_size == 256 && bits == 3) DISPATCH(256, 3, 256)
        else if (group_size == 256 && bits == 4) DISPATCH(256, 4, 256)
        else TORCH_CHECK(false, "Unsupported group_size/bits combo");
    } else {
        if      (group_size == 128 && bits == 3 && block_size == 64)  DISPATCH(128, 3, 64)
        else if (group_size == 128 && bits == 4 && block_size == 64)  DISPATCH(128, 4, 64)
        else if (group_size == 128 && bits == 2 && block_size == 64)  DISPATCH(128, 2, 64)
        else if (group_size == 128 && bits == 3 && block_size == 32)  DISPATCH(128, 3, 32)
        else if (group_size == 256 && bits == 3 && block_size == 128) DISPATCH(256, 3, 128)
        else if (group_size == 256 && bits == 3 && block_size == 64)  DISPATCH(256, 3, 64)
        else TORCH_CHECK(
            false, "Unsupported (group_size=", group_size,
            ", bits=", bits, ", block_size=", block_size, ") combo");
    }

    #undef DISPATCH
    #undef LAUNCH
}


// Sparse MoE variant: grid's X axis indexes into `active_expert_ids`;
// output slots for non-active experts are left untouched.

template <typename OutputT, int GROUP_SIZE, int BITS, int BLOCK_SIZE = GROUP_SIZE>
__global__ void tq_weight_dequant_sparse_3d_kernel(
    const uint8_t* __restrict__ packed_weight,
    const float*   __restrict__ norms,
    const int32_t* __restrict__ active_expert_ids,
    OutputT*       __restrict__ output,
    int n_experts,
    int n_groups,
    int out_dim,
    int in_dim,
    int packed_group_bytes
) {
    const int active_idx = blockIdx.x / out_dim;
    const int out_row = blockIdx.x % out_dim;
    const int group = blockIdx.y;
    const int tid = threadIdx.x;

    if (tid >= GROUP_SIZE) return;

    const int expert_id = active_expert_ids[active_idx];
    if (expert_id < 0 || expert_id >= n_experts) return;

    const int flat_row = expert_id * out_dim + out_row;

    extern __shared__ float smem[];

    const int flat_group = flat_row * n_groups + group;
    const uint8_t* group_ptr = packed_weight + flat_group * packed_group_bytes;

    int index;
    if constexpr (BITS == 4) {
        uint8_t byte = group_ptr[tid / 2];
        index = (tid & 1) ? (byte >> 4) & 0xF : byte & 0xF;
    } else if constexpr (BITS == 3) {
        int group_of_8 = tid / 8;
        int pos_in_group = tid % 8;
        const uint8_t* base = group_ptr + group_of_8 * 3;
        uint8_t b0 = base[0], b1 = base[1], b2 = base[2];
        switch (pos_in_group) {
            case 0: index = b0 & 0x7; break;
            case 1: index = (b0 >> 3) & 0x7; break;
            case 2: index = ((b0 >> 6) | (b1 << 2)) & 0x7; break;
            case 3: index = (b1 >> 1) & 0x7; break;
            case 4: index = (b1 >> 4) & 0x7; break;
            case 5: index = ((b1 >> 7) | (b2 << 1)) & 0x7; break;
            case 6: index = (b2 >> 2) & 0x7; break;
            case 7: index = (b2 >> 5) & 0x7; break;
            default: index = 0;
        }
    } else if constexpr (BITS == 2) {
        uint8_t byte = group_ptr[tid / 4];
        index = (byte >> ((tid & 3) * 2)) & 0x3;
    } else {
        index = group_ptr[tid];
    }

    float val = c_centroids[index] * c_signs2[tid];

    constexpr int LOG2_WARP = 5;
    constexpr int NUM_STAGES = (BLOCK_SIZE == 256) ? 8
                             : (BLOCK_SIZE == 128) ? 7
                             : (BLOCK_SIZE == 64)  ? 6
                             : (BLOCK_SIZE == 32)  ? 5
                             : (BLOCK_SIZE == 16)  ? 4
                             : 0;

    #pragma unroll
    for (int stage = 0; stage < LOG2_WARP && stage < NUM_STAGES; stage++) {
        int h = 1 << stage;
        float other = __shfl_xor_sync(0xFFFFFFFF, val, h);
        val = (tid & h) ? (other - val) : (val + other);
    }

    if constexpr (NUM_STAGES > LOG2_WARP) {
        smem[tid] = val;
        __syncthreads();

        #pragma unroll
        for (int stage = LOG2_WARP; stage < NUM_STAGES; stage++) {
            int h = 1 << stage;
            int pair = ((tid % (2 * h)) < h) ? (tid + h) : (tid - h);
            float a = smem[tid];
            float b = smem[pair];
            __syncthreads();
            smem[tid] = ((tid % (2 * h)) < h) ? (a + b) : (b - a);
            __syncthreads();
        }
        val = smem[tid];
    }

    val = val * rsqrtf((float)BLOCK_SIZE) * c_signs1[tid]
        * norms[flat_row * n_groups + group];

    int col = group * GROUP_SIZE + tid;
    if (col < in_dim) {
        const int out_offset =
            expert_id * (out_dim * in_dim) + out_row * in_dim + col;
        if constexpr (std::is_same_v<OutputT, __half>)
            output[out_offset] = __float2half(val);
        else if constexpr (std::is_same_v<OutputT, __nv_bfloat16>)
            output[out_offset] = __float2bfloat16(val);
        else
            output[out_offset] = val;
    }
}


// ---------------------------------------------------------------------------
// 3D variant for MoE experts (reshapes to 2D, no data copy)
// ---------------------------------------------------------------------------

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
    int64_t block_size
) {
    int total_rows = n_experts * out_dim;
    auto output_2d = output.reshape({total_rows, in_dim});
    tq_weight_dequant(packed_weight, norms.reshape({total_rows, -1}),
                      signs1, signs2, centroids, output_2d,
                      group_size, bits, total_rows, in_dim, block_size);
}


// ---------------------------------------------------------------------------
// Sparse 3D variant — decompress only active_expert_ids into their slots
// ---------------------------------------------------------------------------

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
    int64_t block_size
) {
    TORCH_CHECK(packed_weight.is_cuda() && norms.is_cuda() && output.is_cuda()
                && active_expert_ids.is_cuda(),
                "all inputs must be CUDA");
    TORCH_CHECK(active_expert_ids.scalar_type() == at::ScalarType::Int,
                "active_expert_ids must be int32");
    TORCH_CHECK(group_size == 64 || group_size == 128 || group_size == 256,
                "group_size must be 64, 128, or 256");
    TORCH_CHECK(bits >= 2 && bits <= 4, "bits must be 2-4");

    const int n_active = (int)active_expert_ids.numel();
    if (n_active == 0) return;

    const int n_groups = ((int)in_dim + (int)group_size - 1) / (int)group_size;
    const int packed_group_bytes = packed_group_bytes_for(bits, group_size);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream(
        packed_weight.device().index()).stream();
    maybe_upload_constants(
        centroids.data_ptr<float>(),
        signs1.data_ptr<float>(),
        signs2.data_ptr<float>(),
        bits, group_size, stream);

    dim3 grid((unsigned)n_active * (unsigned)out_dim, (unsigned)n_groups);
    dim3 block((unsigned)group_size);
    size_t smem_bytes = group_size * sizeof(float);

    TORCH_CHECK(block_size > 0 && block_size <= group_size
                && (block_size & (block_size - 1)) == 0,
                "block_size must be a power of two ≤ group_size");

    const int32_t* active_ptr = active_expert_ids.data_ptr<int32_t>();
    const auto out_dtype = output.scalar_type();

    #define LAUNCH(OutT, GS, B, BS)                                                    \
        tq_weight_dequant_sparse_3d_kernel<OutT, GS, B, BS>                            \
            <<<grid, block, smem_bytes, stream>>>(                                     \
                packed_weight.data_ptr<uint8_t>(), norms.data_ptr<float>(),            \
                active_ptr, reinterpret_cast<OutT*>(output.data_ptr()),                \
                (int)n_experts, n_groups, (int)out_dim, (int)in_dim,                   \
                packed_group_bytes);

    #define DISPATCH_SPARSE(GS, B, BS)                                                 \
        if      (out_dtype == at::ScalarType::Half)     LAUNCH(__half, GS, B, BS)      \
        else if (out_dtype == at::ScalarType::BFloat16) LAUNCH(__nv_bfloat16, GS, B, BS)\
        else                                            LAUNCH(float, GS, B, BS)

    if (block_size == group_size) {
        if      (group_size == 64  && bits == 2) DISPATCH_SPARSE(64, 2, 64)
        else if (group_size == 64  && bits == 3) DISPATCH_SPARSE(64, 3, 64)
        else if (group_size == 64  && bits == 4) DISPATCH_SPARSE(64, 4, 64)
        else if (group_size == 128 && bits == 2) DISPATCH_SPARSE(128, 2, 128)
        else if (group_size == 128 && bits == 3) DISPATCH_SPARSE(128, 3, 128)
        else if (group_size == 128 && bits == 4) DISPATCH_SPARSE(128, 4, 128)
        else if (group_size == 256 && bits == 2) DISPATCH_SPARSE(256, 2, 256)
        else if (group_size == 256 && bits == 3) DISPATCH_SPARSE(256, 3, 256)
        else if (group_size == 256 && bits == 4) DISPATCH_SPARSE(256, 4, 256)
        else TORCH_CHECK(false, "Unsupported group_size/bits combo");
    } else {
        if      (group_size == 128 && bits == 3 && block_size == 64) DISPATCH_SPARSE(128, 3, 64)
        else if (group_size == 128 && bits == 4 && block_size == 64) DISPATCH_SPARSE(128, 4, 64)
        else if (group_size == 128 && bits == 2 && block_size == 64) DISPATCH_SPARSE(128, 2, 64)
        else if (group_size == 128 && bits == 3 && block_size == 32) DISPATCH_SPARSE(128, 3, 32)
        else if (group_size == 256 && bits == 3 && block_size == 128) DISPATCH_SPARSE(256, 3, 128)
        else if (group_size == 256 && bits == 3 && block_size == 64) DISPATCH_SPARSE(256, 3, 64)
        else TORCH_CHECK(
            false, "Unsupported (group_size=", group_size,
            ", bits=", bits, ", block_size=", block_size, ") combo");
    }

    #undef DISPATCH_SPARSE
    #undef LAUNCH
}
