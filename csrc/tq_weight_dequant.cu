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


// ---------------------------------------------------------------------------
// Unified dequant kernel: templated on output type, group size, and bits
// ---------------------------------------------------------------------------

template <typename OutputT, int GROUP_SIZE, int BITS>
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

    // Unpack index from packed bytes
    const int flat_group = row * n_groups + group;
    const uint8_t* group_ptr = packed_weight + flat_group * packed_group_bytes;

    int index;
    if constexpr (BITS == 4) {
        uint8_t byte = group_ptr[tid / 2];
        index = (tid & 1) ? (byte >> 4) & 0xF : byte & 0xF;
    } else if constexpr (BITS == 3) {
        // 3-bit sub-byte packing: 8 indices per 3 bytes (24 bits).
        // Layout: byte0 = idx0|(idx1<<3)|(idx2[0:2]<<6)
        //         byte1 = idx2[2]|(idx3<<1)|(idx4<<4)|(idx5[0:1]<<7)
        //         byte2 = idx5[1:3]|(idx6<<2)|(idx7<<5)
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

    // Codebook lookup + apply sign vector D2
    float val = c_centroids[index] * c_signs2[tid];

    // Inverse WHT butterfly: warp shuffles for intra-warp stages,
    // shared memory only for cross-warp stages.
    // For group_size=128: stages 0-4 use warp shuffles (h=1..16, pairs within warp),
    // stages 5-6 use shared memory (h=32..64, pairs across warps).
    // This eliminates ~10 __syncthreads() calls vs the pure shared memory approach.

    constexpr int LOG2_WARP = 5;  // log2(32)
    constexpr int NUM_STAGES = (GROUP_SIZE == 256) ? 8 : (GROUP_SIZE == 128) ? 7 : 6;

    // Stages 0..LOG2_WARP-1: intra-warp butterfly via warp shuffles (no sync needed)
    // __shfl_xor_sync(mask, val, h) returns val from thread (tid ^ h).
    // For butterfly: lower half (bit h clear) gets a+b, upper half gets b-a.
    #pragma unroll
    for (int stage = 0; stage < LOG2_WARP && stage < NUM_STAGES; stage++) {
        int h = 1 << stage;
        float other = __shfl_xor_sync(0xFFFFFFFF, val, h);
        val = (tid & h) ? (other - val) : (val + other);
    }

    // Stages LOG2_WARP..NUM_STAGES-1: cross-warp butterfly via shared memory
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

    // Normalize, apply D1, rescale by group norm
    val = val * rsqrtf((float)GROUP_SIZE) * c_signs1[tid]
        * norms[row * n_groups + group];

    // Write output
    int col = group * GROUP_SIZE + tid;
    if (col < in_dim) {
        if constexpr (std::is_same_v<OutputT, __half>)
            output[row * in_dim + col] = __float2half(val);
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
    int64_t in_dim
) {
    TORCH_CHECK(packed_weight.is_cuda(), "packed_weight must be CUDA");
    TORCH_CHECK(norms.is_cuda(), "norms must be CUDA");
    TORCH_CHECK(output.is_cuda(), "output must be CUDA");
    TORCH_CHECK(group_size == 64 || group_size == 128 || group_size == 256,
                "group_size must be 64, 128, or 256");
    TORCH_CHECK(bits >= 2 && bits <= 4, "bits must be 2-4");

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

    bool use_fp16 = (output.scalar_type() == at::ScalarType::Half);

    // Dispatch by output type, group_size, and bits.
    // Stream argument ensures kernels land on PyTorch's current stream.
    #define DISPATCH(GS, B)                                                    \
        if (use_fp16)                                                          \
            tq_weight_dequant_kernel<__half, GS, B><<<grid, block, smem_bytes, stream>>>( \
                packed_weight.data_ptr<uint8_t>(), norms.data_ptr<float>(),    \
                reinterpret_cast<__half*>(output.data_ptr()),                  \
                n_groups, (int)in_dim, packed_group_bytes);                    \
        else                                                                   \
            tq_weight_dequant_kernel<float, GS, B><<<grid, block, smem_bytes, stream>>>( \
                packed_weight.data_ptr<uint8_t>(), norms.data_ptr<float>(),    \
                output.data_ptr<float>(),                                      \
                n_groups, (int)in_dim, packed_group_bytes);

    if      (group_size == 64  && bits == 2) DISPATCH(64, 2)
    else if (group_size == 64  && bits == 3) DISPATCH(64, 3)
    else if (group_size == 64  && bits == 4) DISPATCH(64, 4)
    else if (group_size == 128 && bits == 2) DISPATCH(128, 2)
    else if (group_size == 128 && bits == 3) DISPATCH(128, 3)
    else if (group_size == 128 && bits == 4) DISPATCH(128, 4)
    else if (group_size == 256 && bits == 2) DISPATCH(256, 2)
    else if (group_size == 256 && bits == 3) DISPATCH(256, 3)
    else if (group_size == 256 && bits == 4) DISPATCH(256, 4)
    else TORCH_CHECK(false, "Unsupported group_size/bits combo");

    #undef DISPATCH
}


// ---------------------------------------------------------------------------
// Sparse MoE kernel — decompress only listed active experts
// ---------------------------------------------------------------------------
//
// Same decode / butterfly / norm logic as tq_weight_dequant_kernel, but the
// grid's X axis indexes into `active_expert_ids` instead of directly into
// all experts' rows. Reads packed/norms from the expert's full-tensor
// offset, writes into the expert's slot in the 3D output. Output slots for
// non-active experts are left untouched.

template <typename OutputT, int GROUP_SIZE, int BITS>
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
    constexpr int NUM_STAGES = (GROUP_SIZE == 256) ? 8 : (GROUP_SIZE == 128) ? 7 : 6;

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

    val = val * rsqrtf((float)GROUP_SIZE) * c_signs1[tid]
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
    int64_t in_dim
) {
    int total_rows = n_experts * out_dim;
    auto output_2d = output.reshape({total_rows, in_dim});
    tq_weight_dequant(packed_weight, norms.reshape({total_rows, -1}),
                      signs1, signs2, centroids, output_2d,
                      group_size, bits, total_rows, in_dim);
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
    int64_t in_dim
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

    const int32_t* active_ptr = active_expert_ids.data_ptr<int32_t>();
    const auto out_dtype = output.scalar_type();

    // Writing bf16/fp16 directly from the kernel avoids a Python-side
    // fp32 scratch + post-copy (which required a D2H sync that broke
    // CUDA graph capture before this change).
    #define LAUNCH(OutT, GS, B)                                                       \
        tq_weight_dequant_sparse_3d_kernel<OutT, GS, B>                               \
            <<<grid, block, smem_bytes, stream>>>(                                    \
                packed_weight.data_ptr<uint8_t>(), norms.data_ptr<float>(),           \
                active_ptr, reinterpret_cast<OutT*>(output.data_ptr()),               \
                (int)n_experts, n_groups, (int)out_dim, (int)in_dim,                  \
                packed_group_bytes);

    #define DISPATCH_SPARSE(GS, B)                                                    \
        if      (out_dtype == at::ScalarType::Half)     LAUNCH(__half, GS, B)         \
        else if (out_dtype == at::ScalarType::BFloat16) LAUNCH(__nv_bfloat16, GS, B)  \
        else                                            LAUNCH(float, GS, B)

    if      (group_size == 64  && bits == 2) DISPATCH_SPARSE(64, 2)
    else if (group_size == 64  && bits == 3) DISPATCH_SPARSE(64, 3)
    else if (group_size == 64  && bits == 4) DISPATCH_SPARSE(64, 4)
    else if (group_size == 128 && bits == 2) DISPATCH_SPARSE(128, 2)
    else if (group_size == 128 && bits == 3) DISPATCH_SPARSE(128, 3)
    else if (group_size == 128 && bits == 4) DISPATCH_SPARSE(128, 4)
    else if (group_size == 256 && bits == 2) DISPATCH_SPARSE(256, 2)
    else if (group_size == 256 && bits == 3) DISPATCH_SPARSE(256, 3)
    else if (group_size == 256 && bits == 4) DISPATCH_SPARSE(256, 4)
    else TORCH_CHECK(false, "Unsupported group_size/bits combo");

    #undef DISPATCH_SPARSE
    #undef LAUNCH
}
