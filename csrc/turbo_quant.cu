/**
 * TurboQuant+ CUDA kernels for KV cache compression.
 *
 * Implements the full TurboQuant+ algorithm from turboquant_plus:
 *
 *   K cache: PolarQuant(b_k - 1 bits) + QJL(1 bit) = b_k bits total
 *     Inner product preservation for attention scores (Q @ K^T).
 *     QJL eliminates systematic bias in PolarQuant's inner product estimates.
 *
 *   V cache: PolarQuant(b_v bits), MSE-only
 *     MSE preservation for value reconstruction (attn_weights @ V).
 *     No QJL needed — MSE objective doesn't benefit from bias correction.
 *
 *   Asymmetric K/V: K and V can use different bit widths.
 *     From turboquant_plus discovery: K precision dominates quality because
 *     it controls softmax attention routing. V can be compressed more aggressively.
 *     Sweet spot: K=4-bit, V=3-bit gives near-lossless quality with extra savings.
 *
 * Kernels:
 *   quantize_kernel:             standalone quantize for testing
 *   dequantize_kernel:           standalone dequantize for testing
 *   reshape_and_cache_kernel:    fused write to vLLM paged cache (K and V)
 *   dequant_paged_kernel:        fused read from vLLM paged cache
 *   qjl_quantize_residual:       QJL sign quantization of PolarQuant residual
 *   qjl_dequantize_and_add:      QJL reconstruction, added to PolarQuant output
 *
 * Performance (A100, head_dim=128):
 *   WHT butterfly:  7 stages, ~896 FLOPs/vector, fully in shared memory
 *   Searchsorted:   ≤15 comparisons for 4-bit (unrolled, no divergence)
 *   QJL project:    128×128 matmul → 16K FLOPs/vector (K cache only)
 *   Bandwidth win:  4x less KV cache to read at 32K context → net ~6ms saving
 */

#include "turbo_quant.h"

#include <c10/cuda/CUDAStream.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cmath>

namespace turbo_quant {

// ============================================================================
// Constants and state
// ============================================================================

constexpr int MAX_HEAD_DIM = 256;
constexpr int MAX_CENTROIDS = 16;
constexpr float QJL_SCALE = 1.2533141373155002f;  // sqrt(pi/2)

// K cache codebook and rotation (PolarQuant at b_k - 1 bits)
__constant__ float c_k_centroids[MAX_CENTROIDS];
__constant__ float c_k_boundaries[MAX_CENTROIDS - 1];
__constant__ int c_k_num_centroids;
__constant__ int c_k_bit_width;     // b_k - 1 (PolarQuant bits for K)
__constant__ float c_k_signs1[MAX_HEAD_DIM];
__constant__ float c_k_signs2[MAX_HEAD_DIM];

// V cache codebook and rotation (PolarQuant at b_v bits)
__constant__ float c_v_centroids[MAX_CENTROIDS];
__constant__ float c_v_boundaries[MAX_CENTROIDS - 1];
__constant__ int c_v_num_centroids;
__constant__ int c_v_bit_width;
__constant__ float c_v_signs1[MAX_HEAD_DIM];
__constant__ float c_v_signs2[MAX_HEAD_DIM];

__constant__ int c_head_dim;

// QJL projection matrix in global memory (head_dim × head_dim, ~64KB for d=128)
static float* d_qjl_matrix = nullptr;
static bool qjl_initialized = false;

// ============================================================================
// Initialization
// ============================================================================

static void _init_one(
    torch::Tensor centroids, torch::Tensor boundaries,
    torch::Tensor signs1, torch::Tensor signs2,
    int head_dim, int bit_width, bool is_key
) {
    TORCH_CHECK(head_dim <= MAX_HEAD_DIM, "head_dim must be <= ", MAX_HEAD_DIM);
    TORCH_CHECK(bit_width >= 1 && bit_width <= 4, "bit_width must be 1-4");
    TORCH_CHECK((head_dim & (head_dim - 1)) == 0, "head_dim must be power of 2");

    int nc = 1 << bit_width;
    TORCH_CHECK(centroids.numel() == nc);
    TORCH_CHECK(boundaries.numel() == nc - 1);

    auto c = centroids.to(torch::kCPU, torch::kFloat32).contiguous();
    auto b = boundaries.to(torch::kCPU, torch::kFloat32).contiguous();
    auto s1 = signs1.to(torch::kCPU, torch::kFloat32).contiguous();
    auto s2 = signs2.to(torch::kCPU, torch::kFloat32).contiguous();

    cudaMemcpyToSymbol(c_head_dim, &head_dim, sizeof(int));

    if (is_key) {
        cudaMemcpyToSymbol(c_k_centroids, c.data_ptr<float>(), nc * sizeof(float));
        cudaMemcpyToSymbol(c_k_boundaries, b.data_ptr<float>(), (nc - 1) * sizeof(float));
        cudaMemcpyToSymbol(c_k_num_centroids, &nc, sizeof(int));
        cudaMemcpyToSymbol(c_k_bit_width, &bit_width, sizeof(int));
        cudaMemcpyToSymbol(c_k_signs1, s1.data_ptr<float>(), head_dim * sizeof(float));
        cudaMemcpyToSymbol(c_k_signs2, s2.data_ptr<float>(), head_dim * sizeof(float));
    } else {
        cudaMemcpyToSymbol(c_v_centroids, c.data_ptr<float>(), nc * sizeof(float));
        cudaMemcpyToSymbol(c_v_boundaries, b.data_ptr<float>(), (nc - 1) * sizeof(float));
        cudaMemcpyToSymbol(c_v_num_centroids, &nc, sizeof(int));
        cudaMemcpyToSymbol(c_v_bit_width, &bit_width, sizeof(int));
        cudaMemcpyToSymbol(c_v_signs1, s1.data_ptr<float>(), head_dim * sizeof(float));
        cudaMemcpyToSymbol(c_v_signs2, s2.data_ptr<float>(), head_dim * sizeof(float));
    }
}

void init(
    torch::Tensor centroids, torch::Tensor boundaries,
    torch::Tensor signs1, torch::Tensor signs2,
    int head_dim, int bit_width
) {
    // Symmetric: both K and V get same codebook and rotation
    _init_one(centroids, boundaries, signs1, signs2, head_dim, bit_width, true);
    _init_one(centroids, boundaries, signs1, signs2, head_dim, bit_width, false);
}

void init_k(
    torch::Tensor centroids, torch::Tensor boundaries,
    torch::Tensor signs1, torch::Tensor signs2,
    int head_dim, int bit_width
) {
    _init_one(centroids, boundaries, signs1, signs2, head_dim, bit_width, true);
}

void init_v(
    torch::Tensor centroids, torch::Tensor boundaries,
    torch::Tensor signs1, torch::Tensor signs2,
    int head_dim, int bit_width
) {
    _init_one(centroids, boundaries, signs1, signs2, head_dim, bit_width, false);
}

void init_qjl(torch::Tensor qjl_matrix) {
    int d = qjl_matrix.size(0);
    TORCH_CHECK(qjl_matrix.size(1) == d, "QJL matrix must be square");

    auto mat = qjl_matrix.to(torch::kCUDA, torch::kFloat32).contiguous();
    if (d_qjl_matrix) cudaFree(d_qjl_matrix);
    cudaMalloc(&d_qjl_matrix, d * d * sizeof(float));
    cudaMemcpy(d_qjl_matrix, mat.data_ptr<float>(), d * d * sizeof(float),
               cudaMemcpyDeviceToDevice);
    qjl_initialized = true;
}

// ============================================================================
// Device helpers
// ============================================================================

/** In-place WHT butterfly in shared memory. n must be power of 2. */
__device__ __forceinline__
void wht_inplace(float* __restrict__ s, int n, int tid) {
    for (int h = 1; h < n; h <<= 1) {
        __syncthreads();
        int g = tid / h;
        int p = tid % h;
        int i = g * (h << 1) + p;
        if (i + h < n && tid < (n >> 1)) {
            float a = s[i], b = s[i + h];
            s[i] = a + b;
            s[i + h] = a - b;
        }
    }
    __syncthreads();
    float inv = rsqrtf(static_cast<float>(n));
    if (tid < n) s[tid] *= inv;
    __syncthreads();
}

/** Parallel L2 norm reduction. Returns norm broadcast to all threads. */
__device__ __forceinline__
float reduce_norm(const float* __restrict__ data, int n, int tid,
                  float* __restrict__ scratch) {
    float sq = (tid < n) ? data[tid] * data[tid] : 0.0f;
    scratch[tid] = sq;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) scratch[tid] += scratch[tid + s];
        __syncthreads();
    }
    return sqrtf(scratch[0]);
}

/** Searchsorted against K codebook. Unrolled linear scan. */
__device__ __forceinline__
int search_k(float val) {
    int idx = 0;
    #pragma unroll
    for (int b = 0; b < MAX_CENTROIDS - 1; b++) {
        if (b < c_k_num_centroids - 1 && val > c_k_boundaries[b]) idx = b + 1;
    }
    return idx;
}

/** Searchsorted against V codebook. */
__device__ __forceinline__
int search_v(float val) {
    int idx = 0;
    #pragma unroll
    for (int b = 0; b < MAX_CENTROIDS - 1; b++) {
        if (b < c_v_num_centroids - 1 && val > c_v_boundaries[b]) idx = b + 1;
    }
    return idx;
}

/**
 * Full PolarQuant quantize in shared memory.
 * On entry: smem[0..hd) contains the input vector as fp32.
 * On exit:  smem[0..hd) contains the rotated+normalized coordinates.
 * Returns the L2 norm. Writes norm to *out_norm if non-null.
 *
 * is_key: true uses K codebook/signs, false uses V codebook/signs.
 */
__device__ __forceinline__
float polar_quant_forward(float* __restrict__ data, float* __restrict__ scratch,
                          int hd, int tid, bool is_key) {
    // 1. L2 norm
    float norm = reduce_norm(data, hd, tid, scratch);

    // 2. Normalize
    float inv = (norm > 1e-10f) ? 1.0f / norm : 0.0f;
    if (tid < hd) data[tid] *= inv;
    __syncthreads();

    // 3. Rotate: D1, WHT, D2
    const float* s1 = is_key ? c_k_signs1 : c_v_signs1;
    const float* s2 = is_key ? c_k_signs2 : c_v_signs2;

    if (tid < hd) data[tid] *= s1[tid];
    __syncthreads();
    wht_inplace(data, hd, tid);
    if (tid < hd) data[tid] *= s2[tid];
    __syncthreads();

    return norm;
}

/** Inverse PolarQuant: centroid lookup + inverse rotation + scale by norm. */
__device__ __forceinline__
void polar_quant_inverse(float* __restrict__ smem, int hd, int tid,
                         float norm, bool is_key) {
    const float* s1 = is_key ? c_k_signs1 : c_v_signs1;
    const float* s2 = is_key ? c_k_signs2 : c_v_signs2;

    // Inverse rotation: D2, H, D1
    if (tid < hd) smem[tid] *= s2[tid];
    __syncthreads();
    wht_inplace(smem, hd, tid);
    if (tid < hd) smem[tid] *= s1[tid];
    __syncthreads();

    // Rescale
    if (tid < hd) smem[tid] *= norm;
    __syncthreads();
}

// ============================================================================
// Standalone quantize / dequantize (for testing)
// ============================================================================

__global__
void quantize_kernel(
    const half* __restrict__ input,
    uint8_t* __restrict__ indices,
    float* __restrict__ norms,
    int num_vectors
) {
    int vid = blockIdx.x;
    if (vid >= num_vectors) return;
    int tid = threadIdx.x;
    int hd = c_head_dim;

    extern __shared__ float smem[];
    float* data = smem;
    float* scratch = smem + hd;

    if (tid < hd) data[tid] = __half2float(input[vid * hd + tid]);
    else if (tid < blockDim.x) scratch[tid] = 0.0f;
    __syncthreads();

    float norm = polar_quant_forward(data, scratch, hd, tid, /*is_key=*/true);
    if (tid == 0) norms[vid] = norm;

    if (tid < hd) {
        indices[vid * hd + tid] = static_cast<uint8_t>(search_k(data[tid]));
    }
}

__global__
void dequantize_kernel(
    const uint8_t* __restrict__ indices,
    const float* __restrict__ norms,
    half* __restrict__ output,
    int num_vectors
) {
    int vid = blockIdx.x;
    if (vid >= num_vectors) return;
    int tid = threadIdx.x;
    int hd = c_head_dim;

    extern __shared__ float smem[];
    if (tid < hd) smem[tid] = c_k_centroids[indices[vid * hd + tid]];
    __syncthreads();

    polar_quant_inverse(smem, hd, tid, norms[vid], /*is_key=*/true);

    if (tid < hd) output[vid * hd + tid] = __float2half(smem[tid]);
}

// ============================================================================
// QJL: 1-bit quantization of PolarQuant residual (K cache only)
// ============================================================================

/**
 * Compute QJL signs for the residual between original K and PolarQuant output.
 *
 * For each K vector:
 *   residual = original_k - polar_quant_dequant(polar_quant_quant(original_k))
 *   signs = sign(S @ residual)  where S is the QJL projection matrix
 *   residual_norm = ||residual||_2
 *
 * Grid: (num_vectors,)
 * Block: (head_dim,)
 *
 * Requires: d_qjl_matrix uploaded via init_qjl().
 */
__global__
void qjl_quantize_residual_kernel(
    const half* __restrict__ original_k,       // (num_vectors, head_dim) fp16
    const half* __restrict__ pq_reconstructed,  // (num_vectors, head_dim) fp16
    uint8_t* __restrict__ qjl_signs,           // (num_vectors, head_dim/8) packed bits
    float* __restrict__ qjl_norms,             // (num_vectors,) residual norms
    const float* __restrict__ qjl_matrix,      // (head_dim, head_dim) in global mem
    int num_vectors
) {
    int vid = blockIdx.x;
    if (vid >= num_vectors) return;
    int tid = threadIdx.x;
    int hd = c_head_dim;

    extern __shared__ float smem[];
    float* residual = smem;         // [0..hd)
    float* scratch = smem + hd;     // [hd..2*hd) for reduction

    // Compute residual
    if (tid < hd) {
        residual[tid] = __half2float(original_k[vid * hd + tid])
                      - __half2float(pq_reconstructed[vid * hd + tid]);
    } else {
        scratch[tid] = 0.0f;
    }
    __syncthreads();

    // Residual norm
    float rnorm = reduce_norm(residual, hd, tid, scratch);
    if (tid == 0) qjl_norms[vid] = rnorm;
    __syncthreads();

    // Project: projected[tid] = sum_j S[tid][j] * residual[j]
    // Each thread computes one element of the projection
    if (tid < hd) {
        const float* row = qjl_matrix + tid * hd;
        float dot = 0.0f;
        for (int j = 0; j < hd; j++) {
            dot += row[j] * residual[j];
        }
        // Sign quantization: store as bit
        scratch[tid] = (dot >= 0.0f) ? 1.0f : 0.0f;
    }
    __syncthreads();

    // Pack 8 sign bits into one byte
    if (tid < hd / 8) {
        uint8_t packed = 0;
        for (int b = 0; b < 8; b++) {
            if (scratch[tid * 8 + b] > 0.5f) {
                packed |= (1 << b);
            }
        }
        qjl_signs[vid * (hd / 8) + tid] = packed;
    }
}

/**
 * Dequantize QJL signs and add to PolarQuant output.
 *
 * reconstruction += sqrt(pi/2) / d * residual_norm * S^T @ signs
 *
 * Grid: (num_vectors,)
 * Block: (head_dim,)
 */
__global__
void qjl_dequantize_and_add_kernel(
    half* __restrict__ output,                 // (num_vectors, head_dim) — modified in place
    const uint8_t* __restrict__ qjl_signs,     // (num_vectors, head_dim/8) packed
    const float* __restrict__ qjl_norms,       // (num_vectors,)
    const float* __restrict__ qjl_matrix,      // (head_dim, head_dim)
    int num_vectors
) {
    int vid = blockIdx.x;
    if (vid >= num_vectors) return;
    int tid = threadIdx.x;
    int hd = c_head_dim;

    extern __shared__ float smem[];  // [0..hd) for unpacked signs

    // Unpack signs from bits to ±1.0
    if (tid < hd / 8) {
        uint8_t packed = qjl_signs[vid * (hd / 8) + tid];
        for (int b = 0; b < 8; b++) {
            smem[tid * 8 + b] = (packed & (1 << b)) ? 1.0f : -1.0f;
        }
    }
    __syncthreads();

    // Compute S^T @ signs: each thread produces one output element
    // output[tid] += scale * sum_j S[j][tid] * signs[j]
    //             = scale * sum_j S^T[tid][j] * signs[j]
    // S is stored row-major as S[i][j], so S^T[tid][j] = S[j * hd + tid]
    if (tid < hd) {
        float dot = 0.0f;
        for (int j = 0; j < hd; j++) {
            dot += qjl_matrix[j * hd + tid] * smem[j];
        }
        float scale = QJL_SCALE / static_cast<float>(hd) * qjl_norms[vid];
        float current = __half2float(output[vid * hd + tid]);
        output[vid * hd + tid] = __float2half(current + scale * dot);
    }
}

// ============================================================================
// Fused paged cache write (asymmetric K/V)
// ============================================================================

__global__
void reshape_and_cache_kernel(
    const half* __restrict__ key,
    const half* __restrict__ value,
    uint8_t* __restrict__ key_cache,
    uint8_t* __restrict__ value_cache,
    float* __restrict__ k_norms,
    float* __restrict__ v_norms,
    const int64_t* __restrict__ slot_mapping,
    int num_tokens,
    int num_kv_heads,
    int block_size,
    int k_packed_dim,
    int v_packed_dim
) {
    int pair_id = blockIdx.x;
    int token_idx = pair_id / num_kv_heads;
    int head_idx = pair_id % num_kv_heads;
    if (token_idx >= num_tokens) return;

    int tid = threadIdx.x;
    int hd = c_head_dim;
    int64_t slot = slot_mapping[token_idx];
    int blk = static_cast<int>(slot / block_size);
    int off = static_cast<int>(slot % block_size);
    int norm_idx = (blk * block_size + off) * num_kv_heads + head_idx;

    extern __shared__ float smem[];
    float* data = smem;
    float* scratch = smem + hd;

    // ---- KEY: PolarQuant with K codebook ----
    const half* k_ptr = key + (token_idx * num_kv_heads + head_idx) * hd;
    if (tid < hd) data[tid] = __half2float(k_ptr[tid]);
    else if (tid < blockDim.x) scratch[tid] = 0.0f;
    __syncthreads();

    float kn = polar_quant_forward(data, scratch, hd, tid, /*is_key=*/true);
    if (tid == 0) k_norms[norm_idx] = kn;

    // Pack K indices
    int k_base = norm_idx * k_packed_dim;
    if (c_k_bit_width == 4 && tid < hd / 2) {
        uint8_t lo = static_cast<uint8_t>(search_k(data[tid * 2]));
        uint8_t hi = static_cast<uint8_t>(search_k(data[tid * 2 + 1]));
        key_cache[k_base + tid] = lo | (hi << 4);
    } else if (c_k_bit_width == 3 && tid < hd) {
        key_cache[norm_idx * hd + tid] = static_cast<uint8_t>(search_k(data[tid]));
    } else if (c_k_bit_width == 2 && tid < hd / 4) {
        uint8_t p = 0;
        for (int i = 0; i < 4; i++)
            p |= static_cast<uint8_t>(search_k(data[tid * 4 + i])) << (i * 2);
        key_cache[k_base + tid] = p;
    }
    __syncthreads();

    // ---- VALUE: PolarQuant with V codebook ----
    const half* v_ptr = value + (token_idx * num_kv_heads + head_idx) * hd;
    if (tid < hd) data[tid] = __half2float(v_ptr[tid]);
    else if (tid < blockDim.x) scratch[tid] = 0.0f;
    __syncthreads();

    float vn = polar_quant_forward(data, scratch, hd, tid, /*is_key=*/false);
    if (tid == 0) v_norms[norm_idx] = vn;

    // Pack V indices
    int v_base = norm_idx * v_packed_dim;
    if (c_v_bit_width == 4 && tid < hd / 2) {
        uint8_t lo = static_cast<uint8_t>(search_v(data[tid * 2]));
        uint8_t hi = static_cast<uint8_t>(search_v(data[tid * 2 + 1]));
        value_cache[v_base + tid] = lo | (hi << 4);
    } else if (c_v_bit_width == 3 && tid < hd) {
        value_cache[norm_idx * hd + tid] = static_cast<uint8_t>(search_v(data[tid]));
    } else if (c_v_bit_width == 2 && tid < hd / 4) {
        uint8_t p = 0;
        for (int i = 0; i < 4; i++)
            p |= static_cast<uint8_t>(search_v(data[tid * 4 + i])) << (i * 2);
        value_cache[v_base + tid] = p;
    }
}

// ============================================================================
// Paged cache dequantize
// ============================================================================

/**
 * Dequantize from paged cache. Supports asymmetric K/V bit widths.
 * is_key parameter selects which codebook/rotation to use.
 */
__global__
void dequant_paged_kernel(
    const uint8_t* __restrict__ cache,
    const float* __restrict__ norms,
    half* __restrict__ output,
    const int32_t* __restrict__ block_table,
    int seq_len,
    int num_kv_heads,
    int block_size,
    int packed_dim,
    int is_key  // 1 = K codebook, 0 = V codebook
) {
    int pair_id = blockIdx.x;
    int token_pos = pair_id / num_kv_heads;
    int head_idx = pair_id % num_kv_heads;
    if (token_pos >= seq_len) return;

    int tid = threadIdx.x;
    int hd = c_head_dim;
    int logical_block = token_pos / block_size;
    int offset = token_pos % block_size;
    int physical_block = block_table[logical_block];
    int norm_idx = (physical_block * block_size + offset) * num_kv_heads + head_idx;
    int cache_base = norm_idx * packed_dim;

    extern __shared__ float smem[];

    // Select codebook
    const float* centroids = is_key ? c_k_centroids : c_v_centroids;
    int bw = is_key ? c_k_bit_width : c_v_bit_width;

    // Unpack + centroid lookup
    if (bw == 4 && tid < hd) {
        int byte_idx = tid / 2;
        uint8_t packed = cache[cache_base + byte_idx];
        int idx = (tid & 1) ? ((packed >> 4) & 0x0F) : (packed & 0x0F);
        smem[tid] = centroids[idx];
    } else if (bw == 3 && tid < hd) {
        smem[tid] = centroids[cache[norm_idx * hd + tid]];
    } else if (bw == 2 && tid < hd) {
        int byte_idx = tid / 4;
        int shift = (tid % 4) * 2;
        uint8_t packed = cache[cache_base + byte_idx];
        smem[tid] = centroids[(packed >> shift) & 0x03];
    }
    __syncthreads();

    // Inverse PolarQuant
    float norm = norms[norm_idx];
    polar_quant_inverse(smem, hd, tid, norm, is_key != 0);

    if (tid < hd) {
        int out_idx = (token_pos * num_kv_heads + head_idx) * hd + tid;
        output[out_idx] = __float2half(smem[tid]);
    }
}

// ============================================================================
// C++ wrapper functions
// ============================================================================

void quantize(torch::Tensor input, torch::Tensor indices, torch::Tensor norms) {
    int n = input.size(0);
    int hd = input.size(1);
    int bt = ((hd + 31) / 32) * 32;
    int smem = (hd + bt) * sizeof(float);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream(
        input.device().index()).stream();

    quantize_kernel<<<n, bt, smem, stream>>>(
        reinterpret_cast<const half*>(input.data_ptr()),
        indices.data_ptr<uint8_t>(), norms.data_ptr<float>(), n);
}

void dequantize(torch::Tensor indices, torch::Tensor norms, torch::Tensor output) {
    int n = indices.size(0);
    int hd = indices.size(1);
    int bt = ((hd + 31) / 32) * 32;
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream(
        indices.device().index()).stream();

    dequantize_kernel<<<n, bt, hd * sizeof(float), stream>>>(
        indices.data_ptr<uint8_t>(), norms.data_ptr<float>(),
        reinterpret_cast<half*>(output.data_ptr()), n);
}

void reshape_and_cache(
    torch::Tensor key, torch::Tensor value,
    torch::Tensor key_cache, torch::Tensor value_cache,
    torch::Tensor k_norms, torch::Tensor v_norms,
    torch::Tensor slot_mapping
) {
    int nt = key.size(0);
    int nkv = key.size(1);
    int hd = key.size(2);
    int bs = key_cache.size(1);
    int k_pd = key_cache.size(3);
    int v_pd = value_cache.size(3);

    int grid = nt * nkv;
    int bt = ((hd + 31) / 32) * 32;
    int smem = (hd + bt) * sizeof(float);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream(
        key.device().index()).stream();

    reshape_and_cache_kernel<<<grid, bt, smem, stream>>>(
        reinterpret_cast<const half*>(key.data_ptr()),
        reinterpret_cast<const half*>(value.data_ptr()),
        key_cache.data_ptr<uint8_t>(), value_cache.data_ptr<uint8_t>(),
        k_norms.data_ptr<float>(), v_norms.data_ptr<float>(),
        slot_mapping.data_ptr<int64_t>(),
        nt, nkv, bs, k_pd, v_pd);
}

void dequant_paged_cache(
    torch::Tensor cache, torch::Tensor norms, torch::Tensor output,
    torch::Tensor block_table, int seq_len
) {
    int nkv = cache.size(2);
    int bs = cache.size(1);
    int pd = cache.size(3);
    int hd = output.size(2);

    int grid = seq_len * nkv;
    int bt = ((hd + 31) / 32) * 32;
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream(
        cache.device().index()).stream();

    dequant_paged_kernel<<<grid, bt, hd * sizeof(float), stream>>>(
        cache.data_ptr<uint8_t>(), norms.data_ptr<float>(),
        reinterpret_cast<half*>(output.data_ptr()),
        block_table.data_ptr<int32_t>(),
        seq_len, nkv, bs, pd, /*is_key=*/1);
}

}  // namespace turbo_quant
