# turboquant-vllm

TurboQuant+ KV cache compression for vLLM. Compress the KV cache 3.7x during inference — serve more concurrent conversations on the same GPU, or use longer context at the same cost.

In early benchmarks on A100 80GB (Qwen3-30B, 5 multi-turn conversation scenarios), TQ+ **matched or outscored the uncompressed baseline on every scenario** while reducing KV cache memory to 27% of FP16. The impact scales with how much of your VRAM is KV cache — for large models at long context, that's most of it.

```python
from turboquant_vllm import patch_vllm_attention

patch_vllm_attention(k_bits=4, v_bits=4)  # before starting vLLM engine

# Then start vLLM as usual — KV cache is compressed transparently
```

## Why this exists

vLLM offers FP8 KV cache (2x compression). For large MoE models at production context lengths, the KV cache is the memory bottleneck — not the weights. TurboQuant+ gives 3.7-4.7x compression with minimal quality loss:

| KV cache type | Compression | Per-vector overhead | Quality impact |
|---------------|-------------|---------------------|----------------|
| FP16 (default) | 1x | 512 bytes | baseline |
| FP8 (vLLM built-in) | 2x | 256 bytes | negligible |
| **TQ+ turbo4** | **3.7x** | 140 bytes (K: 72 + V: 68) | **+0.23% PPL** |
| TQ+ turbo3 | 4.7x* | 108 bytes (K: 56 + V: 52) | +1.06% PPL |
| TQ+ asymmetric K4/V3 | ~4.0x | 124 bytes (K: 72 + V: 52) | K precision preserved |

*turbo3 with proper 3-bit sub-byte packing. Current implementation stores 3-bit as 1 byte per index (2.7x) — packing is a known TODO.

Norm storage is already optimal: one fp32 norm per 128-element vector (head_dim = block_size), matching the [block-size optimization](https://github.com/TheTom/turboquant_plus/blob/main/docs/papers/block-size-experiment.md) finding from turboquant_plus that block_size=128 eliminates redundant norm storage for free.

### Full benchmark (20 scenarios, Verda GPU cloud)

Tested on H100 80GB and A100 80GB on [Verda](https://verda.ai) (Helsinki). 20 multi-turn conversation scenarios scored by Llama-3.3-70B judge. Results so far (more configs in progress):

| Config | Model | KV Cache | Avg Score | Latency |
|--------|-------|----------|-----------|---------|
| 4 | GLM-4.7-Flash BF16 | FP16 (baseline) | **4.61** | 5993ms |
| 11 | GLM-4.7-Flash BF16 | **TQ+ turbo4** | **4.58** | 6042ms |
| 12 | GLM-4.7-Flash BF16 | **TQ+ turbo3** | **4.63** | 5998ms |
| 5 | GLM-4.7-Flash BF16 | FP8 | 1.07 | 6299ms |
| 7 | Qwen3-30B FP16 | FP16 (baseline) | **4.73** | 4396ms |
| 8 | Qwen3-30B AWQ | FP16 | **4.67** | 3721ms |
| 3 | Qwen3-235B AWQ | FP16 (baseline) | **4.74** | 29415ms |
| 6 | Qwen3-235B AWQ | FP8 | **4.71** | 29971ms |

**Key findings:**
- **TQ+ preserves quality on MLA models.** GLM-4.7-Flash uses Multi-head Latent Attention. TQ+ turbo4 (4.58) and turbo3 (4.63) both match the FP16 baseline (4.61).
- **FP8 KV cache breaks MLA.** GLM-Flash with FP8 KV scores 1.07/5 — catastrophically broken. FP8 works fine on standard attention (Qwen3-235B: 4.71 vs 4.74 baseline).
- **AWQ weight quantization has minimal impact.** Qwen3-30B AWQ (4.67) vs FP16 (4.73).

Qwen3-235B + TQ+ results pending. GLM-4.7 355B and DeepSeek-V3 671B require larger disk provisioning.

## Install

```bash
pip install turboquant-plus-vllm@git+https://github.com/varjoranta/turboquant-vllm.git
```

For vLLM integration:
```bash
pip install "turboquant-plus-vllm[vllm] @ git+https://github.com/varjoranta/turboquant-vllm.git"
```

PyPI release coming soon.

## How it works

Based on [TurboQuant](https://arxiv.org/abs/2504.19874) (Zandieh et al., 2025) — data-oblivious vector quantization with near-optimal distortion. After a random rotation, vector coordinates follow a known Gaussian distribution, so **precomputed optimal centroids** replace learned codebooks. No calibration data needed.

Extended by [turboquant_plus](https://github.com/TheTom/turboquant_plus) for KV cache:

- **K cache:** PolarQuant at (b-1) bits + QJL at 1 bit = b bits total. QJL corrects inner product bias — critical because attention scores are inner products (Q @ K^T).
- **V cache:** PolarQuant MSE-only at full b bits. No QJL needed — V is used in a weighted sum, not inner products.
- **Asymmetric K/V:** K precision dominates quality (controls softmax routing). V can be compressed more aggressively. K4/V3 gives better compression AND better quality than symmetric turbo3.

## Usage

### Patch vLLM (simplest)

```python
from turboquant_vllm import patch_vllm_attention

# Symmetric 4-bit
patch_vllm_attention(k_bits=4, v_bits=4)

# Asymmetric: K precision preserved, V compressed more
patch_vllm_attention(k_bits=4, v_bits=3)
```

Then start vLLM normally. The patch intercepts `FlashAttentionImpl` — works with any model that uses standard FlashAttention (Qwen3, Llama, Mistral, etc.).

**Note:** Models using Multi-head Latent Attention (MLA) — DeepSeek-V3, GLM-4.7-Flash — are not yet supported. MLA already compresses KV into a latent representation with a different interface. GLM-4.7 (non-Flash) uses GQA and is supported.

### Standalone compression (without vLLM)

```python
from turboquant_vllm import KVCacheCompressorTorch

compressor = KVCacheCompressorTorch(
    head_dim=128, k_bits=4, v_bits=4, device="cuda"
)

# Compress
ck = compressor.compress_k(key_vectors)   # (num_tokens, head_dim) → CompressedKV
cv = compressor.compress_v(value_vectors)

# Decompress
k_restored = compressor.decompress_k(ck)  # → (num_tokens, head_dim) float32
v_restored = compressor.decompress_v(cv)
```

### CUDA kernel compilation

CUDA kernels are JIT-compiled on first use (requires nvcc):

```bash
# Verify CUDA kernels compile and run
python -m turboquant_vllm.build
```

If CUDA compilation fails, the system automatically falls back to PyTorch ops (slower but functionally identical).

## The CUDA kernels

Four fused kernels in `csrc/turbo_quant.cu`:

| Kernel | Purpose | Key operation |
|--------|---------|---------------|
| `reshape_and_cache_kernel` | Write path | Fused: norm → normalize → WHT rotate → searchsorted → pack 4-bit |
| `dequant_paged_kernel` | Read path | Fused: unpack → centroid lookup → inverse WHT → rescale |
| `qjl_quantize_residual_kernel` | K cache QJL | PolarQuant residual → 128×128 projection → pack sign bits |
| `qjl_dequantize_and_add_kernel` | K cache QJL | Reconstruct QJL contribution, add to PolarQuant output |

Design choices:
- **Walsh-Hadamard Transform** over dense rotation: O(d log d) vs O(d²). 896 FLOPs vs 16,384 for d=128. Fits entirely in shared memory.
- **Separate K/V codebooks** in constant memory for asymmetric bit widths.
- **4-bit packing**: two indices per byte, halves cache bandwidth.
- Targets A100 (sm_80), L40S/RTX4090 (sm_89), H100 (sm_90).

## Bandwidth argument

At 32K context, 32 layers, 32 KV heads (typical large model):

Assuming 32 layers, 32 KV heads, head_dim=128 (typical for Qwen3-235B, Llama-70B class models):

| | FP16 | Turbo4 |
|---|---|---|
| KV cache size | 17.2 GB | 4.6 GB |
| Read time at 2TB/s (A100) | 8.6 ms | 2.3 ms |
| Dequant overhead | 0 | ~0.2 ms |
| **Net per decode step** | **8.6 ms** | **2.5 ms** |

71% reduction in KV cache access time. Models with fewer KV heads (GQA) have proportionally smaller caches, but the 3.8x ratio holds.

## Compatibility

| Model family | Attention type | TQ+ support |
|-------------|---------------|-------------|
| Qwen3, Llama, Mistral | FlashAttention (GQA/MHA) | **Yes** |
| GLM-4.7 | FlashAttention (GQA) | **Yes** |
| DeepSeek-V3, GLM-4.7-Flash | Multi-head Latent Attention (MLA) | **Yes** (new) |

MLA models store a compressed latent vector (`kv_c_normed`) plus positional encoding (`k_pe`) instead of standard K/V. The patch compresses `kv_c_normed` with PolarQuant MSE-only and passes `k_pe` through uncompressed. Both `forward_mha` and `forward_mqa` paths are patched via `TritonMLAImpl`. GPU validation pending.

## Related projects

- **[turboquant-vllm on PyPI](https://pypi.org/project/turboquant-vllm/)** — A separate, independent implementation of TurboQuant for vLLM by Alberto-Codes. Uses Triton kernels and HuggingFace `DynamicCache`, targeting consumer GPUs (RTX 4090). This project differs: fused CUDA kernels for production A100/H100, asymmetric K/V bit widths (required for quantized weight models), and vLLM paged cache integration. The PyPI package for this project will be published as `turboquant-plus-vllm` to avoid confusion.
- **[turbo-quant-lite](https://pypi.org/project/turbo-quant-lite/)** — Numpy-only TurboQuant for embedding compression in databases. Same math, different codebook and use case.
- **[turboquant_plus](https://github.com/TheTom/turboquant_plus)** — Research implementation of the KV cache algorithm. This package builds production CUDA kernels on top of that work.
- **[TurboQuant paper](https://arxiv.org/abs/2504.19874)** — Zandieh et al., 2025. The underlying algorithm.

## License

MIT — Varjosoft Oy
