# turboquant-vllm

TurboQuant+ KV cache compression for vLLM. Compress the KV cache 3.8x during inference — serve more concurrent conversations on the same GPU, or use longer context at the same cost.

In early benchmarks on A100 80GB (Qwen3-30B, 5 multi-turn conversation scenarios), TQ+ **matched or outscored the uncompressed baseline on every scenario** while reducing KV cache memory to 26% of FP16. The impact scales with how much of your VRAM is KV cache — for large models at long context, that's most of it.

```python
from turboquant_vllm import patch_vllm_attention

patch_vllm_attention(k_bits=4, v_bits=4)  # before starting vLLM engine

# Then start vLLM as usual — KV cache is compressed transparently
```

## Why this exists

vLLM offers FP8 KV cache (2x compression). For large MoE models at production context lengths, the KV cache is the memory bottleneck — not the weights. TurboQuant+ gives 3.8-4.6x compression with minimal quality loss:

| KV cache type | Compression | Quality impact |
|---------------|-------------|----------------|
| FP16 (default) | 1x | baseline |
| FP8 (vLLM built-in) | 2x | negligible |
| **TQ+ turbo4** | **3.8x** | **+0.23% PPL** |
| TQ+ turbo3 | 4.6x | +1.06% PPL |
| TQ+ asymmetric K4/V3 | ~4.2x | K precision preserved |

Tested on A100 80GB with Qwen3-30B-A3B-AWQ across 5 multi-turn conversation scenarios (product inquiry, technical support, adversarial injection, reasoning, multilingual), scored by Llama-3.3-70B judge:

| Scenario | Baseline | TQ+ turbo4 |
|----------|----------|------------|
| Product inquiry (EN) | 4.25 | **4.50** |
| Technical support (EN) | 4.25 | **4.50** |
| Product inquiry (FI) | 4.75 | 4.75 |
| Adversarial injection | 5.00 | 5.00 |
| Debate/reasoning | 5.00 | 5.00 |
| **Average** | **4.65** | **4.75** |

Quality preserved or better on every scenario. Full benchmark across 15 model configs and 20 scenarios coming soon.

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

Based on [TurboQuant](https://arxiv.org/abs/2504.19874) (Zandieh et al., ICLR 2026) — data-oblivious vector quantization with near-optimal distortion. After a random rotation, vector coordinates follow a known Gaussian distribution, so **precomputed optimal centroids** replace learned codebooks. No calibration data needed.

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

**Note:** Models using Multi-head Latent Attention (MLA) — GLM-4.7, DeepSeek-V3 — are not yet supported. MLA already compresses KV into a latent representation with a different interface.

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
| GLM-4.7, DeepSeek-V3 | Multi-head Latent Attention (MLA) | Coming soon |

**MLA support:** MLA models store a compressed latent vector (`kv_c_normed`) plus positional encoding (`k_pe`) instead of standard K/V. The fix is straightforward — patch `MLACommonImpl.do_kv_cache_update` and apply TurboQuant+ to the latent vector, which is just a different-shaped tensor going through the same compress/decompress pipeline. The main thing to verify is that MLA's latent vectors follow the Gaussian distribution assumption after rotation, which is empirically testable. Targeting next release.

## Related projects

- **[turboquant-vllm on PyPI](https://pypi.org/project/turboquant-vllm/)** — A separate, independent implementation of TurboQuant for vLLM by Alberto-Codes. Uses Triton kernels and HuggingFace `DynamicCache`, targeting consumer GPUs (RTX 4090). This project differs: fused CUDA kernels for production A100/H100, asymmetric K/V bit widths (required for quantized weight models), and vLLM paged cache integration. The PyPI package for this project will be published as `turboquant-plus-vllm` to avoid confusion.
- **[turbo-quant-lite](https://pypi.org/project/turbo-quant-lite/)** — Numpy-only TurboQuant for embedding compression in databases. Same math, different codebook and use case.
- **[turboquant_plus](https://github.com/TheTom/turboquant_plus)** — Research implementation of the KV cache algorithm. This package builds production CUDA kernels on top of that work.
- **[TurboQuant paper](https://arxiv.org/abs/2504.19874)** — Zandieh et al., ICLR 2026. The underlying algorithm.

## License

MIT — Varjosoft Oy
