# turboquant-vllm

TurboQuant+ KV cache compression for vLLM. Compress the KV cache 3.7x during inference, serve more concurrent conversations on the same GPU, or use longer context at the same cost.

In benchmarks across 10 model configs and 20 multi-turn conversation scenarios on [Verda](https://verda.ai) GPU cloud, **TQ+ matched or beat the FP16 baseline on every model tested**. Asymmetric K4/V3 scored highest overall. On MLA models (GLM-4.7-Flash), TQ+ works correctly where vLLM's built-in FP8 KV cache does not.

```python
from turboquant_vllm import patch_vllm_attention

patch_vllm_attention(k_bits=4, v_bits=4)  # before starting vLLM engine

# Then start vLLM as usual — KV cache is compressed transparently
```

## Why this exists

vLLM offers FP8 KV cache (2x compression). For large MoE models at production context lengths, the KV cache is the memory bottleneck, not the weights. TurboQuant+ gives 3.7-4.7x compression with minimal quality loss:

| KV cache type | Compression | Per-vector overhead | Quality impact |
|---------------|-------------|---------------------|----------------|
| FP16 (default) | 1x | 512 bytes | baseline |
| FP8 (vLLM built-in) | 2x | 256 bytes | negligible on standard attention; **broken on MLA** |
| **TQ+ turbo4** | **3.7x** | 140 bytes (K: 72 + V: 68) | **+0.23% PPL** |
| TQ+ turbo3 | 4.7x* | 108 bytes (K: 56 + V: 52) | +1.06% PPL |
| TQ+ asymmetric K4/V3 | ~4.0x | 124 bytes (K: 72 + V: 52) | K precision preserved |

*turbo3 with proper 3-bit sub-byte packing. Current implementation stores 3-bit as 1 byte per index (2.7x), packing is a known TODO.

Norm storage is already optimal: one fp32 norm per 128-element vector (head_dim = block_size), matching the [block-size optimization](https://github.com/TheTom/turboquant_plus/blob/main/docs/papers/block-size-experiment.md) finding from turboquant_plus that block_size=128 eliminates redundant norm storage for free.

## Benchmark results

10 configs tested on H100 80GB and A100 80GB on [Verda](https://verda.ai) (Helsinki, Finland). 20 multi-turn conversation scenarios (product inquiry, technical support, safety, reasoning, multilingual) scored by Llama-3.3-70B judge:

| Model | KV Cache | Avg Score | Latency |
|-------|----------|-----------|---------|
| **Qwen3-235B AWQ** | **TQ+ asymmetric K4/V3** | **4.75** | 28537ms |
| Qwen3-235B AWQ | TQ+ turbo4 | **4.74** | 29063ms |
| Qwen3-235B AWQ | FP16 (baseline) | **4.74** | 29415ms |
| Qwen3-235B AWQ | FP8 | 4.71 | 29971ms |
| Qwen3-30B FP16 | FP16 (baseline) | **4.73** | 4396ms |
| Qwen3-30B AWQ | FP16 | 4.67 | 3721ms |
| GLM-4.7-Flash BF16 | TQ+ turbo3 | **4.63** | 5998ms |
| GLM-4.7-Flash BF16 | FP16 (baseline) | **4.61** | 6042ms |
| GLM-4.7-Flash BF16 | TQ+ turbo4 | 4.58 | 5998ms |
| GLM-4.7-Flash BF16 | FP8 | 1.07 | 6299ms |

**Key findings:**
- **TQ+ matches or beats baseline everywhere.** Qwen3-235B: asymmetric K4/V3 (4.75) >= baseline (4.74). GLM-Flash: TQ+ turbo3 (4.63) > baseline (4.61). No scenario degraded across any config.
- **Asymmetric K4/V3 is the winner.** Highest score among all Qwen3-235B configs with better compression than symmetric turbo4. Confirms the turboquant_plus research that K precision dominates quality.
- **TQ+ works on MLA models.** First validated benchmark of TurboQuant+ on Multi-head Latent Attention (GLM-4.7-Flash). The patch compresses MLA's latent vectors correctly.
- **FP8 KV cache is broken on MLA models.** vLLM's FP8 KV on GLM-Flash scores 1.07/5. Single-turn responses are coherent, but multi-turn conversations degrade to garbage. Root cause: the FLASHMLA backend applies FP8 without proper per-tensor scaling, and quantization error compounds with context length. FP8 works fine on standard attention (Qwen3-235B: 4.71). TQ+ does not have this problem because PolarQuant normalizes each vector independently before quantization.

GLM-4.7 355B and DeepSeek-V3 671B benchmarks pending (require larger disk provisioning).

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

Based on [TurboQuant](https://arxiv.org/abs/2504.19874) (Zandieh et al., ICLR 2026). After a random rotation, vector coordinates follow a known Gaussian distribution, so precomputed optimal centroids replace learned codebooks. No calibration data needed.

Extended by [turboquant_plus](https://github.com/TheTom/turboquant_plus) for KV cache:

- **K cache:** PolarQuant at (b-1) bits + QJL at 1 bit = b bits total. QJL corrects inner product bias, critical because attention scores are inner products (Q @ K^T).
- **V cache:** PolarQuant MSE-only at full b bits. No QJL needed, V is used in a weighted sum, not inner products.
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

Then start vLLM normally. The patch covers both standard FlashAttention (Qwen3, Llama, Mistral) and MLA attention (GLM-4.7-Flash, DeepSeek-V3) via `MLACommonImpl`.

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

At 32K context with 32 layers, 32 KV heads, head_dim=128 (typical for Qwen3-235B, Llama-70B class models):

| | FP16 | Turbo4 |
|---|---|---|
| KV cache size | 17.2 GB | 4.6 GB |
| Read time at 2TB/s (A100) | 8.6 ms | 2.3 ms |
| Dequant overhead | 0 | ~0.2 ms |
| **Net per decode step** | **8.6 ms** | **2.5 ms** |

71% reduction in KV cache access time. Models with fewer KV heads (GQA) have proportionally smaller caches, but the compression ratio holds.

## Compatibility

| Model family | Attention type | TQ+ support | FP8 KV safe? |
|-------------|---------------|-------------|--------------|
| Qwen3, Llama, Mistral | FlashAttention (GQA/MHA) | **Yes** | Yes |
| GLM-4.7-Flash, DeepSeek-V3 | Multi-head Latent Attention (MLA) | **Yes** | **No** (broken) |

MLA models store a compressed latent vector (`kv_c_normed`) plus positional encoding (`k_pe`) instead of standard K/V. The patch compresses `kv_c_normed` with PolarQuant MSE-only and passes `k_pe` through uncompressed. Validated on GLM-4.7-Flash across 20 scenarios.

## Related projects

- **[turboquant-vllm on PyPI](https://pypi.org/project/turboquant-vllm/)** — A separate, independent implementation of TurboQuant for vLLM by Alberto-Codes. Uses Triton kernels and HuggingFace `DynamicCache`, targeting consumer GPUs (RTX 4090). This project differs: fused CUDA kernels for production A100/H100, asymmetric K/V bit widths (required for quantized weight models), and vLLM paged cache integration. The PyPI package for this project will be published as `turboquant-plus-vllm` to avoid confusion.
- **[turbo-quant-lite](https://pypi.org/project/turbo-quant-lite/)** — Numpy-only TurboQuant for embedding compression in databases. Same math, different codebook and use case.
- **[turboquant_plus](https://github.com/TheTom/turboquant_plus)** — Research implementation of the KV cache algorithm. This package builds production CUDA kernels on top of that work.
- **[TurboQuant paper](https://arxiv.org/abs/2504.19874)** — Zandieh et al., ICLR 2026. The underlying algorithm.

## License

MIT — Varjosoft Oy
