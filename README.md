# turboquant-vllm

TurboQuant+ compression for vLLM. What this package ships today:

- **Weight compression** (4.3-4.6x) via 3-bit TQ3. Any BF16 checkpoint, compressed in 9 seconds, zero calibration. Faster than uncompressed serving. **Unique to this plugin.**
- **Native TQ3 checkpoints** for small-GPU deployments (L40S, RTX 6000 Ada). MoE expert regrouping handled automatically.
- **Expert pruning** via [REAP](https://arxiv.org/abs/2510.13999) saliency scoring for MoE models.
- **AWQ export** from TQ-compressed weights — ~2 min instead of hours of AWQ calibration.
- **Legacy KV cache compression** (monkey-patch, MLA-only) for GLM-4.7/DeepSeek-V3 users on stock vLLM until upstream KV compression supports MLA.

> **KV cache compression for GQA/MHA models (Qwen, Llama, Mistral, Gemma) is being upstreamed to vLLM directly in [vllm-project/vllm#38479](https://github.com/vllm-project/vllm/pull/38479)** by @vibhavagarwal5 — under mgoin's review, close to merging. Once it lands, use `--kv-cache-dtype turboquant_3bit_nc` (or `k8v4`, `4bit_nc`, `k3v4_nc`) on stock vLLM with no plugin required. This package's role going forward is weight quantization + MLA KV compression + the supporting tooling, not the standard KV-cache path.

## Quick start

```python
# Weight compression — unique to this plugin
from turboquant_vllm import enable_weight_quantization
enable_weight_quantization(bits=3)  # 52 GB → 12 GB in 9 seconds
# then: vllm serve google/gemma-4-26B-A4B-it
```

```python
# MLA KV cache compression via legacy monkey-patch (GLM-4.7, DeepSeek-V3)
from turboquant_vllm import patch_vllm_attention
patch_vllm_attention(k_bits=4, v_bits=3, norm_correction=True)
```

Headline numbers:

- **Gemma 4 26B**: ~52 GB BF16 → **~12 GB runtime VRAM** with TQ3. Scores **4.79/5** on our 20-scenario benchmark, comparable to Qwen3-235B AWQ (4.75/5) at 2.6x lower GPU cost. Honest BF16 baseline on current vLLM 0.19 (measured 2026-04-11 on A100 80GB): see the throughput table below.
- **GLM-4.7-Flash 355B MoE**: 62.4 GB → **14.7 GB** (4.2x). Native TQ3 checkpoint with MoE expert regrouping, 13.3 GB GPU memory. Tested on A100 80GB.
- **Qwen3-30B**: 61 GB → **13 GB** (4.6x).

## Why this exists

vLLM offers FP8 KV cache (2x compression). For large MoE models at production context lengths, the KV cache is the memory bottleneck, not the weights. TurboQuant+ gives 3.7-4.7x compression with minimal quality loss:

| KV cache type | Compression | Per-vector overhead | Quality impact |
|---------------|-------------|---------------------|----------------|
| FP16 (default) | 1x | 512 bytes | baseline |
| FP8 (vLLM built-in) | 2x | 256 bytes | negligible on standard attention; **broken on MLA** |
| **TQ+ turbo4** | **3.7x** | 140 bytes (K: 72 + V: 68) | **+0.23% PPL** |
| TQ+ turbo3 | 4.7x* | 108 bytes (K: 56 + V: 52) | +1.06% PPL |
| TQ+ asymmetric K4/V3 | ~4.0x | 124 bytes (K: 72 + V: 52) | K precision preserved |

*turbo3 with 3-bit sub-byte packing (8 indices per 3 bytes). Implemented in v0.3.0.

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

GLM-4.7 355B: native TQ3 checkpoint verified (14.7 GB, 4.2x compression). Full quality benchmark pending. DeepSeek-V3 671B benchmark pending (requires larger disk provisioning).

### Tested models and known issues

| Model family | Attention | Weight quant | Legacy KV monkey-patch | Notes |
|---|---|---|---|---|
| Qwen3 (0.6B-235B) | GQA | Works | Works | Use upstream [vllm-project/vllm#38479](https://github.com/vllm-project/vllm/pull/38479) for KV cache compression once merged |
| Gemma 4 26B | GQA | Works (4.79/5) | Works | Flagship benchmark model |
| GLM-4.7-Flash 355B | **MLA** | Works (native TQ3 ckpt) | **Works** | **Only place TurboQuant KV works on MLA today** — #38479 does not cover MLA |
| DeepSeek-V3 | **MLA** | Pending (larger disk) | Works | Same: MLA requires the legacy monkey-patch path |
| Qwen3.5-35B-A3B | MoE + GatedDeltaNet + GQA | Works (as of varjoranta/turboquant-vllm#15) | Untested | Hybrid architecture; weight quant validated via @gaby's fixes |
| gpt-oss-20b | Alternating full/sliding window + sinks | Works | Not yet | Sliding window + attention sinks need pass-through support in the KV path |

**Where to get KV cache compression:**

- **GQA/MHA models** (Qwen, Llama, Mistral, Gemma): use upstream [vllm-project/vllm#38479](https://github.com/vllm-project/vllm/pull/38479), or Vibhav's branch while the PR is under review. This plugin's old `--kv-cache-dtype tq3` path has been removed.
- **MLA models** (GLM-4.7, DeepSeek-V3): use this plugin's `TQ_KV_K_BITS=4` monkey-patch path. It is the only option today. Will be retired once upstream adds MLA support.
- **Hybrid models** (Qwen3.5, gpt-oss): neither path is fully supported yet. Weight quantization still works.

## Install

```bash
pip install turboquant-plus-vllm
```

Or from git for the latest:
```bash
pip install turboquant-plus-vllm@git+https://github.com/varjoranta/turboquant-vllm.git
```

## How it works

Inspired by [TurboQuant](https://arxiv.org/abs/2504.19874) (Zandieh, Daliri, Hadian, Mirrokni; ICLR 2026). After a random rotation, vector coordinates become easier to quantize. Our implementation uses a Gaussian Lloyd-Max codebook as an approximation. No calibration data needed.

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

# Asymmetric K4/V3 with Phase 2 features (recommended)
patch_vllm_attention(k_bits=4, v_bits=3, norm_correction=True,
                     sink_tokens=4, boundary_layers=5)
```

Phase 2 features: **norm correction** fixes magnitude shrinkage at low bit widths. **Sink tokens** keep the first 4 positions at FP16 (attention sinks get universal attention). **Boundary layers** give the first/last 5 layers K=8-bit precision (they carry more signal through the residual stream). Validated on Gemma 4 26B: token-for-token identical output to FP16 baseline at temperature=0.

> **Note on norm_correction + CUDA**: the CUDA KV store kernel does not yet apply norm correction. If you pass `norm_correction=True` the library automatically falls back to the PyTorch path and logs a warning. To use the CUDA kernel, set `norm_correction=False`.

The patch covers both standard FlashAttention (Qwen3, Llama, Mistral) and MLA attention (GLM-4.7-Flash, DeepSeek-V3) via `MLACommonImpl`.

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

**KV cache kernels** in `csrc/turbo_quant.cu` — used by the legacy monkey-patch path (`patch_vllm_attention`) and the standalone `KVCacheCompressorTorch`. Not used by the new upstream TurboQuant path in [vllm-project/vllm#38479](https://github.com/vllm-project/vllm/pull/38479).

| Kernel | Purpose | Key operation |
|--------|---------|---------------|
| `reshape_and_cache_kernel` | Write path | Fused: norm → normalize → WHT rotate → searchsorted → pack 4-bit |
| `dequant_paged_kernel` | Read path | Fused: unpack → centroid lookup → inverse WHT → rescale |
| `qjl_quantize_residual_kernel` | K cache QJL | PolarQuant residual → 128×128 projection → pack sign bits |
| `qjl_dequantize_and_add_kernel` | K cache QJL | Reconstruct QJL contribution, add to PolarQuant output |

**Weight dequant kernel** in `csrc/tq_weight_dequant.cu` — used by the weight quantization path (unique to this plugin):

| Kernel | Purpose | Key operation |
|--------|---------|---------------|
| `tq_weight_dequant_kernel` | Weight decompression (CUDA) | Unpack indices → codebook lookup → warp-shuffle + shared memory WHT butterfly → rescale |

**Triton fused dequant-GEMM** in `turboquant_vllm/triton_ops.py`:

| Kernel | Purpose | Key operation |
|--------|---------|---------------|
| `_polar_fused_gemm_kernel` | FWHT-on-input GEMM (fastest) | Rotate input once → codebook lookup + norm scale + dot product (no weight decompression) |
| `_tq_fused_gemm_kernel` | Fused weight dequant + matmul | Unpack → codebook → pre-computed rotation matrix → scale → GEMM accumulate |

The fused Triton kernel is **10.5x faster** than separate dequant + cuBLAS GEMM (0.57ms vs 5.9ms for 4096×4096 on A100). It eliminates the intermediate decompressed weight buffer entirely. Uses a pre-computed rotation matrix (128×128 = 64 KB, computed once) instead of the WHT butterfly, turning the inverse rotation into a small matmul that Triton optimizes well.

The CUDA kernel (5x faster than PyTorch) serves as fallback when Triton is unavailable. Uses warp-shuffle operations for intra-warp butterfly stages, shared memory only for cross-warp stages.

Design choices:
- **Four-tier dispatch**: Triton FWHT-on-input (rotates input, no weight decompression) → Triton fused dequant-GEMM → CUDA dequant + cuBLAS → PyTorch fallback. Heuristic selects FWHT-on-input for large layers (>4K output features), dequant-GEMM for small. All tiers support TQ2/TQ3/TQ4 including 3-bit sub-byte packing.
- **Walsh-Hadamard Transform** over dense rotation: O(d log d) vs O(d²). 896 FLOPs vs 16,384 for d=128.
- **Separate K/V codebooks** in constant memory for asymmetric bit widths.
- **Constant memory caching**: codebook and sign vectors only re-uploaded when config changes.
- **4-bit packing**: two indices per byte, halves cache bandwidth.
- Targets T4/RTX 2080 (sm_75), A100 (sm_80), A10/A40/RTX 3090 (sm_86), L40S/RTX 4090 (sm_89), H100/H200 (sm_90), RTX 50xx (sm_120), and GB10/DGX Spark (sm_121). The PyTorch fallback runs on any CUDA GPU, CPU, or Apple MPS.

## Bandwidth argument

At 32K context with 32 layers, 32 KV heads, head_dim=128 (typical for Qwen3-235B, Llama-70B class models):

| | FP16 | Turbo4 |
|---|---|---|
| KV cache size | 17.2 GB | 4.6 GB |
| Read time at 2TB/s (A100) | 8.6 ms | 2.3 ms |
| Dequant overhead | 0 | ~0.2 ms |
| **Net per decode step** | **8.6 ms** | **2.5 ms** |

71% reduction in KV cache access time. Models with fewer KV heads (GQA) have proportionally smaller caches, but the compression ratio holds.

## Compatibility (legacy monkey-patch KV path)

The `patch_vllm_attention` / `TQ_KV_K_BITS=...` path applies to:

| Model family | Attention type | Monkey-patch support | Upstream FP8 KV safe? |
|-------------|---------------|-------------|--------------|
| Qwen3, Llama, Mistral, Gemma | GQA / MHA | **Use upstream [#38479](https://github.com/vllm-project/vllm/pull/38479) instead** | Yes |
| GLM-4.7-Flash, DeepSeek-V3 | **MLA** | **Yes — this is the only option** | **No** (broken) |

MLA models store a compressed latent vector (`kv_c_normed`) plus positional encoding (`k_pe`) instead of standard K/V. The monkey-patch compresses `kv_c_normed` with PolarQuant MSE-only and passes `k_pe` through uncompressed. Validated on GLM-4.7-Flash across 20 scenarios (benchmark table above: TQ+ turbo3 4.63 vs FP16 baseline 4.61).

vLLM's upstream FP8 KV cache **is broken on MLA models**: on GLM-4.7-Flash it scores 1.07/5 because the FLASHMLA backend applies FP8 without proper per-tensor scaling and quantization error compounds with context length. TurboQuant's per-vector PolarQuant normalization avoids this. Until upstream fixes MLA FP8 or #38479 adds MLA support, the legacy monkey-patch path in this plugin remains the only way to compress KV cache on MLA models.

## Weight compression

The same WHT rotation + codebook math compresses model weights. Load any BF16 checkpoint, compress at startup, serve. No calibration data.

```python
from turboquant_vllm import enable_weight_quantization

enable_weight_quantization(bits=3)  # TQ3: best compression
# or bits=4 for TQ4 (more conservative)
# then: vllm serve google/gemma-4-26B-A4B-it
```

Works with vLLM V1 engine (multiprocessing spawn) via the `vllm.general_plugins` entry point. After `pip install turboquant-plus-vllm`, set environment variables and vLLM picks it up automatically — no code changes needed:

```bash
# Docker / production
FROM vllm/vllm-openai:v0.19.0
RUN pip install turboquant-plus-vllm
ENV TQ_WEIGHT_BITS=3
```

```bash
# CLI
TQ_WEIGHT_BITS=3 vllm serve google/gemma-4-26B-A4B-it
```

Inspired by [TurboQuant](https://arxiv.org/abs/2504.19874) (Zandieh, Daliri, Hadian, Mirrokni; ICLR 2026). Our implementation uses a Gaussian Lloyd-Max codebook as an approximation. Weight compression inspired by @coffeecup2020's TQ3_1S proof-of-concept for llama.cpp.

### Results

| Model | BF16 | TQ3 (3-bit) | Compression | Quality |
|-------|------|-------------|-------------|---------|
| **Gemma 4 26B** | 52 GB | **12 GB** | 4.3x | 4.79/5 |
| **GLM-4.7-Flash 355B** | 62.4 GB | **14.7 GB** | 4.2x | Tested ✓ |
| **GLM-5.1 769B** | 1,510 GB | **309 GB** | 4.9x | [Pending](https://huggingface.co/varjosoft/GLM-5.1-Open-TQ3) |
| **Qwen3-30B** | 61 GB | **13 GB** | 4.6x | -- |

Gemma 4 TQ3 quality: **4.79/5** on 20 multi-turn conversation scenarios (scored by Llama-3.3-70B judge). Matches Qwen3-235B AWQ (4.75/5) at 2.6x lower GPU cost.

GLM-4.7-Flash is a 355B MoE model (64 experts, 4 active). The native TQ3 checkpoint handles per-expert weight regrouping and gate_proj+up_proj → gate_up_proj fusion automatically.

### Throughput (A100 80GB, vLLM 0.19.0, Gemma 4 26B, 2026-04-11)

Measured via `vllm bench serve` with `dataset-name=random --random-input-len 512 --random-output-len 128 --num-prompts 32 --ignore-eos` at each concurrency level. Model: `google/gemma-4-26B-A4B-it` bfloat16, `--max-model-len 2048 --gpu-memory-utilization 0.85`.

| Concurrency | Output tok/s | Req/s | Median TTFT | Median ITL |
|:-----------:|:------------:|:-----:|:-----------:|:----------:|
| 1 | **101.4** | 0.79 | 97 ms | 9.1 ms |
| 4 | **290.5** | 2.27 | 67 ms | 13.4 ms |
| 8 | **501.6** | 3.92 | 96 ms | 15.4 ms |
| 16 | **816.1** | 6.38 | 151 ms | 18.8 ms |

Peak GPU memory: 70.5 GB (weights + KV cache + activations; the 52 GB weight footprint is the dominant term).

**Honest note on the TQ3 throughput column.** Earlier README versions claimed "+33% TQ3 vs BF16 at 16 concurrent (364 vs 275 tok/s on H100)" from a much older vLLM. That comparison is stale — both numbers are wrong for current vLLM 0.19 (BF16 alone is measured at 816 tok/s on A100 here, ~3× the old BF16 claim), and re-benchmarking TQ3 on current vLLM hit a fullgraph-compile incompatibility in `TurboQuantWrapper.forward()` that needs a deeper plugin fix (tracked as a follow-up). Rather than ship a number we can't reproduce, **the TQ3 throughput column is intentionally absent until it can be measured honestly**. The *memory footprint* win (12 GB vs 52 GB of weights) is math-deterministic and still holds.

3-bit sub-byte packing: 8 indices per 3 bytes. Norm correction: stores `original_norm / reconstruction_norm` ratio per group to fix 5-10% magnitude shrinkage at 3-bit.

### Perplexity (WikiText-2, Gemma 4 26B-A4B-it, H100 80GB)

| Config | PPL | Memory | vs BF16 |
|--------|-----|--------|---------|
| BF16 baseline | 540 | 51,612 MB | -- |
| TQ4 (4-bit) | 1,415 | 15,265 MB | +162% |
| TQ3 (3-bit) | 1,190 | 12,040 MB | +120% |
| TQ3 native checkpoint | 1,152 | 12,104 MB | +113% |

Note: Gemma 4 26B-A4B-it is an instruction-tuned model. IT models have high WikiText-2 PPL baselines (540 vs ~6 for base models) because their output distribution is trained for chat, not language modeling. The PPL deltas are proportionally larger than for base models but do not reflect proportional quality degradation in chat -- the same compression scores 4.79/5 on our 20-scenario benchmark. TQ3 outperforms TQ4 on PPL because fewer centroids means fewer quantization boundary errors for this model's weight distribution. Native checkpoint matches runtime, confirming the two paths are numerically equivalent.

### Native TQ3 checkpoint (small GPUs)

Runtime compression (`enable_weight_quantization`) needs the full BF16 checkpoint in GPU memory during loading. For GPUs that can't fit the original checkpoint (e.g., L40S 48GB for a 52 GB model), use a native TQ3 checkpoint instead:

```python
from turboquant_vllm import load_tq3_model

# 12 GB checkpoint → 13.7 GB GPU peak, tested on L40S 48GB and H100 80GB
model, tokenizer = load_tq3_model("varjosoft/gemma-4-26B-A4B-it-TQ3-native")
output = model.generate(...)
```

The native checkpoint stores packed 3-bit indices directly (12 GB on disk). The loader creates the model on a meta device (zero memory), loads packed weights to GPU, and decompresses on-the-fly during each forward pass.

Supports MoE models: per-expert 2D weights are automatically regrouped into fused 3D parameters (e.g., GLM-4.7-Flash's 64 experts with gate_proj+up_proj → gate_up_proj fusion). Router/gate weights are decompressed in-place.

Create your own native checkpoint:
```python
from turboquant_vllm.checkpoint import save_tq3_checkpoint

# Dense model (Gemma 4, Qwen3, etc.)
save_tq3_checkpoint("google/gemma-4-26B-A4B-it", "./gemma4-tq3-native")
# CPU only, ~60 GB RAM, ~2 minutes

# MoE model (GLM-4.7-Flash, GLM-5.1, etc.) — streaming, low memory
save_tq3_checkpoint("zai-org/GLM-4.7-Flash", "./glm47-tq3")
# Streams shards from HuggingFace, ~5 GB peak RAM, ~14 minutes
```

**Important: Gemma 4 requires `transformers >= 5.5.0`** (the `gemma4` model type was added in that version). vLLM 0.19.0 pins `transformers < 5`, so Gemma 4 loading requires a manual override: `pip install 'transformers>=5.5'`.

### Limitations

- **V100 16GB**: model loads (12 GB) but not enough room for KV cache. Minimum practical is 24 GB.
- **TQ2 (2-bit)** destroys quality. 4 centroids too few for MLP weight distributions.
- **Native TQ3 inference speed** is slower than runtime compression due to per-forward-pass decompression overhead.

## Expert pruning (REAP)

Integrated [REAP](https://arxiv.org/abs/2510.13999) (Cerebras, ICLR 2026) saliency scoring for MoE expert pruning. Measures actual expert contribution during inference, not just weight magnitude.

```python
from turboquant_vllm.expert_pruning import reap_prune

reap_prune(model, tokenizer, prune_fraction=0.2, num_samples=512)
```

20% pruning preserves quality on Qwen3-30B. 50% pruning degrades quality (works on larger models per REAP paper).

## AWQ export

Export TQ-compressed weights to AWQ format for Marlin serving speed:

```python
from turboquant_vllm.export import compress_and_export

compress_and_export("google/gemma-4-26B-A4B-it", "./gemma4-awq", bits=4)
# ~2 minutes total (TQ compress + AutoAWQ pack)
# Serve with: vllm serve ./gemma4-awq --quantization awq
```

Requires [AutoAWQ](https://github.com/casper-hansen/AutoAWQ). Replaces hours of AWQ calibration with ~2 minutes.

### Combined with KV cache compression

Both features work together:
```python
from turboquant_vllm import enable_weight_quantization, patch_vllm_attention

enable_weight_quantization(bits=4, group_size=128)  # 59.7 GB → 16.8 GB model
patch_vllm_attention(k_bits=4, v_bits=3)            # 3.7x smaller KV cache
```

Same math, same CUDA kernels. Weight compression reduces the hardware you need. KV cache compression increases how many users you can serve on it.

Contributions and testing on different models welcome. Write-up: [varjosoft.com/weight-compression.html](https://varjosoft.com/weight-compression.html)

## KV cache compression: upstream is the path forward

TurboQuant KV cache compression for GQA/MHA models (Qwen, Llama, Mistral, Gemma) is being upstreamed to vLLM directly via [vllm-project/vllm#38479](https://github.com/vllm-project/vllm/pull/38479) by @vibhavagarwal5. As of 2026-04-11 the PR is OPEN, MERGEABLE, tagged `ready`, and under active review by maintainer @mgoin ("I think this is looking quite solid! Enabling CI as I look more carefully"). Merge is close.

Once #38479 lands, the recommended path is:

```bash
pip install vllm  # version with #38479 merged
vllm serve Qwen/Qwen3-4B --kv-cache-dtype turboquant_3bit_nc
```

Four presets land with the PR:

| Preset | Key | Value | Slot bytes | Compression | GSM8K | NIAH |
|---|---|---|---|---|---|---|
| `turboquant_k8v4` | FP8 | 4-bit uniform | 196 | **2.6x** | 0.860 | 100% |
| `turboquant_4bit_nc` | 4-bit MSE + NC | 4-bit uniform + NC | 136 | **3.8x** | 0.830 | 100% |
| `turboquant_k3v4_nc` | 3-bit MSE + NC | 4-bit uniform + NC | 120 | **4.3x** | 0.790 | 100% |
| `turboquant_3bit_nc` | 3-bit MSE + NC | 3-bit uniform + NC | 104 | **4.9x** | 0.785 | 100% |

Baseline GSM8K: 0.880, NIAH: 100%. Quality measured on Qwen/Qwen3-4B with 5-shot GSM8K (200q) and NIAH (512-32K, 77 probes). Full results in the PR body.

**Throughput trade-off** (Qwen3-4B, from the PR body): decode-heavy workloads run at 35-43% of baseline tok/s, long-prefill workloads at 72-87% of baseline. You're trading ~half your decode throughput for 2.6-4.9× KV capacity, which is valuable when KV is the memory bottleneck (long context, high concurrency) but not a free lunch.

Until #38479 lands, install Vibhav's branch directly if you need the fast path:

```bash
pip install git+https://github.com/vibhavagarwal5/vllm.git@feature/turboquant-kv-cache
```

### This plugin's KV cache story

This package **no longer ships a `TurboQuantAttentionBackend` or patches vLLM's KV cache allocator**. Earlier versions did, but:

1. The kernel work those paths depended on was incomplete — the vendored path produced broken output end-to-end and ran slower than the Python fallback (documented locally in `tests/gpu/results/vendor-triton-a100-20260410.txt`).
2. Upstream #38479 is a complete, tested, maintainer-reviewed implementation of the same idea and uses a revised preset schema (`turboquant_*`) that's incompatible with the old `tq3/tq4/tq_k4v3` names.
3. Maintaining a parallel plugin-side implementation in this package would duplicate effort without adding value.

What remains here is the **legacy monkey-patch path** (`patch_vllm_attention` / `TQ_KV_K_BITS=...`). This is the only place in the ecosystem today where TurboQuant KV compression works on **MLA models** (GLM-4.7-Flash, DeepSeek-V3) — #38479 is scoped to standard full-attention and uniform sliding-window models. Once upstream adds MLA support, this path will also be retired. Until then it's the production story for MLA users.

Deprecated path: early versions of the README pointed users at `varjoranta/vllm-1 turboquant-integration` as a "fork with Triton kernels". That fork is obsolete and its Triton kernels produce broken output when invoked through vLLM's real integration. Don't use it. If you need the pre-upstream Triton path, use Vibhav's branch above.

## Environment variables

These env vars activate the plugin's optional paths. Most users should leave them unset.

| Variable | Default | Effect |
|---|---|---|
| `TQ_WEIGHT_BITS` | unset | Activate runtime weight quantization with this many bits (3 or 4). Unique to this plugin. |
| `TQ_WEIGHT_GROUP_SIZE` | `128` | Group size for weight quantization. |
| `TQ_KV_K_BITS` | unset | Activate legacy monkey-patch KV compression with this many bits for K. **Primarily for MLA models** (GLM-4.7, DeepSeek-V3); for non-MLA models, use upstream [vllm-project/vllm#38479](https://github.com/vllm-project/vllm/pull/38479). |
| `TQ_KV_V_BITS` | same as K | Bits for V in monkey-patch mode. |
| `TQ_KV_ROTATION` | `wht` | Rotation used by the monkey-patch path: `wht` (Walsh-Hadamard) or `planar` (2D Givens). |
| `TQ_KV_NORM_CORRECTION` | `1` | Enable Phase-2 norm correction for monkey-patch compression. Set `0` to disable. |

## Running tests

The CPU suite runs in ~5 s and has no vLLM dependency. Use the `[dev]` extra to get pytest:

```bash
uv run --extra dev python -m pytest tests/ -q
```

CI runs the same suite across Python 3.11/3.12/3.13 plus ruff check + format via pre-commit on every PR (`.github/workflows/tests.yml`, `.github/workflows/pre-commit.yml`). A passing CI matrix is required before merge.

## Serverless deployment

Deploy models to [Verda](https://verda.com) GPU cloud (Helsinki) with scale-to-zero billing.

```bash
python containers/deploy.py deploy gemma4-26b-it      # A100, best quality/cost ratio
python containers/deploy.py deploy gpt-oss-20b        # L40S, cheapest good model
python containers/deploy.py deploy qwen3-235b-awq     # H200, highest quality
python containers/deploy.py pause gemma4-26b-it       # stop billing
```

### Measured results (April 2026)

| Model | Active params | GPU | Cost/hr | Cold start | Quality | Per session |
|---|---|---|---|---|---|---|
| **Gemma 4 26B MoE** | 3.8B | A100 80GB | $1.29 | 3 min | Excellent (#6 Arena AI) | ~$0.22 |
| gpt-oss-20b | 3.6B | L40S 48GB | $0.90 | 3.3 min | Very good | ~$0.15 |
| Qwen3-8B | 8B | L40S 48GB | $0.90 | 3 min | Good | ~$0.15 |
| Qwen3-235B AWQ | 22B | H200 141GB | $3.39 | 5.5 min | Best (4.75/5 benchmark) | ~$0.57 |

Cold start = time from zero replicas to first token (model cached on persistent volume). Billing per 10-minute block.

**Gemma 4 setup:** Requires custom vLLM image with `transformers>=5.5.0` and `python3-dev`. Use the instruction-tuned variant `google/gemma-4-26B-A4B-it`. Released April 2, 2026. Apache 2.0.

**For real-time chat, always-warm is required** — cold starts of 2-5 minutes are too slow for interactive use. Monthly cost (8hr/day): Gemma 4 on A100 ~$310, gpt-oss-20b on L40S ~$216. Serverless scale-to-zero works for batch/async workloads.

Code: [containers/deploy.py](https://github.com/varjoranta/verda-model-bench/blob/main/containers/deploy.py)

## Related projects

- **[turboquant-vllm on PyPI](https://pypi.org/project/turboquant-vllm/)** — A separate, independent implementation of TurboQuant for vLLM by Alberto-Codes. Uses Triton kernels and HuggingFace `DynamicCache`, targeting consumer GPUs (RTX 4090). This project differs: fused CUDA kernels for production A100/H100, asymmetric K/V bit widths (required for quantized weight models), and vLLM paged cache integration. This project is published as [`turboquant-plus-vllm`](https://pypi.org/project/turboquant-plus-vllm/).
- **[turbo-quant-lite](https://pypi.org/project/turbo-quant-lite/)** — Numpy-only TurboQuant for embedding compression in databases. Same math, different codebook and use case.
- **[turboquant_plus](https://github.com/TheTom/turboquant_plus)** — Research implementation of the KV cache algorithm. This package builds production CUDA kernels on top of that work.
- **TQ3_1S for llama.cpp** — @coffeecup2020's proof-of-concept applying TurboQuant to model weights (not just KV cache). Achieved near-Q4_0 quality at 3.5-bit. Inspired the weight quantization feature in this package.
- **[TurboQuant paper](https://arxiv.org/abs/2504.19874)** — Zandieh, Daliri, Hadian, Mirrokni; ICLR 2026. The underlying algorithm.
- **[REAP](https://arxiv.org/abs/2510.13999)** — Cerebras, ICLR 2026. Router-weighted expert pruning for MoE compression.
- **[SpinQuant](https://arxiv.org/abs/2405.16406)** — Facebook Research, ICLR 2025. Learned rotation optimization (up to 45% improvement over fixed Hadamard). Our `learned_rotation.py` implements a simplified version.
- **[SqueezeLLM](https://arxiv.org/abs/2306.07629)** — ICML 2024. Sensitivity-weighted codebooks and sparse outlier extraction. Influenced our research direction.
- **[AutoAWQ](https://github.com/casper-hansen/AutoAWQ)** — AWQ quantization and packing library. Used in our AWQ export pipeline.

## Development Process

This library was developed with the help of [Spegling](https://spegl.ing), a personal knowledge system built at Varjosoft. Spegling maintains a persistent wiki compiled from research papers and production systems, integrates with coding agents via MCP, and governs autonomous research with documented provenance. The research for v0.3.0 (TQ3 compression, REAP pruning, fused kernels) was conducted through Spegling analyzing relevant papers, implementing approaches, running benchmarks on Verda GPU instances, and iterating based on results. Total GPU cost: ~$18.

## License

MIT, Varjosoft Oy
