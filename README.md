# turboquant-vllm

Model compression for vLLM. What this package ships today:

- **Weight compression** (4.3-4.6x) via 3-bit TQ3. Any BF16 checkpoint, compressed in seconds, zero calibration. Algorithm is the scalar case of HIGGS; see "How it works" below.
- **bs=1 CUDA GEMV kernel** (v0.9.0) for TQ3 Linear layers: warp-per-output-channel, sm_80+, bf16. Triton's GEMV pads M=1 up to the tensor-core tile and wastes most of the ALU at batch size 1; this kernel replaces that path through a runtime-dispatching custom op so CUDA graphs capture the right path per batch size. Measured **2.12x decode speedup over the Triton-only path** on Qwen3-8B A100 bs=1. Motivation in [When Triton Stops Being the Right Tool](https://varjosoft.com/when-triton-stops).
- **Native TQ3 checkpoints** for small-GPU deployments (L40S, RTX 6000 Ada) and large-model deployments (GLM-5.1 754B on 2×H200). MoE expert regrouping handled automatically.
- **MLX port for Apple Silicon** — loads TQ3 checkpoints through `mlx-lm` on Mac (dense + MoE). Qwen2.5-0.5B at 26 tok/s, Granite-1B MoE at 84 tok/s, Qwen3.5-35B (256 experts) fits in 19 GB on a 48 GB MacBook. [Write-up](https://varjosoft.com/70gb-on-48gb-mac.html).
- **bs=1 Metal GEMV kernel** (v0.10.0) for TQ3 Linear + MoE SwitchLinear on Apple Silicon: SIMD-group-per-output-channel kernel via `mx.fast.metal_kernel`, fuses 3-bit unpack + codebook lookup + norm scaling + matmul into one pass. Three variants — single (dense), batched-shared-x (MoE gate/up_proj), batched-per-x (MoE down_proj).
- **Kurtosis-aware mixed-precision** (v0.11.0): per-tensor κ profile drives TQ3 / TQ4 / FP16 assignment per tensor family. On Qwen3.6-35B-A3B, the `varjosoft/Qwen3.6-35B-A3B-TQ-apex3` checkpoint (18 GB) hits **96.5 % gsm8k-200 — +2.0 ppt over `mlx-community/Qwen3.6-35B-A3B-4bit`** at 1 GB smaller on disk. Full mixed-bits loader + TQ4 Metal kernels ship in this release.
- **Sparse MoE dequant** (v0.12.0): the MoE apply path previously decompressed all N experts per forward even though the downstream `fused_moe` kernel only reads the active top-k. On Qwen3-30B-A3B (128 experts, top-8), 93.75% of weight-dequant GPU time was wasted work. Fixed by threading `topk_ids` through `TurboQuantFusedMoEMethod.apply()` and dequanting only the active experts. Measured A/B on H100, same enforce_eager=True both runs: **1.22 → 10.23 tok/s at bs=1 decode on Qwen3-30B-A3B-Instruct-2507 (8.4× speedup)**. Phase 1 uses a Python loop over active experts and requires `enforce_eager=True`; Phase 2 (follow-up) will add a GPU-resident kernel that preserves CUDA graph capture.
- **Expert pruning** via [REAP](https://arxiv.org/abs/2510.13999) saliency scoring for MoE models.
- **AWQ export** from TQ-compressed weights — ~2 min instead of hours of AWQ calibration.
- **Legacy KV cache compression** (monkey-patch, MLA-only) for GLM-4.7/DeepSeek-V3 users on stock vLLM until upstream KV compression supports MLA.

> **KV cache compression for GQA/MHA models (Qwen, Llama, Mistral, Gemma) is now upstream in vLLM via [vllm-project/vllm#38479](https://github.com/vllm-project/vllm/pull/38479)** by @vibhavagarwal5 (merged 2026-04-15). Use `--kv-cache-dtype turboquant_3bit_nc` (or `k8v4`, `4bit_nc`, `k3v4_nc`) on stock vLLM with no plugin required. This package's role going forward is weight quantization + MLA KV compression + the supporting tooling, not the standard KV-cache path.

> **Weight compression is also going upstream**: [vllm-project/vllm#39970](https://github.com/vllm-project/vllm/pull/39970) adds the Linear-only weight-compression path as `--quantization turboquant` (scalar HIGGS). Opened 2026-04-16, awaiting maintainer triage. Once merged, upstream vLLM will have the weight path; MoE compression stays here until the follow-up upstream PR lands.

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

- **GLM-5.1 (754B MoE)**: 1,508 GB BF16 → **309 GB native TQ3** → serves on **2×H200** (282 GB VRAM). Without TQ3, this requires 8×H200. **7× hardware cost reduction.**
- **Gemma 4 26B**: ~52 GB BF16 → **~12 GB runtime VRAM** with TQ3. Scores **4.79/5** on our 20-scenario benchmark, comparable to Qwen3-235B AWQ (4.75/5) at 2.6x lower GPU cost.
- **GLM-4.7-Flash 355B MoE**: 62.4 GB → **14.7 GB** (4.2x). Native TQ3 checkpoint with MoE expert regrouping, 13.3 GB GPU memory.
- **Qwen3-30B**: 61 GB → **13 GB** (4.6x).
- **Qwen3-30B-A3B-Instruct-2507 MoE decode** on H100 (v0.12.0): 1.22 → **10.23 tok/s at bs=1, 1k ctx** with sparse dequant (8.4× over v0.11.0). Requires `LLM(..., enforce_eager=True, kernel_config={"moe_backend": "triton"})`.

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
| GLM-5.1 754B | **MLA (DSA)** | Works (native TQ3 ckpt, 2×H200) | Untested | Requires DeepGEMM + `--quantization turboquant` |
| GLM-4.7-Flash 355B | **MLA** | Works (native TQ3 ckpt) | **Works** | **Only place TurboQuant KV works on MLA today** — #38479 does not cover MLA |
| DeepSeek-V3 | **MLA** | Pending (larger disk) | Works | Same: MLA requires the legacy monkey-patch path |
| Qwen3.5-35B-A3B | MoE + GatedDeltaNet + GQA | Works (as of varjoranta/turboquant-vllm#15) | Untested | Hybrid architecture; weight quant validated via @gaby's fixes |
| Qwen3-30B-A3B | MoE + GQA | Works | Works | Full CUDA graph capture, 123 tok/s at c=1 on A100 (matching BF16 baseline) |
| gpt-oss-20b | Alternating full/sliding window + sinks | Works | Not yet | Sliding window + attention sinks need pass-through support in the KV path |

**Where to get KV cache compression:**

- **GQA/MHA models** (Qwen, Llama, Mistral, Gemma): use upstream `--kv-cache-dtype turboquant_3bit_nc` on stock vLLM (merged in [vllm-project/vllm#38479](https://github.com/vllm-project/vllm/pull/38479)). This plugin's old `--kv-cache-dtype tq3` path has been removed.
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

The weight-compression algorithm is the scalar case of **HIGGS** (Malinovskii, Panferov, Ilin, Guo, Richtárik, Alistarh — *Pushing the Limits of LLM Quantization via the Linearity Theorem*, [NAACL 2025](https://aclanthology.org/2025.naacl-long.543/), preprint [arXiv:2411.17525](https://arxiv.org/abs/2411.17525)): Random Hadamard Transform + MSE-optimal Lloyd-Max grid + per-group normalization. No calibration data. A reference implementation also exists in HuggingFace transformers as `HiggsConfig`.

The project was originally based on [TurboQuant](https://arxiv.org/abs/2504.19874) (Zandieh, Daliri, Hadian, Mirrokni; ICLR 2026), which is a more general online **vector** quantizer targeted at KV cache and ANN vector search — not weight compression. Engineering simplifications during development (scalar over vector, WHT over general random rotations, Lloyd-Max over learned grids) converged the weight path onto HIGGS. The `turboquant` naming is kept for API and package compatibility; the actual TurboQuant application is the KV-cache path in [vllm-project/vllm#38479](https://github.com/vllm-project/vllm/pull/38479). Thanks to @dalistarh for the attribution catch.

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

The weight path implements the scalar case of HIGGS (Malinovskii et al., NAACL 2025) — see "How it works" above for the full attribution. Weight compression was seeded by @coffeecup2020's TQ3_1S proof-of-concept for llama.cpp.

**Partial-rotary models** (`partial_rotary_factor < 1.0` on the HF text config — MiniMax M2.5/M2.7, Qwen3.6-A3B family) use a **block-diagonal WHT** so the RoPE-rotated head prefix and the content-only suffix stay under separate rotations inside a quantization group. Auto-detected at load time from the `VllmConfig` — `_derive_rotary_dim` resolves `partial_rotary_factor` / `rotary_pct` / `rotary_emb_fraction` aliases, both directly and inside a `rope_parameters` dict, and picks the largest power-of-two block size that divides `head_dim`. Block-diag weights route through the PolarQuant PyTorch dequant path; the Triton and CUDA kernels assume full-width WHT and are bypassed for the block-diag case (correctness over speed — a block-diag kernel variant is a follow-up).

### Two paths: this plugin vs upstream vLLM

Two independent implementations of the same algorithm exist, with different trade-offs:

- **This plugin (`pip install turboquant-plus-vllm`)** — monkey-patches vLLM at import time via the `vllm.general_plugins` entry point. Works with any stable vLLM release, includes the CUDA bs=1 fused-dequant kernel, and the MoE path via `FusedMoE._replace_quant_method`. Enabled through the `TQ_WEIGHT_BITS` env var; fully standalone.
- **Upstream `--quantization turboquant`** ([vLLM PR #39970](https://github.com/vllm-project/vllm/pull/39970) + the MoE follow-up) — native vLLM scheme. Same algorithm, routed through `OnlineQuantizationConfig`. Not yet merged. When it lands you get `vllm serve <model> --quantization turboquant` without any plugin install, at the cost of tracking vLLM versions.

The **upstream MoE path requires forcing the Triton MoE backend** (FlashInfer-CUTLASS and AITER permute expert weight storage during setup, which breaks the shared scratch-pool invariant both paths rely on):

```python
from vllm import LLM

llm = LLM(
    model="Qwen/Qwen3-30B-A3B-Instruct-2507",
    quantization="turboquant",
    kernel_config={"moe_backend": "triton"},
)
# or via CLI: vllm serve ... --quantization turboquant -cc.moe_backend=triton
```

**Known quirk (upstream path only):** set `VLLM_USE_DEEP_GEMM=0` if the `deep_gemm` package isn't installed on the serving box. vLLM's `kernel_warmup` probes every Linear for DeepGEMM compatibility and crashes on missing import even though TurboQuant doesn't use FP8 kernels. This is a vLLM infrastructure behavior unrelated to the algorithm.

### Results

| Model | BF16 | TQ3 (3-bit) | Compression | Quality |
|-------|------|-------------|-------------|---------|
| **Gemma 4 26B** | 52 GB | **12 GB** | 4.3x | 4.79/5 |
| **GLM-4.7-Flash 355B** | 62.4 GB | **14.7 GB** | 4.2x | Tested ✓ |
| **GLM-5.1 754B** | 1,508 GB | **309 GB** | 4.9x | Validated on GLM-4.7-Flash (same arch) |
| **Qwen3-30B-A3B-Instruct-2507** | ~60 GB | **13.69 GB** | 4.4x | **GSM8K-200 91.5%** |

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

**TQ3 throughput status on current vLLM 0.19.** The fullgraph compile incompatibility flagged in earlier README versions is now fixed on branch `fix/dynamo-clean-forward-path`: the two Triton launchers are registered as `torch.library.custom_op` with `register_fake` meta impls, the FWHT-on-input cache (which did a host-sync `.cpu()` fingerprint and broke CUDA graph capture) is removed, and `TurboQuantWrapper.forward()` is split into a dynamo-clean GPU path and an eager CPU fallback. TQ3 weights + TQ3 forward now serve under vLLM 0.19 with piecewise CUDA graphs enabled.

The throughput picture is less rosy than the memory story. Measured on A100 80GB with an 8B dense reference (`Qwen/Qwen3-8B`, `input=512 output=128 num_prompts=32 ignore_eos`):

| Qwen3-8B A100 80GB | BF16 out tok/s | TQ3 out tok/s | TQ3/BF16 | BF16 model mem | TQ3 model mem |
|:---:|:---:|:---:|:---:|:---:|:---:|
| c=1 | 84.1 | 6.3 | 0.07× | 15.3 GiB | **4.97 GiB** (3.07×) |
| c=4 | 329.6 | 25.0 | 0.08× | 15.3 GiB | 4.97 GiB |
| c=8 | 276.2 | 46.2 | 0.17× | 15.3 GiB | 4.97 GiB |
| c=16 | 1210.7 | 98.8 | 0.08× | 15.3 GiB | 4.97 GiB |

The **3.07× live weight-memory compression** is real and matches the math. The **TQ3 decode throughput is ~7–17% of BF16** on vLLM 0.19 — a real performance gap that landed with the dynamo-fullgraph rewrite and is **not recoverable from host-side orchestration alone**. The bottleneck is the Triton compressed-GEMM kernel itself: loading 3-bit codes, centroid lookup, rotation, FMA — all materially slower than cuBLAS BF16 on modern hardware. Recovery needs kernel-level work (better autotune configs, possibly a hybrid cuBLAS-GEMM + Triton-dequant path, or fusion into attention/MLP).

**Startup time, however, has been brought back to sane**. The first TQ3 benchmarks had a 10–25 minute "first capture" stall caused by Triton's `@autotune` re-running for every one of vLLM's ~51 CUDA-graph capture batch sizes. Dropping `batch_size` from the autotune key on `_polar_fused_gemm_kernel` (keyed only on `out_f`, `in_f_padded` now) fixes that:

| Startup (A100 80GB) | Before fix | After fix | Speedup |
|:---|:---:|:---:|:---:|
| Qwen2.5-0.5B TQ3 — total | 310 s | 217 s | 1.4× |
| Qwen2.5-0.5B TQ3 — CUDA graph capture phase alone | ~250 s | ~16 s | **~16×** |
| Qwen3-8B TQ3 — total | 1060 s | 382 s | **2.8×** |

Until a faster compressed-GEMM kernel lands, TQ3 is a **memory win, not a speed win**, on current vLLM 0.19. This plugin still unblocks 52 GB → 12 GB loading scenarios (Gemma 4 26B on L40S 48GB, GLM-4.7-Flash 355B MoE on A100 80GB) where BF16 simply does not fit, and the quality/PPL results below are unchanged.

**MoE weight quantization under vLLM 0.19 — full CUDA graph capture.** `FusedMoE` expert weights are compressed via a `FusedMoEMethodBase` subclass installed through `FusedMoE._replace_quant_method` ([varjoranta/turboquant-vllm#14](https://github.com/varjoranta/turboquant-vllm/issues/14)). `apply()` decompresses both `w13` / `w2` per layer per forward into a shared scratch pool and delegates to the base unquantized method. All CUDA dequant kernels launch on PyTorch's current stream via `c10::cuda::getCurrentCUDAStream`, so piecewise CUDA graph capture works correctly. Validated on Qwen3-30B-A3B (48 layers, 128 experts): coherent output, 123 tok/s at c=1 on A100 80GB — matching the BF16 baseline. No `--enforce-eager` required.

3-bit sub-byte packing: 8 indices per 3 bytes. Per-group shape-gain decomposition (Gray 1984): stores `original_norm / reconstruction_norm` ratio per group to fix 5-10% magnitude shrinkage at 3-bit. (The "shape-gain" name is the classical VQ term — `vector = gain · unit_direction`; confirmed by @dalistarh as the established attribution.)

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

### Serving native TQ3 checkpoints with vLLM

For models too large to fit in GPU memory even as BF16 (e.g., GLM-5.1 at 1,508 GB), native TQ3 checkpoints can be served directly through vLLM with `--quantization turboquant`. This uses **meta-device initialization** — the model architecture is allocated with zero GPU memory, then weights are decompressed from TQ3→BF16 and re-compressed to TQ3 on GPU one layer at a time:

```bash
# GLM-5.1 (754B) on 2×H200 — impossible without TQ3 (needs 754 GB BF16)
vllm serve varjosoft/GLM-5.1-Open-TQ3 \
    --quantization turboquant \
    --tensor-parallel-size 2 \
    --max-model-len 4096 \
    --dtype bfloat16
```

How it works:
1. `TurboQuantOnlineLinearMethod` and `TurboQuantOnlineMoEMethod` set `uses_meta_device=True`
2. Model init allocates zero GPU memory (all weights on meta device)
3. `get_all_weights` hook decompresses `.tq_packed`/`.tq_norms` → BF16 per tensor
4. vLLM's online processing buffers one layer at a time, materializes on GPU
5. `process_weights_after_loading` compresses BF16→TQ3 on GPU per layer
6. Peak memory: ~1 layer BF16 + all previous layers compressed

This is the same pattern vLLM uses for online FP8 quantization. No `TQ_WEIGHT_BITS` env var needed — the quantization config is read from `tq_config.json` in the checkpoint.

**GLM-5.1 (754B MoE):** 1,508 GB BF16 → 309 GB native TQ3 checkpoint. Fits on 2×H200 (282 GB VRAM) with `--quantization turboquant --tensor-parallel-size 2`. Without TQ3, this model requires 8×H200 (€350K) — a **7× hardware cost reduction**.

### Limitations

- **V100 16GB**: model loads (12 GB) but not enough room for KV cache. Minimum practical is 24 GB.
- **TQ2 (2-bit)** destroys quality. 4 centroids too few for MLP weight distributions.
- **Native TQ3 inference speed** is slower than runtime compression due to per-forward-pass decompression overhead.

## Mac / MLX

Loads TQ3 native checkpoints through `mlx-lm` on Apple Silicon — dense architectures and MoE. No paid GPU needed for local development, quality validation, or small-model inference.

```python
from turboquant_vllm.mlx_loader import load_tq3
from mlx_lm import generate

model, tokenizer = load_tq3("./my-tq3-checkpoint")
text = generate(model, tokenizer, "The capital of France is", max_tokens=64)
```

End-to-end results on M4 Pro 48 GB:

| Model | Type | tok/s | Model state |
|---|---|---|---|
| Qwen2.5-0.5B | Dense | 26 | 0.4 GB |
| IBM Granite 1B-A400M | MoE (40 experts) | 84 | 2.5 GB |
| Qwen3.5-35B-A3B | MoE (256 experts) | ~1 | 19 GB |
| **Qwen3.6-35B-A3B-TQ-apex3** | MoE (256 experts) | **29** | **18 GB** |

### Kurtosis-aware mixed-precision (v0.11.0)

Uniform TQ3 trades quality for size on MoE models with heavy-tailed tensors (router gates, gated-delta-net projections, shared-expert down-projections). `v0.11.0` ships **PAT-0349 kurtosis-aware bit-width selection** — compute per-tensor excess kurtosis (κ) once at compression time, then:

- κ < 4 → **TQ3** (default, routed experts)
- 4 ≤ κ < 8 → **TQ4** (attention Q/K/V/O, shared-expert gate/up)
- κ ≥ 15 → **FP16** (router gates, GDN `in_proj_b`/`out_proj`/`linear_fc`, shared-expert `down_proj`)

The mixed-bits MLX loader (`load_tq3`) handles all three precisions transparently — TQ3 goes through the 48-byte Metal kernel, TQ4 through a matching 64-byte nibble kernel, FP16 tensors pass through `mlx_lm`'s standard weight-load path.

Result on Qwen3.6-35B-A3B, gsm8k-200 @ 1024 tok:

| Checkpoint | Disk | gsm8k-200 | Decode (bs=1) |
|---|---|---|---|
| uniform TQ3-native | 16 GB | 85.5 % | 29 tok/s |
| TQ-apex | 17 GB | 93.0 % | 29 tok/s |
| TQ-apex2 | 18 GB | 96.0 % | 29 tok/s |
| **TQ-apex3** | **18 GB** | **96.5 %** | **29 tok/s** |
| `mlx-community/*-4bit` | 19 GB | 94.5 % | 73 tok/s |

**Quality leader**: `varjosoft/Qwen3.6-35B-A3B-TQ-apex3` is the first TurboQuant checkpoint to beat the MLX-community 4-bit reference on accuracy, while being 1 GB smaller on disk. Speed on M4 Pro is capped at ~29 tok/s by the Lloyd-Max codebook lookup cost (measured floor ~54 µs/call at async steady state) — MLX's affine quantization is ~2× faster per kernel but ships at lower quality. This trade-off is fundamental to the codebook choice, not the kernel implementation.

The 1 tok/s earlier result on Qwen3.5-35B-A3B was before the compiled `mx.compile` path + active-only expert dequant landed. The 29 tok/s apex3 number is the current steady-state.

### Usage

```bash
pip install git+https://github.com/varjoranta/turboquant-vllm.git@feat/mixed-bits-mlx-loader
huggingface-cli download varjosoft/Qwen3.6-35B-A3B-TQ-apex3 \
    --local-dir ~/models/qwen3.6-35b-a3b-tq-apex3
```

```python
from turboquant_vllm.mlx_loader import load_tq3
from mlx_lm import generate
model, tokenizer = load_tq3("~/models/qwen3.6-35b-a3b-tq-apex3")
print(generate(model, tokenizer, "The capital of France is", max_tokens=64))
```

Or serve via OpenAI-compatible HTTP: `python examples/mac-serve-tq3.py --model ~/models/qwen3.6-35b-a3b-tq-apex3 --port 8080`.

Key implementation choices:

- **FWHT-on-input** — rotate input once per token instead of inverse-rotating every weight row. 49× faster than the PyTorch CPU fallback per layer.
- **Active-only expert dequant** — for MoE models, gather only the top-k active experts' packed uint8 bytes via `mx.take`, unpack on the fly. Avoids the 120 GB int32 blowup that happens when all 256 experts are pre-unpacked.
- **Fused gate_up_proj split** — handles the HF → mlx_lm weight layout translation for Qwen3.5-MoE style architectures post-sanitize.
- **RoPE compat** — translates transformers 4.52+ `rope_parameters` back to the legacy `rope_theta`/`rope_scaling` keys mlx_lm expects.

The MLX path is **opt-in** — nothing in the CUDA/Triton/CPU paths imports it, so Linux/vLLM users are unaffected.

### Serving via OpenAI-compatible HTTP (opencode, aider, etc.)

[`examples/mac-serve-tq3.py`](examples/mac-serve-tq3.py) wraps `mlx_lm.server` with three patches needed to survive agent clients on a 48 GiB Mac: it monkey-patches the server's loader to route TQ3 checkpoints through `load_tq3`, serializes requests behind a single lock (so parallel agent calls don't double up Metal command buffer allocations), and caps MLX's wired/memory/cache limits with a `mx.clear_cache()` between requests. ~40 lines total.

```bash
python examples/mac-serve-tq3.py --model ~/models/qwen3-coder-30b-a3b-tq3 --port 8080
```

Point any OpenAI-compatible client at `http://127.0.0.1:8080/v1`. Validated end-to-end with opencode against a 30B MoE — see [34 tok/s on a MacBook](https://varjosoft.com/34-tokens-per-second).

### Related MLX work

For **KV-cache** compression on MLX (the original TurboQuant paper's actual subject — Zandieh et al., ICLR 2026), see [sharpner/turboquant-mlx](https://github.com/sharpner/turboquant-mlx). That project implements V2 (hardware-accelerated `mx.quantized_matmul`) and V3 (Lloyd-Max paper-correct) KV-cache compression paths, with perplexity + throughput benchmarks across Llama 3.1/3.2, Mistral, and Gemma 3. Our MLX path above compresses **weights** via HIGGS-scalar; sharpner's compresses the **KV cache** via TurboQuant. The two are orthogonal and can coexist in the same serving stack.

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

TurboQuant KV cache compression for GQA/MHA models (Qwen, Llama, Mistral, Gemma) is now upstream in vLLM via [vllm-project/vllm#38479](https://github.com/vllm-project/vllm/pull/38479) by @vibhavagarwal5 (merged 2026-04-15). Use `--kv-cache-dtype turboquant_3bit_nc` on stock vLLM — no plugin needed.

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

Deprecated path: early versions of the README pointed users at `varjoranta/vllm-1 turboquant-integration` as a "fork with Triton kernels". That fork has been **deleted** as of 2026-04-11 — its kernels produced broken output end-to-end when invoked through vLLM's real integration (gibberish generation + 10× slowdown vs the Python fallback, validated on both A100 sm_80 and RTX 6000 Ada sm_89). If you have an old clone, discard it. For the pre-upstream Triton path, use @vibhavagarwal5's branch above.

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

- **[vllm-project/vllm#39970](https://github.com/vllm-project/vllm/pull/39970)** — Our upstream vLLM PR adding weight compression as `--quantization turboquant` (Linear-only, MoE deferred). Mirrors what this plugin does, landed into vLLM's `OnlineQuantScheme` framework. Awaiting maintainer review.
- **[vllm-project/vllm#38479](https://github.com/vllm-project/vllm/pull/38479)** — @vibhavagarwal5's upstream TurboQuant KV-cache PR (merged 2026-04-15). The actual TurboQuant algorithm for KV + ANN.
- **[HIGGS paper](https://aclanthology.org/2025.naacl-long.543/)** — Malinovskii, Panferov, Ilin, Guo, Richtárik, Alistarh; NAACL 2025. Primary algorithm citation for this plugin's weight path (preprint [arXiv:2411.17525](https://arxiv.org/abs/2411.17525)). Reference implementation also in HuggingFace transformers as `HiggsConfig`.
- **[TurboQuant paper](https://arxiv.org/abs/2504.19874)** — Zandieh, Daliri, Hadian, Mirrokni; ICLR 2026. Online vector quantizer for KV cache / ANN — the framework this project started from, and the algorithm behind #38479's KV path.
- **[Vector Quantization (Gray 1984)](https://www.csd.uoc.gr/~hy474/bibliography/VectorQuantizationGray.pdf)** — classical reference for shape-gain decomposition, the per-group norm refinement we use.
- **[turboquant-vllm on PyPI](https://pypi.org/project/turboquant-vllm/)** — A separate, independent TurboQuant-for-vLLM implementation by Alberto-Codes. Uses Triton kernels and HuggingFace `DynamicCache`, targeting consumer GPUs (RTX 4090). This project differs: fused CUDA kernels for production A100/H100, asymmetric K/V bit widths, and vLLM paged cache integration. Published as [`turboquant-plus-vllm`](https://pypi.org/project/turboquant-plus-vllm/).
- **[turbo-quant-lite](https://pypi.org/project/turbo-quant-lite/)** — Numpy-only TurboQuant for embedding compression in databases. Same math, different codebook and use case.
- **[turboquant_plus](https://github.com/TheTom/turboquant_plus)** — Research implementation of the KV cache algorithm. This package builds production CUDA kernels on top of that work.
- **TQ3_1S for llama.cpp** — @coffeecup2020's proof-of-concept applying TurboQuant to model weights (not just KV cache). Achieved near-Q4_0 quality at 3.5-bit. Inspired the weight quantization feature in this package.
- **[ITQ3_S](https://arxiv.org/abs/2603.27914)** — Yoon, March 2026. Single-author preprint describing interleaved ternary (`{-1, 0, +1}`) quantization over FWHT-rotated weights, with the inverse FWHT fused into the CUDA SMEM-load stage (claimed ~2.1% compute overhead). Blackwell-tuned (256-point blocks); LLaMA-3 8B benchmarks on RTX 5090; code not released. Different quant-grid axis than our Lloyd-Max scalar — ternary vs 16-centroid — and a kernel idea worth mining (FWHT-in-SMEM fusion) for our own Blackwell path.
- **[REAP](https://arxiv.org/abs/2510.13999)** — Cerebras, ICLR 2026. Router-weighted expert pruning for MoE compression.
- **[SpinQuant](https://arxiv.org/abs/2405.16406)** — Facebook Research, ICLR 2025. Learned rotation optimization (up to 45% improvement over fixed Hadamard). Our `learned_rotation.py` implements a simplified version.
- **[SqueezeLLM](https://arxiv.org/abs/2306.07629)** — ICML 2024. Sensitivity-weighted codebooks and sparse outlier extraction. Influenced our research direction.
- **[AutoAWQ](https://github.com/casper-hansen/AutoAWQ)** — AWQ quantization and packing library. Used in our AWQ export pipeline.

## Development Process

This library was developed with the help of [Spegling](https://spegl.ing), a personal knowledge system built at Varjosoft. Spegling maintains a persistent wiki compiled from research papers and production systems, integrates with coding agents via MCP, and governs autonomous research with documented provenance. The research for v0.3.0 (TQ3 compression, REAP pruning, fused kernels) was conducted through Spegling analyzing relevant papers, implementing approaches, running benchmarks on Verda GPU instances, and iterating based on results. Total GPU cost: ~$18.

## License

MIT, Varjosoft Oy
