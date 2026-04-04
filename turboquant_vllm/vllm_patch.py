"""Monkey-patch vLLM's attention backends to use TurboQuant+ KV cache.

Supports two attention backends:
1. FlashAttentionImpl (standard GQA/MHA) — Qwen3, Llama, Mistral, etc.
2. TritonMLAImpl (Multi-head Latent Attention) — GLM-4.7, DeepSeek-V3

For FlashAttention, intercepts do_kv_cache_update(key, value, ...) and forward().
For MLA, intercepts do_kv_cache_update(kv_c_normed, k_pe, ...) — compresses
the latent vector kv_c_normed, passes k_pe through uncompressed.

The compressed data lives in a sidecar store (Python dict), not in vLLM's
paged cache. vLLM's cache is still allocated and written to (for memory
accounting), but attention reads from our decompressed tensors.

Known limitations of the monkey-patch approach:
- Per-token Python loops on compress/decompress (CUDA fused kernels solve this)
- Full cache decompression on every forward call (dirty-tracking would help)
- Sidecar cache grows with sequence length and is not evicted when vLLM frees
  blocks. Long-running servers need periodic restarts until native backend lands.
"""

import torch
import logging
from functools import wraps

from turboquant_vllm.torch_ops import KVCacheCompressorTorch, CompressedKV

logger = logging.getLogger(__name__)

_compressors: dict[int, KVCacheCompressorTorch] = {}
_cache: dict[int, dict[tuple[int, int, int], tuple[CompressedKV, CompressedKV]]] = {}
_mla_cache: dict[int, dict[tuple[int, int], CompressedKV]] = {}

_k_bits = 4
_v_bits = 4
_use_cuda = False
_norm_correction = True
_use_qjl = False


def _try_cuda_init() -> bool:
    """Try to load the JIT-compiled CUDA extension. Returns True if available."""
    global _use_cuda
    try:
        from turboquant_vllm.build import build
        build()
        _use_cuda = True
        logger.info("TurboQuant+ using CUDA kernels")
        return True
    except Exception as e:
        logger.info("TurboQuant+ CUDA not available (%s), using PyTorch fallback", e)
        _use_cuda = False
        return False


def _get_compressor(dim: int, device: torch.device) -> KVCacheCompressorTorch:
    """One compressor per vector dimension (all layers share the same codebook)."""
    if dim not in _compressors:
        _compressors[dim] = KVCacheCompressorTorch(
            dim, k_bits=_k_bits, v_bits=_v_bits,
            seed=42, device=str(device),
            use_cuda=_use_cuda,
            norm_correction=_norm_correction,
            use_qjl=_use_qjl,
        )
        stats = _compressors[dim].memory_stats()
        backend = "CUDA" if _use_cuda else "PyTorch"
        logger.info(
            "TurboQuant+ compressor [%s]: dim=%d K=%d-bit V=%d-bit → %.1fx compression",
            backend, dim, _k_bits, _v_bits, stats["compression_ratio"],
        )
    return _compressors[dim]


def _iter_slots(slot_mapping, block_size):
    """Yield (token_idx, block_idx, offset) for each token in slot_mapping."""
    for t in range(slot_mapping.shape[0]):
        slot = slot_mapping[t].item()
        yield t, slot // block_size, slot % block_size


# ============================================================================
# FlashAttention patches (standard GQA/MHA)
# ============================================================================

def _make_patched_cache_update(original_fn):
    """Wrap do_kv_cache_update to compress K/V with TurboQuant+."""

    @wraps(original_fn)
    def patched(self, layer, key, value, kv_cache, slot_mapping):
        original_fn(self, layer, key, value, kv_cache, slot_mapping)

        layer_id = id(layer)
        if layer_id not in _cache:
            _cache[layer_id] = {}

        head_dim = key.shape[-1]
        num_kv_heads = key.shape[1] if key.dim() == 3 else 1
        compressor = _get_compressor(head_dim, key.device)

        key_cache, _ = kv_cache.unbind(0)
        block_size = key_cache.shape[1]

        for t, block_idx, offset in _iter_slots(slot_mapping, block_size):
            for h in range(num_kv_heads):
                ck = compressor.compress_k(key[t, h].unsqueeze(0))
                cv = compressor.compress_v(value[t, h].unsqueeze(0))
                _cache[layer_id][(block_idx, offset, h)] = (ck, cv)

    return patched


def _make_patched_forward(original_fn):
    """Wrap forward to decompress K/V from TurboQuant+ before attention."""

    @wraps(original_fn)
    def patched(self, layer, query, key, value, kv_cache, attn_metadata, output=None):
        if kv_cache is None:
            return original_fn(self, layer, query, key, value, kv_cache, attn_metadata, output)

        store = _cache.get(id(layer))
        if not store:
            return original_fn(self, layer, query, key, value, kv_cache, attn_metadata, output)

        compressor = _get_compressor(query.shape[-1], query.device)
        key_cache, value_cache = kv_cache.unbind(0)

        for (block_idx, offset, head_idx), (ck, cv) in store.items():
            key_cache[block_idx, offset, head_idx] = compressor.decompress_k(ck).squeeze(0).to(key_cache.dtype)
            value_cache[block_idx, offset, head_idx] = compressor.decompress_v(cv).squeeze(0).to(value_cache.dtype)

        return original_fn(self, layer, query, key, value, kv_cache, attn_metadata, output)

    return patched


# ============================================================================
# MLA patches (GLM-4.7, DeepSeek-V3)
# MLA uses id(self) as layer key because its do_kv_cache_update doesn't
# receive a separate layer argument (unlike FlashAttention).
# ============================================================================

def _make_mla_patched_cache_update(original_fn):
    """Wrap MLA do_kv_cache_update to compress kv_c_normed with TurboQuant+.

    kv_c_normed: the latent KV vector (compressed representation of K+V)
    k_pe: positional encoding for K (small, passed through uncompressed)
    """

    @wraps(original_fn)
    def patched(self, kv_c_normed, k_pe, kv_cache, slot_mapping, kv_cache_dtype, k_scale):
        original_fn(self, kv_c_normed, k_pe, kv_cache, slot_mapping, kv_cache_dtype, k_scale)

        layer_id = id(self)
        if layer_id not in _mla_cache:
            _mla_cache[layer_id] = {}

        latent_dim = kv_c_normed.shape[-1]
        compressor = _get_compressor(latent_dim, kv_c_normed.device)
        block_size = kv_cache.shape[1]

        for t, block_idx, offset in _iter_slots(slot_mapping, block_size):
            # MSE-only compression for the latent vector (no QJL needed)
            compressed = compressor.compress_v(kv_c_normed[t].unsqueeze(0))
            _mla_cache[layer_id][(block_idx, offset)] = compressed

    return patched


def _make_mla_patched_forward(original_fn, cache_arg_idx):
    """Wrap MLA forward to decompress kv_c_normed before attention.

    cache_arg_idx: positional index of kv_c_and_k_pe_cache in the method args.
    Determined at patch time from inspect.signature to avoid fragile hardcoding.
    """

    @wraps(original_fn)
    def patched(self, *args, **kwargs):
        store = _mla_cache.get(id(self))
        if not store:
            return original_fn(self, *args, **kwargs)

        kv_cache = args[cache_arg_idx] if len(args) > cache_arg_idx else kwargs.get("kv_c_and_k_pe_cache")
        if kv_cache is None:
            return original_fn(self, *args, **kwargs)

        latent_dim = next(iter(store.values())).indices.shape[-1]
        compressor = _get_compressor(latent_dim, kv_cache.device)

        for (block_idx, offset), compressed in store.items():
            dec = compressor.decompress_v(compressed).squeeze(0)
            kv_cache[block_idx, offset, :latent_dim] = dec.to(kv_cache.dtype)

        return original_fn(self, *args, **kwargs)

    return patched


# ============================================================================
# Public API
# ============================================================================

def patch_vllm_attention(
    k_bits: int = 4,
    v_bits: int = 4,
    norm_correction: bool = True,
    use_qjl: bool = False,
):
    """Monkey-patch vLLM attention backends for TurboQuant+ KV cache.

    Call before starting the vLLM engine. Patches both FlashAttention (standard
    models) and TritonMLA (GLM-4.7, DeepSeek-V3) if available.

    Args:
        k_bits: Bits for key compression (default 4, 16 centroids).
        v_bits: Bits for value compression (default 4).
        norm_correction: Correct reconstruction magnitude error (default True).
        use_qjl: Use QJL residual correction for K cache (default False).
    """
    global _k_bits, _v_bits, _norm_correction, _use_qjl
    _k_bits = k_bits
    _v_bits = v_bits
    _norm_correction = norm_correction
    _use_qjl = use_qjl

    _try_cuda_init()

    patched_backends = []

    try:
        from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl
        FlashAttentionImpl.do_kv_cache_update = _make_patched_cache_update(
            FlashAttentionImpl.do_kv_cache_update
        )
        FlashAttentionImpl.forward = _make_patched_forward(
            FlashAttentionImpl.forward
        )
        patched_backends.append("FlashAttention")
    except ImportError:
        logger.warning("FlashAttentionImpl not found, skipping FlashAttention patch")

    # Patch MLA at the common base class so all MLA backends
    # (TritonMLA, FlashAttnMLA, FlashMLA, etc.) are covered.
    try:
        import inspect
        from vllm.model_executor.layers.attention.mla_attention import MLACommonImpl
        MLACommonImpl.do_kv_cache_update = _make_mla_patched_cache_update(
            MLACommonImpl.do_kv_cache_update
        )
        for method_name in ("forward_mha", "forward_mqa"):
            if hasattr(MLACommonImpl, method_name):
                fn = getattr(MLACommonImpl, method_name)
                params = list(inspect.signature(fn).parameters.keys())
                cache_idx = next(
                    (i for i, p in enumerate(params) if "cache" in p.lower()),
                    None
                )
                if cache_idx is None:
                    logger.warning("Could not find cache param in %s, skipping", method_name)
                    continue
                setattr(MLACommonImpl, method_name, _make_mla_patched_forward(
                    fn, cache_idx - 1
                ))
        patched_backends.append("MLA")
    except ImportError:
        logger.warning("MLACommonImpl not found, skipping MLA patch")

    if not patched_backends:
        raise ImportError("No vLLM attention backends found. Is vLLM installed?")

    logger.info(
        "TurboQuant+ patched vLLM [%s]: K=%d-bit V=%d-bit",
        "+".join(patched_backends), k_bits, v_bits,
    )
