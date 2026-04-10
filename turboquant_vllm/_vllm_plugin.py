"""vLLM plugin: auto-registers TQ weight and KV cache compression.

Activated via environment variables. This plugin is loaded by vLLM in
the main process AND in spawned subprocesses (V1 engine).

KV cache compression supports two modes:
  1. Native backend (--kv-cache-dtype tq3/tq4/tq_k4v3): Registers
     TurboQuantAttentionBackend as vLLM's CUSTOM attention backend.
     CUDA-graph-compatible. Works in V1 subprocess. Recommended.
  2. Monkey-patch (TQ_KV_K_BITS env var): Patches FlashAttentionImpl.
     Kept for backward compatibility but breaks in vLLM V1 with CUDA graphs.

Environment variables:
    TQ_WEIGHT_BITS: weight quantization bits (2-8)
    TQ_WEIGHT_GROUP_SIZE: weight group size (default 128)
    TQ_KV_K_BITS: monkey-patch mode — key bits (deprecated, use --kv-cache-dtype)
    TQ_KV_V_BITS: monkey-patch mode — value bits
    TQ_KV_ROTATION: rotation mode for monkey-patch ('wht' or 'planar')
"""

import logging
import os
import sys

try:
    from vllm.logger import init_logger

    logger = init_logger("turboquant_vllm.plugin")
except ImportError:
    logger = logging.getLogger("turboquant_vllm.plugin")
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

_patched = False
_native_backend_registered = False


def _register_native_backend() -> bool:
    """Register TurboQuantAttentionBackend as vLLM's CUSTOM attention backend.

    Also patches:
    - CudaPlatform.get_valid_backends to route kv_cache_dtype=tq* → CUSTOM
    - CacheConfig validation to accept tq3/tq4/tq_k4v3 dtype strings
    - AttentionLayer._init_turboquant_buffers to attach rotation matrices

    Returns True if registration succeeded.
    """
    global _native_backend_registered
    if _native_backend_registered:
        return True

    try:
        from vllm.v1.attention.backends.registry import (
            AttentionBackendEnum,
            register_backend,
        )

        register_backend(
            AttentionBackendEnum.CUSTOM,
            "turboquant_vllm.native_backend.TurboQuantAttentionBackend",
        )
        logger.info("TurboQuant native backend registered as CUSTOM (pid=%d)", os.getpid())
    except Exception as e:
        logger.warning("Failed to register TurboQuant native backend: %s", e)
        return False

    # Patch CudaPlatform.get_valid_backends to route tq* → CUSTOM.
    # In vLLM this is an instance method, not classmethod, so the attribute
    # is a regular function — no .__func__ needed.
    try:
        from vllm.platforms.cuda import CudaPlatform

        _orig_get_valid = CudaPlatform.get_valid_backends

        def _tq_get_valid_backends(self, device_capability, attn_selector_config, num_heads=None):
            kv_cache_dtype = getattr(attn_selector_config, "kv_cache_dtype", None)
            if kv_cache_dtype is not None and str(kv_cache_dtype).startswith("tq"):
                from vllm.v1.attention.backends.registry import AttentionBackendEnum

                return [(AttentionBackendEnum.CUSTOM, 0)], {}
            return _orig_get_valid(self, device_capability, attn_selector_config, num_heads)

        CudaPlatform.get_valid_backends = _tq_get_valid_backends
        logger.debug("TurboQuant patched CudaPlatform.get_valid_backends")
    except Exception as e:
        logger.warning("Could not patch CudaPlatform.get_valid_backends: %s", e)

    # Patch CacheConfig to accept tq* kv_cache_dtype without validation error.
    # vLLM uses a Literal type hint but doesn't enforce it at runtime via
    # __set_name__ — however some versions validate in __post_init__.
    # We patch by adding tq* to the allowed set if a validator exists.
    try:
        _patch_cache_dtype_validation()
    except Exception as e:
        logger.debug("Cache dtype validation patch skipped: %s", e)

    # Patch AttentionLayer to initialize TQ buffers (Pi, S, centroids) when
    # kv_cache_dtype is tq*. In stock vLLM, this method doesn't exist, so we
    # add it via monkey-patch.
    try:
        _patch_attention_layer_init()
    except Exception as e:
        logger.warning("Could not patch AttentionLayer for TQ buffer init: %s", e)

    _native_backend_registered = True
    return True


def _patch_cache_dtype_validation() -> None:
    """Allow tq3/tq4/tq_k4v3 as valid kv_cache_dtype values in CacheConfig."""
    try:
        import vllm.config.cache as cache_mod

        # Some vLLM versions have explicit validation in validate_cache_dtype
        if hasattr(cache_mod, "validate_cache_dtype"):
            _orig_validate = cache_mod.validate_cache_dtype

            def _tq_validate(v):
                if isinstance(v, str) and v.startswith("tq"):
                    return v
                return _orig_validate(v)

            cache_mod.validate_cache_dtype = _tq_validate
            return
    except Exception:
        pass

    # Fallback: look for the CacheConfig class validator
    try:
        from vllm.config.cache import CacheConfig

        if hasattr(CacheConfig, "__validators__"):
            # Remove dtype validator if it rejects unknown types
            pass
        # Most likely the Literal is only a type hint — no runtime enforcement
    except Exception:
        pass


def _patch_attention_layer_init() -> None:
    """Inject _init_turboquant_buffers into vLLM's AttentionLayer.

    In the vllm fork, AttentionLayer.__init__ calls self._init_turboquant_buffers()
    when kv_cache_dtype.startswith('tq'). In stock vLLM this code doesn't exist.
    We add it via monkey-patch so the rotation matrices (Pi, S, centroids)
    are attached to the layer before the model runs.
    """
    try:
        from vllm.model_executor.layers.attention.attention import AttentionLayer
    except ImportError:
        return

    if hasattr(AttentionLayer, "_tq_buffers_patched"):
        return

    # Save original __init__
    _orig_init = AttentionLayer.__init__

    def _patched_init(self, *args, **kwargs):
        _orig_init(self, *args, **kwargs)
        kv_cache_dtype = getattr(self, "kv_cache_dtype", "auto")
        if isinstance(kv_cache_dtype, str) and kv_cache_dtype.startswith("tq"):
            head_size = getattr(self, "head_size", None)
            prefix = getattr(self, "layer_name", "") or ""
            if head_size is not None:
                _init_tq_buffers(self, kv_cache_dtype, head_size, prefix)

    AttentionLayer.__init__ = _patched_init
    AttentionLayer._tq_buffers_patched = True
    logger.debug("TurboQuant patched AttentionLayer.__init__ for TQ buffer init")


def _init_tq_buffers(layer, cache_dtype: str, head_size: int, prefix: str) -> None:
    """Attach Pi, S, centroids buffers to an AttentionLayer."""
    from turboquant_vllm.tq_config import (
        TurboQuantConfig,
        generate_rotation_matrix,
        generate_qjl_matrix,
        get_centroids,
    )

    tq_config = TurboQuantConfig.from_cache_dtype(cache_dtype, head_size)

    # Extract layer index from prefix (e.g. "model.layers.5.self_attn")
    layer_idx = 0
    try:
        import re

        m = re.search(r"\.(\d+)\.", prefix)
        if m:
            layer_idx = int(m.group(1))
    except Exception:
        pass

    seed = tq_config.seed + layer_idx * 1337
    # Use register_buffer so tensors are automatically moved to GPU with the model
    layer.register_buffer("_tq_Pi", generate_rotation_matrix(head_size, seed=seed))
    layer.register_buffer("_tq_S", generate_qjl_matrix(head_size, seed=seed + 1))
    layer.register_buffer("_tq_centroids", get_centroids(head_size, tq_config.mse_bits))
    layer._tq_config = tq_config


def register():
    """Called by vLLM's plugin loader in every process."""
    global _patched

    # Always register weight quantization config (needed for TQ3 checkpoint loading)
    try:
        from turboquant_vllm.vllm_quant import register as register_quant_config

        register_quant_config()
    except Exception as e:
        logger.debug("Could not register TurboQuant quant config: %s", e)

    # Always try to register the native backend (no env var needed — it only
    # activates when the user passes --kv-cache-dtype tq3/tq4/tq_k4v3)
    _register_native_backend()

    weight_bits = os.environ.get("TQ_WEIGHT_BITS")
    kv_k_bits = os.environ.get("TQ_KV_K_BITS")

    if weight_bits is not None or kv_k_bits is not None:
        logger.info(
            "TurboQuant plugin activated (pid=%d, TQ_WEIGHT_BITS=%s, TQ_KV_K_BITS=%s)",
            os.getpid(),
            weight_bits,
            kv_k_bits,
        )

    if weight_bits is None and kv_k_bits is None:
        return

    if _patched:
        return
    _patched = True

    # Weight compression
    if weight_bits is not None:
        bits = int(weight_bits)
        group_size = int(os.environ.get("TQ_WEIGHT_GROUP_SIZE", "128"))
        try:
            from turboquant_vllm.weight_quant import patch_vllm_loader

            patch_vllm_loader(bits=bits, group_size=group_size, min_size=128)
            logger.info("TurboQuant TQ%d-g%d weight compression registered (pid=%d)", bits, group_size, os.getpid())
        except ImportError as e:
            logger.warning("Failed to register weight compression: %s", e)

    # KV cache compression (monkey-patch mode — deprecated, kept for compat)
    if kv_k_bits is not None:
        k_bits = int(kv_k_bits)
        v_bits = int(os.environ.get("TQ_KV_V_BITS", str(k_bits)))
        norm_correction = os.environ.get("TQ_KV_NORM_CORRECTION", "1") == "1"
        rotation = os.environ.get("TQ_KV_ROTATION", "wht")
        try:
            from turboquant_vllm.vllm_patch import patch_vllm_attention

            patch_vllm_attention(k_bits=k_bits, v_bits=v_bits, norm_correction=norm_correction, rotation=rotation)
            logger.info(
                "TurboQuant K%d/V%d KV cache compression registered via monkey-patch "
                "(pid=%d) — note: breaks with vLLM V1 CUDA graphs. "
                "Use --kv-cache-dtype tq3 for native backend.",
                k_bits,
                v_bits,
                os.getpid(),
            )
        except ImportError as e:
            logger.warning("Failed to register KV cache compression: %s", e)
