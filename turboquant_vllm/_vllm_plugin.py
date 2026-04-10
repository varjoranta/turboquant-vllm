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
_str_dtype_patched = False


# ---------------------------------------------------------------------------
# STR_DTYPE patch — runs at plugin module load, BEFORE register().
#
# Rationale: vLLM may resolve --kv-cache-dtype during CLI validation in the
# main process before it ever calls load_general_plugins() in a subprocess.
# If we wait for register() to mutate STR_DTYPE_TO_TORCH_DTYPE, the main
# process has already raised KeyError on "tq3". Running as a module-level
# side-effect on first import of this plugin module is the earliest safe
# hook we have.
#
# Belt-and-suspenders: we both mutate the dict AND wrap the resolver
# function. The dict mutation handles call sites that look up
# STR_DTYPE_TO_TORCH_DTYPE at call time; the function wrap handles call
# sites that have already resolved the function by name (closure-captured).
# ---------------------------------------------------------------------------

def _eager_patch_str_dtype_mapping() -> None:
    """Patch STR_DTYPE_TO_TORCH_DTYPE and kv_cache_dtype_str_to_dtype so
    vLLM's dtype resolver accepts tq3/tq4/tq_k4v3 as valid strings.

    Called both at plugin module import time (top of file) and
    redundantly inside _register_native_backend() for safety.
    """
    global _str_dtype_patched
    if _str_dtype_patched:
        return
    try:
        import torch
        import vllm.utils.torch_utils as _tu
    except Exception:
        # vLLM may not be installed (pure test / CI environment).
        # Nothing to patch; downstream register() will report appropriately.
        return

    try:
        # Primary fix: populate the dict with tq* → uint8 entries.
        for name in ("tq3", "tq4", "tq_k4v3"):
            _tu.STR_DTYPE_TO_TORCH_DTYPE.setdefault(name, torch.uint8)

        # Belt-and-suspenders: wrap the resolver function itself. Some
        # callers may have imported it via
        #   from vllm.utils.torch_utils import kv_cache_dtype_str_to_dtype
        # which binds the original function object. Mutating the module-
        # level attribute handles the common "import vllm.utils.torch_utils
        # as X; X.kv_cache_dtype_str_to_dtype(...)" pattern; the dict patch
        # above handles the closure-captured pattern. Doing both is
        # cheap and covers the majority of realistic call sites.
        _orig_resolver = _tu.kv_cache_dtype_str_to_dtype

        # Idempotent — if we've already wrapped, don't wrap again
        if getattr(_orig_resolver, "_tq_wrapped", False):
            _str_dtype_patched = True
            return

        def _tq_aware_resolver(kv_cache_dtype, model_config):
            if isinstance(kv_cache_dtype, str) and kv_cache_dtype in (
                "tq3", "tq4", "tq_k4v3",
            ):
                return torch.uint8
            return _orig_resolver(kv_cache_dtype, model_config)

        _tq_aware_resolver._tq_wrapped = True  # type: ignore[attr-defined]
        _tu.kv_cache_dtype_str_to_dtype = _tq_aware_resolver
        _str_dtype_patched = True
        logger.info(
            "TurboQuant patched STR_DTYPE_TO_TORCH_DTYPE with tq3/tq4/tq_k4v3 "
            "(pid=%d)", os.getpid(),
        )
    except Exception as e:
        logger.warning("Could not patch STR_DTYPE_TO_TORCH_DTYPE: %s", e)


# Fire the eager patch on module load, before register() is called.
_eager_patch_str_dtype_mapping()


def _register_native_backend() -> bool:
    """Register TurboQuantAttentionBackend as vLLM's CUSTOM attention backend.

    Also patches:
    - STR_DTYPE_TO_TORCH_DTYPE (redundant to eager patch, belt-and-suspenders)
    - CudaPlatform.get_valid_backends to route kv_cache_dtype=tq* → CUSTOM
    - AttentionLayer.get_kv_cache_spec to remap head_size for tq* (NEW)
    - MLAAttention.get_kv_cache_spec to fail loud on tq* (NEW)
    - CacheConfig validation to accept tq3/tq4/tq_k4v3 dtype strings
    - AttentionLayer.__init__ to attach rotation matrices

    Returns True if registration succeeded.
    """
    global _native_backend_registered
    if _native_backend_registered:
        return True

    # Redundant STR_DTYPE patch — the eager module-level patch at the top
    # of this file should have already run, but call again here in case
    # vLLM was imported after the plugin module without going through
    # register(). setdefault / _tq_wrapped checks make this a no-op when
    # already patched.
    _eager_patch_str_dtype_mapping()

    try:
        from vllm.v1.attention.backends.registry import (
            AttentionBackendEnum,
            register_backend,
        )
        from turboquant_vllm.native_backend import TurboQuantAttentionBackend

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

    # Patch AttentionLayer.get_kv_cache_spec so --kv-cache-dtype tq*
    # actually shrinks vLLM's cache allocation. Without this the
    # allocator allocates a bf16-sized cache and TQ compression
    # delivers zero token-capacity gain — the bug we're fixing.
    try:
        _patch_get_kv_cache_spec()
    except Exception as e:
        logger.warning("Could not patch AttentionLayer.get_kv_cache_spec: %s", e)

    # Patch MLAAttention.get_kv_cache_spec so tq* + MLA fails loud
    # instead of silently mis-allocating. MLA is handled by the
    # monkey-patch path, not the native backend.
    try:
        _patch_mla_fail_loud()
    except Exception as e:
        logger.warning("Could not install MLA fail-loud guard: %s", e)

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


def _patch_get_kv_cache_spec() -> None:
    """Inject tq* handling into AttentionLayer.get_kv_cache_spec.

    When kv_cache_dtype starts with 'tq', returns a FullAttentionSpec
    with head_size remapped to padded_slot_size // 2, so vLLM's
    per-slot byte accounting — block_size * num_kv_heads *
    (head_size + head_size_v) * sizeof(uint8) — yields exactly the
    compressed slot size. Without this patch, vLLM allocates a
    bf16-sized cache regardless of dtype and TQ compression delivers
    zero token-capacity gain.

    Signature-guarded: if AttentionLayer.get_kv_cache_spec's signature
    doesn't match what we expect, we skip patching with a warning
    rather than apply a broken wrapper.
    """
    try:
        from vllm.model_executor.layers.attention.attention import AttentionLayer
        from vllm.v1.kv_cache_interface import FullAttentionSpec
    except ImportError:
        return

    if getattr(AttentionLayer, "_tq_spec_patched", False):
        return

    _orig_get_spec = AttentionLayer.get_kv_cache_spec

    # Validate signature before patching. Current fork targets
    # (self, vllm_config). If vLLM changes this we want to fail
    # closed (skip + warn) rather than silently apply a broken patch.
    import inspect
    try:
        sig = inspect.signature(_orig_get_spec)
        params = list(sig.parameters.keys())
    except (TypeError, ValueError):
        params = None
    if params != ["self", "vllm_config"]:
        logger.warning(
            "AttentionLayer.get_kv_cache_spec has unexpected signature %s, "
            "expected ['self', 'vllm_config']. Skipping TQ spec patch — "
            "--kv-cache-dtype tq* will not allocate correctly. Please file "
            "an issue at varjoranta/turboquant-vllm with this signature.",
            params,
        )
        return

    def _patched_get_kv_cache_spec(self, vllm_config):
        kv_cache_dtype = getattr(self, "kv_cache_dtype", "auto") or "auto"
        if isinstance(kv_cache_dtype, str) and kv_cache_dtype.startswith("tq"):
            from turboquant_vllm.tq_config import TurboQuantConfig
            block_size = vllm_config.cache_config.block_size
            tq_config = TurboQuantConfig.from_cache_dtype(
                kv_cache_dtype, self.head_size
            )
            padded_slot = tq_config.padded_slot_size
            effective_head_size = padded_slot // 2
            # Allocator arithmetic with dtype=uint8 (1 byte):
            #   block_size * num_kv_heads * (hs + hs_v) * 1
            # = block_size * num_kv_heads * padded_slot
            # which equals the real compressed slot byte count.
            return FullAttentionSpec(
                block_size=block_size,
                num_kv_heads=self.num_kv_heads,
                head_size=effective_head_size,
                head_size_v=effective_head_size,
                dtype=self.kv_cache_torch_dtype,
            )
        return _orig_get_spec(self, vllm_config)

    AttentionLayer.get_kv_cache_spec = _patched_get_kv_cache_spec
    AttentionLayer._tq_spec_patched = True
    logger.info(
        "TurboQuant patched AttentionLayer.get_kv_cache_spec for tq* dtypes "
        "(pid=%d)", os.getpid(),
    )


def _patch_mla_fail_loud() -> None:
    """Fail loudly if --kv-cache-dtype tq* is used with an MLA model.

    MLA uses a separate attention class (MLAAttention) whose cache
    spec is MLAAttentionSpec, not FullAttentionSpec. The native
    backend's effective_head_size trick doesn't apply to MLA's
    latent-compressed cache layout, so silent mis-allocation would
    be the worst outcome. We intercept the MLA spec path and raise
    a clear error pointing users at the monkey-patch alternative
    (TQ_KV_K_BITS env var) which DOES work for MLA because it
    patches MLACommonImpl.do_kv_cache_update at runtime, bypassing
    vLLM's cache allocation entirely.
    """
    try:
        from vllm.model_executor.layers.attention.mla_attention import MLAAttention
    except ImportError:
        return

    if getattr(MLAAttention, "_tq_mla_guard_patched", False):
        return

    _orig_mla_spec = MLAAttention.get_kv_cache_spec

    def _guarded_mla_spec(self, vllm_config):
        kv_cache_dtype = getattr(self, "kv_cache_dtype", "auto") or "auto"
        if isinstance(kv_cache_dtype, str) and kv_cache_dtype.startswith("tq"):
            raise RuntimeError(
                f"--kv-cache-dtype {kv_cache_dtype} is not supported for "
                "MLA models (GLM-4.7, DeepSeek-V3, etc.). The native "
                "TurboQuant backend targets standard GQA/MHA attention "
                "only. For MLA models, use the monkey-patch path instead: "
                "set TQ_KV_K_BITS=3 TQ_KV_V_BITS=3 in the environment and "
                "run vllm serve with --kv-cache-dtype auto. The monkey-patch "
                "intercepts MLACommonImpl.do_kv_cache_update at runtime and "
                "works correctly with MLA's latent-compressed cache."
            )
        return _orig_mla_spec(self, vllm_config)

    MLAAttention.get_kv_cache_spec = _guarded_mla_spec
    MLAAttention._tq_mla_guard_patched = True
    logger.info(
        "TurboQuant installed MLA fail-loud guard (pid=%d)", os.getpid(),
    )


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
        m = re.search(r'\.(\d+)\.', prefix)
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
            os.getpid(), weight_bits, kv_k_bits,
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
            logger.info("TurboQuant TQ%d-g%d weight compression registered (pid=%d)",
                        bits, group_size, os.getpid())
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
            patch_vllm_attention(k_bits=k_bits, v_bits=v_bits,
                                 norm_correction=norm_correction, rotation=rotation)
            logger.info(
                "TurboQuant K%d/V%d KV cache compression registered via monkey-patch "
                "(pid=%d) — note: breaks with vLLM V1 CUDA graphs. "
                "Use --kv-cache-dtype tq3 for native backend.",
                k_bits, v_bits, os.getpid(),
            )
        except ImportError as e:
            logger.warning("Failed to register KV cache compression: %s", e)
