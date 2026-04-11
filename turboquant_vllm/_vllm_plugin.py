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

import functools
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

# Canonical list of TurboQuant KV cache dtype strings. Kept here rather than
# in tq_config.py because tq_config can be imported standalone for CPU testing
# without involving this plugin; the plugin is the right home for the string
# contract with vLLM's --kv-cache-dtype flag.
_TQ_DTYPE_NAMES = ("tq3", "tq4", "tq_k4v3")


def _is_tq_dtype(v) -> bool:
    """True if v is one of the TurboQuant KV cache dtype strings."""
    return isinstance(v, str) and v in _TQ_DTYPE_NAMES


def _is_duplicate_backend_registration_error(exc: Exception) -> bool:
    """Heuristic for vLLM duplicate backend registration errors."""
    msg = str(exc).lower()
    if "backend" not in msg and "custom" not in msg:
        return False
    return "already registered" in msg or "already exists" in msg


_patched = False
_native_backend_registered = False


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

    Fires at plugin module import time AND (redundantly) inside
    _register_native_backend(). Both paths are idempotent via the
    _tq_wrapped marker on the wrapped resolver — no module-level flag
    needed, the wrapper function itself is the single source of truth.
    """
    try:
        import torch
        import vllm.utils.torch_utils as _tu
    except ImportError:
        # vLLM not installed (CPU test / CI environment). Nothing to
        # patch; downstream register() handles reporting. Deliberately
        # narrow: any *other* exception here is a real bug we want to
        # surface instead of swallow.
        return

    # Idempotency gate — if the wrapper is already installed, both the
    # dict and function mutations are already in place from a previous
    # call. Checking the wrapped marker is cheaper and more robust than
    # a separate module-level flag that can drift out of sync.
    if getattr(_tu.kv_cache_dtype_str_to_dtype, "_tq_wrapped", False):
        return

    # Primary fix: populate STR_DTYPE_TO_TORCH_DTYPE with tq* → uint8.
    # setdefault preserves any fork-provided entries (the fork already
    # ships these; setdefault makes us a no-op in that case).
    for name in _TQ_DTYPE_NAMES:
        _tu.STR_DTYPE_TO_TORCH_DTYPE.setdefault(name, torch.uint8)

    # Belt-and-suspenders: wrap the resolver function itself. Some call
    # sites may have imported it via
    #   from vllm.utils.torch_utils import kv_cache_dtype_str_to_dtype
    # which binds the original function object. Mutating the module-level
    # attribute handles the common
    #   import vllm.utils.torch_utils as X; X.kv_cache_dtype_str_to_dtype(...)
    # pattern; the dict patch above handles closure-captured call sites.
    _orig_resolver = _tu.kv_cache_dtype_str_to_dtype

    def _tq_aware_resolver(kv_cache_dtype, model_config):
        if _is_tq_dtype(kv_cache_dtype):
            return torch.uint8
        return _orig_resolver(kv_cache_dtype, model_config)

    _tq_aware_resolver._tq_wrapped = True  # type: ignore[attr-defined]
    _tu.kv_cache_dtype_str_to_dtype = _tq_aware_resolver
    msg = f"TurboQuant patched STR_DTYPE_TO_TORCH_DTYPE with tq3/tq4/tq_k4v3 (pid={os.getpid()})"
    logger.info(msg)
    # Also print to stderr: vLLM's subprocess launcher sometimes runs before
    # the logging handlers are attached, and the CI timing check greps raw
    # server output for this exact string.
    print(msg, file=sys.stderr, flush=True)


def _eager_patch_cache_dtype_literal() -> None:
    """Extend vLLM's CacheDType Literal to accept tq3/tq4/tq_k4v3.

    Ordering invariant: AsyncEngineArgs.add_cli_args() calls
    load_general_plugins() BEFORE EngineArgs.add_cli_args() builds the
    argparse choices list from CacheConfig's type hints. This patch fires
    at plugin import time so the extended Literal is in place when the
    choices are computed. Without it, argparse rejects --kv-cache-dtype tq3
    before any engine code runs.
    """
    try:
        from typing import Literal

        import vllm.config.cache as cache_mod
        from vllm.config.cache import CacheConfig
    except ImportError:
        return

    current = getattr(cache_mod, "CacheDType", None)
    if current is None:
        return

    import typing as _typing

    existing = _typing.get_args(current)
    if all(name in existing for name in _TQ_DTYPE_NAMES):
        return  # already extended

    new_args = tuple(existing) + tuple(n for n in _TQ_DTYPE_NAMES if n not in existing)
    extended = Literal[new_args]  # type: ignore[valid-type]

    cache_mod.CacheDType = extended
    try:
        CacheConfig.__annotations__["cache_dtype"] = extended
    except Exception as e:
        logger.debug("Could not patch CacheConfig.__annotations__: %s", e)

    # dataclasses.fields(CacheConfig) captures each field's .type at class
    # creation time; get_kwargs() reads these, so mutating __annotations__
    # alone is insufficient.
    try:
        from dataclasses import fields

        for f in fields(CacheConfig):
            if f.name == "cache_dtype":
                f.type = extended
                break
    except Exception as e:
        logger.debug("Could not patch CacheConfig fields: %s", e)

    # _compute_kwargs is @lru_cache'd; a prior --help probe may have cached
    # the old choices list.
    try:
        from vllm.engine.arg_utils import _compute_kwargs

        _compute_kwargs.cache_clear()
    except Exception as e:
        logger.debug("Could not clear _compute_kwargs cache: %s", e)

    # CacheConfig is a pydantic dataclass; rebuilding forces a fresh
    # __pydantic_validator__ that honors the patched annotations.
    try:
        from pydantic.dataclasses import rebuild_dataclass

        rebuild_dataclass(CacheConfig, force=True, raise_errors=True)
    except Exception as e:
        logger.debug("Could not rebuild CacheConfig pydantic validator: %s", e)

    logger.info(
        "TurboQuant extended CacheDType Literal with %s (pid=%d)",
        list(_TQ_DTYPE_NAMES),
        os.getpid(),
    )


# Fire the eager patches on module load, before register() is called.
_eager_patch_str_dtype_mapping()
_eager_patch_cache_dtype_literal()


def _tq_reset_patches_for_test() -> None:
    """Reset all plugin-side patch state. Test-only helper.

    Clears:
    - _native_backend_registered module flag
    - AttentionLayer._tq_spec_patched / _tq_buffers_patched class attrs
    - MLAAttention._tq_mla_guard_patched class attr

    Does NOT unwind already-applied monkey patches (that would require
    storing the originals somewhere); the tests that need to restore
    specific patched methods do so via their own save/restore scopes.

    Idempotent; safe to call when vLLM isn't importable.
    """
    global _native_backend_registered
    _native_backend_registered = False
    for mod_path, cls_name, flag in (
        ("vllm.model_executor.layers.attention.attention", "AttentionLayer", "_tq_spec_patched"),
        ("vllm.model_executor.layers.attention.attention", "AttentionLayer", "_tq_buffers_patched"),
        ("vllm.model_executor.layers.attention.mla_attention", "MLAAttention", "_tq_mla_guard_patched"),
    ):
        try:
            mod = __import__(mod_path, fromlist=[cls_name])
            cls = getattr(mod, cls_name, None)
            if cls is not None and flag in cls.__dict__:
                delattr(cls, flag)
        except ImportError:
            pass


@functools.lru_cache(maxsize=8)
def _tq_effective_head_size(cache_dtype: str, head_size: int) -> int:
    """padded_slot_size // 2 for a given (cache_dtype, head_size).

    Cached because get_kv_cache_spec calls this once per attention layer
    at engine init — and the result is deterministic from the arguments.
    Hoisting the TurboQuantConfig import to module scope would pull
    scipy/torch into plugin-load time; this lazy path keeps import cheap
    and the cached result removes the repeated-call cost.
    """
    from turboquant_vllm.tq_config import TurboQuantConfig

    tq_config = TurboQuantConfig.from_cache_dtype(cache_dtype, head_size)
    return tq_config.padded_slot_size // 2


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

        register_backend(
            AttentionBackendEnum.CUSTOM,
            "turboquant_vllm.native_backend.TurboQuantAttentionBackend",
        )
        logger.info("TurboQuant native backend registered as CUSTOM (pid=%d)", os.getpid())
    except Exception as e:
        if _is_duplicate_backend_registration_error(e):
            logger.info("TurboQuant native backend already registered, continuing (pid=%d)", os.getpid())
        else:
            logger.warning("Failed to register TurboQuant native backend: %s", e)
            return False

    # Patch CudaPlatform.get_valid_backends to route tq* → CUSTOM.
    # It's a @classmethod on the class; our replacement must be wrapped in
    # classmethod() too or vLLM's call path crashes with "missing cls".
    # Accessing the original via the class gives a bound method, so
    # _orig_get_valid(...) below can be called without passing cls.
    try:
        import inspect

        from vllm.platforms.cuda import CudaPlatform

        current_get_valid = CudaPlatform.get_valid_backends
        current_fn = getattr(current_get_valid, "__func__", current_get_valid)
        if not getattr(current_fn, "_tq_wrapped", False):
            _orig_get_valid_fn = current_fn
            try:
                param_names = list(inspect.signature(_orig_get_valid_fn).parameters.keys())[1:]
            except (TypeError, ValueError):
                param_names = []

            def _tq_get_valid_backends(cls, *args, **kwargs):
                attn_selector_config = kwargs.get("attn_selector_config")
                if attn_selector_config is None and "attn_selector_config" in param_names:
                    idx = param_names.index("attn_selector_config")
                    if idx < len(args):
                        attn_selector_config = args[idx]
                if attn_selector_config is None:
                    attn_selector_config = next((a for a in args if hasattr(a, "kv_cache_dtype")), None)
                kv_cache_dtype = getattr(attn_selector_config, "kv_cache_dtype", None)
                if _is_tq_dtype(kv_cache_dtype):
                    from vllm.v1.attention.backends.registry import AttentionBackendEnum

                    return [(AttentionBackendEnum.CUSTOM, 0)], {}
                return _orig_get_valid_fn(cls, *args, **kwargs)

            _tq_get_valid_backends._tq_wrapped = True  # type: ignore[attr-defined]
            CudaPlatform.get_valid_backends = classmethod(_tq_get_valid_backends)
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
        if _is_tq_dtype(kv_cache_dtype):
            # Allocator arithmetic with dtype=uint8 (1 byte):
            #   block_size * num_kv_heads * (hs + hs_v) * 1
            # = block_size * num_kv_heads * padded_slot
            # which equals the real compressed slot byte count.
            effective_head_size = _tq_effective_head_size(kv_cache_dtype, self.head_size)
            return FullAttentionSpec(
                block_size=vllm_config.cache_config.block_size,
                num_kv_heads=self.num_kv_heads,
                head_size=effective_head_size,
                head_size_v=effective_head_size,
                dtype=self.kv_cache_torch_dtype,
            )
        return _orig_get_spec(self, vllm_config)

    AttentionLayer.get_kv_cache_spec = _patched_get_kv_cache_spec
    AttentionLayer._tq_spec_patched = True
    logger.info(
        "TurboQuant patched AttentionLayer.get_kv_cache_spec for tq* dtypes (pid=%d)",
        os.getpid(),
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
        if _is_tq_dtype(kv_cache_dtype):
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
        "TurboQuant installed MLA fail-loud guard (pid=%d)",
        os.getpid(),
    )


def _patch_attention_layer_init() -> None:
    """Inject _init_turboquant_buffers into vLLM's Attention layer.

    In the vllm fork, the layer __init__ calls self._init_turboquant_buffers()
    when kv_cache_dtype.startswith('tq'); stock vLLM lacks this, so we
    monkey-patch __init__ to attach the rotation matrices (Pi, S, centroids)
    before the model runs. The class is called Attention in current vLLM
    and AttentionLayer in older versions.
    """
    layer_cls = None
    try:
        from vllm.model_executor.layers.attention.attention import Attention as layer_cls  # type: ignore[no-redef]
    except ImportError:
        try:
            from vllm.model_executor.layers.attention.attention import AttentionLayer as layer_cls  # type: ignore[no-redef]
        except ImportError:
            return

    if hasattr(layer_cls, "_tq_buffers_patched"):
        return

    _orig_init = layer_cls.__init__

    def _patched_init(self, *args, **kwargs):
        _orig_init(self, *args, **kwargs)
        kv_cache_dtype = getattr(self, "kv_cache_dtype", "auto")
        if _is_tq_dtype(kv_cache_dtype):
            head_size = getattr(self, "head_size", None)
            prefix = getattr(self, "layer_name", "") or ""
            if head_size is not None:
                _init_tq_buffers(self, kv_cache_dtype, head_size, prefix)

    layer_cls.__init__ = _patched_init
    layer_cls._tq_buffers_patched = True
    logger.debug(
        "TurboQuant patched %s.__init__ for TQ buffer init",
        layer_cls.__name__,
    )


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
