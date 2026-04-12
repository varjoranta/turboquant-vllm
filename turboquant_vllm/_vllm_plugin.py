"""vLLM plugin entry point for TurboQuant+.

Registers the TurboQuant quantization config (for loading native TQ3
checkpoints via vLLM's quantization registry) and wires up two optional,
env-var-activated runtime paths:

  TQ_WEIGHT_BITS=3 vllm serve <model>
      Runtime weight compression: any BF16 checkpoint → TQ3/TQ4 on load.
      Unique to this plugin — not available upstream.

  TQ_KV_K_BITS=4 vllm serve <model>
      Legacy monkey-patch KV cache compression. Works on MLA models
      (GLM-4.7, DeepSeek-V3) where upstream vLLM's KV quantization does
      not. Breaks with V1 CUDA graphs in the general case; kept for
      MLA-specific deployments until upstream adds MLA support.

The plugin no longer ships a CUSTOM attention backend or patches vLLM's
KV cache allocator. That work has been upstreamed by @vibhavagarwal5 in
https://github.com/vllm-project/vllm/pull/38479 and is more complete,
better tested, and uses a revised preset naming scheme
(``--kv-cache-dtype turboquant_3bit_nc`` etc.). Users who want the
native KV cache path should use that PR once it lands, or install
Vibhav's branch directly:

    pip install git+https://github.com/vibhavagarwal5/vllm.git@feature/turboquant-kv-cache
"""

import logging
import os

try:
    from vllm.logger import init_logger

    logger = init_logger("turboquant_vllm.plugin")
except ImportError:
    import sys

    logger = logging.getLogger("turboquant_vllm.plugin")
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)


_patched = False


def register():
    """Called by vLLM's plugin loader in every process."""
    global _patched

    # Always register TurboQuant quant config — needed for loading native
    # TQ3 checkpoints via vLLM's quantization registry. Cheap, idempotent,
    # no env var required.
    try:
        from turboquant_vllm.vllm_quant import register as register_quant_config

        register_quant_config()
    except Exception as e:
        logger.debug("Could not register TurboQuant quant config: %s", e)

    weight_bits = os.environ.get("TQ_WEIGHT_BITS")
    kv_k_bits = os.environ.get("TQ_KV_K_BITS")

    if weight_bits is None and kv_k_bits is None:
        return

    logger.info(
        "TurboQuant plugin activated (pid=%d, TQ_WEIGHT_BITS=%s, TQ_KV_K_BITS=%s)",
        os.getpid(),
        weight_bits,
        kv_k_bits,
    )

    if _patched:
        return
    _patched = True

    # Weight compression — unique to this plugin.
    if weight_bits is not None:
        bits = int(weight_bits)
        group_size = int(os.environ.get("TQ_WEIGHT_GROUP_SIZE", "128"))
        try:
            from turboquant_vllm.weight_quant import patch_vllm_loader

            patch_vllm_loader(bits=bits, group_size=group_size, min_size=128)
            logger.info(
                "TurboQuant TQ%d-g%d weight compression registered (pid=%d)",
                bits,
                group_size,
                os.getpid(),
            )
        except ImportError as e:
            logger.warning("Failed to register weight compression: %s", e)

    # Legacy monkey-patch KV compression — retained for MLA models where
    # upstream TurboQuant (vllm-project/vllm#38479) does not yet apply.
    if kv_k_bits is not None:
        k_bits = int(kv_k_bits)
        v_bits = int(os.environ.get("TQ_KV_V_BITS", str(k_bits)))
        norm_correction = os.environ.get("TQ_KV_NORM_CORRECTION", "1") == "1"
        rotation = os.environ.get("TQ_KV_ROTATION", "wht")
        boundary_layers = int(os.environ.get("TQ_KV_BOUNDARY_LAYERS", "5"))
        try:
            from turboquant_vllm.vllm_patch import patch_vllm_attention

            patch_vllm_attention(
                k_bits=k_bits,
                v_bits=v_bits,
                norm_correction=norm_correction,
                rotation=rotation,
                boundary_layers=boundary_layers,
            )
            logger.info(
                "TurboQuant K%d/V%d KV monkey-patch registered (pid=%d). "
                "This path breaks with V1 CUDA graphs on non-MLA models — "
                "for non-MLA use vllm-project/vllm#38479 when it lands.",
                k_bits,
                v_bits,
                os.getpid(),
            )
        except ImportError as e:
            logger.warning("Failed to register KV cache compression: %s", e)
