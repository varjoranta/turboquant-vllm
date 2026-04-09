"""vLLM plugin: auto-registers TQ weight and KV cache compression.

Activated via environment variables. This plugin is loaded by vLLM in
the main process AND in spawned subprocesses (V1 engine), so the
monkey-patches survive multiprocessing spawn.

Environment variables:
    TQ_WEIGHT_BITS: weight quantization bits (2-8), triggers weight compression
    TQ_WEIGHT_GROUP_SIZE: weight group size (default 128)
    TQ_KV_K_BITS: KV cache key bits (2-8), triggers KV cache compression
    TQ_KV_V_BITS: KV cache value bits (2-8)
"""
import logging
import os

logger = logging.getLogger("turboquant_vllm.plugin")

_patched = False


def register():
    """Called by vLLM's plugin loader in every process."""
    global _patched

    # Always register the quantization config (needed for TQ3 checkpoint loading)
    try:
        from turboquant_vllm.vllm_quant import register as register_quant_config
        register_quant_config()
    except Exception as e:
        logger.debug("Could not register TurboQuant quant config: %s", e)

    weight_bits = os.environ.get("TQ_WEIGHT_BITS")
    kv_k_bits = os.environ.get("TQ_KV_K_BITS")

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

    # KV cache compression
    if kv_k_bits is not None:
        k_bits = int(kv_k_bits)
        v_bits = int(os.environ.get("TQ_KV_V_BITS", str(k_bits)))
        norm_correction = os.environ.get("TQ_KV_NORM_CORRECTION", "1") == "1"
        try:
            from turboquant_vllm.vllm_patch import patch_vllm_attention
            patch_vllm_attention(k_bits=k_bits, v_bits=v_bits, norm_correction=norm_correction)
            logger.info("TurboQuant K%d/V%d KV cache compression registered (pid=%d)",
                         k_bits, v_bits, os.getpid())
        except ImportError as e:
            logger.warning("Failed to register KV cache compression: %s", e)
