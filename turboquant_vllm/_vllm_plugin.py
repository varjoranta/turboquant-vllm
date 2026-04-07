"""vLLM plugin: auto-registers TQ weight compression in all processes.

Activated via environment variables set by enable_weight_quantization().
This plugin is loaded by vLLM in the main process AND in spawned
subprocesses (V1 engine), so the monkey-patch survives multiprocessing spawn.

Environment variables:
    TQ_WEIGHT_BITS: quantization bits (2-8), triggers the hook
    TQ_WEIGHT_GROUP_SIZE: group size (default 128)
"""
import os

_patched = False


def register():
    """Called by vLLM's plugin loader in every process."""
    global _patched

    bits = os.environ.get("TQ_WEIGHT_BITS")
    if bits is None:
        return

    if _patched:
        return
    _patched = True

    bits = int(bits)
    group_size = int(os.environ.get("TQ_WEIGHT_GROUP_SIZE", "128"))

    try:
        from turboquant_vllm.weight_quant import patch_vllm_loader
        patch_vllm_loader(bits=bits, group_size=group_size, min_size=128)
    except ImportError:
        return
