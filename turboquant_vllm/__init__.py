"""TurboQuant+ for vLLM — KV cache compression, weight quantization, expert pruning.

KV cache compression (3.7x smaller cache, same quality):
    from turboquant_vllm import patch_vllm_attention
    patch_vllm_attention(k_bits=4, v_bits=3, norm_correction=True, sink_tokens=4)

Weight quantization (load any BF16 model, compress to 4-bit on the fly):
    from turboquant_vllm import enable_weight_quantization
    enable_weight_quantization(bits=4)

REAP expert pruning + TQ compression + export for Marlin serving:
    from turboquant_vllm.export import compress_and_export
    compress_and_export("Qwen/Qwen3-30B-A3B", "./output", prune_experts=0.5)

All features can be combined for maximum memory savings.
"""

from turboquant_vllm.vllm_patch import patch_vllm_attention
from turboquant_vllm.weight_quant import enable_weight_quantization
from turboquant_vllm.checkpoint import load_tq3_model, save_tq3_checkpoint
from turboquant_vllm.torch_ops import (
    KVCacheCompressorTorch,
    PolarQuantTorch,
    QJLTorch,
    CompressedKV,
)

__all__ = [
    "patch_vllm_attention",
    "enable_weight_quantization",
    "load_tq3_model",
    "save_tq3_checkpoint",
    "KVCacheCompressorTorch",
    "PolarQuantTorch",
    "QJLTorch",
    "CompressedKV",
]

__version__ = "0.5.1"
