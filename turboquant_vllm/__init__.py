"""TurboQuant+ for vLLM — KV cache compression and weight quantization.

KV cache compression (3.7x smaller cache, same quality):
    from turboquant_vllm import patch_vllm_attention
    patch_vllm_attention(k_bits=4, v_bits=4)

Weight quantization (load any BF16 model, compress to 3-bit on the fly):
    from turboquant_vllm import enable_weight_quantization
    enable_weight_quantization(bits=3)

Both can be used together for maximum memory savings.
"""

from turboquant_vllm.vllm_patch import patch_vllm_attention
from turboquant_vllm.weight_quant import enable_weight_quantization
from turboquant_vllm.torch_ops import (
    KVCacheCompressorTorch,
    PolarQuantTorch,
    QJLTorch,
    CompressedKV,
)

__all__ = [
    "patch_vllm_attention",
    "enable_weight_quantization",
    "KVCacheCompressorTorch",
    "PolarQuantTorch",
    "QJLTorch",
    "CompressedKV",
]

__version__ = "0.1.0"
