"""TurboQuant+ for vLLM — weight quantization plugin.

Primary feature — weight quantization (any BF16 model, 3-bit TQ3 on the fly):
    from turboquant_vllm import enable_weight_quantization
    enable_weight_quantization(bits=3)

Pre-compressed native checkpoint loading:
    from turboquant_vllm import load_tq3_model
    model, tokenizer = load_tq3_model("varjosoft/gemma-4-26B-A4B-it-TQ3-native")

REAP expert pruning + TQ compression + export for Marlin serving:
    from turboquant_vllm.export import compress_and_export
    compress_and_export("Qwen/Qwen3-30B-A3B", "./output", prune_experts=0.5)

Legacy — standalone KV cache compression via attention monkey-patch:
    from turboquant_vllm import patch_vllm_attention
    patch_vllm_attention(k_bits=4, v_bits=3, norm_correction=True, sink_tokens=4)

    The standalone KV path is being superseded by upstream vLLM integration
    (vllm-project/vllm#38479 — 2-bit KV cache compression with 4x capacity).
    Prefer the upstream path once it lands; this plugin's direction is
    weight quantization only.
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

__version__ = "0.13.0"
