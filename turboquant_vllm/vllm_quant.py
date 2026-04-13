"""TurboQuant vLLM integration: quantization config + TQ3 checkpoint loader.

Two roles:
1. Register ``TurboQuantConfig`` so vLLM recognises ``quantization_config``
   in config.json (backward compat for old checkpoints).
2. Patch ``DefaultModelLoader.get_all_weights`` to detect ``tq_config.json``
   and decompress TQ3 packed weights on load, with per-layer GPU compression
   to keep peak VRAM at ~1 layer bf16 + all previously compressed layers.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from torch import nn

logger = logging.getLogger(__name__)


def _lazy_import_vllm():
    """Import vLLM components lazily to avoid import errors when vLLM isn't installed."""
    from vllm.model_executor.layers.linear import LinearBase
    from vllm.model_executor.layers.quantization.base_config import (
        QuantizationConfig,
        QuantizeMethodBase,
    )

    return LinearBase, QuantizationConfig, QuantizeMethodBase


# Deferred class creation — only built when register() is called from the plugin
_registered = False


def register():
    """Register TurboQuant as a vLLM quantization method. Called from the plugin."""
    global _registered
    if _registered:
        return
    _registered = True

    try:
        from vllm.model_executor.layers.quantization import register_quantization_config
    except ImportError:
        logger.debug("vLLM not installed, skipping TurboQuant quant config registration")
        return

    LinearBase, QuantizationConfig, QuantizeMethodBase = _lazy_import_vllm()

    @register_quantization_config("turboquant")
    class TurboQuantConfig(QuantizationConfig):
        """Config for TurboQuant weight quantization (TQ3/TQ4)."""

        def __init__(self, bits: int = 3, group_size: int = 128, sensitive_bits: int | None = None):
            super().__init__()
            self.bits = bits
            self.group_size = group_size
            self.sensitive_bits = sensitive_bits

        def __repr__(self) -> str:
            return (
                f"TurboQuantConfig(bits={self.bits}, group_size={self.group_size}, "
                f"sensitive_bits={self.sensitive_bits})"
            )

        def get_name(self) -> str:
            return "turboquant"

        def get_supported_act_dtypes(self) -> list[torch.dtype]:
            return [torch.float16, torch.bfloat16]

        @classmethod
        def get_min_capability(cls) -> int:
            return 70  # Volta and newer

        @staticmethod
        def get_config_filenames() -> list[str]:
            return ["tq_config.json", "quantize_config.json"]

        @classmethod
        def from_config(cls, config: dict[str, Any]) -> "TurboQuantConfig":
            bits = cls.get_from_keys_or(config, ["bits"], 3)
            group_size = cls.get_from_keys_or(config, ["group_size"], 128)
            sensitive_bits = cls.get_from_keys_or(config, ["sensitive_bits"], None)
            return cls(bits=bits, group_size=group_size, sensitive_bits=sensitive_bits)

        def get_quant_method(
            self, layer: nn.Module, prefix: str
        ) -> "QuantizeMethodBase | None":
            # Native TQ3 checkpoints are decompressed to bf16 during
            # weight loading (see _patch_weight_name_remapping).  All
            # layers receive standard bf16 weights via unquantized
            # methods.  The runtime plugin (enable_weight_quantization)
            # re-compresses on GPU after loading.
            if isinstance(layer, LinearBase):
                from vllm.model_executor.layers.linear import UnquantizedLinearMethod
                return UnquantizedLinearMethod()
            try:
                from vllm.model_executor.layers.fused_moe import FusedMoE
                from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
                    UnquantizedFusedMoEMethod,
                )
                if isinstance(layer, FusedMoE):
                    return UnquantizedFusedMoEMethod(layer.moe_config)
            except ImportError:
                pass
            return None

    _patch_weight_name_remapping()

    logger.info("TurboQuant quantization config registered with vLLM")


def _patch_weight_name_remapping():
    """Monkey-patch vLLM's weight iterator to decompress TQ3 weights on load.

    When a native TQ3 checkpoint is loaded, the checkpoint contains
    ``.tq_packed`` / ``.tq_norms`` tensor pairs instead of standard
    ``.weight`` tensors.  This patch collects each pair, decompresses
    to bf16 on CPU, and yields the result with the original weight name.
    vLLM's model-specific weight loaders (stacked qkv, fused gate_up,
    expert assembly) then work unchanged.

    After loading, the runtime plugin re-compresses weights on GPU via
    ``enable_weight_quantization`` — so the bf16 is transient.
    """
    try:
        from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
    except ImportError:
        return

    from turboquant_vllm.weight_quant import Compressed3D

    _original_get_all_weights = DefaultModelLoader.get_all_weights

    def _decompress_get_all_weights(self, model_config, model):
        """Decompress TQ3 → bf16 per tensor, compress each layer on GPU
        immediately after its weights land.

        This keeps peak GPU memory at ~1 layer bf16 + all previously
        compressed layers, instead of the full model in bf16.  Critical
        for fitting 309 GB models on 2×H200 (282 GB).

        Flow per layer:
          1. Generator yields decompressed bf16 tensors for layer N
          2. vLLM's weight_loader places each on GPU
          3. Generator detects layer N+1 starting → compresses layer N
             on GPU via _replace_linear_layers, freeing bf16 memory
          4. Repeat for layer N+1
        """
        import os as _os

        # Resolve tq_config.json — model_config.model may be a HF repo
        # ID (e.g. "varjosoft/GLM-5.1-Open-TQ3") or a local path.
        tq_config_path = _os.path.join(model_config.model, "tq_config.json")
        if not _os.path.isfile(tq_config_path):
            try:
                from huggingface_hub import hf_hub_download
                tq_config_path = hf_hub_download(
                    model_config.model, "tq_config.json",
                    revision=model_config.revision,
                )
            except Exception:
                yield from _original_get_all_weights(self, model_config, model)
                return

        import json as _json
        import re

        with open(tq_config_path) as f:
            tq_cfg = _json.load(f)
        bits = tq_cfg.get("bits", 3)
        group_size = tq_cfg.get("group_size", 128)
        logger.info(
            "TQ3 native checkpoint detected (bits=%d, group_size=%d), "
            "streaming decompress + per-layer compression",
            bits, group_size,
        )

        # Import compression function for per-layer GPU compression
        from turboquant_vllm.weight_quant import _replace_linear_layers

        # Track which layer is currently being loaded
        _layer_re = re.compile(r"model\.layers\.(\d+)\.")
        prev_layer_idx = None
        compressed_layers = 0

        def _compress_prev_layer(layer_idx):
            """Compress a single decoder layer's weights on GPU."""
            nonlocal compressed_layers
            layers = getattr(getattr(model, "model", None), "layers", None)
            if layers is None or layer_idx >= len(layers):
                return
            count = _replace_linear_layers(
                layers[layer_idx], bits=bits, group_size=group_size,
            )
            if count > 0:
                compressed_layers += count
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if compressed_layers % 50 == 0:
                    mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
                    logger.info(
                        "  Compressed layer %d (%d total, %.1f GB GPU)",
                        layer_idx, compressed_layers, mem,
                    )

        pending_packed = {}
        pending_norms = {}

        for name, tensor in _original_get_all_weights(self, model_config, model):
            # Detect layer boundary
            m = _layer_re.search(name)
            cur_layer_idx = int(m.group(1)) if m else None

            if cur_layer_idx is not None and cur_layer_idx != prev_layer_idx:
                if prev_layer_idx is not None:
                    _compress_prev_layer(prev_layer_idx)
                prev_layer_idx = cur_layer_idx

            # Handle TQ3 packed tensors — collect pairs, decompress when complete
            if name.endswith(".weight.tq_packed"):
                base = name[: -len(".tq_packed")]
                pending_packed[base] = tensor
            elif name.endswith(".weight.tq_norms"):
                base = name[: -len(".tq_norms")]
                pending_norms[base] = tensor
            else:
                yield name, tensor
                continue

            if base in pending_packed and base in pending_norms:
                packed = pending_packed.pop(base)
                norms = pending_norms.pop(base)

                n_rows = norms.shape[0]
                n_groups = norms.shape[1]
                in_dim = n_groups * group_size
                comp = Compressed3D.from_packed(
                    packed, norms, (1, n_rows, in_dim),
                    torch.bfloat16, bits, group_size,
                )
                w = comp.decompress().squeeze(0)
                yield base, w
                del packed, norms, comp, w

        # Compress the last layer
        if prev_layer_idx is not None:
            _compress_prev_layer(prev_layer_idx)

        if compressed_layers > 0:
            mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
            logger.info(
                "Streaming compression complete: %d layers compressed, %.1f GB GPU",
                compressed_layers, mem,
            )

        for base in pending_packed:
            logger.warning("Orphaned .tq_packed without .tq_norms: %s", base)
        for base in pending_norms:
            logger.warning("Orphaned .tq_norms without .tq_packed: %s", base)

    DefaultModelLoader.get_all_weights = _decompress_get_all_weights
