"""TurboQuant quantization config for native vLLM integration.

Registers as a vLLM quantization method so TQ3/TQ4 checkpoints can be loaded
with tensor parallelism. Uses the same dequant kernels as TurboQuantWrapper.

Usage:
    # Checkpoint must have quantization_config in config.json:
    # {"quantization_config": {"quant_method": "turboquant", "bits": 3, "group_size": 128}}
    #
    # Then just:
    vllm serve ./my-tq3-checkpoint --quantization turboquant
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    pass


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
        # Detect TQ3 checkpoint via tq_config.json (not quantization_config
        # in config.json — that would trigger vLLM's quant code path and
        # break CUDA graph capture).
        import os as _os

        tq_config_path = _os.path.join(model_config.model, "tq_config.json")
        if not _os.path.isfile(tq_config_path):
            yield from _original_get_all_weights(self, model_config, model)
            return

        import json as _json

        with open(tq_config_path) as f:
            tq_cfg = _json.load(f)
        bits = tq_cfg.get("bits", 3)
        group_size = tq_cfg.get("group_size", 128)
        logger.info("TQ3 native checkpoint detected (bits=%d, group_size=%d), decompressing on load", bits, group_size)

        # Collect packed/norms pairs, decompress, yield as bf16.
        # Tensors arrive in checkpoint order — packed and norms for the
        # same weight are adjacent (both in the same shard, consecutive).
        pending_packed = {}  # base_name → packed tensor
        pending_norms = {}   # base_name → norms tensor

        for name, tensor in _original_get_all_weights(self, model_config, model):
            if name.endswith(".weight.tq_packed"):
                base = name[: -len(".tq_packed")]  # e.g. "layers.0.q_proj.weight"
                pending_packed[base] = tensor
            elif name.endswith(".weight.tq_norms"):
                base = name[: -len(".tq_norms")]
                pending_norms[base] = tensor
            else:
                # Regular tensor — yield as-is
                yield name, tensor
                continue

            # Check if we have both packed + norms for this weight
            base_p = name[: -len(".tq_packed")] if name.endswith(".tq_packed") else None
            base_n = name[: -len(".tq_norms")] if name.endswith(".tq_norms") else None
            base = base_p or base_n

            if base in pending_packed and base in pending_norms:
                packed = pending_packed.pop(base)
                norms = pending_norms.pop(base)

                # Decompress on CPU to bf16 via Compressed3D (works for
                # 2D by treating as (1, n_rows, in_dim)).
                n_rows = norms.shape[0]
                n_groups = norms.shape[1]
                in_dim = n_groups * group_size
                comp = Compressed3D.from_packed(
                    packed, norms, (1, n_rows, in_dim),
                    torch.bfloat16, bits, group_size,
                )
                w = comp.decompress().squeeze(0)  # (1, n_rows, in_dim) → (n_rows, in_dim)

                if w.isnan().any() or w.isinf().any():
                    logger.error("TQ3 decompress produced NaN/Inf for %s shape=%s", base, w.shape)
                logger.debug("TQ3 decompress: %s → %s (max=%.3f)", base, tuple(w.shape), w.abs().max().item())

                yield base, w
                del packed, norms, comp, w

        # Flush any orphaned packed/norms (shouldn't happen with valid checkpoints)
        for base in pending_packed:
            logger.warning("Orphaned .tq_packed without .tq_norms: %s", base)
        for base in pending_norms:
            logger.warning("Orphaned .tq_norms without .tq_packed: %s", base)

    DefaultModelLoader.get_all_weights = _decompress_get_all_weights
