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
from typing import TYPE_CHECKING, Any, Union

import torch
from torch import nn

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    pass


def _lazy_import_vllm():
    """Import vLLM components lazily to avoid import errors when vLLM isn't installed."""
    from vllm.model_executor.layers.linear import LinearBase, LinearMethodBase
    from vllm.model_executor.layers.quantization.base_config import (
        QuantizationConfig,
        QuantizeMethodBase,
    )
    from vllm.model_executor.parameter import (
        GroupQuantScaleParameter,
        ModelWeightParameter,
        PackedvLLMParameter,
    )

    return (
        LinearBase,
        LinearMethodBase,
        QuantizationConfig,
        QuantizeMethodBase,
        GroupQuantScaleParameter,
        ModelWeightParameter,
        PackedvLLMParameter,
    )


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

    (
        LinearBase,
        LinearMethodBase,
        QuantizationConfig,
        QuantizeMethodBase,
        GroupQuantScaleParameter,
        ModelWeightParameter,
        PackedvLLMParameter,
    ) = _lazy_import_vllm()

    from turboquant_vllm.weight_quant import (
        _SKIP_PATTERNS,
        select_bits,
        _get_quantizer,
        unpack_indices,
    )

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
        ) -> Union["LinearMethodBase", "QuantizeMethodBase"] | None:
            if isinstance(layer, LinearBase):
                # Skip layers that shouldn't be quantized
                if any(p in prefix.lower() for p in _SKIP_PATTERNS):
                    from vllm.model_executor.layers.linear import UnquantizedLinearMethod

                    return UnquantizedLinearMethod()
                # Determine bits for this layer
                layer_bits = select_bits(prefix, self.bits, self.sensitive_bits)
                return TurboQuantLinearMethod(self, layer_bits)

            # FusedMoE expert weights
            try:
                from vllm.model_executor.layers.fused_moe import FusedMoE

                if isinstance(layer, FusedMoE):
                    return TurboQuantFusedMoELoadMethod(
                        layer.moe_config, self.bits, self.group_size
                    )
            except ImportError:
                pass

            return None

    class TurboQuantLinearMethod(LinearMethodBase):
        """Linear method that loads TQ3/TQ4 packed weights and dequants on forward."""

        def __init__(self, quant_config: "TurboQuantConfig", bits: int):
            self.quant_config = quant_config
            self.bits = bits
            self.group_size = quant_config.group_size

        def create_weights(
            self,
            layer: nn.Module,
            input_size_per_partition: int,
            output_partition_sizes: list[int],
            input_size: int,
            output_size: int,
            params_dtype: torch.dtype,
            **extra_weight_attrs,
        ):
            output_size_per_partition = sum(output_partition_sizes)
            weight_loader = extra_weight_attrs.get("weight_loader")

            from fractions import Fraction
            from turboquant_vllm.weight_quant import packed_group_bytes as _pgb

            from turboquant_vllm.weight_quant import padded_size
            padded_in, n_groups = padded_size(input_size_per_partition, self.group_size)
            packed_cols = n_groups * _pgb(self.bits, self.group_size)
            pack_ratio = Fraction(padded_in, packed_cols)

            tq_packed = PackedvLLMParameter(
                data=torch.empty(
                    output_size_per_partition,
                    packed_cols,
                    dtype=torch.uint8,
                ),
                input_dim=1,
                output_dim=0,
                packed_dim=1,
                packed_factor=pack_ratio,
                weight_loader=weight_loader,
            )

            # Norms: (out_features, n_groups) as float32
            # n_groups = padded_in / group_size, so pack ratio is group_size
            tq_norms = PackedvLLMParameter(
                data=torch.empty(
                    output_size_per_partition,
                    n_groups,
                    dtype=torch.float32,
                ),
                input_dim=1,
                output_dim=0,
                packed_dim=1,
                packed_factor=self.group_size,
                weight_loader=weight_loader,
            )

            layer.register_parameter("tq_packed", tq_packed)
            layer.register_parameter("tq_norms", tq_norms)
            layer._tq_bits = self.bits
            layer._tq_group_size = self.group_size
            layer._tq_in_features = input_size_per_partition
            layer._tq_out_features = output_size_per_partition

        def process_weights_after_loading(self, layer: nn.Module) -> None:
            # Ensure weights are contiguous and on the right device
            layer.tq_packed = nn.Parameter(layer.tq_packed.data.contiguous(), requires_grad=False)
            layer.tq_norms = nn.Parameter(layer.tq_norms.data.contiguous(), requires_grad=False)

            # Eagerly populate the module-level rotation matrix cache so
            # the first forward (potentially the warmup pass before CUDA
            # graph capture) does not hit a cache miss and run a
            # butterfly WHT inside the custom_op body. Mirrors the
            # equivalent eager call in TurboQuantWrapper.__init__; see
            # turboquant_vllm.triton_ops._rotation_matrix_cache for the
            # capture-safety invariant.
            try:
                from turboquant_vllm.triton_ops import _get_cached_rotation_matrix

                quantizer = _get_quantizer(self.group_size, self.bits, str(layer.tq_packed.device))
                _get_cached_rotation_matrix(quantizer.signs1, quantizer.signs2, self.group_size)
            except ImportError:
                pass

        def apply(
            self,
            layer: nn.Module,
            x: torch.Tensor,
            bias: torch.Tensor | None = None,
        ) -> torch.Tensor:
            bits = layer._tq_bits
            group_size = layer._tq_group_size
            in_features = layer._tq_in_features
            out_features = layer._tq_out_features

            # Try Triton fused kernels first (fastest)
            try:
                from turboquant_vllm.triton_ops import tq_fused_gemm, tq_fwht_input_gemm

                quantizer = _get_quantizer(group_size, bits, str(x.device))

                args = (
                    x,
                    layer.tq_packed.data,
                    layer.tq_norms.data,
                    quantizer.signs1,
                    quantizer.signs2,
                    quantizer.centroids,
                )
                kwargs = dict(group_size=group_size, bits=bits, bias=bias)

                primary = tq_fwht_input_gemm if out_features >= 4096 else tq_fused_gemm
                fallback = tq_fused_gemm if out_features >= 4096 else tq_fwht_input_gemm
                try:
                    return primary(*args, **kwargs)
                except (ValueError, RuntimeError):
                    try:
                        return fallback(*args, **kwargs)
                    except (ValueError, RuntimeError):
                        pass
            except ImportError:
                pass

            # Fallback: dequant + matmul
            quantizer = _get_quantizer(group_size, bits, str(x.device))
            indices = unpack_indices(layer.tq_packed.data, bits, group_size)
            norms_flat = layer.tq_norms.data.reshape(-1)
            w_groups = quantizer.dequantize(indices, norms_flat)
            padded_in, _ = padded_size(in_features, group_size)
            w_deq = w_groups.reshape(out_features, padded_in)[:, :in_features]
            w_deq = w_deq.to(x.dtype)

            output = torch.matmul(x, w_deq.t())
            del w_deq
            if bias is not None:
                output = output + bias
            return output

    # ------------------------------------------------------------------
    # FusedMoE: load TQ3-packed expert weights from native checkpoint
    # ------------------------------------------------------------------
    from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
        FusedMoEMethodBase,
    )
    from vllm.model_executor.utils import set_weight_attrs

    # Imports used by both linear and MoE loaders, resolved once.
    from turboquant_vllm.weight_quant import Compressed3D, packed_group_bytes, padded_size
    from vllm.model_executor.layers.fused_moe import fused_experts

    class TurboQuantFusedMoELoadMethod(FusedMoEMethodBase):
        """Load TQ3-packed MoE expert weights from a native TQ3 checkpoint.

        Registers packed uint8 + float32 norms parameters.  On first
        forward, builds cached ``Compressed3D`` objects and allocates a
        shared scratch pool.  ``apply()`` decompresses into the scratch
        and delegates to ``fused_experts`` — same kernel as unquantized.
        """

        def __init__(self, moe_config, bits: int, group_size: int):
            import threading

            super().__init__(moe_config)
            self.bits = bits
            self.group_size = group_size
            self._scratch_pool = None
            self._w13_comp = None
            self._w2_comp = None
            self._init_lock = threading.Lock()

        def create_weights(
            self,
            layer: nn.Module,
            num_experts: int,
            hidden_size: int,
            intermediate_size_per_partition: int,
            params_dtype: torch.dtype,
            **extra_weight_attrs,
        ):
            bits = self.bits
            gs = self.group_size

            w13_out = 2 * intermediate_size_per_partition
            w13_in = hidden_size
            w2_out = hidden_size
            w2_in = intermediate_size_per_partition

            def _packed_shape(n_exp, out_dim, in_dim):
                _, n_groups = padded_size(in_dim, gs)
                return (n_exp * out_dim, n_groups * packed_group_bytes(bits, gs))

            def _norms_shape(n_exp, out_dim, in_dim):
                _, n_groups = padded_size(in_dim, gs)
                return (n_exp * out_dim, n_groups)

            weight_loader = extra_weight_attrs.get("weight_loader")

            w13_packed = nn.Parameter(
                torch.empty(*_packed_shape(num_experts, w13_out, w13_in), dtype=torch.uint8),
                requires_grad=False,
            )
            w2_packed = nn.Parameter(
                torch.empty(*_packed_shape(num_experts, w2_out, w2_in), dtype=torch.uint8),
                requires_grad=False,
            )
            w13_norms = nn.Parameter(
                torch.empty(*_norms_shape(num_experts, w13_out, w13_in), dtype=torch.float32),
                requires_grad=False,
            )
            w2_norms = nn.Parameter(
                torch.empty(*_norms_shape(num_experts, w2_out, w2_in), dtype=torch.float32),
                requires_grad=False,
            )

            layer.register_parameter("w13_tq_packed", w13_packed)
            layer.register_parameter("w13_tq_norms", w13_norms)
            layer.register_parameter("w2_tq_packed", w2_packed)
            layer.register_parameter("w2_tq_norms", w2_norms)
            for p in (w13_packed, w13_norms, w2_packed, w2_norms):
                set_weight_attrs(p, {"weight_loader": weight_loader})

            # Store shapes for scratch allocation; use self for bits/gs.
            self._w13_shape = (num_experts, w13_out, w13_in)
            self._w2_shape = (num_experts, w2_out, w2_in)
            self._params_dtype = params_dtype

            # Placeholder bf16 weights — allocated as zero-size to avoid
            # wasting GPU memory.  Re-pointed at scratch on first forward.
            w13_weight = nn.Parameter(torch.empty(0, dtype=params_dtype), requires_grad=False)
            w2_weight = nn.Parameter(torch.empty(0, dtype=params_dtype), requires_grad=False)
            layer.register_parameter("w13_weight", w13_weight)
            layer.register_parameter("w2_weight", w2_weight)
            set_weight_attrs(w13_weight, extra_weight_attrs)
            set_weight_attrs(w2_weight, extra_weight_attrs)

        def get_fused_moe_quant_config(self, layer: nn.Module):
            return None

        def _ensure_ready(self, layer: nn.Module):
            """Lazily allocate scratch and build Compressed3D on first forward."""
            if self._scratch_pool is not None:
                return
            with self._init_lock:
                if self._scratch_pool is not None:
                    return
                self._init_scratch(layer)

        def _init_scratch(self, layer: nn.Module):
            device = layer.w13_tq_packed.device
            dtype = self._params_dtype
            w13_s, w2_s = self._w13_shape, self._w2_shape

            self._scratch_pool = {
                "w13": torch.empty(w13_s, dtype=dtype, device=device),
                "w2": torch.empty(w2_s, dtype=dtype, device=device),
                "w13_fp32": torch.empty(w13_s, dtype=torch.float32, device=device),
                "w2_fp32": torch.empty(w2_s, dtype=torch.float32, device=device),
            }
            layer.w13_weight.data = self._scratch_pool["w13"]
            layer.w2_weight.data = self._scratch_pool["w2"]

            # Cache Compressed3D objects — packed data never changes.
            self._w13_comp = Compressed3D.from_packed(
                layer.w13_tq_packed.data, layer.w13_tq_norms.data,
                w13_s, dtype, self.bits, self.group_size,
            )
            self._w2_comp = Compressed3D.from_packed(
                layer.w2_tq_packed.data, layer.w2_tq_norms.data,
                w2_s, dtype, self.bits, self.group_size,
            )

        def apply(
            self,
            layer: nn.Module,
            x: torch.Tensor,
            topk_weights: torch.Tensor,
            topk_ids: torch.Tensor,
            shared_experts_input: torch.Tensor | None = None,
        ):
            self._ensure_ready(layer)
            pool = self._scratch_pool
            self._w13_comp.decompress_into(pool["w13"], fp32_scratch=pool["w13_fp32"])
            self._w2_comp.decompress_into(pool["w2"], fp32_scratch=pool["w2_fp32"])

            return fused_experts(
                x,
                pool["w13"],
                pool["w2"],
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                inplace=not self.moe.disable_inplace,
            )

    # Patch the weight loader to remap old-style checkpoint names
    # Old: model.layers.0.self_attn.q_proj.weight.tq_packed
    # New: model.layers.0.self_attn.q_proj.tq_packed
    _patch_weight_name_remapping()

    logger.info("TurboQuant quantization config registered with vLLM")


def _patch_weight_name_remapping():
    """Monkey-patch vLLM's weight iterator to remap old TQ checkpoint names."""
    try:
        from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
    except ImportError:
        return

    _original_get_all_weights = DefaultModelLoader.get_all_weights

    def _remapped_get_all_weights(self, model_config, model):
        for name, tensor in _original_get_all_weights(self, model_config, model):
            # Remap checkpoint .weight.tq_* names to registered param names.
            # 2D linears: q_proj.weight.tq_packed → q_proj.tq_packed
            # 3D MoE:     w13_weight.tq_packed → w13_tq_packed
            for suffix in (".tq_packed", ".tq_norms"):
                old = ".weight" + suffix
                if old in name:
                    # MoE experts use underscore join (w13_tq_packed)
                    if "w13_weight" + suffix in name or "w2_weight" + suffix in name:
                        name = name.replace("_weight" + suffix, "_" + suffix[1:])
                    else:
                        name = name.replace(old, suffix)
                    break
            yield name, tensor

    DefaultModelLoader.get_all_weights = _remapped_get_all_weights
