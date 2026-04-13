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
            if isinstance(layer, LinearBase):
                return TurboQuantOnlineLinearMethod(self.bits, self.group_size)
            try:
                from vllm.model_executor.layers.fused_moe import FusedMoE

                if isinstance(layer, FusedMoE):
                    return TurboQuantOnlineMoEMethod(
                        self.bits, self.group_size, layer.moe_config,
                    )
            except ImportError:
                pass
            return None

    # ── Online quant methods (meta-device init, per-layer compression) ──

    class TurboQuantOnlineLinearMethod(QuantizeMethodBase):
        """Load bf16 on meta device, compress to TQ3 per-layer after loading.

        Follows the same pattern as vLLM's online FP8 quantization:
        allocate on meta device during model init (zero GPU memory),
        materialize + compress one layer at a time during weight loading.
        """

        uses_meta_device: bool = True

        def __init__(self, bits: int, group_size: int):
            self.bits = bits
            self.group_size = group_size

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
            from vllm.model_executor.model_loader.reload.layerwise import (
                initialize_online_processing,
            )
            from vllm.model_executor.parameter import ModelWeightParameter

            output_size_per_partition = sum(output_partition_sizes)
            weight_loader = extra_weight_attrs.get("weight_loader")

            weight = ModelWeightParameter(
                data=torch.empty(
                    output_size_per_partition,
                    input_size_per_partition,
                    device="meta",
                    dtype=params_dtype,
                ),
                input_dim=1,
                output_dim=0,
                weight_loader=weight_loader,
            )
            layer.register_parameter("weight", weight)

            # Store config for process_weights_after_loading
            layer.tq_bits = self.bits
            layer.tq_group_size = self.group_size

            initialize_online_processing(layer)

        def process_weights_after_loading(self, layer: nn.Module) -> None:
            """Compress materialized bf16 weight → TQ3 packed format on GPU."""
            from turboquant_vllm.weight_quant import (
                _ensure_triton_backends,
                _get_cuda_module,
                _get_quantizer,
                pack_indices,
                padded_size,
            )

            weight = layer.weight.data
            bits = layer.tq_bits
            group_size = layer.tq_group_size
            device = str(weight.device)

            out_dim, in_dim = weight.shape
            padded_in, n_groups = padded_size(in_dim, group_size)

            if padded_in > in_dim:
                padded = torch.zeros(
                    out_dim, padded_in, dtype=weight.dtype, device=weight.device,
                )
                padded[:, :in_dim] = weight
            else:
                padded = weight

            grouped = padded.reshape(-1, group_size)
            quantizer = _get_quantizer(group_size, bits, device)
            indices, norms_raw = quantizer.quantize(grouped, norm_correction=True)
            packed = pack_indices(indices, bits)
            norms = norms_raw.reshape(out_dim, n_groups)

            # Delete bf16 weight, store packed data as buffers
            delattr(layer, "weight")
            layer.register_buffer("tq_packed_weight", packed)
            layer.register_buffer("tq_norms", norms)
            layer.register_buffer("tq_signs1", quantizer.signs1)
            layer.register_buffer("tq_signs2", quantizer.signs2)
            layer.register_buffer("tq_centroids", quantizer.centroids)
            layer.tq_in_features = in_dim
            layer.tq_out_features = out_dim
            layer.tq_padded_in = padded_in
            layer.tq_n_groups = n_groups

            # Pre-cache Triton rotation matrix (must happen before graph capture)
            _ensure_triton_backends()
            _get_cuda_module()

            del weight, padded, grouped, indices, norms_raw

        def apply(
            self, layer: nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None,
        ) -> torch.Tensor:
            from turboquant_vllm.weight_quant import (
                _tq_fused_gemm_fn,
                _tq_fwht_input_fn,
                _triton_available,
            )

            if _triton_available:
                args = (
                    x, layer.tq_packed_weight, layer.tq_norms,
                    layer.tq_signs1, layer.tq_signs2, layer.tq_centroids,
                )
                kwargs = dict(
                    group_size=self.group_size, bits=self.bits, bias=bias,
                )
                primary = (
                    _tq_fwht_input_fn
                    if layer.tq_out_features >= 4096
                    else _tq_fused_gemm_fn
                )
                fallback = (
                    _tq_fused_gemm_fn
                    if layer.tq_out_features >= 4096
                    else _tq_fwht_input_fn
                )
                try:
                    return primary(*args, **kwargs)
                except (ValueError, RuntimeError):
                    return fallback(*args, **kwargs)

            # CPU/CUDA fallback — dequantize then matmul
            from turboquant_vllm.weight_quant import _get_quantizer, unpack_indices

            indices = unpack_indices(
                layer.tq_packed_weight, self.bits, self.group_size,
            )
            norms_flat = layer.tq_norms.reshape(-1)
            quantizer = _get_quantizer(
                self.group_size, self.bits, str(x.device),
            )
            w_groups = quantizer.dequantize(indices, norms_flat)
            w_deq = w_groups.reshape(
                layer.tq_out_features, layer.tq_padded_in,
            )[:, : layer.tq_in_features].to(x.dtype)
            output = torch.matmul(x, w_deq.t())
            if bias is not None:
                output = output + bias
            return output

    class TurboQuantOnlineMoEMethod:
        """Online TQ3 compression for FusedMoE layers.

        Same meta-device pattern: allocate bf16 on meta, materialize per-layer,
        then compress expert weights to TQ3 after loading.
        """

        uses_meta_device: bool = True

        def __init__(self, bits: int, group_size: int, moe_config: Any):
            self.bits = bits
            self.group_size = group_size
            self.moe_config = moe_config

        def create_weights(
            self, layer: nn.Module, **kwargs,
        ):
            """Delegate to UnquantizedFusedMoEMethod but on meta device."""
            from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
                UnquantizedFusedMoEMethod,
            )
            from vllm.model_executor.model_loader.reload.layerwise import (
                initialize_online_processing,
            )

            # Use the standard unquantized method to create bf16 weight structure
            unquant = UnquantizedFusedMoEMethod(self.moe_config)
            unquant.create_weights(layer, **kwargs)

            # Move all created parameters to meta device
            for name, param in list(layer.named_parameters()):
                if param.device != torch.device("meta"):
                    meta_param = torch.nn.Parameter(
                        torch.empty_like(param, device="meta"),
                        requires_grad=False,
                    )
                    # Preserve weight_loader if present
                    if hasattr(param, "weight_loader"):
                        meta_param.weight_loader = param.weight_loader
                    # Preserve sharding metadata
                    for attr in ("output_dim", "input_dim", "packed_dim", "packed_factor"):
                        if hasattr(param, attr):
                            setattr(meta_param, attr, getattr(param, attr))
                    delattr(layer, name)
                    layer.register_parameter(name, meta_param)

            layer.tq_bits = self.bits
            layer.tq_group_size = self.group_size

            initialize_online_processing(layer)

        def process_weights_after_loading(self, layer: nn.Module) -> None:
            """Compress FusedMoE expert weights to TQ3 after loading."""
            from turboquant_vllm.weight_quant import _replace_linear_layers

            count = _replace_linear_layers(
                layer, bits=layer.tq_bits, group_size=layer.tq_group_size,
            )
            if count > 0:
                logger.info(
                    "TQ%d compressed %d MoE sub-layers", layer.tq_bits, count,
                )

        def apply(
            self,
            layer: nn.Module,
            x: torch.Tensor,
            router_logits: torch.Tensor,
            top_k: int,
            renormalize: bool,
            use_grouped_topk: bool = False,
            topk_group: int | None = None,
            num_expert_group: int | None = None,
            custom_routing_function: Any | None = None,
            scoring_func: str = "softmax",
            e_score_correction_bias: torch.Tensor | None = None,
            activation: str = "silu",
            apply_router_weight_on_input: bool = False,
        ) -> torch.Tensor:
            """After compression, expert weights are TurboQuantWrappers.

            Fall back to standard FusedMoE forward — the compressed weights
            are handled by TurboQuantWrapper.forward() inside the MoE dispatch.
            """
            from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
                UnquantizedFusedMoEMethod,
            )

            # Delegate to unquantized MoE method — works because the expert
            # weight tensors are now TurboQuantWrappers that decompress on
            # the fly during the FusedMoE kernel's matmul calls.
            unquant = UnquantizedFusedMoEMethod(self.moe_config)
            return unquant.apply(
                layer, x, router_logits, top_k, renormalize,
                use_grouped_topk=use_grouped_topk,
                topk_group=topk_group,
                num_expert_group=num_expert_group,
                custom_routing_function=custom_routing_function,
                scoring_func=scoring_func,
                e_score_correction_bias=e_score_correction_bias,
                activation=activation,
                apply_router_weight_on_input=apply_router_weight_on_input,
            )

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
        """Decompress TQ3 → bf16 per tensor for vLLM's weight loader.

        The checkpoint contains ``.tq_packed`` / ``.tq_norms`` tensor pairs.
        This generator collects each pair, decompresses to bf16 on CPU, and
        yields with the original ``.weight`` name so vLLM's model-specific
        weight loaders (stacked qkv, fused gate_up, expert assembly) work
        unchanged.

        GPU compression happens later via ``process_weights_after_loading``
        in ``TurboQuantOnlineLinearMethod`` / ``TurboQuantOnlineMoEMethod``.
        """
        import os as _os

        tq_config_path = _os.path.join(model_config.model, "tq_config.json")
        if not _os.path.isfile(tq_config_path):
            try:
                from huggingface_hub import hf_hub_download

                revision = getattr(model_config, "revision", None)
                tq_config_path = hf_hub_download(
                    model_config.model, "tq_config.json", revision=revision,
                )
            except Exception as e:
                logger.info(
                    "No tq_config.json for %s (%s), passing through",
                    model_config.model, e,
                )
                yield from _original_get_all_weights(self, model_config, model)
                return

        import json as _json

        with open(tq_config_path) as f:
            tq_cfg = _json.load(f)
        bits = tq_cfg.get("bits", 3)
        group_size = tq_cfg.get("group_size", 128)
        logger.info(
            "TQ3 native checkpoint (bits=%d, group_size=%d): "
            "decompressing to bf16 for online quantization",
            bits, group_size,
        )

        pending_packed: dict[str, torch.Tensor] = {}
        pending_norms: dict[str, torch.Tensor] = {}
        decompressed = 0

        for name, tensor in _original_get_all_weights(self, model_config, model):
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
                decompressed += 1
                if decompressed % 50 == 0:
                    logger.info("  Decompressed %d tensors", decompressed)
                yield base, w
                del packed, norms, comp, w

        if decompressed > 0:
            logger.info("TQ3 decompression complete: %d tensors", decompressed)

        for base in pending_packed:
            logger.warning("Orphaned .tq_packed without .tq_norms: %s", base)
        for base in pending_norms:
            logger.warning("Orphaned .tq_norms without .tq_packed: %s", base)

    DefaultModelLoader.get_all_weights = _decompress_get_all_weights
    logger.info("TQ3 decompress-on-load hook installed on DefaultModelLoader.get_all_weights")
