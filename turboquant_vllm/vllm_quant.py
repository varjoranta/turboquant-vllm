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

                if isinstance(layer, FusedMoE) and TurboQuantOnlineMoEMethod is not None:
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

            initialize_online_processing(layer)

        def process_weights_after_loading(self, layer: nn.Module) -> None:
            from turboquant_vllm.weight_quant import (
                _ensure_triton_backends,
                _get_cuda_module,
                _get_quantizer,
                pack_indices,
                padded_size,
            )

            weight = layer.weight.data
            bits = self.bits
            group_size = self.group_size
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

            delattr(layer, "weight")
            layer.register_buffer("tq_packed_weight", packed)
            layer.register_buffer("tq_norms", norms)
            layer.register_buffer("tq_signs1", quantizer.signs1)
            layer.register_buffer("tq_signs2", quantizer.signs2)
            layer.register_buffer("tq_centroids", quantizer.centroids)
            layer.tq_in_features = in_dim
            layer.tq_out_features = out_dim
            layer.tq_padded_in = padded_in

            # Cache dispatch functions — avoids per-forward cross-module lookups
            _ensure_triton_backends()
            _get_cuda_module()
            from turboquant_vllm.weight_quant import (
                _tq_fused_gemm_fn,
                _tq_fwht_input_fn,
                _triton_available,
            )
            if _triton_available:
                layer._tq_primary_fn = (
                    _tq_fwht_input_fn if out_dim >= 4096 else _tq_fused_gemm_fn
                )
                layer._tq_fallback_fn = (
                    _tq_fused_gemm_fn if out_dim >= 4096 else _tq_fwht_input_fn
                )
            else:
                layer._tq_primary_fn = None

            del weight, padded, grouped, indices, norms_raw

        def apply(
            self, layer: nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None,
        ) -> torch.Tensor:
            if layer._tq_primary_fn is not None:
                args = (
                    x, layer.tq_packed_weight, layer.tq_norms,
                    layer.tq_signs1, layer.tq_signs2, layer.tq_centroids,
                )
                try:
                    return layer._tq_primary_fn(
                        *args, group_size=self.group_size, bits=self.bits, bias=bias,
                    )
                except (ValueError, RuntimeError):
                    return layer._tq_fallback_fn(
                        *args, group_size=self.group_size, bits=self.bits, bias=bias,
                    )

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

    # ── MoE online method ──
    # For FusedMoE layers, use the standard unquantized method but with
    # meta-device initialization. After weights load, compress expert
    # tensors to TQ3 via _replace_linear_layers.

    try:
        from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
            FusedMoEMethodBase,
        )
        from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
            UnquantizedFusedMoEMethod,
        )

        # Shared scratch pool across all FusedMoE layers — only one MoE
        # layer runs at a time during forward, so one set of bf16
        # decompression buffers is enough. Per-layer pools would consume
        # 78 × ~5 GB = 390 GB and defeat compression entirely.
        _shared_moe_scratch_pool = None

        class TurboQuantOnlineMoEMethod(FusedMoEMethodBase):
            """Online TQ3 for FusedMoE: meta-device init + per-layer compression."""

            uses_meta_device: bool = True

            def __init__(self, bits: int, group_size: int, moe_config: Any):
                super().__init__(moe_config)
                self.bits = bits
                self.group_size = group_size
                self._unquant = UnquantizedFusedMoEMethod(moe_config)

            def create_weights(self, layer: nn.Module, **kwargs):
                from vllm.model_executor.model_loader.reload.layerwise import (
                    initialize_online_processing,
                )

                self._unquant.create_weights(layer, **kwargs)

                # Move parameters to meta device
                for name, param in list(layer.named_parameters(recurse=False)):
                    if param.device != torch.device("meta"):
                        meta_param = torch.nn.Parameter(
                            torch.empty_like(param, device="meta"),
                            requires_grad=False,
                        )
                        if hasattr(param, "weight_loader"):
                            meta_param.weight_loader = param.weight_loader
                        for attr in ("output_dim", "input_dim", "packed_dim",
                                     "packed_factor", "is_metadata"):
                            if hasattr(param, attr):
                                setattr(meta_param, attr, getattr(param, attr))
                        delattr(layer, name)
                        layer.register_parameter(name, meta_param)

                initialize_online_processing(layer)

            def process_weights_after_loading(self, layer: nn.Module) -> None:
                nonlocal _shared_moe_scratch_pool

                from turboquant_vllm.moe_quant import (
                    TurboQuantFusedMoEMethod,
                    TurboQuantFusedMoEScratchPool,
                )
                from turboquant_vllm.weight_quant import _compress_3d_param

                bits = self.bits
                group_size = self.group_size

                w13 = getattr(layer, "w13_weight", None)
                w2 = getattr(layer, "w2_weight", None)
                if w13 is None or w2 is None or w13.dim() != 3 or w2.dim() != 3:
                    logger.warning(
                        "FusedMoE layer missing w13/w2 3D weights, skipping TQ3",
                    )
                    return

                _compress_3d_param(layer, "w13_weight", bits, group_size)
                _compress_3d_param(layer, "w2_weight", bits, group_size)

                w13_c = layer._tq_w13_weight
                w2_c = layer._tq_w2_weight

                if _shared_moe_scratch_pool is None:
                    _shared_moe_scratch_pool = TurboQuantFusedMoEScratchPool(
                        w13_c, w2_c,
                    )
                else:
                    _shared_moe_scratch_pool.assert_matches(w13_c, w2_c)

                # Point w13/w2 data at the shared scratch pool
                layer.w13_weight.data = _shared_moe_scratch_pool.w13
                layer.w2_weight.data = _shared_moe_scratch_pool.w2

                new_method = TurboQuantFusedMoEMethod(
                    layer.moe_config, w13_c, w2_c, _shared_moe_scratch_pool,
                )
                layer._replace_quant_method(new_method)

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            def get_fused_moe_quant_config(self, layer: nn.Module):
                return self._unquant.get_fused_moe_quant_config(layer)

            def apply(self, layer: nn.Module, x: torch.Tensor, **kwargs) -> torch.Tensor:
                return self._unquant.apply(layer, x, **kwargs)

    except ImportError:
        TurboQuantOnlineMoEMethod = None  # type: ignore[assignment,misc]

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
        """Decompress TQ3 → bf16, layer-by-layer to bound CPU memory.

        Two-phase approach to avoid OOM on large MoE models:

        Phase 1 — collect: iterate all checkpoint tensors. Non-TQ3
        tensors (layernorms, embeddings) are yielded immediately.
        TQ3 pairs (.tq_packed/.tq_norms) are collected as references
        (mmap'd from safetensors, near-zero RSS).

        Phase 2 — decompress by layer: sort collected TQ3 pairs by
        decoder layer index, decompress one layer at a time. Online
        processing completes each layer and frees its bf16 buffer
        before the next layer is decompressed. Peak RSS: ~1 layer
        bf16 instead of the full model.
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
        import re
        from collections import defaultdict

        with open(tq_config_path) as f:
            tq_cfg = _json.load(f)
        bits = tq_cfg.get("bits", 3)
        group_size = tq_cfg.get("group_size", 128)
        logger.info(
            "TQ3 native checkpoint (bits=%d, group_size=%d): "
            "two-phase layer-by-layer decompression",
            bits, group_size,
        )

        # ── Phase 1: collect TQ3 pairs, yield non-TQ3 immediately ──
        tq_pairs: dict[str, dict[str, torch.Tensor]] = {}

        for name, tensor in _original_get_all_weights(self, model_config, model):
            if name.endswith(".weight.tq_packed"):
                base = name[: -len(".tq_packed")]
                tq_pairs.setdefault(base, {})["packed"] = tensor
            elif name.endswith(".weight.tq_norms"):
                base = name[: -len(".tq_norms")]
                tq_pairs.setdefault(base, {})["norms"] = tensor
            else:
                yield name, tensor

        logger.info(
            "Phase 1 complete: collected %d TQ3 pairs", len(tq_pairs),
        )

        # ── Phase 2: group by layer, decompress in order ──
        layer_re = re.compile(r"model\.layers\.(\d+)\.")
        by_layer: dict[int, list[tuple[str, dict]]] = defaultdict(list)
        no_layer: list[tuple[str, dict]] = []

        for base, data in tq_pairs.items():
            m = layer_re.search(base)
            if m:
                by_layer[int(m.group(1))].append((base, data))
            else:
                no_layer.append((base, data))
        del tq_pairs  # free dict; data dicts now only referenced from by_layer/no_layer

        def _decompress_pair(base, data):
            packed = data.get("packed")
            norms = data.get("norms")
            if packed is None or norms is None:
                if packed is None:
                    logger.warning("Missing .tq_packed for %s", base)
                if norms is None:
                    logger.warning("Missing .tq_norms for %s", base)
                return None, None
            n_rows = norms.shape[0]
            n_groups = norms.shape[1]
            in_dim = n_groups * group_size
            comp = Compressed3D.from_packed(
                packed, norms, (1, n_rows, in_dim),
                torch.bfloat16, bits, group_size,
            )
            return base, comp.decompress().squeeze(0)

        decompressed = 0
        for layer_idx in sorted(by_layer.keys()):
            for base, data in by_layer[layer_idx]:
                name, w = _decompress_pair(base, data)
                if w is not None:
                    yield name, w
                    decompressed += 1
                    del w
            # sorted() returned a snapshot list, safe to mutate during iteration
            del by_layer[layer_idx]
            if decompressed % 100 == 0 and decompressed > 0:
                logger.info("  Decompressed %d tensors (layer %d)", decompressed, layer_idx)

        # Non-layer tensors (embeddings, lm_head, etc.)
        for base, data in no_layer:
            name, w = _decompress_pair(base, data)
            if w is not None:
                yield name, w
                decompressed += 1
                del w

        logger.info("TQ3 decompression complete: %d tensors", decompressed)

    DefaultModelLoader.get_all_weights = _decompress_get_all_weights
    logger.info("TQ3 decompress-on-load hook installed on DefaultModelLoader.get_all_weights")
