"""TurboQuant vLLM integration: quantization config + TQ3 checkpoint loader.

Three roles:
1. Register ``TurboQuantConfig`` with ``--quantization turboquant`` so
   vLLM allocates model weights on meta device (zero GPU at init).
2. Online quant methods (``TurboQuantOnlineLinearMethod``,
   ``TurboQuantOnlineMoEMethod``) compress bf16 → TQ3 per-layer after
   weight loading, keeping peak GPU memory at ~1 layer bf16.
3. Patch ``DefaultModelLoader.get_all_weights`` to decompress native
   TQ3 checkpoints (``.tq_packed`` / ``.tq_norms``) to bf16 on the fly.
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
        """Meta-device init + per-layer TQ3 compression for Linear layers.

        Allocates bf16 weight on meta device (zero GPU at init). After
        weight loading materializes the bf16 on GPU, compress to TQ3
        packed format. Single-pass decompression in get_all_weights
        feeds bf16 to vLLM's standard weight routing (QKV stacking,
        gate_up fusion) unchanged.
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
            # Guard: called twice (online processing + global sweep).
            # After first call, weight is emptied; packed data is in tq_packed_weight.
            if not hasattr(layer, "weight") or layer.weight.numel() == 0:
                return

            from turboquant_vllm.weight_quant import (
                _ensure_triton_backends,
                _get_cuda_module,
                _get_quantizer,
                _tq_fused_gemm_fn,
                _tq_fwht_input_fn,
                _triton_available,
                pack_indices,
                padded_size,
            )

            weight = layer.weight.data
            bits = self.bits
            group_size = self.group_size

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
            quantizer = _get_quantizer(group_size, bits, str(weight.device))
            indices, norms_raw = quantizer.quantize(grouped, norm_correction=True)
            packed = pack_indices(indices, bits)
            norms = norms_raw.reshape(out_dim, n_groups)

            # Keep weight for vLLM's MLA/attention post-processing,
            # but zero it to free most GPU memory. Full deletion breaks
            # MLAAttention.process_weights_after_loading which accesses
            # sub-layer weights after our quant method runs.
            layer.weight.data = torch.empty(0, device=weight.device, dtype=weight.dtype)
            layer.register_buffer("tq_packed_weight", packed)
            layer.register_buffer("tq_norms", norms)
            layer.register_buffer("tq_signs1", quantizer.signs1)
            layer.register_buffer("tq_signs2", quantizer.signs2)
            layer.register_buffer("tq_centroids", quantizer.centroids)
            layer.tq_in_features = in_dim
            layer.tq_out_features = out_dim
            layer.tq_padded_in = padded_in

            # Cache dispatch — must run before CUDA graph capture
            _ensure_triton_backends()
            _get_cuda_module()
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

            # CPU/CUDA fallback
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

    try:
        from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
            FusedMoEMethodBase,
        )
        from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
            UnquantizedFusedMoEMethod,
        )

        def _materialize_and_process(
            layer, buffer, orig_loaders, param_shapes, param_dtypes, method,
        ):
            """Materialize meta params on GPU, replay buffered loads, compress."""
            import sys

            # 1. Materialize meta → real tensors on GPU
            for name, param in list(layer.named_parameters(recurse=False)):
                if param.device == torch.device("meta") and name in param_shapes:
                    real = torch.empty(
                        param_shapes[name], dtype=param_dtypes[name],
                        device="cuda",
                    )
                    real_param = torch.nn.Parameter(real, requires_grad=False)
                    # Preserve weight_loader for replay
                    if name in orig_loaders:
                        real_param.weight_loader = orig_loaders[name]
                    for attr in ("output_dim", "input_dim", "packed_dim",
                                 "packed_factor", "is_metadata"):
                        if hasattr(param, attr):
                            setattr(real_param, attr, getattr(param, attr))
                    delattr(layer, name)
                    layer.register_parameter(name, real_param)

            # 2. Replay all buffered weight_loader calls
            for pname, args, kwargs in buffer:
                loader = orig_loaders.get(pname)
                if loader is not None:
                    # Update the param reference in args
                    param = getattr(layer, pname)
                    new_args = (param,) + args[1:]
                    loader(*new_args, **kwargs)
            buffer.clear()

            # 3. Run process_weights_after_loading (kernel setup + compress)
            method._do_compress(layer)

            gpu_gb = torch.cuda.memory_allocated() / 1e9
            print(
                f"[TQ-MOE] Module materialized+compressed GPU={gpu_gb:.1f}GB",
                file=sys.stderr, flush=True,
            )

        # Shared scratch pool across all FusedMoE layers — only one MoE
        # layer runs at a time during forward, so one set of bf16
        # decompression buffers is enough. Per-layer pools would consume
        # 78 × ~5 GB = 390 GB and defeat compression entirely.
        _shared_moe_scratch_pool = None

        class TurboQuantOnlineMoEMethod(FusedMoEMethodBase):
            """Meta-device MoE: compress after loading, decompress per forward.

            The MoE kernel is initialized by the underlying unquantized
            method's ``process_weights_after_loading``. After compression,
            ``apply()`` decompresses into a shared scratch pool and
            delegates to the unquantized method (which has the kernel).
            """

            uses_meta_device: bool = True

            def __init__(self, bits: int, group_size: int, moe_config: Any):
                super().__init__(moe_config)
                self.bits = bits
                self.group_size = group_size
                self._unquant = UnquantizedFusedMoEMethod(moe_config)
                self._pool = None
                self._w13_c = None
                self._w2_c = None

            def create_weights(self, layer: nn.Module, **kwargs):
                self._unquant.create_weights(layer, **kwargs)

                # Compute expected total numel for completion tracking
                total_numel = sum(
                    p.numel() for p in layer.parameters(recurse=False)
                )

                # Save original weight_loaders + shapes BEFORE meta move
                orig_loaders: dict[str, Any] = {}
                param_shapes: dict[str, tuple] = {}
                param_dtypes: dict[str, torch.dtype] = {}
                for name, param in list(layer.named_parameters(recurse=False)):
                    if hasattr(param, "weight_loader"):
                        orig_loaders[name] = param.weight_loader
                    param_shapes[name] = tuple(param.shape)
                    param_dtypes[name] = param.dtype

                # Move parameters to meta device (zero GPU at init)
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

                # Custom per-module buffering — bypass initialize_online_processing.
                # vLLM's CopyCounter may not count copy_() into meta tensors
                # correctly, preventing module completion. We track loaded
                # numel directly from each weight_loader call.
                buffer = []  # [(param_name, args, kwargs, numel)]
                loaded_numel = [0]
                materialized = [False]

                def _make_buffering_loader(param_name, orig_loader):
                    def _buffering_loader(*args, **kwargs):
                        if materialized[0]:
                            # After materialization, run directly
                            return orig_loader(*args, **kwargs)
                        # Buffer the call + track loaded numel
                        loaded_weight = args[1] if len(args) > 1 else None
                        numel = (
                            loaded_weight.numel()
                            if isinstance(loaded_weight, torch.Tensor)
                            else 0
                        )
                        buffer.append((param_name, args, kwargs))
                        loaded_numel[0] += numel

                        if loaded_numel[0] >= total_numel:
                            # All weights arrived — materialize + replay
                            materialized[0] = True
                            _materialize_and_process(
                                layer, buffer, orig_loaders,
                                param_shapes, param_dtypes, self,
                            )
                    return _buffering_loader

                for pname, param in layer.named_parameters(recurse=False):
                    if pname in orig_loaders:
                        param.weight_loader = _make_buffering_loader(
                            pname, orig_loaders[pname],
                        )

            def _do_compress(self, layer: nn.Module) -> None:
                """Kernel setup + TQ3 compression. Called after materialization."""
                import sys

                # Set up the MoE kernel (via _setup_kernel in vLLM 0.19)
                self._unquant.process_weights_after_loading(layer)

                # TEMPORARY: skip compression to isolate Triton crash
                print(
                    f"[TQ-MOE] Skipping compression (debug), "
                    f"w13={layer.w13_weight.shape} GPU={torch.cuda.memory_allocated()/1e9:.1f}GB",
                    file=sys.stderr, flush=True,
                )

            def process_weights_after_loading(self, layer: nn.Module) -> None:
                # Compression handled by _materialize_and_process (triggered
                # by buffering loader). This guard handles the global sweep.
                if not hasattr(layer, "_tq_w13_weight"):
                    # Not yet compressed — run compression now (fallback
                    # for modules where buffering didn't trigger)
                    if hasattr(layer, "w13_weight") and layer.w13_weight.numel() > 0:
                        self._do_compress(layer)

            def get_fused_moe_quant_config(self, layer: nn.Module):
                return self._unquant.get_fused_moe_quant_config(layer)

            def apply(self, layer: nn.Module, x: torch.Tensor, **kwargs) -> torch.Tensor:
                # Decompress into shared scratch pool, then delegate to
                # the unquantized method which has the MoE kernel.
                if self._pool is not None and self._w13_c is not None:
                    self._w13_c.decompress_into(
                        self._pool.w13, fp32_scratch=self._pool.w13_fp32,
                    )
                    self._w2_c.decompress_into(
                        self._pool.w2, fp32_scratch=self._pool.w2_fp32,
                    )
                return self._unquant.apply(layer, x, **kwargs)

    except ImportError:
        TurboQuantOnlineMoEMethod = None  # type: ignore[assignment,misc]

    _patch_weight_name_remapping()

    logger.info("TurboQuant quantization config registered with vLLM")


def _patch_weight_name_remapping():
    """Monkey-patch vLLM's weight iterator to decompress TQ3 weights on load.

    Single-pass: as each ``.tq_packed`` / ``.tq_norms`` pair arrives
    from the checkpoint iterator, decompress to bf16 and yield with the
    original ``.weight`` name.  vLLM's model-specific weight loaders
    (stacked qkv, fused gate_up, expert assembly) work unchanged.

    CPU memory is bounded by the online processing buffer for currently-
    loading modules (typically 1-2 decoder layers).  The bf16 is transient
    — ``process_weights_after_loading`` compresses to TQ3 on GPU.
    """
    try:
        from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
    except ImportError:
        return

    from turboquant_vllm.weight_quant import Compressed3D

    _original_get_all_weights = DefaultModelLoader.get_all_weights

    def _decompress_get_all_weights(self, model_config, model):
        """Decompress TQ3 → bf16 per tensor, single-pass.

        Pairs ``.tq_packed`` + ``.tq_norms`` as they arrive from the
        checkpoint iterator, decompresses to bf16 immediately, and yields
        with the original ``.weight`` name. No collection / buffering of
        packed tensors — CPU memory is bounded by whichever tensors the
        online processing is currently accumulating for incomplete modules
        (typically 1-2 decoder layers worth of bf16).
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
            "single-pass decompress-on-load",
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

            # When both halves of a pair arrive, decompress and yield
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
                if decompressed % 200 == 0:
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
