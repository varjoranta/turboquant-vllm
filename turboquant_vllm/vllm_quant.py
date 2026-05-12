"""TurboQuant vLLM integration: quantization config + TQ3 checkpoint loader.

Three roles:
1. Register ``TurboQuantConfig`` with ``--quantization turboquant`` so
   vLLM allocates model weights on meta device (zero GPU at init).
2. Online quant methods (``TurboQuantOnlineLinearMethod``,
   ``TurboQuantOnlineMoEMethod``) compress bf16 → TQ3 per-layer after
   weight loading, keeping peak GPU memory at ~1 layer bf16.
3. Patch ``DefaultModelLoader.get_all_weights`` to decompress native
   TQ3 checkpoints (``.tq_packed`` / ``.tq_norms``) to bf16 on the fly.

``TurboQuantConfig`` MUST live at module top level. cloudpickle
serializes closure-defined classes by value, transitively pulling in
``torch.ops.turboquant.*`` and crashing vLLM worker startup with
``cannot pickle '_OpNamespace'`` (issue #39).
"""

from __future__ import annotations

import logging
import re
from typing import Any

import torch
from torch import nn

logger = logging.getLogger(__name__)

# vLLM is an optional dependency — the package imports cleanly without
# it (Mac/MLX-only paths). Class definitions below are guarded on the
# imported symbols being non-None.
try:
    from vllm.model_executor.layers.linear import LinearBase
    from vllm.model_executor.layers.quantization.base_config import (
        QuantizationConfig,
        QuantizeMethodBase,
    )
except ImportError:
    LinearBase = None  # type: ignore[assignment,misc]
    QuantizationConfig = object  # type: ignore[assignment,misc]
    QuantizeMethodBase = object  # type: ignore[assignment,misc]

try:
    from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
        FusedMoEMethodBase,
    )
    from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
        UnquantizedFusedMoEMethod,
    )
except ImportError:
    FusedMoEMethodBase = object  # type: ignore[assignment,misc]
    UnquantizedFusedMoEMethod = None  # type: ignore[assignment,misc]


# Shared scratch pool across all FusedMoE layers — only one MoE layer
# runs at a time during forward, so one set of bf16 decompression
# buffers is enough. Per-layer pools would consume 78 × ~5 GB = 390 GB
# and defeat compression entirely.
_shared_moe_scratch_pool = None


# ── TurboQuantConfig: registered as `--quantization turboquant` ──

if LinearBase is not None:

    class TurboQuantConfig(QuantizationConfig):
        """Config for TurboQuant weight quantization (TQ3/TQ4)."""

        def __init__(
            self,
            bits: int = 3,
            group_size: int = 128,
            sensitive_bits: int | None = None,
            native_packed: bool = False,
        ):
            super().__init__()
            if bits not in (2, 3, 4):
                raise ValueError(f"turboquant bits must be 2, 3, or 4; got {bits}")
            if group_size <= 0 or group_size % 8 != 0:
                raise ValueError(f"turboquant group_size must be a positive multiple of 8; got {group_size}")
            if sensitive_bits is not None and sensitive_bits not in (2, 3, 4):
                raise ValueError(f"turboquant sensitive_bits must be 2, 3, or 4 or None; got {sensitive_bits}")
            self.bits = bits
            self.group_size = group_size
            self.sensitive_bits = sensitive_bits
            self.native_packed = native_packed

        def __repr__(self) -> str:
            return (
                f"TurboQuantConfig(bits={self.bits}, group_size={self.group_size}, "
                f"sensitive_bits={self.sensitive_bits}, native_packed={self.native_packed})"
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
            native_packed = config.get("format") == "tq3_native"
            return cls(
                bits=bits,
                group_size=group_size,
                sensitive_bits=sensitive_bits,
                native_packed=native_packed,
            )

        def get_quant_method(self, layer: nn.Module, prefix: str) -> "QuantizeMethodBase | None":
            if isinstance(layer, LinearBase):
                return TurboQuantOnlineLinearMethod(self.bits, self.group_size)
            try:
                from vllm.model_executor.layers.fused_moe import FusedMoE

                if isinstance(layer, FusedMoE) and TurboQuantOnlineMoEMethod is not None:
                    return TurboQuantOnlineMoEMethod(
                        self.bits,
                        self.group_size,
                        layer.moe_config,
                        native_packed=self.native_packed,
                    )
            except ImportError:
                pass
            return None

else:
    TurboQuantConfig = None  # type: ignore[assignment,misc]


# ── Online Linear quant method (meta-device init, per-layer compression) ──

if LinearBase is not None:

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
            if getattr(layer, "_already_called_process_weights_after_loading", False):
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

            bits = self.bits
            group_size = self.group_size

            weight = layer.weight.data

            out_dim, in_dim = weight.shape
            padded_in, n_groups = padded_size(in_dim, group_size)

            if padded_in > in_dim:
                padded = torch.zeros(
                    out_dim,
                    padded_in,
                    dtype=weight.dtype,
                    device=weight.device,
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
            # Pre-cast bf16 companions consumed by the bs=1 CUDA GEMV fast path.
            # Casting once at load time avoids per-decode-step HBM traffic.
            # Gate registration on the arch requirement so apply()'s fast-path
            # check collapses to a single hasattr() rather than a per-call
            # cudaGetDeviceProperties query.
            arch_ok = torch.cuda.is_available() and torch.cuda.get_device_capability(weight.device)[0] >= 8
            if bits == 3 and group_size == 128 and arch_ok:
                bytes_per_group = group_size * bits // 8
                layer.register_buffer(
                    "tq_packed_bs1",
                    packed.view(out_dim * n_groups, bytes_per_group),
                )
                layer.register_buffer("tq_norms_bf16", norms.to(torch.bfloat16))
                layer.register_buffer(
                    "tq_centroids_bf16",
                    quantizer.centroids.to(torch.bfloat16),
                )
            layer.tq_in_features = in_dim
            layer.tq_out_features = out_dim
            layer.tq_padded_in = padded_in

            # Cache dispatch — must run before CUDA graph capture
            _ensure_triton_backends()
            _get_cuda_module()
            if _triton_available:
                layer._tq_primary_fn = _tq_fwht_input_fn if out_dim >= 4096 else _tq_fused_gemm_fn
                layer._tq_fallback_fn = _tq_fused_gemm_fn if out_dim >= 4096 else _tq_fwht_input_fn
            else:
                layer._tq_primary_fn = None

            layer._already_called_process_weights_after_loading = True
            del weight, padded, grouped, indices, norms_raw

        def apply(
            self,
            layer: nn.Module,
            x: torch.Tensor,
            bias: torch.Tensor | None = None,
        ) -> torch.Tensor:
            # Pad input if in_dim was not a multiple of group_size
            if x.shape[-1] != layer.tq_padded_in:
                x = torch.nn.functional.pad(x, (0, layer.tq_padded_in - x.shape[-1]))

            # Route TQ3 bf16 through a runtime-dispatching custom op so the
            # bs=1 CUDA GEMV gets captured inside each size-specific CUDA
            # graph. Dynamo traces the model once (batch >> 1 on
            # profile_run) and would specialize a Python-level M==1 branch
            # against that shape, so the branch must live inside the op.
            if bias is None and self.bits == 3 and x.dtype == torch.bfloat16 and hasattr(layer, "tq_packed_bs1"):
                return torch.ops.turboquant.tq3_apply(
                    x,
                    layer.tq_packed_weight,
                    layer.tq_norms,
                    layer.tq_signs1,
                    layer.tq_signs2,
                    layer.tq_centroids,
                    layer.tq_packed_bs1,
                    layer.tq_norms_bf16,
                    layer.tq_centroids_bf16,
                    self.group_size,
                    self.bits,
                )

            if layer._tq_primary_fn is not None:
                args = (
                    x,
                    layer.tq_packed_weight,
                    layer.tq_norms,
                    layer.tq_signs1,
                    layer.tq_signs2,
                    layer.tq_centroids,
                )
                try:
                    return layer._tq_primary_fn(
                        *args,
                        group_size=self.group_size,
                        bits=self.bits,
                        bias=bias,
                    )
                except (ValueError, RuntimeError) as e:
                    logger.warning("TurboQuant primary kernel failed, using fallback: %s", e)
                    return layer._tq_fallback_fn(
                        *args,
                        group_size=self.group_size,
                        bits=self.bits,
                        bias=bias,
                    )

            # CPU/CUDA fallback
            from turboquant_vllm.weight_quant import _get_quantizer, unpack_indices

            indices = unpack_indices(
                layer.tq_packed_weight,
                self.bits,
                self.group_size,
            )
            norms_flat = layer.tq_norms.reshape(-1)
            quantizer = _get_quantizer(
                self.group_size,
                self.bits,
                str(x.device),
            )
            w_groups = quantizer.dequantize(indices, norms_flat)
            w_deq = w_groups.reshape(
                layer.tq_out_features,
                layer.tq_padded_in,
            ).to(x.dtype)
            output = torch.matmul(x, w_deq.t())
            if bias is not None:
                output = output + bias
            return output

else:
    TurboQuantOnlineLinearMethod = None  # type: ignore[assignment,misc]


# ── MoE online method ──


def _materialize_and_process(
    layer,
    buffer,
    orig_loaders,
    param_shapes,
    param_dtypes,
    method,
):
    """Materialize meta params on GPU, replay buffered loads, compress."""
    # 1. Materialize meta → real tensors on GPU
    for name, param in list(layer.named_parameters(recurse=False)):
        if param.device == torch.device("meta") and name in param_shapes:
            real = torch.empty(
                param_shapes[name],
                dtype=param_dtypes[name],
                device="cuda",
            )
            real_param = torch.nn.Parameter(real, requires_grad=False)
            if name in orig_loaders:
                real_param.weight_loader = orig_loaders[name]
            for attr in ("output_dim", "input_dim", "packed_dim", "packed_factor", "is_metadata"):
                if hasattr(param, attr):
                    setattr(real_param, attr, getattr(param, attr))
            delattr(layer, name)
            layer.register_parameter(name, real_param)

    # 2. Replay all buffered weight_loader calls
    for pname, args, kwargs in buffer:
        loader = orig_loaders.get(pname)
        if loader is not None:
            param = getattr(layer, pname)
            new_args = (param,) + args[1:]
            loader(*new_args, **kwargs)
    buffer.clear()

    # 3. Kernel setup + compress
    method._do_compress(layer)


def _finalize_native_packed_moe(
    layer,
    method,
    param_shapes,
    param_dtypes,
):
    """Bind native packed MoE tensors directly to Compressed3D objects."""
    global _shared_moe_scratch_pool

    from turboquant_vllm.moe_quant import (
        _HAS_FUSED_MOE,
        TurboQuantFusedMoEMethod,
        TurboQuantFusedMoEScratchPool,
        _collect_meta_tensors,
    )
    from turboquant_vllm.weight_quant import Compressed3D, packed_group_bytes, padded_size

    def _bind_real_weight_param(name: str, tensor: torch.Tensor) -> None:
        real_param = torch.nn.Parameter(tensor, requires_grad=False)
        if hasattr(layer, name):
            delattr(layer, name)
        layer.register_parameter(name, real_param)

    def _normalize_packed_layout(packed: torch.Tensor, shape: tuple[int, int, int]) -> torch.Tensor:
        n_experts, out_dim, in_dim = shape
        total_rows = n_experts * out_dim
        _, n_groups = padded_size(in_dim, method.group_size)
        pgb = packed_group_bytes(method.bits, method.group_size)

        if packed.ndim != 2:
            raise ValueError(f"Expected 2D packed tensor for shape {shape}, got {tuple(packed.shape)}")
        if packed.shape == (total_rows * n_groups, pgb):
            return packed
        if packed.shape == (total_rows, n_groups * pgb):
            return packed.reshape(total_rows * n_groups, pgb)
        raise ValueError(
            "Unsupported native packed layout for "
            f"{shape}: got {tuple(packed.shape)}, expected "
            f"({total_rows * n_groups}, {pgb}) or ({total_rows}, {n_groups * pgb})"
        )

    w13_c = Compressed3D.from_packed(
        _normalize_packed_layout(getattr(layer, "w13_weight_tq_packed").data, param_shapes["w13_weight"]),
        getattr(layer, "w13_weight_tq_norms").data,
        shape=param_shapes["w13_weight"],
        dtype=param_dtypes["w13_weight"],
        bits=method.bits,
        group_size=method.group_size,
    )
    w2_c = Compressed3D.from_packed(
        _normalize_packed_layout(getattr(layer, "w2_weight_tq_packed").data, param_shapes["w2_weight"]),
        getattr(layer, "w2_weight_tq_norms").data,
        shape=param_shapes["w2_weight"],
        dtype=param_dtypes["w2_weight"],
        bits=method.bits,
        group_size=method.group_size,
    )

    method._w13_c = w13_c
    method._w2_c = w2_c
    setattr(layer, "_tq_w13_weight", w13_c)
    setattr(layer, "_tq_w2_weight", w2_c)

    if _shared_moe_scratch_pool is None:
        _shared_moe_scratch_pool = TurboQuantFusedMoEScratchPool(w13_c, w2_c)
    else:
        _shared_moe_scratch_pool.assert_matches(w13_c, w2_c)

    pool = _shared_moe_scratch_pool
    method._pool = pool
    _bind_real_weight_param("w13_weight", pool.w13)
    _bind_real_weight_param("w2_weight", pool.w2)

    method._unquant.process_weights_after_loading(layer)
    # vLLM's MoE post-processing may replace the parameter objects while
    # setting up the kernel/runtime layout. Re-bind the final parameters to
    # real CUDA tensors here so later flashinfer packing never sees meta.
    _bind_real_weight_param("w13_weight", pool.w13)
    _bind_real_weight_param("w2_weight", pool.w2)
    if _HAS_FUSED_MOE and hasattr(layer, "_replace_quant_method"):
        layer.base_quant_method = method._unquant
        layer._replace_quant_method(TurboQuantFusedMoEMethod(layer.moe_config, w13_c, w2_c, pool))

    if logger.isEnabledFor(logging.DEBUG):
        meta_hits = []
        meta_hits.extend(_collect_meta_tensors(layer.w13_weight, "layer.w13_weight"))
        meta_hits.extend(_collect_meta_tensors(layer.w2_weight, "layer.w2_weight"))
        meta_hits.extend(_collect_meta_tensors(getattr(layer, "expert_map", None), "layer.expert_map"))
        meta_hits.extend(_collect_meta_tensors(getattr(layer, "base_quant_method", None), "layer.base_quant_method"))
        if meta_hits:
            raise RuntimeError(
                "TurboQuant detected meta tensors after native MoE finalize: " + "; ".join(meta_hits[:20])
            )

    for name in (
        "w13_weight_tq_packed",
        "w13_weight_tq_norms",
        "w2_weight_tq_packed",
        "w2_weight_tq_norms",
    ):
        if hasattr(layer, name):
            delattr(layer, name)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if UnquantizedFusedMoEMethod is not None and LinearBase is not None:

    class TurboQuantOnlineMoEMethod(FusedMoEMethodBase):
        """Meta-device MoE: compress after loading, decompress per forward.

        The MoE kernel is initialized by the underlying unquantized
        method's ``process_weights_after_loading``. After compression,
        ``apply()`` decompresses into a shared scratch pool and
        delegates to the unquantized method (which has the kernel).
        """

        uses_meta_device: bool = True

        def __init__(self, bits: int, group_size: int, moe_config: Any, native_packed: bool = False):
            super().__init__(moe_config)
            self.bits = bits
            self.group_size = group_size
            self.native_packed = native_packed
            self._unquant = UnquantizedFusedMoEMethod(moe_config)
            self._pool = None
            self._w13_c = None
            self._w2_c = None

        def create_weights(self, layer: nn.Module, **kwargs):
            self._unquant.create_weights(layer, **kwargs)

            # Compute expected total numel for completion tracking
            total_numel = sum(p.numel() for p in layer.parameters(recurse=False))

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
                    for attr in ("output_dim", "input_dim", "packed_dim", "packed_factor", "is_metadata"):
                        if hasattr(param, attr):
                            setattr(meta_param, attr, getattr(param, attr))
                    delattr(layer, name)
                    layer.register_parameter(name, meta_param)

            if self.native_packed:
                from turboquant_vllm.weight_quant import packed_group_bytes, padded_size

                num_experts, w13_out_dim, w13_in_dim = param_shapes["w13_weight"]
                _, w2_out_dim, w2_in_dim = param_shapes["w2_weight"]
                _, w13_groups = padded_size(w13_in_dim, self.group_size)
                _, w2_groups = padded_size(w2_in_dim, self.group_size)
                pgb = packed_group_bytes(self.bits, self.group_size)
                native_required = {
                    "w13_weight_tq_packed",
                    "w13_weight_tq_norms",
                    "w2_weight_tq_packed",
                    "w2_weight_tq_norms",
                }
                native_loaded: set[str] = set()
                native_finalized = [False]

                def _register_native_packed_param(name: str, shape: tuple[int, ...], dtype: torch.dtype):
                    param = torch.nn.Parameter(
                        torch.empty(shape, device="meta", dtype=dtype),
                        requires_grad=False,
                    )

                    def _loader(_param, loaded_weight, **_kwargs):
                        target_device = loaded_weight.device
                        if torch.cuda.is_available() and target_device.type != "cuda":
                            target_device = torch.device("cuda", torch.cuda.current_device())
                        materialized = loaded_weight.to(
                            device=target_device, copy=(loaded_weight.device != target_device)
                        )
                        real_param = torch.nn.Parameter(materialized, requires_grad=False)
                        real_param.weight_loader = _loader
                        delattr(layer, name)
                        layer.register_parameter(name, real_param)
                        native_loaded.add(name)
                        if not native_finalized[0] and native_loaded >= native_required:
                            native_finalized[0] = True
                            _finalize_native_packed_moe(
                                layer,
                                self,
                                {
                                    "w13_weight": param_shapes["w13_weight"],
                                    "w2_weight": param_shapes["w2_weight"],
                                },
                                {
                                    "w13_weight": param_dtypes["w13_weight"],
                                    "w2_weight": param_dtypes["w2_weight"],
                                },
                            )
                        del loaded_weight
                        return True

                    param.weight_loader = _loader
                    layer.register_parameter(name, param)

                _register_native_packed_param(
                    "w13_weight_tq_packed",
                    (num_experts * w13_out_dim, w13_groups * pgb),
                    torch.uint8,
                )
                _register_native_packed_param(
                    "w13_weight_tq_norms",
                    (num_experts * w13_out_dim, w13_groups),
                    torch.float32,
                )
                _register_native_packed_param(
                    "w2_weight_tq_packed",
                    (num_experts * w2_out_dim, w2_groups * pgb),
                    torch.uint8,
                )
                _register_native_packed_param(
                    "w2_weight_tq_norms",
                    (num_experts * w2_out_dim, w2_groups),
                    torch.float32,
                )
                return

            # Custom per-module buffering — bypass initialize_online_processing.
            # vLLM's online processing (CopyCounter) doesn't reliably
            # complete FusedMoE modules on meta device. We track loaded
            # numel directly from each weight_loader call instead.
            buffer: list[tuple[str, tuple, dict]] = []
            loaded_numel = [0]
            materialized = [False]

            def _make_buffering_loader(param_name, orig_loader):
                def _buffering_loader(*args, **kwargs):
                    if materialized[0]:
                        return orig_loader(*args, **kwargs)
                    loaded_weight = args[1] if len(args) > 1 else None
                    numel = loaded_weight.numel() if isinstance(loaded_weight, torch.Tensor) else 0
                    buffer.append((param_name, args, kwargs))
                    loaded_numel[0] += numel
                    if loaded_numel[0] >= total_numel:
                        materialized[0] = True
                        _materialize_and_process(
                            layer,
                            buffer,
                            orig_loaders,
                            param_shapes,
                            param_dtypes,
                            self,
                        )
                    # Signal success so model.load_weights commits the expert
                    return True

                return _buffering_loader

            for pname, param in layer.named_parameters(recurse=False):
                if pname in orig_loaders:
                    param.weight_loader = _make_buffering_loader(
                        pname,
                        orig_loaders[pname],
                    )

        def _do_compress(self, layer: nn.Module) -> None:
            """Kernel setup + TQ3 compression. Called after materialization."""
            global _shared_moe_scratch_pool

            from turboquant_vllm.moe_quant import TurboQuantFusedMoEScratchPool
            from turboquant_vllm.weight_quant import _compress_3d_param

            self._unquant.process_weights_after_loading(layer)

            w13 = getattr(layer, "w13_weight", None)
            w2 = getattr(layer, "w2_weight", None)
            if w13 is None or w2 is None or w13.dim() != 3 or w2.dim() != 3:
                return

            _compress_3d_param(layer, "w13_weight", self.bits, self.group_size)
            _compress_3d_param(layer, "w2_weight", self.bits, self.group_size)

            self._w13_c = layer._tq_w13_weight
            self._w2_c = layer._tq_w2_weight

            if _shared_moe_scratch_pool is None:
                _shared_moe_scratch_pool = TurboQuantFusedMoEScratchPool(
                    self._w13_c,
                    self._w2_c,
                )
            else:
                _shared_moe_scratch_pool.assert_matches(
                    self._w13_c,
                    self._w2_c,
                )

            self._pool = _shared_moe_scratch_pool
            layer.w13_weight.data = self._pool.w13
            layer.w2_weight.data = self._pool.w2

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        def process_weights_after_loading(self, layer: nn.Module) -> None:
            if self.native_packed:
                if not hasattr(layer, "_tq_w13_weight"):
                    _finalize_native_packed_moe(
                        layer,
                        self,
                        {
                            "w13_weight": tuple(layer.w13_weight.shape),
                            "w2_weight": tuple(layer.w2_weight.shape),
                        },
                        {
                            "w13_weight": layer.w13_weight.dtype,
                            "w2_weight": layer.w2_weight.dtype,
                        },
                    )
                return

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
            if self._pool is None or self._w13_c is None or self._w2_c is None:
                raise AssertionError(
                    "TurboQuantOnlineMoEMethod.apply requires compressed MoE weights and scratch pool. "
                    "Expected process_weights_after_loading to initialize the fallback state."
                )

            self._w13_c.decompress_into(
                self._pool.w13,
                fp32_scratch=self._pool.w13_fp32,
            )
            self._w2_c.decompress_into(
                self._pool.w2,
                fp32_scratch=self._pool.w2_fp32,
            )
            return self._unquant.apply(layer, x, **kwargs)

else:
    TurboQuantOnlineMoEMethod = None  # type: ignore[assignment,misc]


_registered = False


def register():
    """Register TurboQuant as a vLLM quantization method. Called from the plugin."""
    global _registered
    if _registered:
        return
    _registered = True

    if LinearBase is None:
        logger.debug("vLLM not installed, skipping TurboQuant quant config registration")
        return

    from vllm.model_executor.layers.quantization import register_quantization_config

    register_quantization_config("turboquant")(TurboQuantConfig)
    _patch_weight_name_remapping()
    logger.info("TurboQuant quantization config registered with vLLM")


# FP8 metadata that survives a re-quantization to TQ3 as dead bytes.
_FP8_LEFTOVER_SCALE_SUFFIXES = (
    ".weight_scale_inv",
    ".weight_scale",
    ".input_scale",
)

_EXPERT_INDEX_PATTERN = re.compile(r"^(.+?)\.experts\.(\d+)\.(.+)$")
_NATIVE_MOE_PROJ_FUSION = {
    "gate_proj": "w13_weight",
    "up_proj": "w13_weight",
    "down_proj": "w2_weight",
    "w1": "w13_weight",
    "w3": "w13_weight",
    "w2": "w2_weight",
}
_NATIVE_MOE_PROJ_ORDER = {
    "gate_proj": 0,
    "up_proj": 1,
    "down_proj": 0,
    "w1": 0,
    "w3": 1,
    "w2": 0,
}
_NATIVE_MOE_REQUIRED_ORDERS = {
    "w13_weight": {0, 1},
    "w2_weight": {0},
}


def _resolve_module(root, dotted_path: str):
    obj = root
    for part in dotted_path.split("."):
        try:
            obj = getattr(obj, part)
        except (AttributeError, TypeError):
            obj = obj[int(part)]
    return obj


def _resolve_parent_and_attr(root, dotted_path: str):
    parts = dotted_path.split(".")
    parent = _resolve_module(root, ".".join(parts[:-1])) if len(parts) > 1 else root
    return parent, parts[-1]


def _collect_meta_params(model) -> dict[str, tuple[nn.Module, str, torch.Tensor]]:
    meta_params: dict[str, tuple[nn.Module, str, torch.Tensor]] = {}
    for name, param in model.named_parameters():
        try:
            owner, attr = _resolve_parent_and_attr(model, name)
        except (AttributeError, IndexError, TypeError, ValueError):
            continue
        meta_params[name] = (owner, attr, param)
    for name, buf in model.named_buffers():
        try:
            owner, attr = _resolve_parent_and_attr(model, name)
        except (AttributeError, IndexError, TypeError, ValueError):
            continue
        meta_params[name] = (owner, attr, buf)
    return meta_params


def _regroup_native_moe_packed_tensors(
    model,
    packed_pairs: dict[str, dict[str, torch.Tensor]],
) -> list[tuple[str, torch.Tensor]]:
    """Regroup native per-expert TQ3 tensors into fused vLLM MoE targets."""
    meta_params = _collect_meta_params(model)

    regroup_map: dict[str, list[tuple[int, int, str]]] = {}
    direct_targets: list[tuple[str, torch.Tensor]] = []
    handled: set[str] = set()

    for base_name, tensors in packed_pairs.items():
        if "packed" not in tensors or "norms" not in tensors:
            continue

        if base_name in meta_params:
            _, attr, meta_param = meta_params[base_name]
            if len(meta_param.shape) == 3:
                direct_targets.append((f"{base_name}_tq_packed", tensors["packed"]))
                direct_targets.append((f"{base_name}_tq_norms", tensors["norms"]))
                handled.add(base_name)
                continue

        match = _EXPERT_INDEX_PATTERN.match(base_name)
        if not match:
            continue

        container_path = match.group(1) + ".experts"
        expert_idx = int(match.group(2))
        proj_suffix = match.group(3)
        proj_name = proj_suffix.split(".")[0]
        target_name = _NATIVE_MOE_PROJ_FUSION.get(proj_name)
        if target_name is None:
            continue

        target_key = f"{container_path}.{target_name}"
        if target_key not in meta_params:
            continue

        regroup_map.setdefault(target_key, []).append((_NATIVE_MOE_PROJ_ORDER[proj_name], expert_idx, base_name))
        handled.add(base_name)

    for target_key, entries in regroup_map.items():
        _, _, meta_param = meta_params[target_key]
        if len(meta_param.shape) != 3:
            continue

        n_experts_expected = meta_param.shape[0]
        entries.sort()
        expert_data: dict[int, tuple[list[torch.Tensor], list[torch.Tensor]]] = {}
        for order, expert_idx, base_name in entries:
            del order
            tensors = packed_pairs.get(base_name)
            if tensors is None:
                continue
            if expert_idx not in expert_data:
                expert_data[expert_idx] = ([], [])
            expert_data[expert_idx][0].append(tensors["packed"])
            expert_data[expert_idx][1].append(tensors["norms"])

        if len(expert_data) != n_experts_expected:
            logger.warning(
                "Native TQ3 MoE regroup skipped %s: model expects %d experts, saw %d",
                target_key,
                n_experts_expected,
                len(expert_data),
            )
            continue

        all_packed = []
        all_norms = []
        for expert_idx in sorted(expert_data):
            packed_parts, norm_parts = expert_data[expert_idx]
            all_packed.append(torch.cat(packed_parts, dim=0) if len(packed_parts) > 1 else packed_parts[0])
            all_norms.append(torch.cat(norm_parts, dim=0) if len(norm_parts) > 1 else norm_parts[0])

        direct_targets.append((f"{target_key}_tq_packed", torch.cat(all_packed, dim=0)))
        direct_targets.append((f"{target_key}_tq_norms", torch.cat(all_norms, dim=0)))

    return direct_targets


def _maybe_flush_native_moe_target(
    model,
    base_name: str,
    tensors: dict[str, torch.Tensor],
    meta_params: dict[str, tuple[nn.Module, str, torch.Tensor]],
    target_state: dict[str, dict[int, dict[int, tuple[torch.Tensor, torch.Tensor]]]],
) -> list[tuple[str, torch.Tensor]]:
    """Incrementally regroup one completed native MoE expert tensor pair.

    This keeps memory bounded to roughly one fused MoE target at a time
    instead of retaining every expert tensor in the checkpoint until the
    iterator is exhausted.
    """
    if "packed" not in tensors or "norms" not in tensors:
        return []

    if base_name in meta_params:
        _, _, meta_param = meta_params[base_name]
        if len(meta_param.shape) == 3:
            return [
                (f"{base_name}_tq_packed", tensors["packed"]),
                (f"{base_name}_tq_norms", tensors["norms"]),
            ]

    match = _EXPERT_INDEX_PATTERN.match(base_name)
    if not match:
        return []

    container_path = match.group(1) + ".experts"
    expert_idx = int(match.group(2))
    proj_suffix = match.group(3)
    proj_name = proj_suffix.split(".")[0]
    target_name = _NATIVE_MOE_PROJ_FUSION.get(proj_name)
    if target_name is None:
        return []

    target_key = f"{container_path}.{target_name}"
    meta_entry = meta_params.get(target_key)
    if meta_entry is None:
        return []

    _, _, meta_param = meta_entry
    if len(meta_param.shape) != 3:
        return []

    order = _NATIVE_MOE_PROJ_ORDER[proj_name]
    expert_map = target_state.setdefault(target_key, {})
    expert_parts = expert_map.setdefault(expert_idx, {})
    expert_parts[order] = (tensors["packed"], tensors["norms"])

    required_orders = _NATIVE_MOE_REQUIRED_ORDERS[target_name]
    n_experts_expected = meta_param.shape[0]
    if len(expert_map) != n_experts_expected:
        return []
    if any(set(parts) != required_orders for parts in expert_map.values()):
        return []

    all_packed = []
    all_norms = []
    for idx in range(n_experts_expected):
        parts = expert_map[idx]
        packed_parts = [parts[o][0] for o in sorted(required_orders)]
        norm_parts = [parts[o][1] for o in sorted(required_orders)]
        all_packed.append(torch.cat(packed_parts, dim=0) if len(packed_parts) > 1 else packed_parts[0])
        all_norms.append(torch.cat(norm_parts, dim=0) if len(norm_parts) > 1 else norm_parts[0])

    del target_state[target_key]
    packed_out = torch.cat(all_packed, dim=0)
    norms_out = torch.cat(all_norms, dim=0)
    return [
        (f"{target_key}_tq_packed", packed_out),
        (f"{target_key}_tq_norms", norms_out),
    ]


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
                    model_config.model,
                    "tq_config.json",
                    revision=revision,
                )
            except Exception as e:
                logger.info(
                    "No tq_config.json for %s (%s), passing through",
                    model_config.model,
                    e,
                )
                yield from _original_get_all_weights(self, model_config, model)
                return

        import json as _json

        with open(tq_config_path) as f:
            tq_cfg = _json.load(f)
        bits = tq_cfg.get("bits", 3)
        group_size = tq_cfg.get("group_size", 128)
        native_packed = tq_cfg.get("format") == "tq3_native"
        logger.info(
            "TQ3 native checkpoint (bits=%d, group_size=%d): single-pass decompress-on-load",
            bits,
            group_size,
        )
        pending_packed: dict[str, torch.Tensor] = {}
        pending_norms: dict[str, torch.Tensor] = {}
        pending_moe_pairs: dict[str, dict[str, torch.Tensor]] = {}
        moe_meta_params = _collect_meta_params(model) if native_packed else {}
        moe_target_state: dict[str, dict[int, dict[int, tuple[torch.Tensor, torch.Tensor]]]] = {}
        decompressed = 0
        yielded_native_moe = 0
        skipped_fp8_scales = 0

        for name, tensor in _original_get_all_weights(self, model_config, model):
            if name.endswith(".tq_packed") and ".experts." in name:
                base = name[: -len(".tq_packed")]
                pending_moe_pairs.setdefault(base, {})["packed"] = tensor
                if "norms" in pending_moe_pairs[base]:
                    ready_tensors = pending_moe_pairs.pop(base)
                    for out_name, out_tensor in _maybe_flush_native_moe_target(
                        model,
                        base,
                        ready_tensors,
                        moe_meta_params,
                        moe_target_state,
                    ):
                        yielded_native_moe += 1
                        yield out_name, out_tensor
                continue
            elif name.endswith(".tq_norms") and ".experts." in name:
                base = name[: -len(".tq_norms")]
                pending_moe_pairs.setdefault(base, {})["norms"] = tensor
                if "packed" in pending_moe_pairs[base]:
                    ready_tensors = pending_moe_pairs.pop(base)
                    for out_name, out_tensor in _maybe_flush_native_moe_target(
                        model,
                        base,
                        ready_tensors,
                        moe_meta_params,
                        moe_target_state,
                    ):
                        yielded_native_moe += 1
                        yield out_name, out_tensor
                continue
            elif name.endswith(".weight.tq_packed"):
                base = name[: -len(".tq_packed")]
                pending_packed[base] = tensor
            elif name.endswith(".weight.tq_norms"):
                base = name[: -len(".tq_norms")]
                pending_norms[base] = tensor
            elif name.endswith(_FP8_LEFTOVER_SCALE_SUFFIXES):
                skipped_fp8_scales += 1
                continue
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
                    packed,
                    norms,
                    (1, n_rows, in_dim),
                    torch.bfloat16,
                    bits,
                    group_size,
                )
                w = comp.decompress().squeeze(0)
                decompressed += 1
                if decompressed % 200 == 0:
                    logger.info("  Decompressed %d tensors", decompressed)
                yield base, w
                del packed, norms, comp, w

        if decompressed > 0:
            logger.info("TQ3 decompression complete: %d tensors", decompressed)
        if skipped_fp8_scales > 0:
            logger.info(
                "TQ3 native: dropped %d FP8 leftover scale tensors",
                skipped_fp8_scales,
            )

        for base in pending_packed:
            logger.warning("Orphaned .tq_packed without .tq_norms: %s", base)
        for base in pending_norms:
            logger.warning("Orphaned .tq_norms without .tq_packed: %s", base)
        for base, tensors in pending_moe_pairs.items():
            if "packed" not in tensors or "norms" not in tensors:
                logger.warning("Orphaned native MoE packed pair: %s", base)
        for target_key, expert_map in moe_target_state.items():
            _, _, meta_param = moe_meta_params[target_key]
            logger.warning(
                "Incomplete native MoE regroup for %s: expected %d experts, saw %d",
                target_key,
                meta_param.shape[0],
                len(expert_map),
            )

    DefaultModelLoader.get_all_weights = _decompress_get_all_weights
    logger.info("TQ3 decompress-on-load hook installed on DefaultModelLoader.get_all_weights")
