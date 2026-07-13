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


# Fallback scratch pool for direct tests/legacy construction. Normal vLLM
# model loads keep the pool on the per-model TurboQuantConfig instance so
# separate LLM objects in one process never share writable dequant buffers.
_shared_moe_scratch_pool = None

# The four per-layer placeholder params that native-packed TQ3 MoE
# checkpoints populate from disk and that _finalize_native_packed_moe
# regroups into the fused w13/w2 Compressed3D objects.
_NATIVE_PACKED_PARAM_NAMES = (
    "w13_weight_tq_packed",
    "w13_weight_tq_norms",
    "w2_weight_tq_packed",
    "w2_weight_tq_norms",
)


def _normalize_sensitive_patterns(patterns, sensitive_bits: "int | None" = None) -> "tuple[str, ...]":
    """Module-level alias so config construction stays cloudpickle-safe."""
    from turboquant_vllm.weight_quant import normalize_sensitive_patterns

    return normalize_sensitive_patterns(patterns, sensitive_bits)


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
            sensitive_patterns: "tuple[str, ...] | list[str] | None" = None,
        ):
            super().__init__()
            if bits not in (2, 3, 4):
                raise ValueError(f"turboquant bits must be 2, 3, or 4; got {bits}")
            # The WHT rotation requires a power of two — non-power-of-two
            # multiples of 8 (e.g. 96) used to fail deep in the kernels with
            # an obscure error. Sizes outside 64/128/256 stay valid (tests
            # and tiny-tensor checkpoints use 8) but run on the pure-PyTorch
            # path: process_weights_after_loading only binds the Triton/CUDA
            # kernels for the sizes they are compiled for.
            if group_size < 8 or group_size & (group_size - 1) != 0:
                raise ValueError(f"turboquant group_size must be a power of two >= 8; got {group_size}")
            if sensitive_bits is not None and sensitive_bits not in (2, 3, 4):
                raise ValueError(f"turboquant sensitive_bits must be 2, 3, or 4 or None; got {sensitive_bits}")
            self.bits = bits
            self.group_size = group_size
            self.sensitive_bits = sensitive_bits
            self.sensitive_patterns = _normalize_sensitive_patterns(sensitive_patterns, sensitive_bits)
            self.native_packed = native_packed
            self._moe_scratch_pool = None

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
            sensitive_patterns = cls.get_from_keys_or(config, ["sensitive_patterns"], None)
            native_packed = config.get("format") == "tq3_native"
            return cls(
                bits=bits,
                group_size=group_size,
                sensitive_bits=sensitive_bits,
                native_packed=native_packed,
                sensitive_patterns=sensitive_patterns,
            )

        def _bits_for(self, name: str) -> int:
            from turboquant_vllm.weight_quant import select_bits

            return select_bits(name, self.bits, self.sensitive_bits, self.sensitive_patterns)

        def get_quant_method(self, layer: nn.Module, prefix: str) -> "QuantizeMethodBase | None":
            if isinstance(layer, LinearBase):
                return TurboQuantOnlineLinearMethod(self._bits_for(prefix), self.group_size)
            try:
                from turboquant_vllm.weight_quant import _moe_module_types

                # vLLM 0.25 made FusedMoE a factory function; the expert-weight
                # module is RoutedExperts. Match whichever class(es) exist.
                moe_types = _moe_module_types()

                if moe_types and isinstance(layer, moe_types) and TurboQuantOnlineMoEMethod is not None:
                    # The fused MoE params don't carry per-projection names,
                    # so match the patterns against synthetic proj names under
                    # this prefix — the same substrings the checkpoint packer
                    # saw in the per-expert on-disk names (e.g.
                    # "…experts.5.down_proj.weight").
                    return TurboQuantOnlineMoEMethod(
                        self.bits,
                        self.group_size,
                        layer.moe_config,
                        native_packed=self.native_packed,
                        scratch_pool_owner=self,
                        w13_bits=self._bits_for(f"{prefix}.gate_up_proj"),
                        w2_bits=self._bits_for(f"{prefix}.down_proj"),
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

            # Backend state (_triton_available, _tq_*_fn) is initialized
            # lazily by _ensure_triton_backends/_get_cuda_module below, so it
            # must be read through the module AFTER those calls — a
            # `from ... import` here would capture the pre-initialization
            # None values on the first layer processed.
            from turboquant_vllm import weight_quant as wq
            from turboquant_vllm.weight_quant import (
                _get_quantizer,
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

            # Backend probes + dispatch inputs — must run before CUDA graph
            # capture. The Triton/CUDA kernels are built for exactly these
            # group sizes; anything else (e.g. group_size=8 test configs)
            # must take the pure-PyTorch path in apply() rather than crash
            # at dispatch.
            kernels_support_group = group_size in wq._CUDA_KERNEL_GROUP_SIZES
            triton_ok = wq._ensure_triton_backends()
            cuda_mod = wq._get_cuda_module()

            # Pre-cast bf16 companions consumed by the bs=1 CUDA GEMV fast path.
            # Casting once at load time avoids per-decode-step HBM traffic.
            # Gate registration on the arch requirement so apply()'s fast-path
            # check collapses to a single hasattr() rather than a per-call
            # cudaGetDeviceProperties query. Also require at least one M>1
            # backend (Triton or the CUDA extension): the tq3_apply route
            # this registration enables covers only M == 1 with the GEMV and
            # needs one of them for prefill.
            # weight may sit on CPU (offload paths) — get_device_capability
            # raises on non-CUDA devices even when CUDA is available.
            arch_ok = (
                weight.device.type == "cuda"
                and torch.cuda.is_available()
                and torch.cuda.get_device_capability(weight.device)[0] >= 8
            )
            if bits == 3 and group_size == 128 and arch_ok and (triton_ok or cuda_mod is not None):
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

            if triton_ok and kernels_support_group:
                # Shared selector — encodes both the ~4K crossover and the
                # group_size > 128 fused-GEMM exclusion (whose OutOfResources
                # failure apply()'s fallback handler cannot catch).
                layer._tq_primary_fn, layer._tq_fallback_fn = wq._select_triton_gemm_fns(out_dim, group_size)
            else:
                layer._tq_primary_fn = None
            # Without Triton, prefer the CUDA dequant-GEMM extension over the
            # last-resort per-forward Python unpack loop (same routing as
            # TurboQuantWrapper._forward_gpu). Bind the callable here so
            # apply() doesn't run an import statement per forward.
            layer._tq_cuda_gemm_fn = (
                wq._tq_cuda_dequant_gemm_fn
                if (
                    layer._tq_primary_fn is None
                    and kernels_support_group
                    and cuda_mod is not None
                    and wq._tq_cuda_dequant_gemm_fn is not None
                )
                else None
            )

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
                if layer._tq_fallback_fn is None:
                    return layer._tq_primary_fn(
                        *args,
                        group_size=self.group_size,
                        bits=self.bits,
                        bias=bias,
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

            # CUDA extension fallback (no Triton): fused dequant + cuBLAS GEMM,
            # same routing as TurboQuantWrapper._forward_gpu. Availability is
            # decided once in process_weights_after_loading; the custom op is
            # opaque to dynamo like the Triton launchers.
            cuda_gemm_fn = getattr(layer, "_tq_cuda_gemm_fn", None)
            if cuda_gemm_fn is not None and x.is_cuda:
                return cuda_gemm_fn(
                    x,
                    layer.tq_packed_weight,
                    layer.tq_norms,
                    layer.tq_signs1,
                    layer.tq_signs2,
                    layer.tq_centroids,
                    bias,
                    self.group_size,
                    self.bits,
                    layer.tq_out_features,
                    layer.tq_padded_in,  # x is already padded to this width
                    # block_size == group_size: the online method always
                    # quantizes with a full-width WHT (no rotary_dim support,
                    # unlike TurboQuantWrapper, which passes a rotary-aware
                    # block size for partial-rotary layers).
                    self.group_size,
                )

            # Pure-PyTorch fallback (CPU, or CUDA without any extension)
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


def _materialize_target_device() -> str:
    """Device that meta-initialized weights materialize onto."""
    return "cuda" if torch.cuda.is_available() else "cpu"


# The expert weight tensors whose checkpoint-side arrival marks an online
# MoE layer's load as complete. Biases and other auxiliary params shard
# differently than the uniform tp scaling in _moe_expected_checkpoint_numel
# assumes (e.g. w2_bias is replicated across TP), so they are buffered but
# never counted toward the completion threshold.
_TRACKED_MOE_WEIGHT_PARAMS = ("w13_weight", "w2_weight")


def _positive_int(primary, fallback) -> "int | None":
    """First of the two values that is a positive plain int (bool excluded)."""
    for value in (primary, fallback):
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    return None


def _moe_expected_checkpoint_numel(layer, moe_config, partition_numel: int) -> "int | None":
    """Checkpoint-side numel that marks this layer's expert load as complete.

    vLLM's FusedMoE ``weight_loader`` receives FULL (unsharded, global-expert)
    tensors from the checkpoint and shards them into the local param: under TP
    the intermediate dim arrives ``tp_size``× larger than the param slice, and
    under EP the loader fires for every global expert while the param only
    holds the local ones. Measuring completion against the partition-side
    param numel therefore fires the materialize+compress trigger at ~1/tp of
    the load and compresses partially-loaded experts. Scale the partition-side
    sum by both factors instead.

    ``partition_numel`` must cover only uniformly-TP-sharded expert weights
    (see ``_TRACKED_MOE_WEIGHT_PARAMS``) — replicated params like ``w2_bias``
    would inflate the expectation and make the threshold unreachable.

    Returns None when the parallel factors cannot be derived — the caller must
    then skip threshold-based triggering and let
    ``process_weights_after_loading`` replay the buffered loads.
    """
    tp_size = _positive_int(
        getattr(layer, "tp_size", None),
        getattr(moe_config, "tp_size", None),
    )
    global_experts = _positive_int(
        getattr(layer, "global_num_experts", None),
        getattr(moe_config, "num_experts", None),
    )
    local_experts = _positive_int(
        getattr(layer, "local_num_experts", None),
        getattr(moe_config, "num_local_experts", None),
    )
    if tp_size is None or global_experts is None or local_experts is None:
        return None
    if global_experts % local_experts != 0:
        return None
    return partition_numel * tp_size * (global_experts // local_experts)


def _materialize_and_process(layer, state, method):
    """Materialize meta params on GPU, replay buffered loads, compress.

    ``state`` is the pending-load dict built in ``create_weights``
    (buffer / orig_loaders / param_shapes / param_dtypes / materialized) —
    the same object both completion paths (threshold fire and
    ``_finish_online_moe_load``) operate on.
    """
    buffer = state["buffer"]
    orig_loaders = state["orig_loaders"]
    param_shapes = state["param_shapes"]
    param_dtypes = state["param_dtypes"]
    target_device = _materialize_target_device()
    # 1. Materialize meta → real tensors on GPU. Zero-filled, not empty:
    # untracked params (e.g. biases arriving after the w13/w2 completion
    # threshold fires) and loader-less params may have no buffered load to
    # replay, and the unquant process_weights_after_loading inside
    # _do_compress reads them — zeros are the safe neutral value, and any
    # late checkpoint load still overwrites them through the live param.
    for name, param in list(layer.named_parameters(recurse=False)):
        if param.device == torch.device("meta") and name in param_shapes:
            real = torch.zeros(
                param_shapes[name],
                dtype=param_dtypes[name],
                device=target_device,
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


def _finish_online_moe_load(layer, method, pending_state) -> None:
    """Complete an online MoE load from ``process_weights_after_loading``.

    Covers layers where the checkpoint-side numel threshold never fired
    (unknown TP/EP factors, short checkpoints): replay the buffered loads
    and compress. Raises instead of compressing meta/unreplayed params —
    ``_materialize_and_process`` allocates meta params via ``torch.empty``,
    so any param with zero buffered loads would otherwise be compressed
    from uninitialized memory with no error.
    """
    if pending_state is not None and not pending_state["materialized"][0]:
        pending_state["materialized"][0] = True
        buffer = pending_state["buffer"]
        if buffer:
            # A non-empty buffer is not proof of completeness: every meta
            # param that CAN be checkpoint-loaded (i.e. has a weight_loader)
            # must have at least one buffered load before compression bakes
            # its values. Loader-less params are exempt — the checkpoint can
            # never populate them and materialize zero-fills them instead.
            buffered_params = {pname for pname, _, _ in buffer}
            unloaded = [
                name
                for name, param in layer.named_parameters(recurse=False)
                if param.is_meta
                and name in pending_state["param_shapes"]
                and name in pending_state["orig_loaders"]
                and name not in buffered_params
            ]
            if unloaded:
                raise RuntimeError(
                    f"TurboQuant online MoE: checkpoint provided no weights for "
                    f"{unloaded} on this layer ({len(buffer)} buffered loads for "
                    f"{sorted(buffered_params)}). Refusing to compress "
                    f"uninitialized parameters."
                )
            logger.warning(
                "TurboQuant online MoE: load-completion threshold never fired for %s; "
                "replaying %d buffered loads at process_weights_after_loading",
                type(layer).__name__,
                len(buffer),
            )
            _materialize_and_process(layer, pending_state, method)
            return

    w13 = getattr(layer, "w13_weight", None)
    if w13 is None:
        return
    if w13.is_meta:
        raise RuntimeError(
            "TurboQuant online MoE: w13_weight is still on the meta device after "
            "weight loading and there are no buffered loads to replay. The "
            "checkpoint did not populate this layer's experts."
        )
    if w13.numel() > 0:
        method._do_compress(layer)


_META_MATERIALIZE_SKIP_TENSORS = {
    "_expert_map",
    "expert_mask",
    "expert_global_to_physical",
    "expert_physical_to_global",
    "expert_local_to_global",
    "e_score_correction_bias",
}


def _materialize_meta_tensor_like(meta_tensor: torch.Tensor, target_device: str) -> torch.Tensor:
    """Materialize a meta tensor without reading from meta storage.

    Mirrors vLLM's reload.meta.materialize_meta_tensor pattern: construct new
    storage with the same size/stride/dtype, then preserve tensor subclass and
    custom attrs. Do not use ``.data =`` or ``empty_like(meta, device=...)``;
    both can route through meta copy/set_data paths and fail for vLLM Parameter
    subclasses.
    """
    tensor = torch.empty_strided(
        size=tuple(meta_tensor.size()),
        stride=tuple(meta_tensor.stride()),
        dtype=meta_tensor.dtype,
        device=target_device,
        requires_grad=False,
    )
    tensor.zero_()
    tensor.__class__ = meta_tensor.__class__
    tensor.__dict__ = meta_tensor.__dict__.copy()
    return tensor


def _materialize_meta_tensors(layer, label: str = ""):
    """Walk every parameter and buffer on ``layer`` and submodules.

    For each tensor still on ``meta``, replace the owning module slot with a
    real zero tensor on the active device while preserving stride, subclass and
    tensor attrs.

    Why: vLLM's FusedMoE creates parameter slots up front (some on meta until
    first use). PR #44's native-packed loader rebinds w13_weight/w2_weight to
    real CUDA tensors, but vLLM 0.20+ FlashInfer CUTLASS MoE backend reads
    additional tensors (scales, packing tables, FP8 staging buffers) — and its
    `run_moe` DLPack conversion fails with "Cannot pack tensors on meta" if
    any of those still live on the meta device when the first forward fires.

    Logs every name materialized so the run output documents which slots
    needed the rescue. Returns the list of materialized names for callers
    that want to assert on the result.
    """
    target_device = _materialize_target_device()
    materialized: list[str] = []
    failed: list[str] = []

    def _try_materialize(owner_module, store_name: str, attr_name: str, tensor: torch.Tensor | None):
        if tensor is None or not isinstance(tensor, torch.Tensor) or not tensor.is_meta:
            return
        if attr_name in _META_MATERIALIZE_SKIP_TENSORS:
            return
        try:
            new_tensor = _materialize_meta_tensor_like(tensor, target_device)
        except Exception as e:
            failed.append(f"{store_name}:{attr_name} ({type(e).__name__}: {e})")
            return
        getattr(owner_module, store_name)[attr_name] = new_tensor
        materialized.append(f"{store_name}:{owner_module.__class__.__name__}.{attr_name}")

    for _mod_name, sub in layer.named_modules():
        for p_name, param in list(sub._parameters.items()):
            _try_materialize(sub, "_parameters", p_name, param)
        for b_name, buf in list(sub._buffers.items()):
            _try_materialize(sub, "_buffers", b_name, buf)

    if failed:
        logger.warning(
            "TurboQuant native-packed MoE finalize (%s): could not materialize %d tensors: %s",
            label,
            len(failed),
            failed[:10],
        )

    if materialized:
        logger.info(
            "TurboQuant native-packed MoE finalize (%s): materialized %d meta tensors: %s",
            label,
            len(materialized),
            materialized[:20],
        )
    return materialized


def _collect_residual_meta_tensors(obj, prefix: str, max_depth: int = 4) -> list[str]:
    """Debug collector for meta tensors reachable from MoE runtime objects."""
    seen: set[int] = set()
    hits: list[str] = []

    def _walk(value, path: str, depth: int) -> None:
        obj_id = id(value)
        if obj_id in seen:
            return
        seen.add(obj_id)
        if isinstance(value, torch.Tensor):
            if value.is_meta:
                hits.append(f"{path}: shape={tuple(value.shape)} dtype={value.dtype}")
            return
        if value is None or depth >= max_depth:
            return
        if isinstance(value, (str, bytes, int, float, bool, torch.dtype, torch.device)):
            return
        if isinstance(value, dict):
            for key, item in value.items():
                _walk(item, f"{path}.{key}", depth + 1)
            return
        if isinstance(value, (list, tuple, set)):
            for idx, item in enumerate(value):
                _walk(item, f"{path}[{idx}]", depth + 1)
            return
        if isinstance(value, torch.nn.Module):
            for name, param in value._parameters.items():
                if name in _META_MATERIALIZE_SKIP_TENSORS:
                    continue
                _walk(param, f"{path}._parameters.{name}", depth + 1)
            for name, buf in value._buffers.items():
                if name in _META_MATERIALIZE_SKIP_TENSORS:
                    continue
                _walk(buf, f"{path}._buffers.{name}", depth + 1)
            for name, sub in value._modules.items():
                _walk(sub, f"{path}.{name}", depth + 1)
            return
        if hasattr(value, "__dict__"):
            for name, item in vars(value).items():
                if name.startswith("__") or name in _META_MATERIALIZE_SKIP_TENSORS:
                    continue
                _walk(item, f"{path}.{name}", depth + 1)

    _walk(obj, prefix, 0)
    return hits


def _resolve_native_moe_shape(
    packed: torch.Tensor,
    norms: torch.Tensor,
    shape: tuple[int, int, int],
    bits: int,
    group_size: int,
) -> tuple[int, int, int]:
    """Resolve the true (n_experts, out_dim, in_dim) of a native-packed MoE
    weight from the norms tensor (the authoritative quant metadata), not the
    registered ``shape``.

    DeepSeek-V4-Flash registers MoE weights with shapes that mis-report
    out_dim (w13 is registered un-fused / half) and in_dim (w2), while the
    native TQ3 checkpoint packs the true gate_up-fused, full-width tensors.
    The packer always lays norms out as exactly ``(n_experts * out_dim,
    n_groups)``, so it recovers both dims independent of the registered
    shape. n_experts (expert count) and group_size are structurally
    unambiguous and trusted. If the packed tensor is inconsistent with the
    norms-derived layout, the original ``shape`` is returned unchanged so
    the downstream validators raise their precise errors instead of masking
    corruption. For well-formed (standard) models norms already matches the
    registered shape, so this is a no-op.
    """
    from turboquant_vllm.weight_quant import packed_group_bytes, padded_size

    n_experts, _p_out, p_in = shape
    if norms.ndim != 2 or packed.ndim != 2 or n_experts <= 0:
        return shape
    total_rows, n_groups = norms.shape
    if n_groups <= 0 or total_rows <= 0 or total_rows % n_experts != 0:
        return shape
    out_dim = total_rows // n_experts
    pgb = packed_group_bytes(bits, group_size)
    if pgb <= 0 or packed.numel() != total_rows * n_groups * pgb:
        return shape  # packed inconsistent with norms -> let from_packed raise
    # Keep the registered in_dim when it is consistent with the
    # norms-derived n_groups (preserves the exact in_dim, incl. padding,
    # for well-formed models); otherwise the registered shape is unreliable
    # (DSV4) and DSV4 MoE dims are group_size-aligned.
    if padded_size(p_in, group_size)[1] == n_groups:
        in_dim = p_in
    else:
        in_dim = n_groups * group_size
    return (n_experts, out_dim, in_dim)


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
    )
    from turboquant_vllm.weight_quant import (
        Compressed3D,
        bits_from_packed_group_bytes,
        packed_group_bytes,
        padded_size,
    )

    def _bind_real_weight_param(name: str, tensor: torch.Tensor) -> None:
        real_param = torch.nn.Parameter(tensor, requires_grad=False)
        if hasattr(layer, name):
            delattr(layer, name)
        layer.register_parameter(name, real_param)

    def _normalize_packed_layout(packed: torch.Tensor, shape: tuple[int, int, int], bits: int) -> torch.Tensor:
        n_experts, out_dim, in_dim = shape
        total_rows = n_experts * out_dim
        _, n_groups = padded_size(in_dim, method.group_size)
        pgb = packed_group_bytes(bits, method.group_size)

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

    def _backend_name() -> str:
        backend = getattr(method._unquant, "unquantized_backend", None)
        return str(getattr(backend, "name", backend))

    def _needs_w13_w31_layout() -> bool:
        return _backend_name() == "FLASHINFER_CUTLASS" and bool(getattr(layer.moe_config, "is_act_and_mul", True))

    def _swap_w13_to_w31_compressed(w13: Compressed3D) -> Compressed3D:
        n_experts, out_dim, _in_dim = w13.shape
        if out_dim % 2 != 0:
            raise ValueError(f"Cannot swap gated w13 with odd out_dim: {w13.shape}")
        half = out_dim // 2
        packed = w13.packed.reshape(n_experts, out_dim, w13.n_groups, -1)
        norms = w13.norms.reshape(n_experts, out_dim, w13.n_groups)
        packed = torch.cat((packed[:, half:], packed[:, :half]), dim=1).reshape_as(w13.packed)
        norms = torch.cat((norms[:, half:], norms[:, :half]), dim=1).reshape_as(w13.norms)
        return Compressed3D.from_packed(
            packed.contiguous(),
            norms.contiguous(),
            shape=w13.shape,
            dtype=w13.dtype,
            bits=w13.bits,
            group_size=w13.group_size,
        )

    def _load_projection(name: str) -> Compressed3D:
        packed = getattr(layer, f"{name}_tq_packed").data
        norms = getattr(layer, f"{name}_tq_norms").data
        # Per-projection bit widths: sensitive_bits checkpoints pack w2
        # (down_proj) at a different width than w13. The packed byte
        # geometry is the ground truth — pgb = packed bytes / norms entries
        # inverts packed_group_bytes bijectively over bits 2/3/4 — so the
        # decode width cannot be desynced by config-side pattern matching
        # (the method's w13_bits/w2_bits come from synthetic proj names,
        # which mismatch checkpoints whose experts are named w1/w2/w3).
        # Method attrs are only the fallback when geometry is unreadable,
        # letting the shape validators raise their precise errors.
        bits = getattr(method, f"{name[: -len('_weight')]}_bits", method.bits)
        if norms.ndim == 2 and norms.numel() > 0 and packed.numel() % norms.numel() == 0:
            derived = bits_from_packed_group_bytes(packed.numel() // norms.numel(), method.group_size)
            if derived is not None:
                if derived != bits and (name, derived, bits) not in _geometry_bits_notices:
                    # Once per (projection, widths) combo, not per layer: the
                    # benign cause — expert names like w1/w2/w3 not matching
                    # the config-side sensitive patterns — repeats identically
                    # for every MoE layer of the model. The trace still
                    # matters because the same geometry would also result
                    # from a truncated shard whose byte width happens to
                    # equal another supported width.
                    _geometry_bits_notices.add((name, derived, bits))
                    logger.warning(
                        "TurboQuant native MoE %s: packed geometry says %d-bit but config "
                        "expected %d-bit; trusting the packed data. Benign when the "
                        "checkpoint's expert names don't match the sensitive patterns; "
                        "investigate if the checkpoint may be corrupt. (Logged once.)",
                        name,
                        derived,
                        bits,
                    )
                bits = derived
        shape = _resolve_native_moe_shape(packed, norms, param_shapes[name], bits, method.group_size)
        return Compressed3D.from_packed(
            _normalize_packed_layout(packed, shape, bits),
            norms,
            shape=shape,
            dtype=param_dtypes[name],
            bits=bits,
            group_size=method.group_size,
        )

    w13_c = _load_projection("w13_weight")
    w2_c = _load_projection("w2_weight")
    if _backend_name() == "FLASHINFER_TRTLLM":
        raise NotImplementedError(
            "TurboQuant native-packed MoE does not support FlashInfer TRTLLM's "
            "block-layout BF16 backend. Use VLLM_FLASHINFER_MOE_BACKEND=throughput "
            "or moe_backend=flashinfer_cutlass/triton for native TQ3 checkpoints."
        )
    if _needs_w13_w31_layout():
        w13_c = _swap_w13_to_w31_compressed(w13_c)

    method._w13_c = w13_c
    method._w2_c = w2_c
    setattr(layer, "_tq_w13_weight", w13_c)
    setattr(layer, "_tq_w2_weight", w2_c)

    get_pool = getattr(method, "_get_moe_scratch_pool", None)
    set_pool = getattr(method, "_set_moe_scratch_pool", None)
    current_pool = get_pool() if callable(get_pool) else _shared_moe_scratch_pool
    if current_pool is None:
        current_pool = TurboQuantFusedMoEScratchPool(w13_c, w2_c)
        if callable(set_pool):
            set_pool(current_pool)
        else:
            _shared_moe_scratch_pool = current_pool
    else:
        current_pool.assert_matches(w13_c, w2_c)

    pool = current_pool
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
        layer._replace_quant_method(
            TurboQuantFusedMoEMethod(
                layer.moe_config,
                w13_c,
                w2_c,
                pool,
                base_method=method._unquant,
            )
        )

    # Full-coverage meta-tensor sweep. The earlier targeted walk (w13/w2/
    # expert_map/base_quant_method) only flagged the parameters PR #44 itself
    # rebinds. vLLM 0.20+ FlashInfer CUTLASS MoE backend reads ADDITIONAL
    # tensors (per-expert scales, packing tables, FP8 staging buffers) created
    # by the unquant's process_weights_after_loading or by `_replace_quant_method`
    # — and `run_moe`'s DLPack conversion fails with "Cannot pack tensors on meta"
    # if any of those still live on the meta device.
    #
    # Strategy: walk every parameter + buffer on the layer (and recursively on
    # sub-modules), and if any are on meta, materialize them as a zero tensor
    # on the active CUDA device. Zeros are safe because the FlashInfer path
    # uses these slots only as buffers/scales that the kernel rewrites or for
    # FP8 staging (which we don't quantize to, but vLLM allocates regardless).
    _materialize_meta_tensors(layer, label="post-finalize")

    # The materialize sweep above replaces ANY meta param with empty zeros.
    # If `_replace_quant_method` put w13_weight/w2_weight on meta, the sweep
    # just clobbered our pool binding with empty zeros. Re-bind one final time
    # so the kernel sees the live pool tensors.
    _bind_real_weight_param("w13_weight", pool.w13)
    _bind_real_weight_param("w2_weight", pool.w2)

    residual_meta = []
    residual_meta.extend(_collect_residual_meta_tensors(layer, "layer"))
    runner = getattr(layer, "runner", None)
    if runner is not None:
        residual_meta.extend(_collect_residual_meta_tensors(runner, "layer.runner"))
    moe_kernel = getattr(method._unquant, "moe_kernel", None)
    if moe_kernel is not None:
        residual_meta.extend(_collect_residual_meta_tensors(moe_kernel, "method._unquant.moe_kernel"))
    if residual_meta:
        logger.warning(
            "TurboQuant native-packed MoE finalize: residual meta tensors after materialization: %s",
            residual_meta[:30],
        )

    for name in _NATIVE_PACKED_PARAM_NAMES:
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

        def __init__(
            self,
            bits: int,
            group_size: int,
            moe_config: Any,
            native_packed: bool = False,
            scratch_pool_owner: Any | None = None,
            w13_bits: int | None = None,
            w2_bits: int | None = None,
        ):
            super().__init__(moe_config)
            self.bits = bits
            # Per-projection bit widths: sensitive_bits configs pack w2
            # (down_proj) at a different width than w13 (gate_up).
            self.w13_bits = w13_bits if w13_bits is not None else bits
            self.w2_bits = w2_bits if w2_bits is not None else bits
            self.group_size = group_size
            self.native_packed = native_packed
            self._scratch_pool_owner = scratch_pool_owner
            self._local_moe_scratch_pool = None
            self._unquant = UnquantizedFusedMoEMethod(moe_config)
            self._moe_config = moe_config
            self._pending_load = None
            self._pool = None
            self._w13_c = None
            self._w2_c = None

        def _get_moe_scratch_pool(self):
            if self._scratch_pool_owner is not None:
                return getattr(self._scratch_pool_owner, "_moe_scratch_pool", None)
            return self._local_moe_scratch_pool

        def _set_moe_scratch_pool(self, pool) -> None:
            if self._scratch_pool_owner is not None:
                setattr(self._scratch_pool_owner, "_moe_scratch_pool", pool)
            else:
                self._local_moe_scratch_pool = pool

        @property
        def supports_eplb(self) -> bool:
            return bool(getattr(self._unquant, "supports_eplb", False))

        def create_weights(self, layer: nn.Module, **kwargs):
            self._unquant.create_weights(layer, **kwargs)

            # Completion tracking counts only the uniformly-TP-sharded expert
            # weights; biases (e.g. replicated w2_bias) would break the
            # uniform scaling in _moe_expected_checkpoint_numel and leave the
            # threshold unreachable.
            tracked_numel = sum(
                p.numel() for name, p in layer.named_parameters(recurse=False) if name in _TRACKED_MOE_WEIGHT_PARAMS
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
                w13_pgb = packed_group_bytes(self.w13_bits, self.group_size)
                w2_pgb = packed_group_bytes(self.w2_bits, self.group_size)
                native_required = set(_NATIVE_PACKED_PARAM_NAMES)
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
                    (num_experts * w13_out_dim, w13_groups * w13_pgb),
                    torch.uint8,
                )
                _register_native_packed_param(
                    "w13_weight_tq_norms",
                    (num_experts * w13_out_dim, w13_groups),
                    torch.float32,
                )
                _register_native_packed_param(
                    "w2_weight_tq_packed",
                    (num_experts * w2_out_dim, w2_groups * w2_pgb),
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
            #
            # Completion is measured in CHECKPOINT-side numel: loader calls
            # carry full (unsharded, global-expert) tensors, so comparing
            # against the partition-side param sum would fire the compress
            # trigger at ~1/tp of the load under TP/EP and freeze partially
            # loaded experts. When the parallel factors can't be derived,
            # the threshold stays disabled and process_weights_after_loading
            # replays the buffer instead (correct, but holds this layer's
            # checkpoint shards in host memory until the load finishes).
            expected_numel = (
                _moe_expected_checkpoint_numel(layer, self._moe_config, tracked_numel) if tracked_numel > 0 else None
            )
            if expected_numel is None:
                logger.warning(
                    "TurboQuant online MoE: could not derive TP/EP factors for %s; "
                    "deferring compression to process_weights_after_loading",
                    type(layer).__name__,
                )

            buffer: list[tuple[str, tuple, dict]] = []
            loaded_numel = [0]
            materialized = [False]
            pending_state = {
                "buffer": buffer,
                "orig_loaders": orig_loaders,
                "param_shapes": param_shapes,
                "param_dtypes": param_dtypes,
                "materialized": materialized,
            }
            self._pending_load = pending_state

            def _make_buffering_loader(param_name, orig_loader):
                counted = param_name in _TRACKED_MOE_WEIGHT_PARAMS

                def _buffering_loader(*args, **kwargs):
                    if materialized[0]:
                        # Loads can legitimately arrive after the threshold
                        # fired (untracked biases ordered after the last
                        # expert weight). vLLM's caller may still hold the
                        # pre-materialization meta param (params_dict is
                        # built once per load pass), so route the write into
                        # the live registered param. args[0] is the param in
                        # every call shape; loaded_weight may be positional
                        # or a kwarg.
                        current = getattr(layer, param_name, None)
                        if current is not None and args:
                            return orig_loader(current, *args[1:], **kwargs)
                        return orig_loader(*args, **kwargs)
                    buffer.append((param_name, args, kwargs))
                    if counted:
                        loaded_weight = args[1] if len(args) > 1 else kwargs.get("loaded_weight")
                        if isinstance(loaded_weight, torch.Tensor):
                            loaded_numel[0] += loaded_weight.numel()
                    if expected_numel is not None and loaded_numel[0] >= expected_numel:
                        materialized[0] = True
                        _materialize_and_process(layer, pending_state, self)
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
            from turboquant_vllm.moe_quant import TurboQuantFusedMoEScratchPool
            from turboquant_vllm.weight_quant import _compress_3d_param

            self._unquant.process_weights_after_loading(layer)

            w13 = getattr(layer, "w13_weight", None)
            w2 = getattr(layer, "w2_weight", None)
            if w13 is None or w2 is None or w13.dim() != 3 or w2.dim() != 3:
                return

            _compress_3d_param(layer, "w13_weight", self.w13_bits, self.group_size)
            _compress_3d_param(layer, "w2_weight", self.w2_bits, self.group_size)

            self._w13_c = layer._tq_w13_weight
            self._w2_c = layer._tq_w2_weight

            shared_pool = self._get_moe_scratch_pool()
            if shared_pool is None:
                shared_pool = TurboQuantFusedMoEScratchPool(
                    self._w13_c,
                    self._w2_c,
                )
                self._set_moe_scratch_pool(shared_pool)
            else:
                shared_pool.assert_matches(
                    self._w13_c,
                    self._w2_c,
                )

            self._pool = shared_pool
            layer.w13_weight.data = self._pool.w13
            layer.w2_weight.data = self._pool.w2

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        def process_weights_after_loading(self, layer: nn.Module) -> None:
            if self.native_packed:
                if not hasattr(layer, "_tq_w13_weight"):
                    # Why: nn.Module._apply does not walk Compressed3D's
                    # internal tensors (stored via setattr), so a meta leak
                    # at this entry point becomes invisible until decode.
                    meta_placeholders = [
                        n for n in _NATIVE_PACKED_PARAM_NAMES if hasattr(layer, n) and getattr(layer, n).is_meta
                    ]
                    if meta_placeholders:
                        raise RuntimeError(
                            f"TQ3 native packed placeholders still on meta after "
                            f"load: {meta_placeholders}. Regroup did not populate them."
                        )
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
                # Not yet compressed — replay any buffered loads and run
                # compression now. `numel() > 0` alone is NOT a valid check
                # here: meta tensors report their full shape, so compressing
                # without the replay would bake uninitialized data.
                _finish_online_moe_load(layer, self, self._pending_load)

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
# Qwen3.6-style native-packed checkpoints store per-layer experts pre-fused
# on disk: `.experts.gate_up_proj.tq_packed` (gate+up stacked across all
# experts) and `.experts.down_proj.tq_packed`. No per-expert index. The
# optional `.weight` covers authors who saved with the suffix preserved.
# Bare `w13`/`w2` aliases are safe in this path because pre-fused names
# never carry an expert index (no collision with per-expert `experts.0.w2`).
_NATIVE_MOE_PRE_FUSED_PATTERN = re.compile(
    r"^(.+?\.experts)\.(gate_up_proj|down_proj|w13_weight|w2_weight|w13|w2)(?:\.weight)?$"
)
_NATIVE_MOE_PRE_FUSED_TO_TARGET = {
    "gate_up_proj": "w13_weight",
    "down_proj": "w2_weight",
    "w13_weight": "w13_weight",
    "w2_weight": "w2_weight",
    "w13": "w13_weight",
    "w2": "w2_weight",
}


def _try_pre_fused_rename(base: str) -> str | None:
    """Return the placeholder target path for an already-fused expert base.

    Returns None when base doesn't match the pre-fused pattern (caller must
    fall through to per-expert regroup).
    """
    m = _NATIVE_MOE_PRE_FUSED_PATTERN.match(base)
    if not m:
        return None
    target = _NATIVE_MOE_PRE_FUSED_TO_TARGET.get(m.group(2))
    if target is None:
        return None
    return f"{m.group(1)}.{target}"


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
            widths = {p.shape[-1] for p in packed_parts}
            if len(widths) > 1:
                # Same invariant as _maybe_flush_native_moe_target: one fused
                # target stores everything at one bit width.
                raise ValueError(
                    f"Native MoE regroup for {target_key}: expert {expert_idx} projection "
                    f"halves have mismatched packed widths {sorted(widths)} — gate/up must "
                    "be packed at the same bit width."
                )
            all_packed.append(torch.cat(packed_parts, dim=0) if len(packed_parts) > 1 else packed_parts[0])
            all_norms.append(torch.cat(norm_parts, dim=0) if len(norm_parts) > 1 else norm_parts[0])

        expert_widths = {t.shape[-1] for t in all_packed}
        if len(expert_widths) > 1:
            raise ValueError(
                f"Native MoE regroup for {target_key}: experts have mismatched packed "
                f"widths {sorted(expert_widths)} — all experts of a fused projection "
                "must be packed at the same bit width."
            )
        direct_targets.append((f"{target_key}_tq_packed", torch.cat(all_packed, dim=0)))
        direct_targets.append((f"{target_key}_tq_norms", torch.cat(all_norms, dim=0)))

    return direct_targets


_FLUSH_DEBUG_LIMIT = 5
_flush_debug_count = {"no_regex": 0, "unknown_proj": 0, "no_target": 0, "wrong_shape": 0}

# (projection, derived_bits, config_bits) combos already warned about in
# _finalize_native_packed_moe — the benign cause repeats per layer.
_geometry_bits_notices: set[tuple[str, int, int]] = set()


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
        if _flush_debug_count["no_regex"] < _FLUSH_DEBUG_LIMIT:
            _flush_debug_count["no_regex"] += 1
            logger.warning("regroup miss [no_regex]: base_name=%r", base_name)
        return []

    container_path = match.group(1) + ".experts"
    expert_idx = int(match.group(2))
    proj_suffix = match.group(3)
    proj_name = proj_suffix.split(".")[0]
    target_name = _NATIVE_MOE_PROJ_FUSION.get(proj_name)
    if target_name is None:
        if _flush_debug_count["unknown_proj"] < _FLUSH_DEBUG_LIMIT:
            _flush_debug_count["unknown_proj"] += 1
            logger.warning(
                "regroup miss [unknown_proj]: proj_name=%r base_name=%r (known: %s)",
                proj_name,
                base_name,
                list(_NATIVE_MOE_PROJ_FUSION.keys()),
            )
        return []

    target_key = f"{container_path}.{target_name}"
    meta_entry = meta_params.get(target_key)
    if meta_entry is None:
        if _flush_debug_count["no_target"] < _FLUSH_DEBUG_LIMIT:
            _flush_debug_count["no_target"] += 1
            sample_experts_keys = [k for k in meta_params if ".experts." in k][:5]
            logger.warning(
                "regroup miss [no_target]: looking for %r, not in meta_params. Sample meta_params .experts. keys: %s",
                target_key,
                sample_experts_keys,
            )
        return []

    _, _, meta_param = meta_entry
    if len(meta_param.shape) != 3:
        if _flush_debug_count["wrong_shape"] < _FLUSH_DEBUG_LIMIT:
            _flush_debug_count["wrong_shape"] += 1
            logger.warning(
                "regroup miss [wrong_shape]: target=%r expected 3D, got shape=%s",
                target_key,
                tuple(meta_param.shape),
            )
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
        widths = {p.shape[-1] for p in packed_parts}
        if len(widths) > 1:
            # A fused w13 target stores gate and up as one packed tensor, so
            # both halves must share one bit width. Mixed widths (e.g. a
            # custom sensitive_patterns entry matching only up_proj at save
            # time) would otherwise die in torch.cat with an error naming no
            # tensor.
            raise ValueError(
                f"Native MoE regroup for {target_key}: expert {idx} projection halves "
                f"have mismatched packed widths {sorted(widths)}. Gate/up (w1/w3) must be "
                "packed at the same bit width — sensitive_patterns matching only one half "
                "of a fused projection cannot be represented in the fused layout."
            )
        all_packed.append(torch.cat(packed_parts, dim=0) if len(packed_parts) > 1 else packed_parts[0])
        all_norms.append(torch.cat(norm_parts, dim=0) if len(norm_parts) > 1 else norm_parts[0])

    del target_state[target_key]
    expert_widths = {t.shape[-1] for t in all_packed}
    if len(expert_widths) > 1:
        # Same invariant as the per-expert halves check above, across
        # experts: one fused target stores every expert at one bit width.
        raise ValueError(
            f"Native MoE regroup for {target_key}: experts have mismatched packed "
            f"widths {sorted(expert_widths)} — all experts of a fused projection must "
            "be packed at the same bit width (check custom sensitive_patterns that "
            "match only a subset of experts)."
        )
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

        from turboquant_vllm.weight_quant import select_bits

        with open(tq_config_path) as f:
            tq_cfg = _json.load(f)
        bits = tq_cfg.get("bits", 3)
        group_size = tq_cfg.get("group_size", 128)
        # Sensitive tensors (o_proj/down_proj by default) are packed at
        # sensitive_bits by save_tq3_checkpoint — decoding them at the
        # uniform width reads the packed bytes with the wrong layout.
        sensitive_bits = tq_cfg.get("sensitive_bits")
        sensitive_patterns = _normalize_sensitive_patterns(tq_cfg.get("sensitive_patterns"), sensitive_bits)
        # True in_dim for weights whose width was padded to a multiple of
        # group_size at pack time (keyed by raw on-disk tensor name).
        orig_in_dims = tq_cfg.get("orig_in_dims") or {}
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
        seen_native_moe_disk = 0
        skipped_fp8_scales = 0

        # Why: _collect_meta_params keys by post-mapper names from
        # named_parameters(), but raw_name from disk is pre-mapper. Without
        # this step, multimodal MoE (Qwen3-VL) silently drops every
        # per-expert tensor because the regroup target lookup never matches.
        mapper = getattr(model, "hf_to_vllm_mapper", None)
        if native_packed:
            experts_meta_keys = [k for k in moe_meta_params if ".experts." in k]
            logger.info(
                "TQ3 native regroup setup: mapper=%s, meta_params has %d keys total, %d with .experts. (sample: %s)",
                "yes" if mapper is not None else "no",
                len(moe_meta_params),
                len(experts_meta_keys),
                experts_meta_keys[:5],
            )

        def _map(n: str) -> str | None:
            if mapper is None:
                return n
            return mapper._map_name(n)

        for raw_name, tensor in _original_get_all_weights(self, model_config, model):
            name = _map(raw_name)
            if name is None:
                continue
            # The regroup interception only applies to native-packed MoE
            # checkpoints (meta placeholder params exist). Without that
            # format, a regroup lookup can never succeed — the pairs would
            # be popped and silently dropped — so per-expert tensors fall
            # through to the dense decompress branch instead.
            if native_packed and name.endswith(".tq_packed") and ".experts." in name:
                seen_native_moe_disk += 1
                base = name[: -len(".tq_packed")]
                pre_fused = _try_pre_fused_rename(base)
                if pre_fused is not None:
                    yielded_native_moe += 1
                    yield f"{pre_fused}_tq_packed", tensor
                    continue
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
            elif native_packed and name.endswith(".tq_norms") and ".experts." in name:
                seen_native_moe_disk += 1
                base = name[: -len(".tq_norms")]
                pre_fused = _try_pre_fused_rename(base)
                if pre_fused is not None:
                    yielded_native_moe += 1
                    yield f"{pre_fused}_tq_norms", tensor
                    continue
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
                # Key (and later yield) by the RAW name: like the
                # pass-through branch below, decompressed weights go through
                # AutoWeightsLoader, which applies hf_to_vllm_mapper itself —
                # yielding the mapped name here would map them twice.
                base = raw_name[: -len(".tq_packed")]
                pending_packed[base] = tensor
            elif name.endswith(".weight.tq_norms"):
                base = raw_name[: -len(".tq_norms")]
                pending_norms[base] = tensor
            elif name.endswith(_FP8_LEFTOVER_SCALE_SUFFIXES):
                skipped_fp8_scales += 1
                continue
            else:
                # Yield the RAW (pre-mapper) name. vLLM's AutoWeightsLoader
                # applies hf_to_vllm_mapper once itself; yielding the mapped
                # `name` here makes it map twice, which corrupts non-idempotent
                # rules (e.g. DSV4 head.weight -> lm_head.weight -> lm_lm_head).
                yield raw_name, tensor
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
                    select_bits(base, bits, sensitive_bits, sensitive_patterns),
                    group_size,
                )
                w = comp.decompress().squeeze(0)
                # n_groups only preserves the padded width; restore the true
                # in_dim recorded at pack time (standalone loaders truncate
                # against the model's param shapes instead). A strided view
                # suffices — vLLM's weight loaders copy_ from the source.
                # Bound the value: a corrupt/hand-edited entry (e.g. negative)
                # would otherwise silently tail-truncate via slice semantics.
                true_in = orig_in_dims.get(base)
                if true_in is not None and 0 < true_in < w.shape[1]:
                    w = w[:, :true_in]
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
        if native_packed:
            logger.info(
                "TQ3 native MoE regroup: saw %d per-expert tensors from disk, yielded %d fused MoE targets",
                seen_native_moe_disk,
                yielded_native_moe,
            )
            if seen_native_moe_disk > 0 and yielded_native_moe == 0:
                raise RuntimeError(
                    f"TQ3 native MoE regroup yielded zero fused targets from "
                    f"{seen_native_moe_disk} per-expert tensors. "
                    f"Likely cause: hf_to_vllm_mapper mismatch with _collect_meta_params keys."
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
