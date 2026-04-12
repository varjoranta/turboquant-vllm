"""TurboQuant MoE expert weight quantization via FusedMoE quant method swap.

``apply()`` runs inside ``torch.ops.vllm.moe_forward`` (a PT2 custom op
with a fake impl), so the pybind11 dequant call and arbitrary Python
are safe there — dynamo never traces into the body. CUDA dequant
kernels launch on PyTorch's current stream via
``c10::cuda::getCurrentCUDAStream``, so CUDA graph capture works.
Installed by ``weight_quant._replace_linear_layers`` Phase 2A.
"""

from __future__ import annotations

import torch

try:
    from vllm.model_executor.custom_op import CustomOp
    from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
        FusedMoEMethodBase,
    )

    _HAS_FUSED_MOE = True
except ImportError:
    # Dev/CI machines without vLLM still need to import this module
    # for CPU tests that monkey-patch the FusedMoE surface. Stub both
    # bases to a single placeholder so the multi-inheritance class
    # declaration below doesn't trip Python's "duplicate base class"
    # check (otherwise both would resolve to ``object``).
    class _StubFusedMoEMethodBase:
        def __init__(self, moe_config):
            self.moe = moe_config

    class _StubCustomOp:
        def __init__(self):
            pass

    FusedMoEMethodBase = _StubFusedMoEMethodBase  # type: ignore[misc,assignment]
    CustomOp = _StubCustomOp  # type: ignore[misc,assignment]
    _HAS_FUSED_MOE = False


class TurboQuantFusedMoEScratchPool:
    """Shared dequant scratch buffers for all FusedMoE layers in a model.

    Only one MoE layer runs at a time during forward, so one set of
    bf16 destinations + fp32 intermediates is enough regardless of
    layer count. Per-layer scratch would hold the uncompressed expert
    weights on the side for every layer and defeat the compression.
    All FusedMoE layers in a model are assumed to share the same
    expert shapes; ``assert_matches`` enforces this.
    """

    __slots__ = ("w13", "w2", "w13_fp32", "w2_fp32", "shape_w13", "shape_w2")

    def __init__(self, w13_compressed, w2_compressed):
        device = w13_compressed.packed.device
        bf16_dtype = w13_compressed.dtype
        self.w13 = torch.empty(w13_compressed.shape, dtype=bf16_dtype, device=device)
        self.w2 = torch.empty(w2_compressed.shape, dtype=bf16_dtype, device=device)
        self.w13_fp32 = torch.empty(w13_compressed.shape, dtype=torch.float32, device=device)
        self.w2_fp32 = torch.empty(w2_compressed.shape, dtype=torch.float32, device=device)
        self.shape_w13 = w13_compressed.shape
        self.shape_w2 = w2_compressed.shape

    def assert_matches(self, w13_compressed, w2_compressed) -> None:
        """Raise if a later FusedMoE layer has a different expert shape.

        All layers of a typical MoE model share the same expert shapes,
        but the walker calls this on every layer past the first to
        catch heterogeneous models early rather than silently producing
        wrong outputs from a mismatched buffer.
        """
        assert w13_compressed.shape == self.shape_w13, (
            f"heterogeneous FusedMoE layer shapes: scratch pool sized "
            f"{tuple(self.shape_w13)} but layer needs {tuple(w13_compressed.shape)}"
        )
        assert w2_compressed.shape == self.shape_w2, (
            f"heterogeneous FusedMoE layer shapes: scratch pool sized "
            f"{tuple(self.shape_w2)} but layer needs {tuple(w2_compressed.shape)}"
        )


# vLLM's CustomOp registry uses the decorator name as a key for
# enable/disable tracking via compilation_config.disabled_custom_ops.
# We need to register so CustomOp.__init__ can read self.__class__.name.
if _HAS_FUSED_MOE:
    _register_custom_op = CustomOp.register("turboquant_fused_moe")
else:
    def _register_custom_op(cls):
        return cls


@_register_custom_op
class TurboQuantFusedMoEMethod(FusedMoEMethodBase, CustomOp):
    """FusedMoE quant method that dequantizes TQ3-packed experts inside apply().

    ``apply()`` decompresses both expert tensors into the shared scratch
    pool and delegates to ``layer.base_quant_method.apply()`` so the
    kernel path is identical to the unquantized BF16 forward.

    Multi-inherits ``CustomOp`` because ``FusedMoE._replace_quant_method``
    reassigns ``self.quant_method`` on an ``nn.Module`` — torch's
    ``Module.__setattr__`` only accepts ``nn.Module`` in that slot, and
    ``FusedMoEMethodBase`` alone isn't one. The
    ``@CustomOp.register("turboquant_fused_moe")`` decorator gives us the
    ``.name`` attribute ``CustomOp.__init__`` reads during dispatch.
    """

    def __init__(
        self,
        moe_config,
        w13_compressed,
        w2_compressed,
        scratch_pool: TurboQuantFusedMoEScratchPool,
    ):
        if not _HAS_FUSED_MOE:
            raise RuntimeError(
                "TurboQuantFusedMoEMethod requires vllm.model_executor.layers."
                "fused_moe — import failed at module load."
            )
        # Both bases must be initialized: FusedMoEMethodBase needs
        # moe_config; CustomOp needs to set up its nn.Module state.
        FusedMoEMethodBase.__init__(self, moe_config)
        CustomOp.__init__(self)

        self._w13 = w13_compressed
        self._w2 = w2_compressed
        self._pool = scratch_pool

    # ------------------------------------------------------------------
    # Abstract methods from FusedMoEMethodBase
    # ------------------------------------------------------------------

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        # Installed via _replace_quant_method after the original
        # UnquantizedFusedMoEMethod has already loaded bf16 parameters.
        # create_weights is never called on us; loud failure if it is.
        raise NotImplementedError(
            "TurboQuantFusedMoEMethod is installed via "
            "FusedMoE._replace_quant_method after weight loading, not as a "
            "create_weights dispatch target. See "
            "turboquant_vllm.weight_quant._replace_linear_layers Phase 2A."
        )

    def get_fused_moe_quant_config(self, layer: torch.nn.Module):
        # We dequantize to bf16 inside apply() and delegate to the
        # base UnquantizedFusedMoEMethod; there is no FusedMoEQuantConfig
        # to plumb through the kernel.
        return None

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts_input: torch.Tensor | None = None,
    ):
        # ``layer.w13_weight.data`` / ``layer.w2_weight.data`` were
        # re-pointed at ``pool.w13`` / ``pool.w2`` at install time, so
        # writing into the pool buffers here makes the freshly
        # dequantized values visible to the base unquantized kernel.
        pool = self._pool
        self._w13.decompress_into(pool.w13, fp32_scratch=pool.w13_fp32)
        self._w2.decompress_into(pool.w2, fp32_scratch=pool.w2_fp32)

        return layer.base_quant_method.apply(
            layer=layer,
            x=x,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            shared_experts_input=shared_experts_input,
        )
