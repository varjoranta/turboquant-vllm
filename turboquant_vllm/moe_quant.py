"""TurboQuant MoE expert weight quantization via FusedMoE quant method swap.

Installs a ``FusedMoEMethodBase`` subclass on each ``FusedMoE`` layer so
its ``apply()`` call decompresses TQ3-packed expert weights on the fly
and hands the resulting bf16 tensors to vLLM's standard ``fused_experts``
path. This replaces the older ``register_forward_pre_hook`` pattern in
``weight_quant.py`` which crashed on vLLM 0.19 because dynamo tried to
trace through the pybind11 dequant call.

The key insight: ``FusedMoE.forward`` delegates to
``DefaultMoERunner.forward`` which calls ``torch.ops.vllm.moe_forward``,
a PT2 custom op with a fake impl. The entire MoE body — including
``quant_method.apply()`` — is opaque to dynamo, so raw pybind11 calls
and arbitrary Python control flow are safe inside ``apply()``.

Installation sequence (see ``weight_quant._replace_linear_layers``
Phase 2A for the caller):

1. Walk the model, find each ``FusedMoE`` layer
2. Compress ``layer.w13_weight`` and ``layer.w2_weight`` into
   ``Compressed3D`` objects via ``_compress_3d_param``. This attaches
   the objects as ``layer._tq_w13_weight`` and ``layer._tq_w2_weight``
   and frees the bf16 originals.
3. Call ``layer._replace_quant_method(TurboQuantFusedMoEMethod(scratch))``
   which swaps ``layer.quant_method`` AND re-inits the runner so the
   runner's captured reference points at our new method.

**Scratch pool sharing**: only one MoE layer runs at a time during
forward, so one pair of bf16 scratch buffers is sufficient for the
whole model regardless of layer count. Allocating per-layer scratch
would wipe out the compression win — for Qwen3-30B-A3B that's ~54 GB
of scratch (= the entire uncompressed expert weight footprint).
``TurboQuantFusedMoEScratchPool`` allocates one shared pair on first
use and all method instances dispatch through it.

See varjoranta/turboquant-vllm#14 for the bug report this fixes.
"""

from __future__ import annotations

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)


try:
    from vllm.model_executor.custom_op import CustomOp
    from vllm.model_executor.layers.fused_moe import fused_experts
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
    fused_experts = None  # type: ignore[assignment]
    _HAS_FUSED_MOE = False


class TurboQuantFusedMoEScratchPool:
    """Single pair of bf16 scratch buffers shared across all FusedMoE layers.

    Only one MoE layer runs at a time during a forward pass, so a single
    pair of ``(w13, w2)`` scratch buffers is enough for the entire model.
    Allocating per-layer scratch is the easy mistake: for a 48-layer MoE
    that ends up holding the uncompressed expert weights in bf16 on the
    side, defeating the entire point of compression.

    All FusedMoE layers in a given model are assumed to share the same
    expert-weight shapes (same ``num_experts``, same ``hidden_size``,
    same ``intermediate_size_per_partition``). Layers with mismatched
    shapes trigger an assertion, not silent corruption.
    """

    __slots__ = ("w13", "w2")

    def __init__(self):
        self.w13: torch.Tensor | None = None
        self.w2: torch.Tensor | None = None

    def ensure(
        self,
        attr: str,
        shape: torch.Size,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        buf = getattr(self, attr)
        if buf is None:
            buf = torch.empty(shape, dtype=dtype, device=device)
            setattr(self, attr, buf)
            return buf
        assert buf.shape == shape, (
            f"heterogeneous FusedMoE layer shapes: scratch was sized {tuple(buf.shape)} "
            f"but a later layer needs {tuple(shape)}. Per-layer scratch not supported."
        )
        assert buf.dtype == dtype and buf.device == device
        return buf


class TurboQuantFusedMoEMethod(FusedMoEMethodBase, CustomOp):
    """FusedMoE quant method that dequantizes TQ3-packed experts inside apply.

    The compressed expert weights are attached to the layer by the runtime
    quantization walker in ``weight_quant._replace_linear_layers`` BEFORE
    this method is installed via ``layer._replace_quant_method``. We read
    them from the layer at forward time, dequantize into the shared
    scratch pool, and hand the result to ``fused_experts``.

    **Why the multi-inheritance**: ``FusedMoE._replace_quant_method`` sets
    ``self.quant_method = mk`` (``layer.py:604``). Because ``FusedMoE`` is
    an ``nn.Module`` and ``quant_method`` was registered as a child module
    by the original ``UnquantizedFusedMoEMethod`` (which itself inherits
    from ``CustomOp`` -> ``nn.Module``), torch's ``Module.__setattr__``
    enforces that the new value also be an ``nn.Module``. Inheriting from
    ``CustomOp`` gives us the ``nn.Module`` MRO via the same path the
    in-tree quant methods use.
    """

    def __init__(self, moe_config, scratch_pool: TurboQuantFusedMoEScratchPool):
        if not _HAS_FUSED_MOE:
            raise RuntimeError(
                "TurboQuantFusedMoEMethod requires vllm.model_executor.layers."
                "fused_moe — import failed at module load."
            )
        # Both bases must be initialized: FusedMoEMethodBase needs
        # moe_config; CustomOp needs to set up its nn.Module state.
        FusedMoEMethodBase.__init__(self, moe_config)
        CustomOp.__init__(self)
        self._scratch_pool = scratch_pool

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
        # We dequantize to bf16 inside apply() and pass the result to
        # fused_experts with quant_config=None (the unquantized path).
        return None

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts_input: torch.Tensor | None = None,
    ):
        """Decompress-on-the-fly expert forward.

        Runs inside ``torch.ops.vllm.moe_forward`` which is opaque to
        dynamo, so the pybind11 ``weight_dequant_3d`` call inside
        ``decompress_into`` is safe here.
        """
        w13_compressed = layer._tq_w13_weight
        w2_compressed = layer._tq_w2_weight

        w13_buf = self._scratch_pool.ensure(
            "w13",
            w13_compressed.shape,
            w13_compressed.dtype,
            w13_compressed.packed.device,
        )
        w2_buf = self._scratch_pool.ensure(
            "w2",
            w2_compressed.shape,
            w2_compressed.dtype,
            w2_compressed.packed.device,
        )

        w13_compressed.decompress_into(w13_buf)
        w2_compressed.decompress_into(w2_buf)

        return fused_experts(
            x,
            w13_buf,
            w2_buf,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            inplace=not self.moe.disable_inplace,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
            global_num_experts=layer.global_num_experts,
            expert_map=layer.expert_map,
            quant_config=None,
        )
