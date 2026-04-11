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
    """Shared scratch buffers for all FusedMoE layers.

    Only one MoE layer runs at a time during forward, so a single set
    of scratch buffers is enough regardless of layer count. Per-layer
    scratch is the easy mistake: for a 48-layer MoE that ends up
    holding the uncompressed expert weights on the side, defeating the
    compression entirely.

    The pool holds five slots:

    - ``w13`` / ``w2`` — bf16 destination for the dequantized expert
      tensors, passed to ``fused_experts`` / the base unquantized
      kernel.
    - ``w13_fp32`` / ``w2_fp32`` — fp32 intermediate for the CUDA
      dequant kernel (which only produces fp16 or fp32). Without
      pooling, ``decompress_into`` would allocate these per-call,
      which under CUDA graph capture bakes ~3 GB of ephemeral memory
      into each captured piecewise graph and easily blows the KV-cache
      budget on Qwen3-30B-A3B.

    All FusedMoE layers in a given model are assumed to share the same
    expert-weight shapes. Mismatched shapes trigger an assertion, not
    silent corruption.
    """

    __slots__ = ("w13", "w2", "w13_fp32", "w2_fp32")

    def __init__(self):
        self.w13: torch.Tensor | None = None
        self.w2: torch.Tensor | None = None
        self.w13_fp32: torch.Tensor | None = None
        self.w2_fp32: torch.Tensor | None = None

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

        **Delegation strategy**: populate ``layer.w13_weight.data`` /
        ``layer.w2_weight.data`` with decompressed bf16 scratch views
        pointing at the shared scratch pool, then delegate to the
        original ``UnquantizedFusedMoEMethod`` captured as
        ``layer.base_quant_method`` during ``FusedMoE.__init__``. This
        walks the exact same code path the unquantized BF16 benchmark
        runs — including ``self.kernel.apply``, activation, and
        routing specialization — which avoids having to reimplement
        any of it here.

        The scratch pool allocates ONE pair of bf16 buffers shared by
        all FusedMoE layers (only one layer runs at a time during
        forward, so one pair is sufficient). We re-point
        ``layer.w13_weight.data`` / ``layer.w2_weight.data`` at the
        scratch buffers before each delegation call.
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
        w13_fp32 = self._scratch_pool.w13_fp32
        w2_fp32 = self._scratch_pool.w2_fp32

        # Dequantize into the shared scratch pool. ``layer.w13_weight.data``
        # and ``layer.w2_weight.data`` were permanently re-pointed at
        # these same scratch buffers at install time, so the base
        # unquantized method will read the freshly dequantized values.
        #
        # **Known limitation**: this design requires ``--enforce-eager``.
        # Under vLLM CUDA graph capture, all FusedMoE layers land in
        # the same captured piece and each layer's dequant + base_method
        # call writes to / reads from the same shared scratch address.
        # The graph's write-after-read dependency tracking doesn't
        # reliably serialize across layers, and the empirical result
        # is correct eager output but gibberish under capture (tested
        # on Qwen3-30B-A3B 2026-04-11). Cloning dequant output per
        # layer avoids the aliasing but OOMs the captured graph's
        # private pool (each clone gets committed, ~1.15 GB × 48
        # layers). The proper fix is a custom fused MoE GEMM that
        # does dequant-inside-kernel, eliminating the scratch — see
        # moe_wna16.py as a reference implementation. Until that
        # lands, the walker in ``_replace_linear_layers`` sets
        # ``enforce_eager=True`` on the vLLM config when it detects
        # any FusedMoE layer has been compressed.
        w13_compressed.decompress_into(w13_buf, fp32_scratch=w13_fp32)
        w2_compressed.decompress_into(w2_buf, fp32_scratch=w2_fp32)

        return layer.base_quant_method.apply(
            layer=layer,
            x=x,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            shared_experts_input=shared_experts_input,
        )
