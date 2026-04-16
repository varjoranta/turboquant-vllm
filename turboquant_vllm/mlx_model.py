"""MLX model-side wrapper for TurboQuant weight compression.

Provides ``TurboQuantMLXLinear`` — a drop-in replacement for
``mlx.nn.Linear`` that stores packed TQ3 weights and dequantizes on
every forward via the primitives in ``mlx_ops``. This is the last
piece needed to serve a TQ3 checkpoint through ``mlx-lm`` on Apple
Silicon without falling back to the CPU path.

v1 scope (this module):
  - Per-Linear wrapper, forward-only
  - Accepts packed weights + shape-gain norms loaded from a TQ3
    checkpoint shard (see ``load_tq3_weights_into_linear``)
  - Shares one ``PolarQuantStateMLX`` instance per (group_size, bits)
    tuple across the whole model

Next (Phase 5):
  - Loader that walks an ``mlx_lm`` model architecture and replaces
    each ``nn.Linear`` with ``TurboQuantMLXLinear`` whose weights come
    from our native TQ3 safetensors shards
  - Integration with ``mlx_lm.server``
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from turboquant_vllm.mlx_ops import (
    PolarQuantStateMLX,
    fwht_on_input_matmul_mlx,
    unpack_indices_3bit_mlx,
)
from turboquant_vllm.weight_quant import padded_size


class TurboQuantMLXLinear(nn.Module):
    """MLX Linear that holds packed TQ3 weights and dequantizes on forward.

    The dequant path is:
        packed uint8 -> unpack_indices_3bit -> codebook lookup ->
        inverse WHT -> shape-gain scale -> matmul with the activation.

    Per-forward dequant is the v1 design (same as the PyTorch CPU path);
    a fused Metal dequant-GEMM kernel is a Phase 6 follow-up once quality
    parity is established.

    Args:
        packed_weight: uint8 array of shape ``(out_features, k_packed)``
            where ``k_packed = (padded_in // 8) * 3`` for 3-bit.
        norms: float32 array of shape ``(out_features, n_groups)``
            with shape-gain scales (original_norm / reconstruction_norm).
        state: shared ``PolarQuantStateMLX`` (dim == group_size).
        in_features: original input dim before padding.
        out_features: output dim.
        bias: optional bias array of shape ``(out_features,)``.
    """

    def __init__(
        self,
        packed_weight: mx.array,
        norms: mx.array,
        state: PolarQuantStateMLX,
        in_features: int,
        out_features: int,
        bias: mx.array | None = None,
    ):
        super().__init__()
        self.norms = norms
        self.quant_state = state
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias

        self.group_size = state.dim
        self.padded_in, self.n_groups = padded_size(in_features, self.group_size)
        self._pad_needed = self.padded_in > in_features

        # Unpack the packed uint8 weights once at init — indices are immutable
        # and repeated unpacking on the hot path dominated early profiling.
        # Keep packed_weight too so introspection / reload paths can see it.
        self.packed_weight = packed_weight
        self._indices_grouped = unpack_indices_3bit_mlx(packed_weight, dim=self.padded_in).reshape(
            self.out_features * self.n_groups, self.group_size
        )

    def __call__(self, x: mx.array) -> mx.array:
        # fwht_on_input_matmul_mlx expects 2D (batch, in_features); flatten
        # leading dims for 3D/4D token-streaming inputs and reshape on the
        # way out.
        orig_shape = x.shape
        x_flat = x.reshape(-1, orig_shape[-1]) if x.ndim > 2 else x

        out_flat = fwht_on_input_matmul_mlx(
            x=x_flat,
            indices_grouped=self._indices_grouped,
            norms=self.norms,
            state=self.quant_state,
            bias=self.bias,
            output_dtype=x.dtype,
        )

        if x.ndim > 2:
            return out_flat.reshape(*orig_shape[:-1], self.out_features)
        return out_flat


class TurboQuantMLXSwitchLinear(nn.Module):
    """TQ3 drop-in for ``mlx_lm.models.switch_layers.SwitchLinear``.

    Stores ``num_experts`` packed TQ3 experts as a single tensor and on
    every forward dequantises them via the FWHT-on-input codebook-lookup
    path, then calls ``mx.gather_mm`` (the same primitive SwitchLinear
    uses) to route tokens to their selected experts.

    Full dequant of all experts per forward, not just the active ones:
    it's the simplest correct implementation and fits comfortably into
    unified memory for the model sizes we care about on Apple Silicon.
    Active-only dequant is a follow-up optimisation.

    Args:
        packed_weight: uint8 ``(num_experts * out_features * n_groups,
            bytes_per_group)`` — the TQ3 native MoE layout where
            ``bytes_per_group = group_size * bits / 8``.
        norms: float32 ``(num_experts * out_features, n_groups)``
            shape-gain scales.
        state: shared ``PolarQuantStateMLX``.
        in_features: original input dim before padding.
        out_features: per-expert output dim.
        num_experts: number of experts in the layer.
        bias: optional ``(num_experts, out_features)`` bias, or ``None``.
    """

    def __init__(
        self,
        packed_weight: mx.array,
        norms: mx.array,
        state: PolarQuantStateMLX,
        in_features: int,
        out_features: int,
        num_experts: int,
        bias: mx.array | None = None,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.in_features = in_features
        self.out_features = out_features
        self.quant_state = state
        self.bias = bias

        self.group_size = state.dim
        self.padded_in, self.n_groups = padded_size(in_features, self.group_size)
        self._pad_needed = self.padded_in > in_features

        self.packed_weight = packed_weight
        self.norms = norms.reshape(num_experts, out_features, self.n_groups)

        # Pre-unpack once. Shape (num_experts, out_features, n_groups, group_size).
        unpacked = unpack_indices_3bit_mlx(packed_weight, dim=self.padded_in).reshape(
            num_experts * out_features * self.n_groups, self.group_size
        )
        self._indices_grouped = unpacked.reshape(num_experts, out_features, self.n_groups, self.group_size)

    def __call__(
        self,
        x: mx.array,
        indices: mx.array,
        sorted_indices: bool = False,
    ) -> mx.array:
        # Pad input if the original in_features wasn't a multiple of group_size
        if self._pad_needed:
            x = mx.pad(x, [(0, 0)] * (x.ndim - 1) + [(0, self.padded_in - self.in_features)])

        # FWHT-on-input: apply the transform once per token, not per weight row
        leading = x.shape[:-1]
        x_groups = x.reshape(*leading, self.n_groups, self.group_size)
        x_groups = x_groups * self.quant_state.signs1
        x_groups = mx.hadamard_transform(x_groups)
        x_groups = x_groups * self.quant_state.signs2
        x_rot = x_groups.reshape(*leading, self.padded_in)

        # Dequantise all experts: codebook lookup + shape-gain scale
        w = self.quant_state.centroids[self._indices_grouped]
        w = w * self.norms[:, :, :, None]
        w = w.reshape(self.num_experts, self.out_features, self.padded_in).astype(x.dtype)

        # Expert-routed matmul (same primitive SwitchLinear uses)
        out = mx.gather_mm(
            x_rot,
            w.swapaxes(-1, -2),
            rhs_indices=indices,
            sorted_indices=sorted_indices,
        )
        if self.bias is not None:
            out = out + mx.expand_dims(self.bias[indices], -2)
        return out
