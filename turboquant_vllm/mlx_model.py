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

from turboquant_vllm.mlx_metal_kernels import (
    tq3_gemv_bs1_batched_mlx,
    tq3_gemv_bs1_batched_per_x_mlx,
    tq3_gemv_bs1_mlx,
)
from turboquant_vllm.mlx_ops import (
    PolarQuantStateMLX,
    fwht_on_input_matmul_mlx,
    rht_on_last_dim_mlx,
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
        orig_shape = x.shape
        x_flat = x.reshape(-1, orig_shape[-1]) if x.ndim > 2 else x

        # bs=1 fp16 fast path via the custom Metal GEMV kernel: skips the
        # full bf16/fp16 weight materialization that fwht_on_input_matmul_mlx
        # does, fusing 3-bit unpack + codebook lookup + norm + matmul into
        # one pass. 5-10x over the existing path on Qwen3-8B layer shapes.
        if (
            x_flat.shape[0] == 1
            and x_flat.dtype == mx.float16
        ):
            x_padded = (
                mx.pad(x_flat, [(0, 0), (0, self.padded_in - self.in_features)])
                if self._pad_needed else x_flat
            )
            x_rot = rht_on_last_dim_mlx(
                x_padded, self.quant_state.signs1, self.quant_state.signs2,
                self.n_groups, self.group_size,
            ).reshape(self.padded_in).astype(mx.float16)
            packed_view = self.packed_weight.reshape(
                self.out_features * self.n_groups, 48,
            )
            out_vec = tq3_gemv_bs1_mlx(
                x_rot,
                packed_view,
                self.norms.astype(mx.float16),
                self.quant_state.centroids.astype(mx.float16),
            )
            if self.bias is not None:
                out_vec = out_vec + self.bias.astype(mx.float16)
            out_flat = out_vec.reshape(1, self.out_features)
        else:
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

        self.norms = norms.reshape(num_experts, out_features, self.n_groups)

        # Keep packed uint8 — do NOT pre-unpack to int32. For 256-expert
        # models the int32 blowup is ~1 GB per SwitchLinear (120 GB total
        # for 35B). Instead reshape packed so axis 0 is per-expert and we
        # can gather only the active experts per forward.
        bytes_per_group = packed_weight.shape[-1]
        self._packed_per_expert = packed_weight.reshape(num_experts, out_features * self.n_groups, bytes_per_group)

    def __call__(
        self,
        x: mx.array,
        indices: mx.array,
        sorted_indices: bool = False,
    ) -> mx.array:
        # ``sorted_indices`` is accepted for API parity with mlx_lm's
        # ``SwitchLinear`` but unused here: we dequant per-slot regardless of
        # sort order, so the hint buys nothing.
        del sorted_indices

        if self._pad_needed:
            x = mx.pad(x, [(0, 0)] * (x.ndim - 1) + [(0, self.padded_in - self.in_features)])

        x_rot = rht_on_last_dim_mlx(
            x, self.quant_state.signs1, self.quant_state.signs2, self.n_groups, self.group_size
        ).reshape(*x.shape[:-1], self.padded_in)

        ids_flat = indices.reshape(-1)
        active_packed = mx.take(self._packed_per_expert, ids_flat, axis=0)
        active_norms = mx.take(self.norms, ids_flat, axis=0)

        # bs=1 fp16 fast path: batched custom Metal GEMV across active experts.
        # Skips the full ``(K_active, OC, K)`` weight materialisation that the
        # einsum path requires, fusing 3-bit unpack + codebook + norm + matmul
        # into one kernel call per Linear.
        # Two cases:
        #   (a) shared-x: gate/up_proj — same activation goes to every expert.
        #       x has shape (1, 1, 1, K), x_rot.size == padded_in.
        #   (b) per-expert-x: down_proj — each expert sees its own activation.
        #       x has shape (1, K_active, 1, K), x_rot.size == K_active * padded_in.
        K_active = ids_flat.size
        if x.dtype == mx.float16 and x_rot.size == self.padded_in:
            x_rot_flat = x_rot.reshape(self.padded_in).astype(mx.float16)
            out_flat = tq3_gemv_bs1_batched_mlx(
                x_rot_flat,
                active_packed,
                active_norms.astype(mx.float16),
                self.quant_state.centroids.astype(mx.float16),
            )
            if self.bias is not None:
                out_flat = out_flat + mx.take(self.bias, ids_flat, axis=0).astype(mx.float16)
            return out_flat.reshape(*indices.shape, 1, self.out_features)
        if x.dtype == mx.float16 and x_rot.size == K_active * self.padded_in:
            x_rot_per_k = x_rot.reshape(K_active, self.padded_in).astype(mx.float16)
            out_flat = tq3_gemv_bs1_batched_per_x_mlx(
                x_rot_per_k,
                active_packed,
                active_norms.astype(mx.float16),
                self.quant_state.centroids.astype(mx.float16),
            )
            if self.bias is not None:
                out_flat = out_flat + mx.take(self.bias, ids_flat, axis=0).astype(mx.float16)
            return out_flat.reshape(*indices.shape, 1, self.out_features)

        # Fallback: full dequant + einsum (existing path).
        active_unpacked = unpack_indices_3bit_mlx(
            active_packed.reshape(-1, active_packed.shape[-1]),
            dim=self.padded_in,
        ).reshape(ids_flat.size, self.out_features, self.n_groups, self.group_size)

        w = self.quant_state.centroids[active_unpacked]
        w = w * active_norms[:, :, :, None]
        w = w.reshape(ids_flat.size, self.out_features, self.padded_in).astype(x.dtype)

        x_rot_per_k = mx.broadcast_to(
            x_rot.squeeze(-2),
            (*indices.shape, self.padded_in),
        ).reshape(ids_flat.size, self.padded_in)
        out_flat = mx.einsum("ki,koi->ko", x_rot_per_k, w)

        if self.bias is not None:
            out_flat = out_flat + mx.take(self.bias, ids_flat, axis=0)

        return out_flat.reshape(*indices.shape, 1, self.out_features)
