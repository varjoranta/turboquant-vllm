"""End-to-end parity: TurboQuantMLXSwitchLinear vs mlx_lm.SwitchLinear.

Compresses a synthetic MoE weight tensor with TQ3, loads it into both the
PyTorch reference and the MLX SwitchLinear drop-in, and checks that the
token-routed matmul output matches within float32 tolerance.

If this passes, the MoE serving path through ``mlx-lm`` is validated for
a single SwitchLinear. Wiring the loader to replace every
``switch_mlp.gate_proj``/``up_proj``/``down_proj`` in a real MoE
architecture is the next step.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import mlx.core as mx
    from mlx_lm.models.switch_layers import SwitchLinear

    HAS_MLX = True
except ImportError:
    HAS_MLX = False


@unittest.skipUnless(HAS_MLX, "MLX not installed (Mac-only)")
class TestTurboQuantMLXSwitchLinearParity(unittest.TestCase):
    """TQ3 SwitchLinear replacement matches SwitchLinear over equivalent weights."""

    def _compress_experts(
        self, num_experts: int, in_features: int, out_features: int, bits: int = 3, group_size: int = 128
    ):
        """Compress a (num_experts, out, in) weight tensor to packed TQ3 form."""
        from turboquant_vllm.torch_ops import PolarQuantTorch
        from turboquant_vllm.weight_quant import pack_indices, padded_size

        torch.manual_seed(42)
        w = torch.randn(num_experts, out_features, in_features, dtype=torch.float32)
        padded_in, n_groups = padded_size(in_features, group_size)

        if padded_in > in_features:
            padded = torch.zeros(num_experts, out_features, padded_in, dtype=w.dtype)
            padded[:, :, :in_features] = w
        else:
            padded = w

        pq = PolarQuantTorch(dim=group_size, bit_width=bits, seed=42, device="cpu")
        grouped = padded.reshape(-1, group_size)
        indices, norms_raw = pq.quantize(grouped, norm_correction=True)
        packed = pack_indices(indices, bits)
        norms = norms_raw.reshape(num_experts * out_features, n_groups)

        # Reconstruct the dequantised 3D weight so the reference path uses
        # the same values TurboQuantMLXSwitchLinear sees after its forward.
        w_groups = pq.dequantize(indices, norms_raw)
        w_deq = w_groups.reshape(num_experts, out_features, padded_in)[:, :, :in_features]

        return {
            "packed": packed,
            "norms": norms,
            "quantizer": pq,
            "w_deq": w_deq,  # (num_experts, out, in)
        }

    def _compare(
        self,
        num_experts: int,
        in_features: int,
        out_features: int,
        bias: bool = False,
    ):
        from turboquant_vllm.mlx_model import TurboQuantMLXSwitchLinear
        from turboquant_vllm.mlx_ops import PolarQuantStateMLX

        comp = self._compress_experts(num_experts, in_features, out_features)
        state = PolarQuantStateMLX.from_torch_quantizer(comp["quantizer"])

        # Reference: uncompressed SwitchLinear with the dequantised weights
        ref_layer = SwitchLinear(in_features, out_features, num_experts, bias=bias)
        ref_layer.weight = mx.array(comp["w_deq"].numpy().astype(np.float32))
        bias_mx = None
        if bias:
            torch.manual_seed(7)
            bias_pt = torch.randn(num_experts, out_features, dtype=torch.float32)
            ref_layer.bias = mx.array(bias_pt.numpy())
            bias_mx = mx.array(bias_pt.numpy())
        mx.eval(ref_layer.parameters())

        # TurboQuant path
        tq_layer = TurboQuantMLXSwitchLinear(
            packed_weight=mx.array(comp["packed"].numpy()),
            norms=mx.array(comp["norms"].numpy()),
            state=state,
            in_features=in_features,
            out_features=out_features,
            num_experts=num_experts,
            bias=bias_mx,
        )

        # SwitchGLU-style call: (batch, seq, 1, 1, in_features) + (batch, seq, k)
        torch.manual_seed(1)
        x_pt = torch.randn(2, 3, 1, 1, in_features, dtype=torch.float32)
        x_mx = mx.array(x_pt.numpy())
        # Two experts per token; pick simple indices
        idx = mx.array([[[0, 1]], [[num_experts - 1, 0]]])
        idx = mx.broadcast_to(idx, (2, 3, 2))

        ref = np.array(ref_layer(x_mx, idx))
        out = np.array(tq_layer(x_mx, idx))
        np.testing.assert_allclose(out, ref, rtol=5e-3, atol=5e-3)

    def test_aligned_dim(self):
        """in_features=128 (== group_size), no input padding."""
        self._compare(num_experts=4, in_features=128, out_features=64)

    def test_multiple_groups(self):
        """in_features=256 = 2 groups, no padding."""
        self._compare(num_experts=4, in_features=256, out_features=64)

    def test_unaligned_dim(self):
        """in_features=200 needs padding to 256."""
        self._compare(num_experts=4, in_features=200, out_features=64)

    def test_with_bias(self):
        """Per-expert bias path."""
        self._compare(num_experts=4, in_features=128, out_features=64, bias=True)

    def test_more_experts(self):
        """Scale to 16 experts."""
        self._compare(num_experts=16, in_features=128, out_features=64)


if __name__ == "__main__":
    unittest.main()
