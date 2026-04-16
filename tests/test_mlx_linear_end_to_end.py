"""End-to-end parity: TurboQuantMLXLinear vs PyTorch TurboQuantWrapper.

Compresses a synthetic Linear weight matrix, loads the packed form into
both the PyTorch ``TurboQuantWrapper`` and the MLX ``TurboQuantMLXLinear``,
and verifies the two produce numerically-equivalent matmul output on the
same input.

If this passes, the MLX serving path is validated end-to-end for a single
Linear — the only remaining work for Phase 5 is wiring a model loader
that walks an architecture and replaces every nn.Linear with our wrapper.
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

    HAS_MLX = True
except ImportError:
    HAS_MLX = False


@unittest.skipUnless(HAS_MLX, "MLX not installed (Mac-only)")
class TestTurboQuantMLXLinearParity(unittest.TestCase):
    """Single-Linear end-to-end parity with the PyTorch wrapper."""

    def _compress_linear(self, in_features: int, out_features: int, bits: int = 3, group_size: int = 128):
        """Compress a synthetic Linear, return the packed form + reference output."""
        from turboquant_vllm.torch_ops import PolarQuantTorch
        from turboquant_vllm.weight_quant import pack_indices, padded_size

        torch.manual_seed(42)
        w = torch.randn(out_features, in_features, dtype=torch.float32)
        padded_in, n_groups = padded_size(in_features, group_size)

        if padded_in > in_features:
            padded = torch.zeros(out_features, padded_in, dtype=w.dtype)
            padded[:, :in_features] = w
        else:
            padded = w

        pq = PolarQuantTorch(dim=group_size, bit_width=bits, seed=42, device="cpu")
        grouped = padded.reshape(-1, group_size)
        indices, norms_raw = pq.quantize(grouped, norm_correction=True)
        packed = pack_indices(indices, bits)
        norms = norms_raw.reshape(out_features, n_groups)

        return {
            "w_original": w,
            "packed": packed,
            "norms": norms,
            "quantizer": pq,
            "padded_in": padded_in,
            "n_groups": n_groups,
        }

    def _compare_to_pytorch_reference(self, in_features: int, out_features: int):
        """Compressed PyTorch matmul output == MLX TurboQuantMLXLinear output."""
        from turboquant_vllm.mlx_model import TurboQuantMLXLinear
        from turboquant_vllm.mlx_ops import PolarQuantStateMLX
        from turboquant_vllm.weight_quant import unpack_indices

        bits, group_size = 3, 128
        comp = self._compress_linear(in_features, out_features, bits, group_size)

        # PyTorch reference: decompress weight + matmul
        pq = comp["quantizer"]
        indices = unpack_indices(comp["packed"], bits, group_size)
        w_groups = pq.dequantize(indices, comp["norms"].reshape(-1))
        w_deq_pt = w_groups.reshape(out_features, comp["padded_in"])[:, :in_features]

        torch.manual_seed(1)
        x = torch.randn(4, in_features, dtype=torch.float32)
        ref = (x @ w_deq_pt.t()).numpy()

        # MLX path: build state + TurboQuantMLXLinear, run forward on the same input
        state = PolarQuantStateMLX.from_torch_quantizer(pq)
        layer = TurboQuantMLXLinear(
            packed_weight=mx.array(comp["packed"].numpy()),
            norms=mx.array(comp["norms"].numpy()),
            state=state,
            in_features=in_features,
            out_features=out_features,
            bias=None,
        )

        out_mx = layer(mx.array(x.numpy()))
        out_np = np.array(out_mx)

        np.testing.assert_allclose(out_np, ref, rtol=5e-3, atol=5e-3)

    def test_aligned_dim(self):
        """in_features=128 (== group_size), no input padding needed."""
        self._compare_to_pytorch_reference(in_features=128, out_features=64)

    def test_multiple_groups(self):
        """in_features=256 = 2 groups, no padding."""
        self._compare_to_pytorch_reference(in_features=256, out_features=128)

    def test_unaligned_dim(self):
        """in_features=200 needs padding to 256."""
        self._compare_to_pytorch_reference(in_features=200, out_features=64)

    def test_fwht_on_input_matches_full_dequant(self):
        """FWHT-on-input matmul matches the dense-dequant + matmul path."""
        from turboquant_vllm.mlx_ops import (
            PolarQuantStateMLX,
            fwht_on_input_matmul_mlx,
            unpack_indices_3bit_mlx,
        )

        for in_features, out_features in [(128, 64), (256, 128), (200, 64)]:
            with self.subTest(in_features=in_features, out_features=out_features):
                comp = self._compress_linear(in_features, out_features)
                pq = comp["quantizer"]
                state = PolarQuantStateMLX.from_torch_quantizer(pq)

                packed_mx = mx.array(comp["packed"].numpy())
                norms_mx = mx.array(comp["norms"].numpy())
                indices_grouped = unpack_indices_3bit_mlx(
                    packed_mx, dim=comp["padded_in"]
                ).reshape(out_features * comp["n_groups"], 128)

                torch.manual_seed(3)
                x_pt = torch.randn(4, in_features, dtype=torch.float32)
                x_mx = mx.array(x_pt.numpy())

                # Reference: PyTorch decompress + matmul
                from turboquant_vllm.weight_quant import unpack_indices

                indices = unpack_indices(comp["packed"], 3, 128)
                w_groups = pq.dequantize(indices, comp["norms"].reshape(-1))
                w_deq = w_groups.reshape(out_features, comp["padded_in"])[
                    :, :in_features
                ]
                ref = (x_pt @ w_deq.t()).numpy()

                out_mx = fwht_on_input_matmul_mlx(
                    x=x_mx,
                    indices_grouped=indices_grouped,
                    norms=norms_mx,
                    state=state,
                )
                out_np = np.array(out_mx)

                np.testing.assert_allclose(out_np, ref, rtol=5e-3, atol=5e-3)

    def test_with_bias(self):
        """Bias is added in fp32 then cast to x.dtype."""
        from turboquant_vllm.mlx_model import TurboQuantMLXLinear
        from turboquant_vllm.mlx_ops import PolarQuantStateMLX

        in_features, out_features = 128, 64
        comp = self._compress_linear(in_features, out_features)
        pq = comp["quantizer"]

        torch.manual_seed(7)
        bias_pt = torch.randn(out_features, dtype=torch.float32)

        state = PolarQuantStateMLX.from_torch_quantizer(pq)
        layer = TurboQuantMLXLinear(
            packed_weight=mx.array(comp["packed"].numpy()),
            norms=mx.array(comp["norms"].numpy()),
            state=state,
            in_features=in_features,
            out_features=out_features,
            bias=mx.array(bias_pt.numpy()),
        )
        x = torch.randn(2, in_features, dtype=torch.float32)
        out_mx = layer(mx.array(x.numpy()))
        out_np = np.array(out_mx)[:, :out_features]

        # Reference: decompress via pq + matmul + bias
        from turboquant_vllm.weight_quant import unpack_indices

        indices = unpack_indices(comp["packed"], 3, 128)
        w_groups = pq.dequantize(indices, comp["norms"].reshape(-1))
        w_deq = w_groups.reshape(out_features, comp["padded_in"])[:, :in_features]
        ref = (x @ w_deq.t() + bias_pt).numpy()

        np.testing.assert_allclose(out_np, ref, rtol=5e-3, atol=5e-3)


if __name__ == "__main__":
    unittest.main()
