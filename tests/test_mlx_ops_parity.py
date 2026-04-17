"""Bitwise parity: MLX dequant path == PyTorch dequant path.

Mac-only test. Skips cleanly on machines without MLX installed. Validates
that ``turboquant_vllm.mlx_ops`` produces numerically-identical output
(within float32 precision) to ``PolarQuantTorch.dequantize``.

If this test passes, we know the MLX port of the dequant math is correct;
the remaining work is wiring it into a model load path so ``mlx_lm.server``
can serve TQ3 checkpoints.
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
class TestMLXDequantParity(unittest.TestCase):
    """MLX dequant path must match PyTorch PolarQuantTorch.dequantize."""

    def _setup_quantizer(self, dim: int = 128, bits: int = 3):
        from turboquant_vllm.torch_ops import PolarQuantTorch

        return PolarQuantTorch(dim=dim, bit_width=bits, seed=42, device="cpu")

    def test_fast_wht_matches(self):
        """MLX fast_wht_batch == PyTorch _fast_wht_batch, within float32 tol."""
        from turboquant_vllm.mlx_ops import fast_wht_batch_mlx
        from turboquant_vllm.torch_ops import _fast_wht_batch

        torch.manual_seed(0)
        x_pt = torch.randn(4, 128, dtype=torch.float32)
        x_np = x_pt.numpy()

        ref = _fast_wht_batch(x_pt.clone()).numpy()

        x_mx = mx.array(x_np)
        out_mx = fast_wht_batch_mlx(x_mx)
        out_np = np.array(out_mx)

        np.testing.assert_allclose(out_np, ref, rtol=1e-5, atol=1e-5)

    def test_unpack_indices_3bit_matches(self):
        """MLX 3-bit unpack == PyTorch _unpack_indices(bits=3)."""
        from turboquant_vllm.mlx_ops import unpack_indices_3bit_mlx
        from turboquant_vllm.weight_quant import unpack_indices as _unpack_indices

        torch.manual_seed(0)
        dim = 128
        n_rows = 4
        # packed shape for 3-bit: (n_rows, n_groups_of_3 * 3) where
        # n_groups_of_3 = ceil(dim / 8). For dim=128 that's 16 groups => 48 bytes.
        n_packed = (dim + 7) // 8 * 3
        packed_pt = torch.randint(0, 256, (n_rows, n_packed), dtype=torch.uint8)

        ref = _unpack_indices(packed_pt, bits=3, dim=dim).numpy()

        packed_mx = mx.array(packed_pt.numpy())
        out_mx = unpack_indices_3bit_mlx(packed_mx, dim=dim)
        out_np = np.array(out_mx)

        np.testing.assert_array_equal(out_np, ref)
        assert out_mx.dtype == mx.uint8

    def test_unpack_3bit_cross_byte_boundary(self):
        """MLX unpack correctly handles cross-byte positions (2 and 5)."""
        from turboquant_vllm.mlx_ops import unpack_indices_3bit_mlx
        from turboquant_vllm.weight_quant import unpack_indices as _unpack_indices

        # All max values: every bit set, stresses every cross-byte boundary.
        packed_pt = torch.full((1, 3), 0xFF, dtype=torch.uint8)
        ref = _unpack_indices(packed_pt, bits=3, dim=8).numpy()

        packed_mx = mx.array(packed_pt.numpy())
        out_mx = unpack_indices_3bit_mlx(packed_mx, dim=8)
        out_np = np.array(out_mx)

        np.testing.assert_array_equal(out_np, ref)
        # Also check the expected value: all 0x7
        np.testing.assert_array_equal(out_np, np.full((1, 8), 7))

    def _mlx_state(self, pq):
        from turboquant_vllm.mlx_ops import PolarQuantStateMLX

        return PolarQuantStateMLX(
            signs1=mx.array(pq.signs1.numpy()),
            signs2=mx.array(pq.signs2.numpy()),
            centroids=mx.array(pq.centroids.numpy()),
            dim=pq.dim,
        )

    def test_full_dequant_matches(self):
        """End-to-end: MLX dequant pipeline == PolarQuantTorch.dequantize."""
        from turboquant_vllm.mlx_ops import polar_quant_dequantize_mlx

        pq = self._setup_quantizer(dim=128, bits=3)

        torch.manual_seed(0)
        x = torch.randn(16, 128, dtype=torch.float32)
        indices_pt, norms_pt = pq.quantize(x, norm_correction=True)
        ref = pq.dequantize(indices_pt, norms_pt).numpy()

        out_mx = polar_quant_dequantize_mlx(
            indices=mx.array(indices_pt.numpy().astype(np.int32)),
            norms=mx.array(norms_pt.numpy()),
            state=self._mlx_state(pq),
        )
        out_np = np.array(out_mx)

        np.testing.assert_allclose(out_np, ref, rtol=1e-4, atol=1e-4)

    def test_full_dequant_non_pow2(self):
        """Dim 200 exercises the padding path (padded_dim=256)."""
        from turboquant_vllm.mlx_ops import polar_quant_dequantize_mlx

        pq = self._setup_quantizer(dim=200, bits=3)
        state = self._mlx_state(pq)
        self.assertEqual(state.padded_dim, 256)

        torch.manual_seed(1)
        x = torch.randn(8, 200, dtype=torch.float32)
        indices_pt, norms_pt = pq.quantize(x, norm_correction=True)
        ref = pq.dequantize(indices_pt, norms_pt).numpy()

        out_mx = polar_quant_dequantize_mlx(
            indices=mx.array(indices_pt.numpy().astype(np.int32)),
            norms=mx.array(norms_pt.numpy()),
            state=state,
        )
        out_np = np.array(out_mx)

        np.testing.assert_allclose(out_np, ref, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
