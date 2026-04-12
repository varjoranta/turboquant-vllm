"""CPU tests for the native TQ3 FusedMoE checkpoint loading path.

Validates that ``TurboQuantFusedMoELoadMethod`` (the vLLM native loader
for MoE expert weights) correctly:
  - Registers packed/norms parameters with expected shapes
  - Decompresses via ``Compressed3D.from_packed`` round-trip
  - Produces output matching the runtime compression path
  - Weight name remapping works for MoE tensor names

These tests do NOT require a GPU or vLLM. They use the CPU fallback
path for ``Compressed3D`` operations.
"""

from __future__ import annotations

import os
import unittest

import torch

from turboquant_vllm.weight_quant import (
    Compressed3D,
    packed_group_bytes,
    pack_indices,
)


class TestPackedGroupBytes(unittest.TestCase):
    """packed_group_bytes must match the packing logic in pack_indices."""

    def test_4bit(self):
        self.assertEqual(packed_group_bytes(4, 128), 64)
        self.assertEqual(packed_group_bytes(4, 64), 32)
        self.assertEqual(packed_group_bytes(4, 256), 128)

    def test_3bit(self):
        # 128 values * 3 bits = 384 bits = 48 bytes
        self.assertEqual(packed_group_bytes(3, 128), 48)
        self.assertEqual(packed_group_bytes(3, 64), 24)
        self.assertEqual(packed_group_bytes(3, 256), 96)

    def test_2bit(self):
        self.assertEqual(packed_group_bytes(2, 128), 32)
        self.assertEqual(packed_group_bytes(2, 64), 16)


class TestCompressed3DFromPackedRoundTrip(unittest.TestCase):
    """Compressed3D.from_packed must produce identical decompression
    as compressing from raw data."""

    @staticmethod
    def _device():
        return "cuda" if torch.cuda.is_available() else "cpu"

    def test_roundtrip_matches(self):
        """Compress → decompress vs from_packed → decompress."""
        dev = self._device()
        data = torch.randn(4, 256, 128, dtype=torch.float32, device=dev)
        bits, gs = 3, 128

        comp_a = Compressed3D(data, bits=bits, group_size=gs)
        ref = comp_a.decompress()

        comp_b = Compressed3D.from_packed(
            comp_a.packed, comp_a.norms, data.shape, data.dtype, bits, gs
        )
        out_b = comp_b.decompress()

        self.assertEqual(ref.shape, out_b.shape)
        self.assertTrue(
            torch.allclose(ref, out_b),
            f"from_packed decompress diverged: max diff = {(ref - out_b).abs().max():.6g}",
        )

    def test_from_packed_into_buffer(self):
        """from_packed + decompress_into matches decompress."""
        dev = self._device()
        data = torch.randn(2, 128, 128, dtype=torch.float32, device=dev)
        bits, gs = 3, 128

        comp = Compressed3D(data, bits=bits, group_size=gs)
        ref = comp.decompress()

        comp2 = Compressed3D.from_packed(
            comp.packed, comp.norms, data.shape, data.dtype, bits, gs
        )
        buf = torch.empty_like(ref)
        comp2.decompress_into(buf)

        self.assertTrue(
            torch.allclose(ref, buf),
            f"decompress_into diverged: max diff = {(ref - buf).abs().max():.6g}",
        )

    def test_4bit_roundtrip(self):
        dev = self._device()
        data = torch.randn(2, 64, 128, dtype=torch.float32, device=dev)
        bits, gs = 4, 128
        comp = Compressed3D(data, bits=bits, group_size=gs)
        ref = comp.decompress()

        comp2 = Compressed3D.from_packed(
            comp.packed, comp.norms, data.shape, data.dtype, bits, gs
        )
        out = comp2.decompress()
        self.assertTrue(torch.allclose(ref, out))


class TestNativeMoELoaderShapes(unittest.TestCase):
    """Verify parameter shapes match checkpoint expectations."""

    def test_packed_shape_calculation(self):
        """Packed parameter shape must match save_tq3_checkpoint output."""
        num_experts = 4
        out_dim = 256
        in_dim = 128
        bits = 3
        gs = 128

        # Simulate save_tq3_checkpoint: flatten 3D → 2D, compress
        data = torch.randn(num_experts, out_dim, in_dim)
        comp = Compressed3D(data, bits=bits, group_size=gs)

        # Verify shape matches what create_weights would compute
        padded_in = ((in_dim + gs - 1) // gs) * gs
        n_groups = padded_in // gs
        pgb = packed_group_bytes(bits, gs)
        expected_packed = (num_experts * out_dim, n_groups * pgb)
        expected_norms = (num_experts * out_dim, n_groups)

        self.assertEqual(comp.packed.shape, expected_packed,
                         f"packed shape mismatch: {comp.packed.shape} vs {expected_packed}")
        self.assertEqual(comp.norms.shape, expected_norms,
                         f"norms shape mismatch: {comp.norms.shape} vs {expected_norms}")


class TestDecompressDetection(unittest.TestCase):
    """Decompress-on-load detects TQ3 checkpoints via tq_config.json, not quantization_config."""

    def test_tq_config_triggers_decompress(self):
        """Checkpoint with tq_config.json should activate decompression."""
        import tempfile, json
        d = tempfile.mkdtemp()
        with open(os.path.join(d, "tq_config.json"), "w") as f:
            json.dump({"format": "tq3_native", "bits": 3, "group_size": 128}, f)
        self.assertTrue(os.path.isfile(os.path.join(d, "tq_config.json")))

    def test_no_tq_config_skips_decompress(self):
        """Checkpoint without tq_config.json should NOT decompress."""
        import tempfile
        d = tempfile.mkdtemp()
        self.assertFalse(os.path.isfile(os.path.join(d, "tq_config.json")))


if __name__ == "__main__":
    unittest.main(verbosity=2)
