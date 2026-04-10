"""Tests for 3-bit sub-byte packing.

Verifies pack_indices/unpack_indices roundtrip for 3-bit values.
Does NOT require GPU — pure Python/PyTorch operations.

Run: pytest tests/test_3bit_packing.py -v
"""

import pytest
import torch

from turboquant_vllm.weight_quant import pack_indices, unpack_indices


class TestThreeBitPacking:
    @pytest.mark.parametrize("n_cols", [8, 16, 64, 128, 256])
    def test_roundtrip(self, n_cols):
        """Pack then unpack should recover original indices."""
        n_rows = 32
        indices = torch.randint(0, 8, (n_rows, n_cols), dtype=torch.int64)

        packed = pack_indices(indices, bits=3)
        unpacked = unpack_indices(packed, bits=3, dim=n_cols)

        assert unpacked.shape == (n_rows, n_cols), f"Shape mismatch: {unpacked.shape}"
        assert (unpacked == indices).all(), f"Values mismatch at {(unpacked != indices).nonzero()[:5]}"

    def test_compression_ratio(self):
        """3-bit packing should use 3/8 bytes per index."""
        n_rows, n_cols = 100, 128
        indices = torch.randint(0, 8, (n_rows, n_cols), dtype=torch.int64)
        packed = pack_indices(indices, bits=3)

        expected_packed_cols = (n_cols // 8) * 3  # 8 indices per 3 bytes
        assert packed.shape == (n_rows, expected_packed_cols), (
            f"Packed shape {packed.shape}, expected ({n_rows}, {expected_packed_cols})"
        )

        # Compression: 128 int64 → 48 uint8, that's 1024 bytes → 48 bytes = 21.3x packing
        # In practice for weight compression: 128 elements at 3 bits = 48 bytes
        # vs FP16: 128 * 2 = 256 bytes → 5.3x per group (before norms)
        ratio = n_cols / packed.shape[1]
        print(f"\n  3-bit packing: {n_cols} indices → {packed.shape[1]} bytes (ratio={ratio:.1f}, expected ~2.67)")
        assert abs(ratio - 8 / 3) < 0.01, f"Unexpected ratio: {ratio}"

    def test_all_values(self):
        """All 8 possible 3-bit values should survive roundtrip."""
        # Create a tensor with all values 0-7 in known positions
        indices = torch.arange(8, dtype=torch.int64).unsqueeze(0).repeat(4, 1)
        packed = pack_indices(indices, bits=3)
        unpacked = unpack_indices(packed, bits=3, dim=8)
        assert (unpacked == indices).all()

    def test_large_groups(self):
        """Test with real weight compression dimensions."""
        # Typical: 4096 output rows, 128 group size
        n_rows, n_cols = 4096, 128
        indices = torch.randint(0, 8, (n_rows, n_cols), dtype=torch.int64)
        packed = pack_indices(indices, bits=3)
        unpacked = unpack_indices(packed, bits=3, dim=n_cols)
        assert (unpacked == indices).all()
        print(
            f"\n  Large test: {n_rows}x{n_cols} → packed {packed.shape} "
            f"({packed.numel()} bytes vs {n_rows * n_cols} original)"
        )

    def test_4bit_unchanged(self):
        """Verify 4-bit packing still works (regression test)."""
        n_rows, n_cols = 32, 128
        indices = torch.randint(0, 16, (n_rows, n_cols), dtype=torch.int64)
        packed = pack_indices(indices, bits=4)
        unpacked = unpack_indices(packed, bits=4, dim=n_cols)
        assert (unpacked == indices).all()

    def test_2bit_unchanged(self):
        """Verify 2-bit packing still works (regression test)."""
        n_rows, n_cols = 32, 128
        indices = torch.randint(0, 4, (n_rows, n_cols), dtype=torch.int64)
        packed = pack_indices(indices, bits=2)
        unpacked = unpack_indices(packed, bits=2, dim=n_cols)
        assert (unpacked == indices).all()
