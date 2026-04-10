"""Regression tests for turboquant_vllm.vllm_patch._get_compressor.

Historical bug: the cache was keyed by (dim, k_bits) but the return used
_compressors[dim], raising KeyError on every call. This broke the monkey-
patch boundary-layer logic silently — no test ever exercised the path.
These tests lock in the correct behaviour so it can't regress.
"""

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestGetCompressor(unittest.TestCase):
    def setUp(self):
        # Reset module-level state between tests
        from turboquant_vllm import vllm_patch

        vllm_patch._compressors.clear()
        vllm_patch._k_bits = 4
        vllm_patch._v_bits = 4
        vllm_patch._norm_correction = False  # avoid CUDA path on Mac
        vllm_patch._use_qjl = False
        vllm_patch._rotation = "wht"
        vllm_patch._boundary_layers = 5
        vllm_patch._total_layers = 32
        vllm_patch._use_cuda = False
        self.vllm_patch = vllm_patch

    def test_returns_without_keyerror(self):
        """The original bug: _compressors[dim] lookup raised KeyError."""
        device = torch.device("cpu")
        comp = self.vllm_patch._get_compressor(dim=128, device=device, layer_idx=10)
        self.assertIsNotNone(comp)
        # And it actually has the right k_bits (matching _k_bits, not boundary)
        self.assertEqual(comp.k_bits, 4)

    def test_boundary_layer_gets_8bit(self):
        """Layers in the boundary range should get K=8-bit precision."""
        device = torch.device("cpu")
        # Layer 2 is in first 5 (boundary)
        boundary_comp = self.vllm_patch._get_compressor(dim=128, device=device, layer_idx=2)
        self.assertEqual(boundary_comp.k_bits, 8)
        # Layer 15 is middle — uses base _k_bits
        middle_comp = self.vllm_patch._get_compressor(dim=128, device=device, layer_idx=15)
        self.assertEqual(middle_comp.k_bits, 4)
        # They must be different instances
        self.assertIsNot(boundary_comp, middle_comp)

    def test_cache_hit_returns_same_instance(self):
        """Second call with same (dim, k) must return the cached instance."""
        device = torch.device("cpu")
        c1 = self.vllm_patch._get_compressor(dim=128, device=device, layer_idx=15)
        c2 = self.vllm_patch._get_compressor(dim=128, device=device, layer_idx=16)
        self.assertIs(c1, c2, "same (dim, k_bits) should hit the cache")

    def test_different_dims_are_separate_instances(self):
        """Different head_dims must not share a compressor."""
        device = torch.device("cpu")
        c128 = self.vllm_patch._get_compressor(dim=128, device=device, layer_idx=15)
        c64 = self.vllm_patch._get_compressor(dim=64, device=device, layer_idx=15)
        self.assertIsNot(c128, c64)
        self.assertEqual(c128.head_dim, 128)
        self.assertEqual(c64.head_dim, 64)

    def test_cache_key_is_tuple(self):
        """Regression: _compressors type hint and cache key must be tuple[int, int]."""
        device = torch.device("cpu")
        self.vllm_patch._get_compressor(dim=128, device=device, layer_idx=15)
        keys = list(self.vllm_patch._compressors.keys())
        self.assertEqual(len(keys), 1)
        self.assertIsInstance(keys[0], tuple)
        self.assertEqual(len(keys[0]), 2)
        self.assertEqual(keys[0], (128, 4))


class TestIterSlots(unittest.TestCase):
    """Regression: _iter_slots must skip -1 (padding) entries in slot_mapping.

    Historical bug: negative slot values were passed through to slot //
    block_size which returned -1, scattering compressed entries into the
    wrong dict key and corrupting cache on read. vLLM uses -1 in slot_mapping
    as a placeholder for unscheduled tokens when a batch is smaller than
    the max sequence length.
    """

    def test_negative_slots_are_skipped(self):
        from turboquant_vllm.vllm_patch import _iter_slots

        # Mixed: valid, padding, valid, padding
        slot_mapping = torch.tensor([5, -1, 17, -1, 32], dtype=torch.int64)
        block_size = 16
        results = list(_iter_slots(slot_mapping, block_size))

        # Only the 3 non-negative entries should be yielded
        self.assertEqual(len(results), 3)
        token_indices = [r[0] for r in results]
        self.assertEqual(token_indices, [0, 2, 4])

        # And each yielded result must have non-negative block_idx/offset
        for t, block_idx, offset in results:
            self.assertGreaterEqual(block_idx, 0, f"token {t}: block_idx must be >= 0")
            self.assertGreaterEqual(offset, 0, f"token {t}: offset must be >= 0")

    def test_all_negative_yields_nothing(self):
        """All-padding batch should be a no-op, not corrupt anything."""
        from turboquant_vllm.vllm_patch import _iter_slots

        slot_mapping = torch.tensor([-1, -1, -1], dtype=torch.int64)
        results = list(_iter_slots(slot_mapping, block_size=16))
        self.assertEqual(results, [])

    def test_valid_slots_unchanged(self):
        """Sanity: positive slots still decompose correctly."""
        from turboquant_vllm.vllm_patch import _iter_slots

        slot_mapping = torch.tensor([0, 15, 16, 17, 31, 32], dtype=torch.int64)
        results = list(_iter_slots(slot_mapping, block_size=16))
        self.assertEqual(len(results), 6)
        expected = [(0, 0, 0), (1, 0, 15), (2, 1, 0), (3, 1, 1), (4, 1, 15), (5, 2, 0)]
        self.assertEqual(results, expected)


if __name__ == "__main__":
    unittest.main(verbosity=2)
