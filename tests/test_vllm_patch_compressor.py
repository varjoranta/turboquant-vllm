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
        vllm_patch._layer_compressor.clear()
        vllm_patch._layer_indices.clear()
        vllm_patch._layer_token_counts.clear()
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


class TestFrozenCompressor(unittest.TestCase):
    """Regression: compressor assigned to a layer must be frozen on first use.

    Historical bug: _get_compressor was called on every cache update, re-deriving
    boundary status from the current _total_layers. During early forward passes,
    _total_layers is small (auto-registration hasn't seen all layers yet), so a
    layer could be classified as boundary (K=8-bit). Once all layers register,
    the same layer falls outside the boundary range (K=4-bit). Without freezing,
    decompress would use a different compressor/codebook than compress, producing
    garbage. The fix: _layer_compressor caches the compressor per layer_id on
    first use and never re-derives.
    """

    def setUp(self):
        from turboquant_vllm import vllm_patch

        vllm_patch._compressors.clear()
        vllm_patch._k_bits = 4
        vllm_patch._v_bits = 4
        vllm_patch._norm_correction = False
        vllm_patch._use_qjl = False
        vllm_patch._rotation = "wht"
        vllm_patch._boundary_layers = 5
        vllm_patch._total_layers = 0
        vllm_patch._use_cuda = False
        vllm_patch._layer_compressor.clear()
        vllm_patch._layer_indices.clear()
        vllm_patch._layer_token_counts.clear()
        vllm_patch._cache.clear()
        vllm_patch._sink_tokens = 0  # disable sink to simplify test
        vllm_patch._fp16_heads = set()
        self.vllm_patch = vllm_patch

    def test_compressor_frozen_across_total_layers_growth(self):
        """Layer that starts as boundary must keep its compressor after _total_layers grows."""
        vp = self.vllm_patch
        device = torch.device("cpu")

        # Use boundary_layers=2 so the math is clean:
        #   _total_layers=6, idx=4: boundary (4 >= 6-2=4) → K=8
        #   _total_layers=28, idx=4: NOT boundary (4 >= 2 and 4 < 26) → K=4
        vp._boundary_layers = 2
        vp._total_layers = 6
        vp._layer_indices[9999] = 4  # layer_id=9999 → layer_idx=4

        first_compressor = vp._get_compressor(dim=128, device=device, layer_idx=4)
        vp._layer_compressor[9999] = first_compressor
        self.assertEqual(first_compressor.k_bits, 8, "boundary layer should get K=8-bit")

        # Now _total_layers grows to 28 (all layers registered).
        # Layer index 4 is NO LONGER boundary (not in first 2, not in last 2 of 28).
        vp._total_layers = 28

        # If we naively re-derive, we'd get a 4-bit compressor:
        naive = vp._get_compressor(dim=128, device=device, layer_idx=4)
        self.assertEqual(naive.k_bits, 4, "re-derived should be 4-bit (non-boundary at 28 layers)")

        # But the frozen lookup must return the ORIGINAL 8-bit compressor.
        frozen = vp._layer_compressor[9999]
        self.assertIs(frozen, first_compressor, "frozen compressor must be the same instance")
        self.assertEqual(frozen.k_bits, 8, "frozen compressor must keep K=8-bit")

    def test_make_patched_cache_update_freezes_compressor(self):
        """_make_patched_cache_update must freeze compressor on first call per layer."""
        vp = self.vllm_patch

        # Build a minimal mock layer and kv_cache
        class FakeLayer:
            pass

        layer = FakeLayer()
        layer_id = id(layer)

        # Register layer as index 4 with small _total_layers (boundary).
        # boundary_layers=2, total=6: idx=4 >= 6-2=4 → boundary (K=8)
        vp._boundary_layers = 2
        vp._total_layers = 6
        vp._layer_indices[layer_id] = 4

        head_dim = 64
        key = torch.randn(1, 1, head_dim)  # (tokens, heads, dim)
        value = torch.randn(1, 1, head_dim)
        # kv_cache: (2, blocks, block_size, heads, dim)
        kv_cache = torch.zeros(2, 4, 16, 1, head_dim)
        slot_mapping = torch.tensor([0], dtype=torch.int64)

        original_called = [False]

        def fake_original(self_, layer_, key_, value_, kv_cache_, slot_mapping_):
            original_called[0] = True

        patched_fn = vp._make_patched_cache_update(fake_original)

        # First call — should freeze compressor as boundary (K=8)
        patched_fn(None, layer, key, value, kv_cache, slot_mapping)
        self.assertTrue(original_called[0])
        self.assertIn(layer_id, vp._layer_compressor)
        first_comp = vp._layer_compressor[layer_id]
        self.assertEqual(first_comp.k_bits, 8)

        # Grow _total_layers — layer 4 is no longer boundary
        vp._total_layers = 28

        # Second call — must reuse frozen compressor
        original_called[0] = False
        patched_fn(None, layer, key, value, kv_cache, slot_mapping)
        self.assertTrue(original_called[0])
        self.assertIs(
            vp._layer_compressor[layer_id], first_comp, "compressor must stay frozen after _total_layers changes"
        )


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
