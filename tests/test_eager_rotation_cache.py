"""Regression test: TurboQuantWrapper must eagerly populate the rotation
matrix cache in ``__init__`` so the first forward never hits a cache miss.

Background — applying lessons from the vllm-project/vllm#38479 review pass
(see ``_ensure_on_device`` lazy init in Vibhav's TurboQuant KV cache PR):
any initialization that runs inside a ``torch.library.custom_op`` body on
first forward is at risk of executing during vLLM's CUDA graph capture
warmup pass. If vLLM's warmup fires the custom op but our rotation matrix
cache is still empty, the cache miss triggers ``torch.eye`` plus a
butterfly WHT inside the captured region — still correct under torch's
capture-aware allocator, but ugly and implicit.

The fix: pre-populate ``_rotation_matrix_cache`` in
``TurboQuantWrapper.__init__`` (and the parallel
``TurboQuantLinearMethod.process_weights_after_loading``) before any
forward can run. This test locks in that behaviour so future refactors
cannot silently regress it.

CPU-only; no Triton, no CUDA, no vLLM.
"""

import unittest

import torch
import torch.nn as nn


class TestEagerRotationMatrixCache(unittest.TestCase):
    """Constructing a TurboQuantWrapper must populate the cache."""

    def setUp(self):
        # Clear the cache at the start of each test so hits/misses are
        # deterministic regardless of test order.
        from turboquant_vllm.triton_ops import _rotation_matrix_cache

        _rotation_matrix_cache.clear()
        self._cache = _rotation_matrix_cache

    def _build_wrapper(self, in_features=256, out_features=128, bits=3, group_size=128):
        from turboquant_vllm.weight_quant import TurboQuantWrapper

        linear = nn.Linear(in_features, out_features, bias=False)
        return TurboQuantWrapper(linear, bits=bits, group_size=group_size)

    def test_cache_empty_before_construction(self):
        self.assertEqual(
            len(self._cache),
            0,
            "precondition: cache must start empty (setUp should have cleared it)",
        )

    def test_construction_populates_cache(self):
        self._build_wrapper()
        self.assertGreaterEqual(
            len(self._cache),
            1,
            "TurboQuantWrapper.__init__ must eagerly populate the rotation "
            "matrix cache so the first forward (potentially during CUDA "
            "graph capture) does not hit a cache miss",
        )

    def test_two_wrappers_same_bits_share_cache_entry(self):
        """Two wrappers with the same (bits, group_size) share sign vectors
        via the _get_quantizer singleton, so the cache should have exactly
        one entry for both."""
        self._build_wrapper(bits=3, group_size=128)
        n_after_first = len(self._cache)
        self._build_wrapper(bits=3, group_size=128)
        n_after_second = len(self._cache)
        self.assertEqual(
            n_after_first,
            n_after_second,
            "wrappers with identical (bits, group_size) should reuse the "
            "same rotation matrix cache entry — second construction added "
            f"a duplicate entry ({n_after_first} -> {n_after_second})",
        )

    def test_different_bits_produce_different_entries(self):
        """Different bit widths use different quantizers, so different
        signs, so different cache keys."""
        self._build_wrapper(bits=3, group_size=128)
        self._build_wrapper(bits=4, group_size=128)
        self.assertGreaterEqual(
            len(self._cache),
            2,
            f"expected ≥2 cache entries for two different bit widths, got {len(self._cache)}",
        )

    def test_cached_matrix_is_correct_shape(self):
        """Built rotation matrix should be (group_size, group_size)."""
        self._build_wrapper(group_size=128)
        (mat,) = list(self._cache.values())[:1]
        self.assertEqual(
            mat.shape,
            (128, 128),
            f"expected (128, 128) rotation matrix, got {tuple(mat.shape)}",
        )

    def test_cached_matrix_is_contiguous(self):
        self._build_wrapper()
        (mat,) = list(self._cache.values())[:1]
        self.assertTrue(
            mat.is_contiguous(),
            "cached rotation matrix must be contiguous — downstream "
            "Triton kernels assume contiguous strides",
        )


class TestEagerCacheSurvivesWithoutTriton(unittest.TestCase):
    """If Triton is not importable (dev box, CI without GPU), the eager
    population must degrade gracefully rather than crashing construction.

    Behaviour: ``_triton_available`` defaults to False at import time and
    only flips to True after ``_ensure_triton_backends()`` succeeds. On a
    Triton-less box that probe returns False, so the eager cache call is
    skipped entirely. We verify TurboQuantWrapper still constructs in
    that scenario.
    """

    def test_construction_does_not_raise_when_triton_unavailable(self):
        from turboquant_vllm import weight_quant

        # Simulate "Triton probe returned False" state.
        saved = weight_quant._triton_available
        weight_quant._triton_available = False
        try:
            linear = nn.Linear(256, 128, bias=False)
            weight_quant.TurboQuantWrapper(linear, bits=3, group_size=128)
        finally:
            weight_quant._triton_available = saved


if __name__ == "__main__":
    unittest.main(verbosity=2)
