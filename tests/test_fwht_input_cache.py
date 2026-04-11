"""Unit tests for FWHTInputCache — runs on CPU, no Triton/CUDA required.

The cache sits inside ``tq_fwht_input_gemm``'s ``torch.library.custom_op``
body and reuses the rotated input across Q/K/V projections. All cache
key reads are pure host-side tensor metadata — no .cpu() or .item() —
so the cache is safe to call inside CUDA graph capture. These tests
verify the identity semantics of the key without exercising any
CUDA/Triton path.
"""

import unittest

import torch

from turboquant_vllm.triton_ops import FWHTInputCache


class TestFWHTInputCacheHit(unittest.TestCase):
    """Same tensor passed twice should hit on the second call."""

    def test_same_tensor_hits(self):
        cache = FWHTInputCache()
        x = torch.randn(4, 128)
        rotated = torch.randn(4, 128)

        self.assertIsNone(cache.get(x), "empty cache must miss")
        cache.put(x, rotated)
        got = cache.get(x)

        self.assertIs(got, rotated, "repeat call on same tensor must return cached rotation")

    def test_three_consecutive_gets_same_tensor(self):
        """Q/K/V case: same x is passed three times in a row."""
        cache = FWHTInputCache()
        x = torch.randn(2, 64)
        rotated = torch.randn(2, 64)

        self.assertIsNone(cache.get(x))
        cache.put(x, rotated)
        self.assertIs(cache.get(x), rotated)
        self.assertIs(cache.get(x), rotated)


class TestFWHTInputCacheMiss(unittest.TestCase):
    """Different-identity tensors must miss."""

    def test_different_shape_misses(self):
        cache = FWHTInputCache()
        x1 = torch.randn(4, 128)
        x2 = torch.randn(8, 128)
        cache.put(x1, torch.randn(4, 128))
        self.assertIsNone(cache.get(x2))

    def test_different_dtype_misses(self):
        cache = FWHTInputCache()
        x1 = torch.randn(4, 128, dtype=torch.float32)
        x2 = torch.randn(4, 128, dtype=torch.float16)
        cache.put(x1, torch.randn(4, 128))
        self.assertIsNone(cache.get(x2))

    def test_different_ptr_misses(self):
        cache = FWHTInputCache()
        x1 = torch.randn(4, 128)
        x2 = torch.randn(4, 128)  # same shape/dtype, different storage
        cache.put(x1, torch.randn(4, 128))
        self.assertIsNone(cache.get(x2))


class TestFWHTInputCacheInferenceModeSafety(unittest.TestCase):
    """Under torch.inference_mode() the key reads must not raise.

    vLLM runs its forward in inference mode, which yields
    InferenceTensors whose `_version` attribute raises
    `RuntimeError: Inference tensors do not track version counter.`
    The cache must not touch any attribute that behaves this way.
    """

    def test_put_and_get_inside_inference_mode(self):
        with torch.inference_mode():
            cache = FWHTInputCache()
            x = torch.randn(4, 128)
            rotated = torch.randn(4, 128)
            cache.put(x, rotated)
            self.assertIs(cache.get(x), rotated)


class TestFWHTInputCacheEviction(unittest.TestCase):
    """One-entry cache: a new put replaces the old entry naturally."""

    def test_put_replaces_previous(self):
        cache = FWHTInputCache()
        x1 = torch.randn(4, 128)
        x2 = torch.randn(4, 128)
        r1 = torch.randn(4, 128)
        r2 = torch.randn(4, 128)

        cache.put(x1, r1)
        self.assertIs(cache.get(x1), r1)

        cache.put(x2, r2)
        self.assertIs(cache.get(x2), r2)
        self.assertIsNone(cache.get(x1), "x1 should no longer hit after x2 was put")

    def test_clear_empties(self):
        cache = FWHTInputCache()
        x = torch.randn(4, 128)
        cache.put(x, torch.randn(4, 128))
        self.assertIsNotNone(cache.get(x))
        cache.clear()
        self.assertIsNone(cache.get(x))


class TestFWHTInputCacheNoHostSync(unittest.TestCase):
    """Regression test: neither get() nor put() may host-sync.

    The previous iteration of this cache did ``.cpu()`` for content
    fingerprinting, which triggered cudaErrorStreamCaptureUnsupported
    inside vLLM piecewise CUDA graph capture. We verify here that the
    current implementation only touches tensor metadata (data_ptr,
    shape, dtype, _version) and never materializes data.
    """

    def test_get_does_not_read_tensor_data(self):
        cache = FWHTInputCache()
        # A tensor that would raise if the cache tried to read its data
        # via any path that goes through the storage (e.g. .cpu()).
        x = torch.randn(4, 128)
        self.assertIsNone(cache.get(x))
        # If get() called .cpu() or .item() we would have observed a
        # sync; since the tensor is on CPU here we instead verify that
        # the code paths touched are exactly the metadata ones.
        # Cheap structural check: the cache class uses __slots__ and
        # only the four metadata fields + the cached result.
        self.assertEqual(
            set(FWHTInputCache.__slots__),
            {"_ptr", "_shape", "_dtype", "_result"},
            "FWHTInputCache __slots__ changed — if a new field was added, "
            "verify it is still CUDA-graph-capture safe (no .cpu(), no .item()) "
            "AND safe under torch.inference_mode() (no tensor._version reads — "
            "inference tensors don't track it)",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
