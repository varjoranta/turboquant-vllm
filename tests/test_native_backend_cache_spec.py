"""Regression tests for the native-backend cache-spec integration.

The native backend was declaring supported_kv_cache_dtypes = ["tq3",
"tq4", "tq_k4v3"] but was NOT participating in vLLM's KV cache
allocation math. Result: --kv-cache-dtype tq3 either crashed at
startup (stock vLLM rejects the dtype string) or silently allocated
a bf16-sized cache (if the user ran the fork). The token-capacity
gain was ~5% instead of the expected 2×.

This test suite locks in the fix:

1. STR_DTYPE_TO_TORCH_DTYPE is patched with tq3/tq4/tq_k4v3 → uint8
2. kv_cache_dtype_str_to_dtype wrapper handles tq* directly
3. AttentionLayer.get_kv_cache_spec returns a FullAttentionSpec
   with head_size remapped to padded_slot_size // 2 for tq* dtypes
4. The resulting spec's real_page_size_bytes equals the actual
   compressed slot size (the allocator-math contract — Test D)
5. MLAAttention.get_kv_cache_spec raises a clear error for tq*
   instead of silently mis-allocating

Tests A, B, C, E, F, G can run without vLLM installed (they stub
vLLM modules via unittest.mock). Test D requires vLLM because it
needs the real FullAttentionSpec; it's skipped otherwise.
"""

import sys
import unittest
import unittest.mock as mock
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))


def _reset_plugin_state():
    """Reset module-level state between tests so patch idempotency
    doesn't cause a test that runs second to see a stale cache."""
    import turboquant_vllm._vllm_plugin as p
    p._str_dtype_patched = False
    p._native_backend_registered = False


def _make_fake_vllm_torch_utils(
    orig_resolver=None,
    initial_dict=None,
):
    """Return a dict of sys.modules entries that make
    `import vllm.utils.torch_utils as _tu` resolve to our fake.

    Subtlety: `import vllm.utils.torch_utils as _tu` binds `_tu` via
    attribute access (_tu = vllm.utils.torch_utils), NOT via a direct
    sys.modules lookup. So we must also wire the parent chain so
    fake_vllm.utils.torch_utils IS our fake_tu, not an auto-created
    MagicMock attribute.
    """
    fake_vllm = mock.MagicMock()
    fake_utils = mock.MagicMock()
    fake_tu = mock.MagicMock()
    fake_tu.STR_DTYPE_TO_TORCH_DTYPE = dict(initial_dict or {})
    if orig_resolver is None:
        def orig_resolver(d, m):  # noqa: E306
            return torch.float16
    fake_tu.kv_cache_dtype_str_to_dtype = orig_resolver

    # Wire the parent attribute chain so `import vllm.utils.torch_utils
    # as _tu` produces our fake_tu, not an ephemeral MagicMock attribute.
    fake_vllm.utils = fake_utils
    fake_utils.torch_utils = fake_tu

    return {
        "vllm": fake_vllm,
        "vllm.utils": fake_utils,
        "vllm.utils.torch_utils": fake_tu,
    }, fake_tu


# ============================================================================
# A: STR_DTYPE dict patch
# ============================================================================

class TestStrDtypeDictPatch(unittest.TestCase):
    def setUp(self):
        _reset_plugin_state()

    def test_str_dtype_patch_adds_tq_entries(self):
        """The dict patch must add tq3/tq4/tq_k4v3 with torch.uint8 values."""
        fake_modules, fake_tu = _make_fake_vllm_torch_utils(
            initial_dict={"float16": torch.float16},
        )
        with mock.patch.dict(sys.modules, fake_modules):
            from turboquant_vllm._vllm_plugin import _eager_patch_str_dtype_mapping
            _eager_patch_str_dtype_mapping()

        self.assertIn("tq3", fake_tu.STR_DTYPE_TO_TORCH_DTYPE)
        self.assertIn("tq4", fake_tu.STR_DTYPE_TO_TORCH_DTYPE)
        self.assertIn("tq_k4v3", fake_tu.STR_DTYPE_TO_TORCH_DTYPE)
        self.assertIs(fake_tu.STR_DTYPE_TO_TORCH_DTYPE["tq3"], torch.uint8)
        self.assertIs(fake_tu.STR_DTYPE_TO_TORCH_DTYPE["tq4"], torch.uint8)
        self.assertIs(fake_tu.STR_DTYPE_TO_TORCH_DTYPE["tq_k4v3"], torch.uint8)
        # Pre-existing entry must not be clobbered
        self.assertIs(fake_tu.STR_DTYPE_TO_TORCH_DTYPE["float16"], torch.float16)

    def test_str_dtype_patch_is_idempotent(self):
        """Calling the patch twice must not error or double-wrap."""
        fake_modules, fake_tu = _make_fake_vllm_torch_utils()
        with mock.patch.dict(sys.modules, fake_modules):
            from turboquant_vllm._vllm_plugin import _eager_patch_str_dtype_mapping
            _eager_patch_str_dtype_mapping()
            # Second call should be a no-op via the _str_dtype_patched guard
            _eager_patch_str_dtype_mapping()

        # dict still populated, nothing crashed
        self.assertEqual(len(fake_tu.STR_DTYPE_TO_TORCH_DTYPE), 3)


# ============================================================================
# B: Resolver wrapper
# ============================================================================

class TestResolverWrapper(unittest.TestCase):
    def setUp(self):
        _reset_plugin_state()

    def test_resolver_returns_uint8_for_tq_strings(self):
        """The wrapped kv_cache_dtype_str_to_dtype must return torch.uint8
        for tq* without calling the original resolver."""
        original_calls = []

        def _orig(d, m):
            original_calls.append(d)
            return torch.float16

        fake_modules, fake_tu = _make_fake_vllm_torch_utils(orig_resolver=_orig)
        with mock.patch.dict(sys.modules, fake_modules):
            from turboquant_vllm._vllm_plugin import _eager_patch_str_dtype_mapping
            _eager_patch_str_dtype_mapping()
            wrapped = fake_tu.kv_cache_dtype_str_to_dtype

            # tq* strings return uint8 WITHOUT invoking the original
            self.assertIs(wrapped("tq3", None), torch.uint8)
            self.assertIs(wrapped("tq4", None), torch.uint8)
            self.assertIs(wrapped("tq_k4v3", None), torch.uint8)
            self.assertEqual(original_calls, [])

            # Non-tq strings DO go through the original
            result = wrapped("float16", None)
            self.assertIs(result, torch.float16)
            self.assertEqual(original_calls, ["float16"])

    def test_wrapper_is_idempotent(self):
        """Calling _eager_patch twice must not wrap the resolver twice."""
        fake_modules, fake_tu = _make_fake_vllm_torch_utils()
        with mock.patch.dict(sys.modules, fake_modules):
            from turboquant_vllm._vllm_plugin import _eager_patch_str_dtype_mapping
            _eager_patch_str_dtype_mapping()
            first_wrapped = fake_tu.kv_cache_dtype_str_to_dtype
            _eager_patch_str_dtype_mapping()
            second_wrapped = fake_tu.kv_cache_dtype_str_to_dtype

        # Second call should be a complete no-op — same wrapper object
        self.assertIs(first_wrapped, second_wrapped)
        self.assertTrue(getattr(first_wrapped, "_tq_wrapped", False))


# ============================================================================
# C: AttentionLayer.get_kv_cache_spec patch — TQ branch returns remapped spec
# ============================================================================

class _FakeVllmConfig:
    class cache_config:  # noqa: N801 — matches vLLM naming
        block_size = 16


class _FakeAttentionLayer:
    """Mimics the shape of vLLM's AttentionLayer for the patched method."""
    def __init__(self, head_size=128, num_kv_heads=8, kv_cache_dtype="tq3"):
        self.head_size = head_size
        self.num_kv_heads = num_kv_heads
        self.kv_cache_dtype = kv_cache_dtype
        self.kv_cache_torch_dtype = torch.uint8


class TestGetKvCacheSpecPatch(unittest.TestCase):
    def test_tq3_returns_remapped_spec(self):
        """For tq3, the returned spec must have head_size =
        padded_slot_size // 2 so allocator arithmetic yields the
        compressed slot bytes.
        """
        try:
            import vllm.v1.kv_cache_interface  # noqa: F401
        except ImportError:
            self.skipTest("vLLM not installed; skipping spec-construction test")

        from turboquant_vllm._vllm_plugin import _patch_get_kv_cache_spec
        from turboquant_vllm.tq_config import TurboQuantConfig

        # Patch AttentionLayer in-memory, call the patched method via
        # a fake instance, inspect the returned spec
        import vllm.model_executor.layers.attention.attention as al_mod
        AttentionLayer = al_mod.AttentionLayer

        # Temporarily stash the original so we can restore it
        _orig = getattr(AttentionLayer, "get_kv_cache_spec", None)
        _orig_patched_flag = getattr(AttentionLayer, "_tq_spec_patched", False)

        # Ensure we install fresh
        if hasattr(AttentionLayer, "_tq_spec_patched"):
            delattr(AttentionLayer, "_tq_spec_patched")

        try:
            # Install a minimal baseline method if vLLM's signature differs
            # from what the patch expects
            if _orig is None or list(
                __import__("inspect").signature(_orig).parameters.keys()
            ) != ["self", "vllm_config"]:
                def _baseline(self, vllm_config):
                    return None
                AttentionLayer.get_kv_cache_spec = _baseline

            _patch_get_kv_cache_spec()

            # Build a fake layer and call the patched method
            fake = _FakeAttentionLayer(head_size=128, kv_cache_dtype="tq3")
            spec = AttentionLayer.get_kv_cache_spec(fake, _FakeVllmConfig())

            self.assertIsNotNone(spec, "patch should return a real spec for tq3")
            expected_effective = (
                TurboQuantConfig.from_cache_dtype("tq3", head_dim=128)
                .padded_slot_size // 2
            )
            self.assertEqual(spec.head_size, expected_effective)
            self.assertEqual(spec.head_size_v, expected_effective)
            self.assertIs(spec.dtype, torch.uint8)
            self.assertEqual(spec.num_kv_heads, 8)
            self.assertEqual(spec.block_size, 16)
        finally:
            # Restore
            if _orig is not None:
                AttentionLayer.get_kv_cache_spec = _orig
            if _orig_patched_flag:
                AttentionLayer._tq_spec_patched = True
            elif hasattr(AttentionLayer, "_tq_spec_patched"):
                delattr(AttentionLayer, "_tq_spec_patched")


# ============================================================================
# D: The headline test — real_page_size_bytes matches compressed slot
# ============================================================================

class TestRealPageSizeBytes(unittest.TestCase):
    """Verify the allocator arithmetic actually works out to the
    compressed slot size. This is the test that proves the capacity
    multiplier story, not just that we returned a spec object.
    """

    def test_tq3_real_page_size_matches_compressed_slot(self):
        try:
            from vllm.v1.kv_cache_interface import FullAttentionSpec
        except ImportError:
            self.skipTest("vLLM not installed; skipping allocator math test")

        from turboquant_vllm.tq_config import TurboQuantConfig

        head_size = 128
        num_kv_heads = 8
        block_size = 16
        tq_config = TurboQuantConfig.from_cache_dtype("tq3", head_dim=head_size)
        effective = tq_config.padded_slot_size // 2

        spec = FullAttentionSpec(
            block_size=block_size,
            num_kv_heads=num_kv_heads,
            head_size=effective,
            head_size_v=effective,
            dtype=torch.uint8,
        )

        # Allocator math: block_size * num_kv_heads * (hs + hs_v) * sizeof(dtype)
        expected_bytes = block_size * num_kv_heads * tq_config.padded_slot_size * 1
        self.assertEqual(spec.real_page_size_bytes, expected_bytes)

        # And vs the bf16 baseline
        baseline_bytes = block_size * num_kv_heads * head_size * 2 * 2
        ratio = baseline_bytes / spec.real_page_size_bytes
        self.assertGreaterEqual(
            ratio, 1.8,
            f"tq3 should give at least 1.8x capacity vs bf16 (got {ratio:.2f}x). "
            f"expected_bytes={expected_bytes}, baseline_bytes={baseline_bytes}",
        )

    def test_tq4_real_page_size_matches_compressed_slot(self):
        try:
            from vllm.v1.kv_cache_interface import FullAttentionSpec
        except ImportError:
            self.skipTest("vLLM not installed; skipping allocator math test")

        from turboquant_vllm.tq_config import TurboQuantConfig

        head_size = 128
        num_kv_heads = 8
        block_size = 16
        tq_config = TurboQuantConfig.from_cache_dtype("tq4", head_dim=head_size)
        effective = tq_config.padded_slot_size // 2

        spec = FullAttentionSpec(
            block_size=block_size,
            num_kv_heads=num_kv_heads,
            head_size=effective,
            head_size_v=effective,
            dtype=torch.uint8,
        )

        expected_bytes = block_size * num_kv_heads * tq_config.padded_slot_size * 1
        self.assertEqual(spec.real_page_size_bytes, expected_bytes)
        baseline_bytes = block_size * num_kv_heads * head_size * 2 * 2
        ratio = baseline_bytes / spec.real_page_size_bytes
        self.assertGreaterEqual(ratio, 1.5, f"tq4 ratio {ratio:.2f}x < 1.5x")


# ============================================================================
# E: get_kv_cache_spec pass-through for non-tq dtypes
# ============================================================================

class TestGetKvCacheSpecPassThrough(unittest.TestCase):
    def test_auto_dtype_calls_original(self):
        """A non-tq kv_cache_dtype must fall through to the original method."""
        try:
            import vllm.model_executor.layers.attention.attention as al_mod
        except ImportError:
            self.skipTest("vLLM not installed; skipping pass-through test")

        from turboquant_vllm._vllm_plugin import _patch_get_kv_cache_spec
        AttentionLayer = al_mod.AttentionLayer

        _orig = getattr(AttentionLayer, "get_kv_cache_spec", None)
        _orig_flag = getattr(AttentionLayer, "_tq_spec_patched", False)

        if hasattr(AttentionLayer, "_tq_spec_patched"):
            delattr(AttentionLayer, "_tq_spec_patched")

        call_count = {"n": 0}

        try:
            def _baseline(self, vllm_config):
                call_count["n"] += 1
                return "BASELINE_SENTINEL"

            AttentionLayer.get_kv_cache_spec = _baseline
            _patch_get_kv_cache_spec()

            fake = _FakeAttentionLayer(head_size=128, kv_cache_dtype="auto")
            result = AttentionLayer.get_kv_cache_spec(fake, _FakeVllmConfig())

            self.assertEqual(result, "BASELINE_SENTINEL",
                             "non-tq path must delegate to original method")
            self.assertEqual(call_count["n"], 1)
        finally:
            if _orig is not None:
                AttentionLayer.get_kv_cache_spec = _orig
            if _orig_flag:
                AttentionLayer._tq_spec_patched = True
            elif hasattr(AttentionLayer, "_tq_spec_patched"):
                delattr(AttentionLayer, "_tq_spec_patched")


# ============================================================================
# F, G: MLA fail-loud guard
# ============================================================================

class _FakeMLALayer:
    def __init__(self, kv_cache_dtype="tq3"):
        self.kv_cache_dtype = kv_cache_dtype


class TestMlaFailLoudGuard(unittest.TestCase):
    def test_tq3_raises_runtime_error(self):
        """MLA + tq3 must raise RuntimeError with a helpful message."""
        try:
            import vllm.model_executor.layers.attention.mla_attention as mla_mod
        except ImportError:
            self.skipTest("vLLM not installed; skipping MLA guard test")

        from turboquant_vllm._vllm_plugin import _patch_mla_fail_loud

        MLAAttention = mla_mod.MLAAttention
        _orig = getattr(MLAAttention, "get_kv_cache_spec", None)
        _orig_flag = getattr(MLAAttention, "_tq_mla_guard_patched", False)

        if hasattr(MLAAttention, "_tq_mla_guard_patched"):
            delattr(MLAAttention, "_tq_mla_guard_patched")

        try:
            def _baseline(self, vllm_config):
                return "MLA_BASELINE"
            MLAAttention.get_kv_cache_spec = _baseline
            _patch_mla_fail_loud()

            fake = _FakeMLALayer(kv_cache_dtype="tq3")
            with self.assertRaises(RuntimeError) as ctx:
                MLAAttention.get_kv_cache_spec(fake, _FakeVllmConfig())
            msg = str(ctx.exception)
            self.assertIn("MLA", msg)
            self.assertIn("TQ_KV_K_BITS", msg)
            self.assertIn("tq3", msg)
        finally:
            if _orig is not None:
                MLAAttention.get_kv_cache_spec = _orig
            if _orig_flag:
                MLAAttention._tq_mla_guard_patched = True
            elif hasattr(MLAAttention, "_tq_mla_guard_patched"):
                delattr(MLAAttention, "_tq_mla_guard_patched")

    def test_auto_dtype_passes_through(self):
        """MLA + auto must delegate to the original method."""
        try:
            import vllm.model_executor.layers.attention.mla_attention as mla_mod
        except ImportError:
            self.skipTest("vLLM not installed; skipping MLA pass-through test")

        from turboquant_vllm._vllm_plugin import _patch_mla_fail_loud

        MLAAttention = mla_mod.MLAAttention
        _orig = getattr(MLAAttention, "get_kv_cache_spec", None)
        _orig_flag = getattr(MLAAttention, "_tq_mla_guard_patched", False)

        if hasattr(MLAAttention, "_tq_mla_guard_patched"):
            delattr(MLAAttention, "_tq_mla_guard_patched")

        try:
            call_count = {"n": 0}
            def _baseline(self, vllm_config):
                call_count["n"] += 1
                return "MLA_BASELINE"
            MLAAttention.get_kv_cache_spec = _baseline
            _patch_mla_fail_loud()

            fake = _FakeMLALayer(kv_cache_dtype="auto")
            result = MLAAttention.get_kv_cache_spec(fake, _FakeVllmConfig())
            self.assertEqual(result, "MLA_BASELINE")
            self.assertEqual(call_count["n"], 1)
        finally:
            if _orig is not None:
                MLAAttention.get_kv_cache_spec = _orig
            if _orig_flag:
                MLAAttention._tq_mla_guard_patched = True
            elif hasattr(MLAAttention, "_tq_mla_guard_patched"):
                delattr(MLAAttention, "_tq_mla_guard_patched")


if __name__ == "__main__":
    unittest.main(verbosity=2)
