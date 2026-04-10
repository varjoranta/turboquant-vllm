"""Regression tests for the native-backend cache-spec integration.

The native backend was declaring supported_kv_cache_dtypes = ["tq3",
"tq4", "tq_k4v3"] but was NOT participating in vLLM's KV cache
allocation math. Result: --kv-cache-dtype tq3 either crashed at
startup (stock vLLM rejects the dtype string) or silently allocated
a bf16-sized cache (if the user ran the fork). The token-capacity
gain was ~5% instead of the expected 2x.

This test suite locks in the fix:

1. STR_DTYPE_TO_TORCH_DTYPE is patched with tq3/tq4/tq_k4v3 -> uint8
2. kv_cache_dtype_str_to_dtype wrapper handles tq* directly
3. AttentionLayer.get_kv_cache_spec returns a FullAttentionSpec
   with head_size remapped to padded_slot_size // 2 for tq* dtypes
4. The resulting spec's real_page_size_bytes equals the actual
   compressed slot size
5. MLAAttention.get_kv_cache_spec raises a clear error for tq*
   instead of silently mis-allocating

Tests that stub vLLM via mock.patch.dict can run without vLLM
installed; tests that need the real FullAttentionSpec /
AttentionLayer / MLAAttention classes skip cleanly otherwise.
"""

import contextlib
import sys
import unittest
import unittest.mock as mock
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))


def _make_fake_vllm_torch_utils(orig_resolver=None, initial_dict=None):
    """Build a sys.modules stub chain for `import vllm.utils.torch_utils`.

    Subtlety: `import vllm.utils.torch_utils as _tu` binds _tu via
    attribute access (_tu = vllm.utils.torch_utils), NOT a direct
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

    fake_vllm.utils = fake_utils
    fake_utils.torch_utils = fake_tu

    return {
        "vllm": fake_vllm,
        "vllm.utils": fake_utils,
        "vllm.utils.torch_utils": fake_tu,
    }, fake_tu


@contextlib.contextmanager
def _temporary_method_patch(cls, method_name, flag_name, replacement):
    """Temporarily replace a method on cls and clear its patch-applied flag.

    Installs `replacement` as cls.<method_name>, clears the idempotency
    flag so the plugin's patcher re-runs, and restores both on exit. Used
    by tests C, E, F, G to stage a baseline method the plugin can wrap,
    then verify the wrapped behavior, without leaking state across tests.
    """
    orig_method = getattr(cls, method_name, None)
    had_flag = flag_name in cls.__dict__
    if had_flag:
        delattr(cls, flag_name)
    setattr(cls, method_name, replacement)
    try:
        yield
    finally:
        if orig_method is not None:
            setattr(cls, method_name, orig_method)
        elif method_name in cls.__dict__:
            delattr(cls, method_name)
        if flag_name in cls.__dict__:
            delattr(cls, flag_name)


def _reset_plugin_state():
    """Clear all plugin patch flags so each test starts from scratch."""
    from turboquant_vllm._vllm_plugin import _tq_reset_patches_for_test
    _tq_reset_patches_for_test()


# ============================================================================
# A: STR_DTYPE dict patch
# ============================================================================

class TestStrDtypeDictPatch(unittest.TestCase):
    def setUp(self):
        _reset_plugin_state()

    def test_str_dtype_patch_adds_tq_entries(self):
        """Adds tq3/tq4/tq_k4v3 -> uint8, preserves pre-existing entries."""
        fake_modules, fake_tu = _make_fake_vllm_torch_utils(
            initial_dict={"float16": torch.float16},
        )
        with mock.patch.dict(sys.modules, fake_modules):
            from turboquant_vllm._vllm_plugin import _eager_patch_str_dtype_mapping
            _eager_patch_str_dtype_mapping()

        self.assertIs(fake_tu.STR_DTYPE_TO_TORCH_DTYPE["tq3"], torch.uint8)
        self.assertIs(fake_tu.STR_DTYPE_TO_TORCH_DTYPE["tq4"], torch.uint8)
        self.assertIs(fake_tu.STR_DTYPE_TO_TORCH_DTYPE["tq_k4v3"], torch.uint8)
        self.assertIs(fake_tu.STR_DTYPE_TO_TORCH_DTYPE["float16"], torch.float16)

    def test_str_dtype_patch_is_idempotent(self):
        """Second call must not error or re-mutate."""
        fake_modules, fake_tu = _make_fake_vllm_torch_utils()
        with mock.patch.dict(sys.modules, fake_modules):
            from turboquant_vllm._vllm_plugin import _eager_patch_str_dtype_mapping
            _eager_patch_str_dtype_mapping()
            _eager_patch_str_dtype_mapping()

        self.assertEqual(len(fake_tu.STR_DTYPE_TO_TORCH_DTYPE), 3)


# ============================================================================
# B: Resolver wrapper
# ============================================================================

class TestResolverWrapper(unittest.TestCase):
    def setUp(self):
        _reset_plugin_state()

    def test_resolver_returns_uint8_for_tq_strings(self):
        """Wrapped resolver short-circuits tq*, delegates everything else."""
        original_calls = []

        def _orig(d, m):
            original_calls.append(d)
            return torch.float16

        fake_modules, fake_tu = _make_fake_vllm_torch_utils(orig_resolver=_orig)
        with mock.patch.dict(sys.modules, fake_modules):
            from turboquant_vllm._vllm_plugin import _eager_patch_str_dtype_mapping
            _eager_patch_str_dtype_mapping()
            wrapped = fake_tu.kv_cache_dtype_str_to_dtype

            self.assertIs(wrapped("tq3", None), torch.uint8)
            self.assertIs(wrapped("tq4", None), torch.uint8)
            self.assertIs(wrapped("tq_k4v3", None), torch.uint8)
            self.assertEqual(original_calls, [])

            self.assertIs(wrapped("float16", None), torch.float16)
            self.assertEqual(original_calls, ["float16"])

    def test_wrapper_is_idempotent(self):
        """Second eager-patch call must not double-wrap the resolver."""
        fake_modules, fake_tu = _make_fake_vllm_torch_utils()
        with mock.patch.dict(sys.modules, fake_modules):
            from turboquant_vllm._vllm_plugin import _eager_patch_str_dtype_mapping
            _eager_patch_str_dtype_mapping()
            first_wrapped = fake_tu.kv_cache_dtype_str_to_dtype
            _eager_patch_str_dtype_mapping()
            second_wrapped = fake_tu.kv_cache_dtype_str_to_dtype

        self.assertIs(first_wrapped, second_wrapped)
        self.assertTrue(getattr(first_wrapped, "_tq_wrapped", False))


# ============================================================================
# Fake vLLM types for the spec-construction tests below
# ============================================================================

class _FakeCacheConfig:
    block_size = 16


class _FakeVllmConfig:
    cache_config = _FakeCacheConfig()


class _FakeAttentionLayer:
    def __init__(self, head_size=128, num_kv_heads=8, kv_cache_dtype="tq3"):
        self.head_size = head_size
        self.num_kv_heads = num_kv_heads
        self.kv_cache_dtype = kv_cache_dtype
        self.kv_cache_torch_dtype = torch.uint8


class _FakeMLALayer:
    def __init__(self, kv_cache_dtype="tq3"):
        self.kv_cache_dtype = kv_cache_dtype


# ============================================================================
# C: AttentionLayer.get_kv_cache_spec patch returns remapped spec
# ============================================================================

class TestGetKvCacheSpecPatch(unittest.TestCase):
    def setUp(self):
        _reset_plugin_state()

    def test_tq3_returns_remapped_spec(self):
        """For tq3 the returned spec has head_size = padded_slot_size // 2."""
        try:
            import vllm.model_executor.layers.attention.attention as al_mod  # noqa: F401
            import vllm.v1.kv_cache_interface  # noqa: F401
        except ImportError:
            self.skipTest("vLLM not installed")

        from turboquant_vllm._vllm_plugin import (
            _patch_get_kv_cache_spec, _tq_effective_head_size,
        )
        from vllm.model_executor.layers.attention.attention import AttentionLayer

        def _baseline(self, vllm_config):
            return None

        with _temporary_method_patch(
            AttentionLayer, "get_kv_cache_spec", "_tq_spec_patched", _baseline
        ):
            _patch_get_kv_cache_spec()
            fake = _FakeAttentionLayer(head_size=128, kv_cache_dtype="tq3")
            spec = AttentionLayer.get_kv_cache_spec(fake, _FakeVllmConfig())

        self.assertIsNotNone(spec)
        expected = _tq_effective_head_size("tq3", 128)
        self.assertEqual(spec.head_size, expected)
        self.assertEqual(spec.head_size_v, expected)
        self.assertIs(spec.dtype, torch.uint8)
        self.assertEqual(spec.num_kv_heads, 8)
        self.assertEqual(spec.block_size, 16)


# ============================================================================
# D: Allocator math sanity check — real_page_size_bytes vs bf16 baseline
# ============================================================================

class TestFullAttentionSpecArithmetic(unittest.TestCase):
    """Verify that passing head_size = padded_slot_size // 2 and
    dtype = torch.uint8 to vLLM's FullAttentionSpec yields a
    real_page_size_bytes equal to the compressed slot size.

    This test does NOT exercise the plugin's patched method; it's a
    sanity check on the *vLLM side* of the contract the plugin relies
    on. If vLLM ever changes how real_page_size_bytes is computed
    from head_size/dtype, this test fails and we know the plugin's
    remap trick needs updating before the patch can be trusted.
    """

    def _check(self, cache_dtype, min_ratio):
        try:
            from vllm.v1.kv_cache_interface import FullAttentionSpec
        except ImportError:
            self.skipTest("vLLM not installed")

        from turboquant_vllm.tq_config import TurboQuantConfig

        head_size = 128
        num_kv_heads = 8
        block_size = 16
        tq_config = TurboQuantConfig.from_cache_dtype(cache_dtype, head_dim=head_size)
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
        self.assertGreaterEqual(
            ratio, min_ratio,
            f"{cache_dtype} ratio {ratio:.2f}x < {min_ratio}x "
            f"(compressed={expected_bytes}, bf16={baseline_bytes})",
        )

    def test_tq3_capacity_ratio(self):
        self._check("tq3", min_ratio=1.8)

    def test_tq4_capacity_ratio(self):
        self._check("tq4", min_ratio=1.5)


# ============================================================================
# E: get_kv_cache_spec pass-through for non-tq dtypes
# ============================================================================

class TestGetKvCacheSpecPassThrough(unittest.TestCase):
    def setUp(self):
        _reset_plugin_state()

    def test_auto_dtype_calls_original(self):
        """A non-tq kv_cache_dtype must delegate to the original method."""
        try:
            import vllm.model_executor.layers.attention.attention as al_mod  # noqa: F401
        except ImportError:
            self.skipTest("vLLM not installed")

        from turboquant_vllm._vllm_plugin import _patch_get_kv_cache_spec
        from vllm.model_executor.layers.attention.attention import AttentionLayer

        call_count = {"n": 0}

        def _baseline(self, vllm_config):
            call_count["n"] += 1
            return "BASELINE_SENTINEL"

        with _temporary_method_patch(
            AttentionLayer, "get_kv_cache_spec", "_tq_spec_patched", _baseline
        ):
            _patch_get_kv_cache_spec()
            fake = _FakeAttentionLayer(head_size=128, kv_cache_dtype="auto")
            result = AttentionLayer.get_kv_cache_spec(fake, _FakeVllmConfig())

        self.assertEqual(result, "BASELINE_SENTINEL")
        self.assertEqual(call_count["n"], 1)


# ============================================================================
# F, G: MLA fail-loud guard
# ============================================================================

class TestMlaFailLoudGuard(unittest.TestCase):
    def setUp(self):
        _reset_plugin_state()

    def test_tq3_raises_runtime_error(self):
        """MLA + tq3 must raise RuntimeError mentioning MLA and TQ_KV_K_BITS."""
        try:
            import vllm.model_executor.layers.attention.mla_attention as mla_mod  # noqa: F401
        except ImportError:
            self.skipTest("vLLM not installed")

        from turboquant_vllm._vllm_plugin import _patch_mla_fail_loud
        from vllm.model_executor.layers.attention.mla_attention import MLAAttention

        def _baseline(self, vllm_config):
            return "MLA_BASELINE"

        with _temporary_method_patch(
            MLAAttention, "get_kv_cache_spec", "_tq_mla_guard_patched", _baseline
        ):
            _patch_mla_fail_loud()
            fake = _FakeMLALayer(kv_cache_dtype="tq3")
            with self.assertRaises(RuntimeError) as ctx:
                MLAAttention.get_kv_cache_spec(fake, _FakeVllmConfig())

        msg = str(ctx.exception)
        self.assertIn("MLA", msg)
        self.assertIn("TQ_KV_K_BITS", msg)
        self.assertIn("tq3", msg)

    def test_auto_dtype_passes_through(self):
        """MLA + auto must delegate to the original method."""
        try:
            import vllm.model_executor.layers.attention.mla_attention as mla_mod  # noqa: F401
        except ImportError:
            self.skipTest("vLLM not installed")

        from turboquant_vllm._vllm_plugin import _patch_mla_fail_loud
        from vllm.model_executor.layers.attention.mla_attention import MLAAttention

        call_count = {"n": 0}

        def _baseline(self, vllm_config):
            call_count["n"] += 1
            return "MLA_BASELINE"

        with _temporary_method_patch(
            MLAAttention, "get_kv_cache_spec", "_tq_mla_guard_patched", _baseline
        ):
            _patch_mla_fail_loud()
            fake = _FakeMLALayer(kv_cache_dtype="auto")
            result = MLAAttention.get_kv_cache_spec(fake, _FakeVllmConfig())

        self.assertEqual(result, "MLA_BASELINE")
        self.assertEqual(call_count["n"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
