"""Local validation of native backend — no GPU, no vLLM required.

Tests:
  1. tq_config: TurboQuantConfig properties for tq3/tq4/tq_k4v3
  2. Centroids: Lloyd-Max solve for small d (d=8 to avoid scipy cost)
  3. Rotation matrices: orthogonality
  4. Buffer init: _init_tq_buffers produces correct shapes
  5. Store+decode round-trip (CPU, small tensors)
  6. Plugin registration logic (mocked vLLM)
"""

import math
import sys
import unittest
import unittest.mock as mock

import torch

# Add parent dir so we can import turboquant_vllm without installing
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))


# ============================================================================
# Helpers
# ============================================================================


def assert_close(a, b, atol=1e-4, msg=""):
    if not torch.allclose(a.float(), b.float(), atol=atol):
        diff = (a.float() - b.float()).abs().max().item()
        raise AssertionError(f"{msg} max_diff={diff:.6f} atol={atol}")


def make_dummy_kv_cache(num_blocks=4, block_size=16, num_kv_heads=4, slot_size=64):
    return torch.zeros(num_blocks, block_size, num_kv_heads, slot_size, dtype=torch.uint8)


# ============================================================================
# 1. TurboQuantConfig
# ============================================================================


class TestTurboQuantConfig(unittest.TestCase):
    def test_tq3_no_qjl(self):
        from turboquant_vllm.tq_config import TurboQuantConfig

        c = TurboQuantConfig.from_cache_dtype("tq3", head_dim=128)
        self.assertEqual(c.total_bits, 3)
        self.assertEqual(c.mse_bits, 3)
        self.assertEqual(c.n_centroids, 8)
        self.assertTrue(c.no_qjl)
        # True 3-bit pack: 8 indices → 3 bytes, so 128 * 3/8 = 48 + 2 = 50
        self.assertEqual(c.key_packed_size, 50)
        # value FP8 = 128 bytes
        self.assertEqual(c.value_packed_size, 128)
        self.assertEqual(c.slot_size, 178)

    def test_tq4_no_qjl(self):
        from turboquant_vllm.tq_config import TurboQuantConfig

        c = TurboQuantConfig.from_cache_dtype("tq4", head_dim=128)
        self.assertEqual(c.mse_bits, 4)
        self.assertEqual(c.n_centroids, 16)
        # key = ceil(128*4/8) + 2 = 64 + 2 = 66 bytes
        self.assertEqual(c.key_packed_size, 66)

    def test_tq_k4v3(self):
        from turboquant_vllm.tq_config import TurboQuantConfig

        c = TurboQuantConfig.from_cache_dtype("tq_k4v3", head_dim=128)
        self.assertTrue(c.asymmetric)
        self.assertEqual(c.total_bits, 4)
        self.assertEqual(c.v_total_bits, 3)
        # Regression: effective_value_quant_bits must honor asymmetric,
        # not return the default value_quant_bits=8. Otherwise tq_k4v3
        # silently stores V as FP8 and is identical to tq4.
        self.assertEqual(c.effective_value_quant_bits, 3)
        self.assertFalse(c.value_fp8)
        # 3-bit V storage: ceil(128 * 3 / 8) = 48 bytes + 4 for fp16 scale/zero
        self.assertEqual(c.value_packed_size, 52)

    def test_tq_k4v3_vs_tq4_differ(self):
        """tq_k4v3 and tq4 must produce different slot sizes (regression)."""
        from turboquant_vllm.tq_config import TurboQuantConfig

        tq4 = TurboQuantConfig.from_cache_dtype("tq4", head_dim=128)
        tq_k4v3 = TurboQuantConfig.from_cache_dtype("tq_k4v3", head_dim=128)
        self.assertNotEqual(
            tq4.slot_size,
            tq_k4v3.slot_size,
            "tq_k4v3 must use 3-bit V storage; if this fails the asymmetric "
            "path is broken and tq_k4v3 is silently aliased to tq4.",
        )
        # tq_k4v3 should be smaller than tq4 (3-bit V vs FP8 V)
        self.assertLess(tq_k4v3.slot_size, tq4.slot_size)

    def test_unknown_raises(self):
        from turboquant_vllm.tq_config import TurboQuantConfig

        with self.assertRaises(ValueError):
            TurboQuantConfig.from_cache_dtype("tq5", head_dim=128)


# ============================================================================
# 2. Centroids
# ============================================================================


class TestCentroids(unittest.TestCase):
    def test_lloyd_max_shape(self):
        from turboquant_vllm.tq_config import get_centroids

        c = get_centroids(d=8, bits=3)
        self.assertEqual(c.shape, (8,))

    def test_centroids_sorted(self):
        from turboquant_vllm.tq_config import get_centroids

        c = get_centroids(d=8, bits=3)
        self.assertTrue((c[1:] > c[:-1]).all(), "centroids must be sorted")

    def test_centroids_symmetric(self):
        from turboquant_vllm.tq_config import get_centroids

        c = get_centroids(d=8, bits=3)
        # Lloyd-Max on symmetric distribution → symmetric centroids
        self.assertTrue(torch.allclose(c, -c.flip(0), atol=1e-5), "centroids should be symmetric around 0")


# ============================================================================
# 3. Rotation matrices
# ============================================================================


class TestRotationMatrices(unittest.TestCase):
    def test_rotation_orthogonal(self):
        from turboquant_vllm.tq_config import generate_rotation_matrix

        Pi = generate_rotation_matrix(d=64, seed=42)
        I = Pi @ Pi.T
        self.assertTrue(torch.allclose(I, torch.eye(64), atol=1e-5), "Pi @ Pi.T should be identity")

    def test_rotation_deterministic(self):
        from turboquant_vllm.tq_config import generate_rotation_matrix

        Pi1 = generate_rotation_matrix(d=32, seed=1337)
        Pi2 = generate_rotation_matrix(d=32, seed=1337)
        self.assertTrue(torch.allclose(Pi1, Pi2), "same seed → same matrix")

    def test_different_seeds_different(self):
        from turboquant_vllm.tq_config import generate_rotation_matrix

        Pi1 = generate_rotation_matrix(d=32, seed=1)
        Pi2 = generate_rotation_matrix(d=32, seed=2)
        self.assertFalse(torch.allclose(Pi1, Pi2), "different seeds → different matrix")

    def test_qjl_matrix_shape(self):
        from turboquant_vllm.tq_config import generate_qjl_matrix

        S = generate_qjl_matrix(d=64, seed=42)
        self.assertEqual(S.shape, (64, 64))


# ============================================================================
# 4. Buffer init
# ============================================================================


class TestBufferInit(unittest.TestCase):
    def test_init_tq_buffers_shapes(self):
        from turboquant_vllm._vllm_plugin import _init_tq_buffers

        class FakeLayer:
            """Minimal nn.Module-like for register_buffer."""

            def register_buffer(self, name, tensor):
                setattr(self, name, tensor)

        layer = FakeLayer()
        _init_tq_buffers(layer, "tq3", head_size=128, prefix="model.layers.3.self_attn")

        self.assertEqual(layer._tq_Pi.shape, (128, 128))
        self.assertEqual(layer._tq_S.shape, (128, 128))
        self.assertEqual(layer._tq_centroids.shape, (8,))  # 2^3 centroids

    def test_layer_idx_from_prefix(self):
        from turboquant_vllm._vllm_plugin import _init_tq_buffers

        class FakeLayer:
            def register_buffer(self, name, tensor):
                setattr(self, name, tensor)

        l0, l5 = FakeLayer(), FakeLayer()
        _init_tq_buffers(l0, "tq3", head_size=64, prefix="model.layers.0.self_attn")
        _init_tq_buffers(l5, "tq3", head_size=64, prefix="model.layers.5.self_attn")
        # Different layer indices → different seeds → different Pi matrices
        self.assertFalse(torch.allclose(l0._tq_Pi, l5._tq_Pi), "layers 0 and 5 should have different rotation matrices")


# ============================================================================
# 5. Store + decode round-trip (CPU)
# ============================================================================


class TestStoreDecodeCPU(unittest.TestCase):
    """Smoke test: compress a few K/V vectors, write to cache, read back, check MSE."""

    def setUp(self):
        from turboquant_vllm.tq_config import (
            TurboQuantConfig,
            generate_rotation_matrix,
            generate_qjl_matrix,
            get_centroids,
        )

        self.D = 64
        self.H = 2
        self.cfg = TurboQuantConfig.from_cache_dtype("tq3", head_dim=self.D)
        seed = 42

        class FakeLayer:
            pass

        layer = FakeLayer()
        Pi = generate_rotation_matrix(self.D, seed=seed)
        layer._tq_Pi = Pi
        layer._tq_S = generate_qjl_matrix(self.D, seed=seed + 1)
        layer._tq_centroids = get_centroids(self.D, self.cfg.mse_bits)
        Pi_f = Pi.float().contiguous()
        layer._tq_PiT = Pi_f.T.contiguous()
        S_f = layer._tq_S.float().contiguous()
        layer._tq_Pi_S_T = (Pi_f @ S_f.T).contiguous()
        c = layer._tq_centroids.float()
        c_sorted, _ = c.sort()
        layer._tq_midpoints = (c_sorted[:-1] + c_sorted[1:]) / 2
        layer._tq_cached = True
        self.layer = layer

    def _make_impl(self):
        from turboquant_vllm.native_backend import TurboQuantAttentionImpl

        impl = TurboQuantAttentionImpl(
            num_heads=self.H, head_size=self.D, scale=1.0 / math.sqrt(self.D), num_kv_heads=self.H, kv_cache_dtype="tq3"
        )
        # Pre-initialize tq_config (normally done in _ensure_on_device)
        impl.tq_config = self.cfg
        impl._shifts_on_device = True
        impl._current_layer = self.layer
        return impl

    def test_store_writes_to_cache(self):
        impl = self._make_impl()
        N = 4
        kps = self.cfg.key_packed_size
        vps = self.cfg.value_packed_size
        slot_size = kps + vps
        # Make cache with enough room
        kv_cache = torch.zeros(2, 16, self.H, slot_size, dtype=torch.uint8)
        slot_mapping = torch.tensor([0, 1, 2, 3])
        key = torch.randn(N, self.H, self.D)
        value = torch.randn(N, self.H, self.D)

        impl._store_kv(
            key, value, kv_cache, slot_mapping, self.layer._tq_Pi, self.layer._tq_S, self.layer._tq_centroids
        )

        # Cache should now be non-zero in written slots
        for i in range(N):
            blk = i // 16
            off = i % 16
            slot_data = kv_cache[blk, off]  # (H, slot_size)
            self.assertFalse(slot_data.sum() == 0, f"slot {i} should be non-zero after store")

    def test_tq3_mse_reasonable(self):
        """TQ3 MSE on random normalized vectors should be < 0.15."""
        from turboquant_vllm.tq_config import get_centroids

        cfg = self.cfg
        D, H = self.D, self.H
        N = 32
        key = torch.randn(N, H, D)
        value = torch.randn(N, H, D)
        kps = cfg.key_packed_size
        vps = cfg.value_packed_size
        slot_size = kps + vps
        kv_cache = torch.zeros(4, 16, H, slot_size, dtype=torch.uint8)
        slot_mapping = torch.arange(N)

        impl = self._make_impl()
        impl._store_kv(
            key, value, kv_cache, slot_mapping, self.layer._tq_Pi, self.layer._tq_S, self.layer._tq_centroids
        )

        # Manually decode one token to check quality
        # Decode slot 0: read packed key, unpack MSE indices, reconstruct
        slot = kv_cache[0, 0, 0]  # (slot_size,) for head 0
        # True 3-bit pack: 8 indices → 3 bytes (3*D/8 total bytes)
        mse_bytes_n = 3 * D // 8
        mse_raw = slot[:mse_bytes_n]

        # Unpack 8 indices from every 3 bytes (inverse of the store layout)
        b0 = mse_raw[0::3].int()  # (D/8,)
        b1 = mse_raw[1::3].int()
        b2 = mse_raw[2::3].int()
        idx = torch.stack(
            [
                b0 & 0x7,
                (b0 >> 3) & 0x7,
                ((b0 >> 6) & 0x3) | ((b1 & 0x1) << 2),
                (b1 >> 1) & 0x7,
                (b1 >> 4) & 0x7,
                ((b1 >> 7) & 0x1) | ((b2 & 0x3) << 1),
                (b2 >> 2) & 0x7,
                (b2 >> 5) & 0x7,
            ],
            dim=-1,
        ).reshape(-1)[:D]

        centroids = get_centroids(D, cfg.mse_bits)
        c_vals = centroids[idx.long()]

        # Recover norm
        noff = mse_bytes_n
        nd = slot[noff : noff + 2].contiguous()
        vec_norm = nd.view(torch.float16).item()

        # Reconstruct: un-rotate c_vals back to original domain.
        # Store: y = x_hat @ Pi.T  (rotate)
        # c_vals ≈ y in rotated domain
        # Decode: x_hat ≈ c_vals @ Pi  (un-rotate, Pi^{-1} = Pi.T but we need Pi here)
        Pi = self.layer._tq_Pi
        reconstructed = vec_norm * (c_vals.float() @ Pi)

        # Compare to original (token 0, head 0)
        original = key[0, 0].float()
        mse = ((original - reconstructed) ** 2).mean().item()
        self.assertLess(mse, 0.15, f"TQ3 MSE too high: {mse:.4f}")


# ============================================================================
# 6. Plugin registration mock test
# ============================================================================


class TestPluginRegistration(unittest.TestCase):
    """Test that _register_native_backend calls the right vLLM APIs."""

    def test_register_backend_called(self):
        """Verify register_backend is called with CUSTOM and correct path."""
        calls = []

        fake_custom = object()

        class FakeEnum:
            CUSTOM = fake_custom

        def fake_register_backend(backend, class_path):
            calls.append((backend, class_path))
            return lambda x: x

        with mock.patch.dict(
            "sys.modules",
            {
                "vllm": mock.MagicMock(),
                "vllm.v1": mock.MagicMock(),
                "vllm.v1.attention": mock.MagicMock(),
                "vllm.v1.attention.backends": mock.MagicMock(),
                "vllm.v1.attention.backends.registry": mock.MagicMock(
                    AttentionBackendEnum=FakeEnum,
                    register_backend=fake_register_backend,
                ),
                "vllm.platforms": mock.MagicMock(),
                "vllm.platforms.cuda": mock.MagicMock(),
                "vllm.model_executor": mock.MagicMock(),
                "vllm.model_executor.layers": mock.MagicMock(),
                "vllm.model_executor.layers.attention": mock.MagicMock(),
                "vllm.model_executor.layers.attention.attention": mock.MagicMock(),
            },
        ):
            import importlib
            import turboquant_vllm._vllm_plugin as plugin

            importlib.reload(plugin)

            plugin._native_backend_registered = False
            result = plugin._register_native_backend()

        self.assertTrue(result)
        self.assertEqual(len(calls), 1)
        backend, path = calls[0]
        self.assertIs(backend, fake_custom)
        self.assertIn("TurboQuantAttentionBackend", path)
        self.assertIn("turboquant_vllm.native_backend", path)

    def test_register_backend_duplicate_error_is_treated_as_success(self):
        """Duplicate CUSTOM registration should still patch selector and return True."""
        fake_custom = object()

        class FakeEnum:
            CUSTOM = fake_custom

        def fake_register_backend(_backend, _class_path):
            raise RuntimeError("CUSTOM backend already registered")

        class FakeCudaPlatform:
            @classmethod
            def get_valid_backends(cls, *_args, **_kwargs):
                return [("BASELINE", 0)], {}

        with mock.patch.dict(
            "sys.modules",
            {
                "vllm": mock.MagicMock(),
                "vllm.v1": mock.MagicMock(),
                "vllm.v1.attention": mock.MagicMock(),
                "vllm.v1.attention.backends": mock.MagicMock(),
                "vllm.v1.attention.backends.registry": mock.MagicMock(
                    AttentionBackendEnum=FakeEnum,
                    register_backend=fake_register_backend,
                ),
                "vllm.platforms": mock.MagicMock(),
                "vllm.platforms.cuda": mock.MagicMock(CudaPlatform=FakeCudaPlatform),
            },
        ):
            import importlib
            import turboquant_vllm._vllm_plugin as plugin

            importlib.reload(plugin)

            plugin._native_backend_registered = False
            result = plugin._register_native_backend()
            selector_cfg = type("SelectorCfg", (), {"kv_cache_dtype": "tq3"})()
            backends, _ = FakeCudaPlatform.get_valid_backends(None, selector_cfg, None)

        self.assertTrue(result)
        self.assertEqual(backends, [(fake_custom, 0)])

    def test_init_tq_buffers_layer_idx_extraction(self):
        """Layer index parsed from prefix correctly."""
        import re

        prefixes = [
            ("model.layers.0.self_attn", 0),
            ("model.layers.12.self_attn", 12),
            ("transformer.h.5.attn", 5),
            ("no_layer_index", 0),  # fallback
        ]
        for prefix, expected_idx in prefixes:
            m = re.search(r"\.(\d+)\.", prefix)
            idx = int(m.group(1)) if m else 0
            self.assertEqual(idx, expected_idx, f"prefix={prefix}")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("TurboQuant native backend local validation")
    print("=" * 60)
    print()

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for cls in [
        TestTurboQuantConfig,
        TestCentroids,
        TestRotationMatrices,
        TestBufferInit,
        TestStoreDecodeCPU,
        TestPluginRegistration,
    ]:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
