"""CPU unit tests for the TurboQuantFusedMoEMethod path.

Goal: lock in the structural contract that ``enable_weight_quantization``
installs ``TurboQuantFusedMoEMethod`` on every ``FusedMoE`` module it
finds, that the compressed expert state ends up on the layer under the
expected attribute names, and that ``Compressed3D.decompress_into``
correctly writes into a pre-allocated buffer.

These tests do NOT require a GPU, a real FusedMoE, or vLLM. They use a
fake ``FusedMoE``-shaped module that implements the minimal surface
``weight_quant.enable_weight_quantization`` reads: ``w13_weight``,
``w2_weight``, ``moe_config``, and ``_replace_quant_method``. A pytest
``autouse`` fixture monkey-patches ``turboquant_vllm.moe_quant`` so our
fake module type passes the ``isinstance(..., FusedMoE)`` check and a
stub ``TurboQuantFusedMoEMethod`` is installed without needing a real
vLLM ``FusedMoEMethodBase`` base class.

The actual ``apply()`` path (with Triton fused_experts) is GPU-only and
validated by the benchmark runs, not by these unit tests.
"""

from __future__ import annotations

import types
import unittest

import torch
import torch.nn as nn

from turboquant_vllm.weight_quant import Compressed3D, _replace_linear_layers


class _FakeMoeConfig:
    """Stands in for vLLM's FusedMoEConfig."""

    disable_inplace = False


class _FakeFusedMoE(nn.Module):
    """Minimal surface that enable_weight_quantization's FusedMoE path reads.

    Fields touched by the walker:
      - ``w13_weight`` (3D nn.Parameter)
      - ``w2_weight``  (3D nn.Parameter)
      - ``moe_config`` (any object, passed to TurboQuantFusedMoEMethod.__init__)
      - ``_replace_quant_method(method)`` — our walker calls this
      - attribute slots for ``_tq_w13_weight`` / ``_tq_w2_compressed``
    """

    def __init__(self, num_experts: int, hidden: int, intermediate: int):
        super().__init__()
        self.w13_weight = nn.Parameter(
            torch.randn(num_experts, 2 * intermediate, hidden, dtype=torch.bfloat16),
            requires_grad=False,
        )
        self.w2_weight = nn.Parameter(
            torch.randn(num_experts, hidden, intermediate, dtype=torch.bfloat16),
            requires_grad=False,
        )
        self.moe_config = _FakeMoeConfig()
        self.installed_method = None
        # Match vLLM FusedMoE's public surface just enough for the walker's
        # debug-logging path to do type(self.quant_method).__name__.
        self.quant_method = types.SimpleNamespace()

    def _replace_quant_method(self, method):
        self.installed_method = method
        self.quant_method = method
        self.runner = types.SimpleNamespace(quant_method=method)


class _FakeTurboQuantFusedMoEMethod:
    """Stand-in for the real TurboQuantFusedMoEMethod that skips the vLLM
    base class requirement. The walker only calls __init__ and passes the
    instance to _replace_quant_method, so this is sufficient."""

    def __init__(self, moe_config, w13_compressed, w2_compressed, scratch_pool):
        self.moe = moe_config
        self.w13_compressed = w13_compressed
        self.w2_compressed = w2_compressed
        self.scratch_pool = scratch_pool


class _FakeScratchPool:
    """Stand-in for TurboQuantFusedMoEScratchPool.

    Eagerly allocates the four scratch slots from the first FusedMoE
    layer's compressed objects — matches the real pool's contract
    (sized up front so vLLM's memory profile sees the scratch bytes
    before CUDA graph capture).
    """

    def __init__(self, w13_compressed, w2_compressed):
        self.w13 = torch.empty(w13_compressed.shape, dtype=w13_compressed.dtype)
        self.w2 = torch.empty(w2_compressed.shape, dtype=w2_compressed.dtype)
        self.w13_fp32 = torch.empty(w13_compressed.shape, dtype=torch.float32)
        self.w2_fp32 = torch.empty(w2_compressed.shape, dtype=torch.float32)
        self.shape_w13 = w13_compressed.shape
        self.shape_w2 = w2_compressed.shape

    def assert_matches(self, w13_compressed, w2_compressed) -> None:
        assert w13_compressed.shape == self.shape_w13
        assert w2_compressed.shape == self.shape_w2


def _patch_moe_walker_to_use_fakes():
    """Patch _replace_linear_layers' FusedMoE imports to use our fakes.

    The walker inside weight_quant.py does:

        from vllm.model_executor.layers.fused_moe import FusedMoE
        from turboquant_vllm.moe_quant import (
            TurboQuantFusedMoEMethod,
            TurboQuantFusedMoEScratchPool,
        )

    inside a try/except ImportError. In the CPU test environment we
    don't have vLLM installed, so it hits the except branch and skips
    the FusedMoE path entirely. To exercise the walker, we inject a
    fake ``vllm.model_executor.layers.fused_moe`` module into sys.modules
    and patch ``turboquant_vllm.moe_quant`` before the walker runs.
    """
    import sys

    root = types.ModuleType("vllm")
    me = types.ModuleType("vllm.model_executor")
    layers = types.ModuleType("vllm.model_executor.layers")
    fused_moe_pkg = types.ModuleType("vllm.model_executor.layers.fused_moe")
    fused_moe_pkg.FusedMoE = _FakeFusedMoE
    sys.modules["vllm"] = root
    sys.modules["vllm.model_executor"] = me
    sys.modules["vllm.model_executor.layers"] = layers
    sys.modules["vllm.model_executor.layers.fused_moe"] = fused_moe_pkg

    import turboquant_vllm.moe_quant as moe_quant

    moe_quant.TurboQuantFusedMoEMethod = _FakeTurboQuantFusedMoEMethod
    moe_quant.TurboQuantFusedMoEScratchPool = _FakeScratchPool


def _unpatch_moe_walker():
    import sys

    for k in list(sys.modules.keys()):
        if k == "vllm" or k.startswith("vllm."):
            del sys.modules[k]


# ---------------------------------------------------------------------------
# Compressed3D.decompress_into
# ---------------------------------------------------------------------------


class TestCompressed3DDecompressInto(unittest.TestCase):
    """``decompress_into`` must write into a pre-allocated buffer matching
    ``self.shape``, ``self.dtype``, and ``self.device`` without allocating
    a new result tensor.

    On the CPU fallback path (no CUDA extension) this uses the chunked
    PyTorch reference via ``decompress()`` + ``copy_()``. We verify shape
    preservation and value equivalence with ``decompress()``.
    """

    def test_into_matches_decompress_value_and_shape(self):
        data = torch.randn(4, 256, 128, dtype=torch.float32)
        compressed = Compressed3D(data, bits=3, group_size=128)

        ref = compressed.decompress()
        out = torch.empty_like(ref)
        compressed.decompress_into(out)

        self.assertEqual(out.shape, ref.shape)
        self.assertEqual(out.dtype, ref.dtype)
        self.assertTrue(
            torch.allclose(out, ref),
            f"decompress_into diverged from decompress: max abs diff = "
            f"{(out - ref).abs().max().item():.6g}",
        )

    def test_into_rejects_wrong_shape(self):
        data = torch.randn(2, 128, 128, dtype=torch.float32)
        compressed = Compressed3D(data, bits=3, group_size=128)
        wrong = torch.empty(2, 128, 256, dtype=torch.float32)
        with self.assertRaises(AssertionError):
            compressed.decompress_into(wrong)

    def test_into_rejects_wrong_dtype(self):
        data = torch.randn(2, 128, 128, dtype=torch.float32)
        compressed = Compressed3D(data, bits=3, group_size=128)
        wrong = torch.empty(2, 128, 128, dtype=torch.float16)
        with self.assertRaises(AssertionError):
            compressed.decompress_into(wrong)


# ---------------------------------------------------------------------------
# enable_weight_quantization → FusedMoE walker
# ---------------------------------------------------------------------------


class TestFusedMoEWalkerInstallation(unittest.TestCase):
    """The walker must compress w13/w2, attach Compressed3D to the layer,
    free the bf16 originals, and install TurboQuantFusedMoEMethod."""

    def setUp(self):
        _patch_moe_walker_to_use_fakes()

    def tearDown(self):
        _unpatch_moe_walker()

    def _build_model(self, n_layers=2):
        """Wrap N fake FusedMoE modules in a parent nn.Module."""
        model = nn.Module()
        for i in range(n_layers):
            expert = _FakeFusedMoE(num_experts=4, hidden=128, intermediate=128)
            model.add_module(f"block_{i}", expert)
        return model

    def test_walker_installs_on_every_fused_moe(self):
        model = self._build_model(n_layers=3)
        _replace_linear_layers(model, bits=3, group_size=128)

        for i in range(3):
            expert = getattr(model, f"block_{i}")
            self.assertIsNotNone(
                expert.installed_method,
                f"block_{i}: _replace_quant_method was never called",
            )
            self.assertIsInstance(
                expert.installed_method, _FakeTurboQuantFusedMoEMethod
            )

    def test_walker_attaches_compressed_tensors(self):
        model = self._build_model(n_layers=2)
        _replace_linear_layers(model, bits=3, group_size=128)

        for i in range(2):
            expert = getattr(model, f"block_{i}")
            self.assertTrue(
                hasattr(expert, "_tq_w13_weight"),
                f"block_{i} missing _tq_w13_weight after walk",
            )
            self.assertTrue(hasattr(expert, "_tq_w2_weight"))
            self.assertIsInstance(expert._tq_w13_weight, Compressed3D)
            self.assertIsInstance(expert._tq_w2_weight, Compressed3D)

    def test_walker_repoints_weight_data_at_scratch_pool(self):
        """After compression the walker must re-point w13/w2 .data at the
        shared scratch buffers so the base unquant method reads freshly
        dequantized values. The original bf16 allocation is freed by
        ``_compress_3d_param`` before the re-point, so no extra copy lingers.
        """
        model = self._build_model(n_layers=2)
        _replace_linear_layers(model, bits=3, group_size=128)

        pool = model.block_0.installed_method.scratch_pool
        for i in range(2):
            expert = getattr(model, f"block_{i}")
            self.assertEqual(
                expert.w13_weight.data.data_ptr(),
                pool.w13.data_ptr(),
                f"block_{i} w13_weight.data must alias the shared scratch buffer",
            )
            self.assertEqual(
                expert.w2_weight.data.data_ptr(), pool.w2.data_ptr()
            )

    def test_walker_shares_scratch_pool_across_layers(self):
        """Scratch pool must be shared across every FusedMoE in the model.

        This is the critical memory-correctness invariant: per-layer
        scratch buffers would hold N × uncompressed-expert-bytes on the
        side, defeating the entire compression story (see the simplify
        efficiency review on fix/moe-fused-quant-method).
        """
        model = self._build_model(n_layers=3)
        _replace_linear_layers(model, bits=3, group_size=128)

        pools = {
            id(getattr(model, f"block_{i}").installed_method.scratch_pool)
            for i in range(3)
        }
        self.assertEqual(
            len(pools),
            1,
            f"expected one shared scratch pool across all 3 FusedMoE "
            f"layers, got {len(pools)} distinct pools — per-layer scratch "
            "would defeat the compression win",
        )

    def test_walker_respects_skip_patterns(self):
        """A FusedMoE whose name matches a skip pattern must NOT be touched."""
        model = nn.Module()
        # "lm_head" is in _SKIP_PATTERNS
        model.add_module("lm_head_experts", _FakeFusedMoE(4, 128, 128))
        _replace_linear_layers(model, bits=3, group_size=128)
        self.assertIsNone(
            model.lm_head_experts.installed_method,
            "lm_head_experts should have been skipped by _SKIP_PATTERNS",
        )

    def test_walker_compression_ratio_is_real(self):
        """Compressed bytes should be roughly 3 bits / 16 bits = ~1/5 of
        original for TQ3 on bf16 tensors (ignoring norm overhead)."""
        model = self._build_model(n_layers=1)
        _replace_linear_layers(model, bits=3, group_size=128)
        expert = model.block_0
        w13 = expert._tq_w13_weight
        ratio = w13.original_bytes / w13.compressed_bytes
        self.assertGreater(
            ratio,
            4.0,
            f"TQ3 compression ratio {ratio:.2f}× is suspiciously low "
            "(should be around 5× for bf16 → 3-bit)",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
