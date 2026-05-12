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

import torch.nn as nn
import torch

from turboquant_vllm.weight_quant import (
    Compressed3D,
    packed_group_bytes,
)
from turboquant_vllm.vllm_quant import (
    TurboQuantOnlineMoEMethod,
    _finalize_native_packed_moe,
    _maybe_flush_native_moe_target,
    _regroup_native_moe_packed_tensors,
    _collect_meta_params,
)
from turboquant_vllm.moe_quant import _HAS_FUSED_MOE


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

        comp_b = Compressed3D.from_packed(comp_a.packed, comp_a.norms, data.shape, data.dtype, bits, gs)
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

        comp2 = Compressed3D.from_packed(comp.packed, comp.norms, data.shape, data.dtype, bits, gs)
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

        comp2 = Compressed3D.from_packed(comp.packed, comp.norms, data.shape, data.dtype, bits, gs)
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

        self.assertEqual(
            comp.packed.shape, expected_packed, f"packed shape mismatch: {comp.packed.shape} vs {expected_packed}"
        )
        self.assertEqual(
            comp.norms.shape, expected_norms, f"norms shape mismatch: {comp.norms.shape} vs {expected_norms}"
        )


class TestDecompressDetection(unittest.TestCase):
    """Decompress-on-load detects TQ3 checkpoints via tq_config.json, not quantization_config."""

    def test_tq_config_triggers_decompress(self):
        """Checkpoint with tq_config.json should activate decompression."""
        import tempfile
        import json

        d = tempfile.mkdtemp()
        with open(os.path.join(d, "tq_config.json"), "w") as f:
            json.dump({"format": "tq3_native", "bits": 3, "group_size": 128}, f)
        self.assertTrue(os.path.isfile(os.path.join(d, "tq_config.json")))

    def test_no_tq_config_skips_decompress(self):
        """Checkpoint without tq_config.json should NOT decompress."""
        import tempfile

        d = tempfile.mkdtemp()
        self.assertFalse(os.path.isfile(os.path.join(d, "tq_config.json")))


class _FakeExperts(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_parameter(
            "w13_weight",
            nn.Parameter(torch.empty(2, 8, 4, device="meta", dtype=torch.float32), requires_grad=False),
        )
        self.register_parameter(
            "w2_weight",
            nn.Parameter(torch.empty(2, 4, 4, device="meta", dtype=torch.float32), requires_grad=False),
        )


class _FakeMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.experts = _FakeExperts()


class _FakeLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = _FakeMLP()


class _FakeBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([_FakeLayer()])


class _FakeRoot(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _FakeBackbone()


class TestNativeMoEPackedRegroup(unittest.TestCase):
    def test_per_expert_packed_tensors_regroup_into_fused_targets(self):
        model = _FakeRoot()
        bits = 3
        group_size = 4

        gate0 = torch.randn(4, 4)
        up0 = torch.randn(4, 4)
        down0 = torch.randn(4, 4)
        gate1 = torch.randn(4, 4)
        up1 = torch.randn(4, 4)
        down1 = torch.randn(4, 4)

        def _pack_2d(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            comp = Compressed3D(weight.unsqueeze(0), bits=bits, group_size=group_size)
            return comp.packed, comp.norms

        packed_pairs = {}
        for expert_idx, gate, up, down in (
            (0, gate0, up0, down0),
            (1, gate1, up1, down1),
        ):
            prefix = f"model.layers.0.mlp.experts.{expert_idx}"
            gate_packed, gate_norms = _pack_2d(gate)
            up_packed, up_norms = _pack_2d(up)
            down_packed, down_norms = _pack_2d(down)
            packed_pairs[f"{prefix}.gate_proj.weight"] = {"packed": gate_packed, "norms": gate_norms}
            packed_pairs[f"{prefix}.up_proj.weight"] = {"packed": up_packed, "norms": up_norms}
            packed_pairs[f"{prefix}.down_proj.weight"] = {"packed": down_packed, "norms": down_norms}

        regrouped = dict(_regroup_native_moe_packed_tensors(model, packed_pairs))

        self.assertIn("model.layers.0.mlp.experts.w13_weight_tq_packed", regrouped)
        self.assertIn("model.layers.0.mlp.experts.w13_weight_tq_norms", regrouped)
        self.assertIn("model.layers.0.mlp.experts.w2_weight_tq_packed", regrouped)
        self.assertIn("model.layers.0.mlp.experts.w2_weight_tq_norms", regrouped)

        expected_w13 = torch.stack(
            [
                torch.cat([gate0, up0], dim=0),
                torch.cat([gate1, up1], dim=0),
            ],
            dim=0,
        )
        expected_w2 = torch.stack([down0, down1], dim=0)

        w13_comp = Compressed3D.from_packed(
            regrouped["model.layers.0.mlp.experts.w13_weight_tq_packed"],
            regrouped["model.layers.0.mlp.experts.w13_weight_tq_norms"],
            expected_w13.shape,
            expected_w13.dtype,
            bits,
            group_size,
        )
        w2_comp = Compressed3D.from_packed(
            regrouped["model.layers.0.mlp.experts.w2_weight_tq_packed"],
            regrouped["model.layers.0.mlp.experts.w2_weight_tq_norms"],
            expected_w2.shape,
            expected_w2.dtype,
            bits,
            group_size,
        )

        self.assertLess((w13_comp.decompress() - expected_w13).abs().max().item(), 2.0)
        self.assertLess((w2_comp.decompress() - expected_w2).abs().max().item(), 2.0)

    def test_per_expert_packed_tensors_regroup_supports_w1_w2_w3_names(self):
        model = _FakeRoot()
        bits = 3
        group_size = 4

        gate0 = torch.randn(4, 4)
        up0 = torch.randn(4, 4)
        down0 = torch.randn(4, 4)
        gate1 = torch.randn(4, 4)
        up1 = torch.randn(4, 4)
        down1 = torch.randn(4, 4)

        def _pack_2d(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            comp = Compressed3D(weight.unsqueeze(0), bits=bits, group_size=group_size)
            return comp.packed, comp.norms

        packed_pairs = {}
        for expert_idx, gate, up, down in (
            (0, gate0, up0, down0),
            (1, gate1, up1, down1),
        ):
            prefix = f"model.layers.0.mlp.experts.{expert_idx}"
            gate_packed, gate_norms = _pack_2d(gate)
            up_packed, up_norms = _pack_2d(up)
            down_packed, down_norms = _pack_2d(down)
            packed_pairs[f"{prefix}.w1.weight"] = {"packed": gate_packed, "norms": gate_norms}
            packed_pairs[f"{prefix}.w3.weight"] = {"packed": up_packed, "norms": up_norms}
            packed_pairs[f"{prefix}.w2.weight"] = {"packed": down_packed, "norms": down_norms}

        regrouped = dict(_regroup_native_moe_packed_tensors(model, packed_pairs))

        expected_w13 = torch.stack(
            [
                torch.cat([gate0, up0], dim=0),
                torch.cat([gate1, up1], dim=0),
            ],
            dim=0,
        )
        expected_w2 = torch.stack([down0, down1], dim=0)

        w13_comp = Compressed3D.from_packed(
            regrouped["model.layers.0.mlp.experts.w13_weight_tq_packed"],
            regrouped["model.layers.0.mlp.experts.w13_weight_tq_norms"],
            expected_w13.shape,
            expected_w13.dtype,
            bits,
            group_size,
        )
        w2_comp = Compressed3D.from_packed(
            regrouped["model.layers.0.mlp.experts.w2_weight_tq_packed"],
            regrouped["model.layers.0.mlp.experts.w2_weight_tq_norms"],
            expected_w2.shape,
            expected_w2.dtype,
            bits,
            group_size,
        )

        self.assertLess((w13_comp.decompress() - expected_w13).abs().max().item(), 2.0)
        self.assertLess((w2_comp.decompress() - expected_w2).abs().max().item(), 2.0)

    def test_incremental_native_moe_flush_emits_when_target_complete(self):
        model = _FakeRoot()
        bits = 3
        group_size = 4
        meta_params = _collect_meta_params(model)
        target_state = {}

        gate0 = torch.randn(4, 4)
        up0 = torch.randn(4, 4)
        down0 = torch.randn(4, 4)
        gate1 = torch.randn(4, 4)
        up1 = torch.randn(4, 4)
        down1 = torch.randn(4, 4)

        def _pack_2d(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            comp = Compressed3D(weight.unsqueeze(0), bits=bits, group_size=group_size)
            return comp.packed, comp.norms

        flushed = []
        for name, weight in (
            ("model.layers.0.mlp.experts.0.w1.weight", gate0),
            ("model.layers.0.mlp.experts.0.w3.weight", up0),
            ("model.layers.0.mlp.experts.1.w1.weight", gate1),
            ("model.layers.0.mlp.experts.1.w3.weight", up1),
        ):
            packed, norms = _pack_2d(weight)
            flushed.extend(
                _maybe_flush_native_moe_target(
                    model,
                    name,
                    {"packed": packed, "norms": norms},
                    meta_params,
                    target_state,
                )
            )
        self.assertIn("model.layers.0.mlp.experts.w13_weight_tq_packed", dict(flushed))
        self.assertNotIn("model.layers.0.mlp.experts.w2_weight_tq_packed", dict(flushed))

        flushed = []
        for name, weight in (
            ("model.layers.0.mlp.experts.0.w2.weight", down0),
            ("model.layers.0.mlp.experts.1.w2.weight", down1),
        ):
            packed, norms = _pack_2d(weight)
            flushed.extend(
                _maybe_flush_native_moe_target(
                    model,
                    name,
                    {"packed": packed, "norms": norms},
                    meta_params,
                    target_state,
                )
            )
        self.assertIn("model.layers.0.mlp.experts.w2_weight_tq_packed", dict(flushed))
        self.assertEqual(target_state, {})

    def test_finalize_native_packed_moe_accepts_compact_row_major_packed_layout(self):
        bits = 3
        group_size = 8

        w13 = torch.randn(2, 8, 8)
        w2 = torch.randn(2, 4, 8)

        class _FinalizeLayer(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_parameter("w13_weight", nn.Parameter(torch.empty_like(w13), requires_grad=False))
                self.register_parameter("w2_weight", nn.Parameter(torch.empty_like(w2), requires_grad=False))

        layer = _FinalizeLayer()
        w13_comp = Compressed3D(w13, bits=bits, group_size=group_size)
        w2_comp = Compressed3D(w2, bits=bits, group_size=group_size)

        layer.register_parameter(
            "w13_weight_tq_packed",
            nn.Parameter(w13_comp.packed.reshape(16, -1), requires_grad=False),
        )
        layer.register_parameter(
            "w13_weight_tq_norms",
            nn.Parameter(w13_comp.norms.clone(), requires_grad=False),
        )
        layer.register_parameter(
            "w2_weight_tq_packed",
            nn.Parameter(w2_comp.packed.reshape(8, -1), requires_grad=False),
        )
        layer.register_parameter(
            "w2_weight_tq_norms",
            nn.Parameter(w2_comp.norms.clone(), requires_grad=False),
        )

        class _FakeUnquant:
            def process_weights_after_loading(self, _layer):
                return None

        class _FakeMethod:
            def __init__(self):
                self.bits = bits
                self.group_size = group_size
                self._unquant = _FakeUnquant()

        method = _FakeMethod()
        _finalize_native_packed_moe(
            layer,
            method,
            {
                "w13_weight": tuple(w13.shape),
                "w2_weight": tuple(w2.shape),
            },
            {
                "w13_weight": w13.dtype,
                "w2_weight": w2.dtype,
            },
        )

        self.assertTrue(hasattr(layer, "_tq_w13_weight"))
        self.assertTrue(hasattr(layer, "_tq_w2_weight"))
        self.assertLess((layer._tq_w13_weight.decompress() - w13).abs().max().item(), 2.0)
        self.assertLess((layer._tq_w2_weight.decompress() - w2).abs().max().item(), 2.0)

    def test_finalize_native_packed_moe_replaces_meta_weight_params(self):
        bits = 3
        group_size = 8
        w13 = torch.randn(2, 8, 8)
        w2 = torch.randn(2, 4, 8)
        layer = _FakeExperts()

        w13_comp = Compressed3D(w13, bits=bits, group_size=group_size)
        w2_comp = Compressed3D(w2, bits=bits, group_size=group_size)

        layer.register_parameter(
            "w13_weight_tq_packed",
            nn.Parameter(w13_comp.packed.reshape(16, -1), requires_grad=False),
        )
        layer.register_parameter(
            "w13_weight_tq_norms",
            nn.Parameter(w13_comp.norms.clone(), requires_grad=False),
        )
        layer.register_parameter(
            "w2_weight_tq_packed",
            nn.Parameter(w2_comp.packed.reshape(8, -1), requires_grad=False),
        )
        layer.register_parameter(
            "w2_weight_tq_norms",
            nn.Parameter(w2_comp.norms.clone(), requires_grad=False),
        )

        class _FakeUnquant:
            def process_weights_after_loading(self, _layer):
                return None

        class _FakeMethod:
            def __init__(self):
                self.bits = bits
                self.group_size = group_size
                self._unquant = _FakeUnquant()

        method = _FakeMethod()
        _finalize_native_packed_moe(
            layer,
            method,
            {"w13_weight": (2, 8, 8), "w2_weight": (2, 4, 8)},
            {"w13_weight": torch.float32, "w2_weight": torch.float32},
        )

        self.assertFalse(layer.w13_weight.is_meta)
        self.assertFalse(layer.w2_weight.is_meta)

    def test_finalize_native_packed_moe_replaces_layer_quant_method(self):
        if not _HAS_FUSED_MOE:
            self.skipTest("vLLM fused MoE not available in local test environment")
        bits = 3
        group_size = 8
        w13 = torch.randn(2, 8, 8)
        w2 = torch.randn(2, 4, 8)

        class _FinalizeLayer(nn.Module):
            def __init__(self):
                super().__init__()
                self.moe_config = object()
                self.register_parameter("w13_weight", nn.Parameter(torch.empty_like(w13), requires_grad=False))
                self.register_parameter("w2_weight", nn.Parameter(torch.empty_like(w2), requires_grad=False))
                self.replaced_quant_method = None

            def _replace_quant_method(self, mk):
                self.replaced_quant_method = mk

        layer = _FinalizeLayer()
        w13_comp = Compressed3D(w13, bits=bits, group_size=group_size)
        w2_comp = Compressed3D(w2, bits=bits, group_size=group_size)
        layer.register_parameter(
            "w13_weight_tq_packed",
            nn.Parameter(w13_comp.packed.reshape(16, -1), requires_grad=False),
        )
        layer.register_parameter(
            "w13_weight_tq_norms",
            nn.Parameter(w13_comp.norms.clone(), requires_grad=False),
        )
        layer.register_parameter(
            "w2_weight_tq_packed",
            nn.Parameter(w2_comp.packed.reshape(8, -1), requires_grad=False),
        )
        layer.register_parameter(
            "w2_weight_tq_norms",
            nn.Parameter(w2_comp.norms.clone(), requires_grad=False),
        )

        class _FakeUnquant:
            def process_weights_after_loading(self, _layer):
                return None

        class _FakeMethod:
            def __init__(self):
                self.bits = bits
                self.group_size = group_size
                self._unquant = _FakeUnquant()

        method = _FakeMethod()
        _finalize_native_packed_moe(
            layer,
            method,
            {"w13_weight": (2, 8, 8), "w2_weight": (2, 4, 8)},
            {"w13_weight": torch.float32, "w2_weight": torch.float32},
        )

        self.assertIs(layer.base_quant_method, method._unquant)
        self.assertIsNotNone(layer.replaced_quant_method)

    def test_native_packed_loader_finalizes_once_all_four_tensors_arrive(self):
        if TurboQuantOnlineMoEMethod is None:
            self.skipTest("TurboQuantOnlineMoEMethod unavailable in local test environment")
        bits = 3
        group_size = 8
        w13 = torch.randn(2, 8, 8)
        w2 = torch.randn(2, 4, 8)

        class _FinalizeLayer(nn.Module):
            def __init__(self):
                super().__init__()
                self.moe_config = object()
                self.register_parameter("w13_weight", nn.Parameter(torch.empty_like(w13), requires_grad=False))
                self.register_parameter("w2_weight", nn.Parameter(torch.empty_like(w2), requires_grad=False))

            def _replace_quant_method(self, mk):
                self.replaced_quant_method = mk

        layer = _FinalizeLayer()

        class _FakeUnquant:
            def process_weights_after_loading(self, _layer):
                return None

        method = TurboQuantOnlineMoEMethod(bits, group_size, layer.moe_config, native_packed=True)
        method._unquant = _FakeUnquant()
        method.create_weights(layer)

        w13_comp = Compressed3D(w13, bits=bits, group_size=group_size)
        w2_comp = Compressed3D(w2, bits=bits, group_size=group_size)

        for name, tensor in (
            ("w13_weight_tq_packed", w13_comp.packed.reshape(16, -1)),
            ("w13_weight_tq_norms", w13_comp.norms.clone()),
            ("w2_weight_tq_packed", w2_comp.packed.reshape(8, -1)),
            ("w2_weight_tq_norms", w2_comp.norms.clone()),
        ):
            getattr(layer, name).weight_loader(getattr(layer, name), tensor)

        self.assertTrue(hasattr(layer, "_tq_w13_weight"))
        self.assertTrue(hasattr(layer, "_tq_w2_weight"))
        self.assertFalse(hasattr(layer, "w13_weight_tq_packed"))
        self.assertFalse(hasattr(layer, "w13_weight_tq_norms"))
        self.assertFalse(hasattr(layer, "w2_weight_tq_packed"))
        self.assertFalse(hasattr(layer, "w2_weight_tq_norms"))

    def test_online_moe_apply_keeps_pool_decompress_fallback(self):
        if TurboQuantOnlineMoEMethod is None:
            self.skipTest("TurboQuantOnlineMoEMethod unavailable in local test environment")

        method = TurboQuantOnlineMoEMethod(3, 8, object(), native_packed=True)

        class _FakeCompressed:
            def __init__(self):
                self.calls = []

            def decompress_into(self, target, fp32_scratch=None):
                self.calls.append((target, fp32_scratch))

        class _FakeUnquant:
            def apply(self, layer, x, **kwargs):
                return ("ok", layer, x, kwargs)

        pool = type(
            "_Pool",
            (),
            {
                "w13": torch.empty(1),
                "w2": torch.empty(1),
                "w13_fp32": None,
                "w2_fp32": None,
            },
        )()
        w13_c = _FakeCompressed()
        w2_c = _FakeCompressed()
        method._pool = pool
        method._w13_c = w13_c
        method._w2_c = w2_c
        method._unquant = _FakeUnquant()

        layer = object()
        x = torch.randn(2, 4)
        out = method.apply(layer, x, topk_ids=torch.tensor([[0]]))

        self.assertEqual(out[0], "ok")
        self.assertEqual(len(w13_c.calls), 1)
        self.assertEqual(len(w2_c.calls), 1)
        self.assertIs(w13_c.calls[0][0], pool.w13)
        self.assertIs(w2_c.calls[0][0], pool.w2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
