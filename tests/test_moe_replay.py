"""Test that _materialize_and_process correctly replays buffered weight loads.

Reproduces the bug where MoE GPU memory stays at 5 GB instead of growing
to expected ~62 GB — the replay doesn't actually fill the materialized params.
"""

import unittest

import torch
import torch.nn as nn


class FakeFusedMoE(nn.Module):
    """Minimal FusedMoE with expert assembly weight_loader."""

    def __init__(self, num_experts: int, out_dim: int, in_dim: int):
        super().__init__()
        w13 = nn.Parameter(
            torch.zeros(num_experts, 2 * out_dim, in_dim), requires_grad=False
        )
        w2 = nn.Parameter(
            torch.zeros(num_experts, in_dim, out_dim), requires_grad=False
        )
        self.register_parameter("w13_weight", w13)
        self.register_parameter("w2_weight", w2)

        # Attach weight_loader (simulating vLLM's set_weight_attrs)
        self.w13_weight.weight_loader = self._make_loader("w13_weight")
        self.w2_weight.weight_loader = self._make_loader("w2_weight")

    def _make_loader(self, param_name: str):
        layer = self

        def weight_loader(param, loaded_weight, expert_id=0, shard_id="gate"):
            """Simulate FusedMoE expert assembly."""
            if param_name == "w13_weight":
                half = param.shape[1] // 2
                if shard_id == "gate":
                    param.data[expert_id, :half] = loaded_weight
                else:
                    param.data[expert_id, half:] = loaded_weight
            else:
                param.data[expert_id] = loaded_weight

        return weight_loader


class TestMoEReplay(unittest.TestCase):
    def test_full_flow_buffer_materialize_replay(self):
        """Simulate the complete flow: create → meta → buffer → materialize → replay."""
        num_experts, out_dim, in_dim = 4, 8, 16
        layer = FakeFusedMoE(num_experts, out_dim, in_dim)

        # Save original state
        orig_loaders = {}
        param_shapes = {}
        param_dtypes = {}
        for name, param in layer.named_parameters(recurse=False):
            if hasattr(param, "weight_loader"):
                orig_loaders[name] = param.weight_loader
            param_shapes[name] = tuple(param.shape)
            param_dtypes[name] = param.dtype

        total_numel = sum(p.numel() for p in layer.parameters())

        # Move to meta
        for name, param in list(layer.named_parameters(recurse=False)):
            meta = nn.Parameter(torch.empty_like(param, device="meta"), requires_grad=False)
            meta.weight_loader = param.weight_loader
            delattr(layer, name)
            layer.register_parameter(name, meta)

        # Verify meta
        for name, param in layer.named_parameters(recurse=False):
            self.assertEqual(param.device, torch.device("meta"))

        # Set up buffering (same as create_weights)
        buffer = []
        loaded_numel = [0]
        materialized = [False]

        def make_buffering_loader(pname, orig):
            def _buf(*args, **kwargs):
                if materialized[0]:
                    return orig(*args, **kwargs)
                loaded = args[1] if len(args) > 1 else None
                numel = loaded.numel() if isinstance(loaded, torch.Tensor) else 0
                buffer.append((pname, args, kwargs))
                loaded_numel[0] += numel
            return _buf

        for name, param in layer.named_parameters(recurse=False):
            if name in orig_loaders:
                param.weight_loader = make_buffering_loader(name, orig_loaders[name])

        # Simulate model.load_weights calling param.weight_loader
        expert_data = {}
        for expert_id in range(num_experts):
            gate = torch.randn(out_dim, in_dim)
            up = torch.randn(out_dim, in_dim)
            down = torch.randn(in_dim, out_dim)
            expert_data[expert_id] = (gate, up, down)

            # These calls go through _buffering_loader
            layer.w13_weight.weight_loader(
                layer.w13_weight, gate, expert_id=expert_id, shard_id="gate"
            )
            layer.w13_weight.weight_loader(
                layer.w13_weight, up, expert_id=expert_id, shard_id="up"
            )
            layer.w2_weight.weight_loader(
                layer.w2_weight, down, expert_id=expert_id
            )

        self.assertEqual(loaded_numel[0], total_numel)
        self.assertEqual(len(buffer), 12)

        # Now simulate _materialize_and_process
        # 1. Materialize
        for name, param in list(layer.named_parameters(recurse=False)):
            if param.device == torch.device("meta") and name in param_shapes:
                real = nn.Parameter(
                    torch.zeros(param_shapes[name], dtype=param_dtypes[name]),
                    requires_grad=False,
                )
                if name in orig_loaders:
                    real.weight_loader = orig_loaders[name]
                delattr(layer, name)
                layer.register_parameter(name, real)

        # Verify materialized params are on CPU (no CUDA in test), zeroed
        for name, param in layer.named_parameters(recurse=False):
            self.assertNotEqual(param.device, torch.device("meta"))
            self.assertEqual(param.sum().item(), 0.0)

        # 2. Replay
        for pname, args, kwargs in buffer:
            loader = orig_loaders.get(pname)
            if loader is not None:
                param = getattr(layer, pname)
                new_args = (param,) + args[1:]
                loader(*new_args, **kwargs)
        buffer.clear()

        # 3. Verify data is loaded
        for expert_id in range(num_experts):
            gate, up, down = expert_data[expert_id]
            # Check w13_weight[expert_id] has gate + up data
            w13 = layer.w13_weight.data[expert_id]
            self.assertTrue(
                torch.allclose(w13[:out_dim], gate),
                f"Expert {expert_id} gate data mismatch",
            )
            self.assertTrue(
                torch.allclose(w13[out_dim:], up),
                f"Expert {expert_id} up data mismatch",
            )
            # Check w2_weight[expert_id] has down data
            self.assertTrue(
                torch.allclose(layer.w2_weight.data[expert_id], down),
                f"Expert {expert_id} down data mismatch",
            )

    def test_replay_with_wrong_param_reference(self):
        """If replay uses stale param reference, data won't land."""
        layer = FakeFusedMoE(num_experts=2, out_dim=4, in_dim=8)

        orig_loader = layer.w13_weight.weight_loader

        # Move to meta
        meta = nn.Parameter(torch.empty(2, 8, 8, device="meta"), requires_grad=False)
        delattr(layer, "w13_weight")
        layer.register_parameter("w13_weight", meta)

        # Buffer a call
        loaded = torch.randn(4, 8)
        buffered_args = (meta, loaded)  # meta param in args

        # Materialize
        real = nn.Parameter(torch.zeros(2, 8, 8), requires_grad=False)
        delattr(layer, "w13_weight")
        layer.register_parameter("w13_weight", real)

        # Replay WITH correct param reference
        param = getattr(layer, "w13_weight")
        new_args = (param,) + buffered_args[1:]
        orig_loader(*new_args, expert_id=0, shard_id="gate")

        # Data should be there
        self.assertFalse(
            torch.allclose(layer.w13_weight.data[0, :4], torch.zeros(4, 8)),
            "Replay should have filled the tensor",
        )
        self.assertTrue(
            torch.allclose(layer.w13_weight.data[0, :4], loaded),
            "Replay data should match loaded weight",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
