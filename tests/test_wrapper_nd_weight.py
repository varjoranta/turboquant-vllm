"""Regression test: TurboQuantWrapper must handle >2-D weight tensors.

Some modules (e.g. vLLM parallel linears, MoE expert layers) expose
weight tensors with 3 or more dimensions.  The wrapper must flatten them
to 2-D before quantization instead of crashing on:

    out_dim, in_dim = weight.shape   # ValueError when ndim > 2
"""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))

from turboquant_vllm.weight_quant import TurboQuantWrapper


def _make_fake_module(weight_shape: tuple[int, ...], include_bias: bool = False):
    """Build a lightweight object that quacks like nn.Linear.

    TurboQuantWrapper.__init__ reads:
        original.weight.data, original.in_features, original.out_features,
        original.bias
    We construct a SimpleNamespace that provides all four.
    """
    weight = nn.Parameter(torch.randn(*weight_shape))
    mod = SimpleNamespace(
        weight=weight,
        in_features=weight_shape[-1],
        out_features=weight_shape[-2],
        bias=nn.Parameter(torch.randn(weight_shape[-2])) if include_bias else None,
    )
    return mod


class TestWrapperNdWeight(unittest.TestCase):
    """TurboQuantWrapper with >2-D weight tensors."""

    def test_3d_weight_does_not_crash(self):
        """A 3-D weight (e.g. num_experts, out, in) must not raise."""
        mod = _make_fake_module((4, 256, 128))
        wrapper = TurboQuantWrapper(mod, bits=3, group_size=128)
        # Flattened out_features = 4 * 256 = 1024
        self.assertEqual(wrapper.out_features, 4 * 256)
        self.assertEqual(wrapper.in_features, 128)

    def test_4d_weight_does_not_crash(self):
        """A 4-D weight must also be handled gracefully."""
        mod = _make_fake_module((2, 4, 256, 128))
        wrapper = TurboQuantWrapper(mod, bits=3, group_size=128)
        self.assertEqual(wrapper.out_features, 2 * 4 * 256)
        self.assertEqual(wrapper.in_features, 128)

    def test_3d_weight_forward_shape(self):
        """Forward pass should produce correct output shape after flattening."""
        mod = _make_fake_module((4, 256, 128))
        wrapper = TurboQuantWrapper(mod, bits=4, group_size=128)
        x = torch.randn(2, 128)
        with torch.no_grad():
            y = wrapper(x)
        self.assertEqual(y.shape, (2, 4 * 256))

    def test_2d_weight_unchanged(self):
        """Normal 2-D weights must still work (no regression)."""
        linear = nn.Linear(256, 128, bias=False)
        wrapper = TurboQuantWrapper(linear, bits=3, group_size=128)
        self.assertEqual(wrapper.in_features, 256)
        self.assertEqual(wrapper.out_features, 128)
        x = torch.randn(1, 256)
        with torch.no_grad():
            y = wrapper(x)
        self.assertEqual(y.shape, (1, 128))


if __name__ == "__main__":
    unittest.main(verbosity=2)
