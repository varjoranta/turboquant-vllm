"""Regression test for weight_quant._replace_linear_layers on CPU-only builds.

Historical bug: _replace_linear_layers ended with an unconditional
torch.cuda.empty_cache(), which raises AssertionError on PyTorch builds
without CUDA (Mac, CI, CPU-only wheels). This made the entire weight
compression path unusable on any machine without a GPU.
"""

import sys
import unittest
from pathlib import Path

import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))


class TinyLinearModel(nn.Module):
    """Model with one Linear layer large enough to trigger replacement."""

    def __init__(self, in_features=1024, out_features=1024):
        super().__init__()
        self.dense = nn.Linear(in_features, out_features)


class TestReplaceLinearLayersCpu(unittest.TestCase):
    def test_cpu_only_does_not_crash(self):
        """_replace_linear_layers must not call torch.cuda.empty_cache()
        when running on a CPU-only PyTorch build.

        Before the fix this raised:
            AssertionError: Torch not compiled with CUDA enabled
        during the post-replacement bookkeeping, regardless of whether
        any layers were actually moved to GPU.
        """
        from turboquant_vllm.weight_quant import _replace_linear_layers

        model = TinyLinearModel(in_features=1024, out_features=1024)
        # default min_size=1024, so our 1024x1024 layer qualifies
        total = _replace_linear_layers(model, bits=3, group_size=128)
        self.assertGreaterEqual(
            total, 1,
            "the 1024x1024 Linear should have been replaced at least once",
        )

    def test_no_replacements_also_safe(self):
        """Even when no layers qualify, the path must be CPU-safe."""
        from turboquant_vllm.weight_quant import _replace_linear_layers

        # Too small for default min_size=1024
        model = TinyLinearModel(in_features=64, out_features=64)
        total = _replace_linear_layers(model, bits=3, group_size=128)
        self.assertEqual(total, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
