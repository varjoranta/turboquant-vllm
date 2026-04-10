"""Regression test for KVCacheCompressorTorch norm_correction + CUDA interaction.

The CUDA store kernel doesn't apply norm_correction — it writes raw norms.
A previous version of KVCacheCompressorTorch silently took the CUDA path
even when norm_correction=True was requested, degrading quality with no
warning.  This test locks in the safe fallback: use_cuda must flip to False
during __init__ when norm_correction=True.
"""

import logging
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestCudaNormCorrectionFallback(unittest.TestCase):
    def test_norm_correction_forces_pytorch_path(self):
        """use_cuda=True + norm_correction=True must fall back to PyTorch."""
        from turboquant_vllm.torch_ops import KVCacheCompressorTorch

        with self.assertLogs("turboquant_vllm.torch_ops", level="WARNING") as logs:
            comp = KVCacheCompressorTorch(
                head_dim=64,
                k_bits=4,
                v_bits=4,
                seed=42,
                device="cpu",
                use_cuda=True,       # caller asks for CUDA
                norm_correction=True,  # but also wants norm correction
                use_qjl=False,
            )
        self.assertFalse(
            comp.use_cuda,
            "norm_correction=True must override use_cuda=True to avoid "
            "silent quality degradation on the CUDA path",
        )
        self.assertIsNone(comp._cuda_mod, "CUDA module must not be loaded")
        # The warning must explain *why* we fell back
        self.assertTrue(
            any("norm_correction" in rec.message for rec in logs.records),
            f"expected a norm_correction warning, got: {[r.message for r in logs.records]}",
        )

    def test_no_warning_when_use_cuda_off(self):
        """Pure CPU construction (use_cuda=False) must not emit the
        norm_correction fallback warning regardless of norm_correction value.

        Note: we don't exercise use_cuda=True + norm_correction=False in
        this test because it triggers CUDA JIT compilation on first call,
        which fails on CPU-only machines — that path is covered by GPU
        integration tests, not this CPU suite.
        """
        from turboquant_vllm.torch_ops import KVCacheCompressorTorch

        logger = logging.getLogger("turboquant_vllm.torch_ops")
        with self.assertLogs(logger, level="WARNING") as logs:
            logger.warning("sentinel")  # keep assertLogs happy
            KVCacheCompressorTorch(
                head_dim=64, k_bits=4, v_bits=4, seed=42, device="cpu",
                use_cuda=False, norm_correction=True, use_qjl=False,
            )
        norm_msgs = [r for r in logs.records if "norm_correction" in r.message]
        self.assertEqual(
            norm_msgs, [],
            "no norm_correction warning should fire when use_cuda=False",
        )

    def test_default_pytorch_path(self):
        """The default construction must not touch CUDA at all."""
        from turboquant_vllm.torch_ops import KVCacheCompressorTorch
        comp = KVCacheCompressorTorch(
            head_dim=64, k_bits=4, v_bits=4, seed=42, device="cpu",
        )
        self.assertFalse(comp.use_cuda)
        self.assertIsNone(comp._cuda_mod)


if __name__ == "__main__":
    unittest.main(verbosity=2)
