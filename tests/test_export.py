from __future__ import annotations

import sys
import types
import unittest
from unittest import mock

import torch
import torch.nn as nn

from turboquant_vllm.export import compress_and_export


class _FakeTokenizer:
    @staticmethod
    def from_pretrained(_model_id):
        return _FakeTokenizer()


class _FakeConfig:
    @staticmethod
    def from_pretrained(_model_id):
        return _FakeConfig()


class _DenseAndMoEModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(128, 128, bias=False)
        self.register_parameter(
            "experts_weight",
            nn.Parameter(torch.randn(2, 128, 128, dtype=torch.bfloat16), requires_grad=False),
        )


class _FakeAutoModelForCausalLM:
    @staticmethod
    def from_pretrained(_model_id, **_kwargs):
        return _DenseAndMoEModel()


class TestExportGuards(unittest.TestCase):
    def test_compress_and_export_rejects_moe_tensors(self):
        fake_transformers = types.SimpleNamespace(
            AutoModelForCausalLM=_FakeAutoModelForCausalLM,
            AutoTokenizer=_FakeTokenizer,
            AutoConfig=_FakeConfig,
        )
        with (
            mock.patch.dict(sys.modules, {"transformers": fake_transformers}),
            mock.patch("torch.cuda.memory_allocated", return_value=0),
        ):
            with self.assertRaisesRegex(NotImplementedError, "does not support MoE expert tensors"):
                compress_and_export("fake/model", "/tmp/tq-export-test")


if __name__ == "__main__":
    unittest.main()
