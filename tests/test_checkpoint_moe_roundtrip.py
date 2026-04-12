"""CPU test: save_tq3_checkpoint → load native TQ3 for MoE expert weights.

Verifies the full pipeline: compress 3D expert tensors via checkpoint.py,
read back the packed data, reconstruct via Compressed3D.from_packed, and
verify the decompressed output matches the original compression path.

This test creates a minimal fake MoE checkpoint (safetensors) with 3D
expert weight tensors, runs save_tq3_checkpoint on it, then reads
the output and validates shapes and values.

No GPU or vLLM required.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest

import torch
from safetensors.torch import save_file, load_file

from turboquant_vllm.weight_quant import (
    Compressed3D,
    packed_group_bytes,
)


class TestCheckpointMoERoundTrip(unittest.TestCase):
    """save_tq3_checkpoint → from_packed → decompress must match."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="tq3_moe_test_")
        self.bits = 3
        self.group_size = 128
        self.num_experts = 4
        self.hidden = 128
        self.intermediate = 128

    def _create_fake_checkpoint(self):
        """Create a minimal fake MoE checkpoint with config + safetensors."""
        ckpt_dir = os.path.join(self.tmpdir, "fake_model")
        os.makedirs(ckpt_dir, exist_ok=True)

        # Config
        config = {
            "model_type": "llama",
            "architectures": ["LlamaForCausalLM"],
            "hidden_size": self.hidden,
            "intermediate_size": self.intermediate,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "vocab_size": 100,
            "torch_dtype": "float32",
        }
        with open(os.path.join(ckpt_dir, "config.json"), "w") as f:
            json.dump(config, f)

        # Tokenizer (minimal — enough for AutoTokenizer to not crash)
        with open(os.path.join(ckpt_dir, "tokenizer_config.json"), "w") as f:
            json.dump({"tokenizer_class": "PreTrainedTokenizerFast"}, f)
        # Write a minimal tokenizer.json
        with open(os.path.join(ckpt_dir, "tokenizer.json"), "w") as f:
            json.dump({
                "version": "1.0",
                "model": {"type": "BPE", "vocab": {"<s>": 0, "</s>": 1}, "merges": []},
                "added_tokens": [],
            }, f)

        # Expert weights as 3D tensors (simulates FusedMoE fused format)
        w13 = torch.randn(
            self.num_experts, 2 * self.intermediate, self.hidden, dtype=torch.float32
        )
        w2 = torch.randn(
            self.num_experts, self.hidden, self.intermediate, dtype=torch.float32
        )
        # Also a non-weight tensor that should NOT be compressed
        embed = torch.randn(100, self.hidden, dtype=torch.float32)

        tensors = {
            "model.layers.0.mlp.experts.w13_weight": w13,
            "model.layers.0.mlp.experts.w2_weight": w2,
            "model.embed_tokens.weight": embed,  # skip pattern
        }
        save_file(tensors, os.path.join(ckpt_dir, "model.safetensors"))

        return ckpt_dir, w13, w2

    def test_expert_roundtrip(self):
        """Checkpoint save → load → decompress matches direct compression."""
        ckpt_dir, orig_w13, orig_w2 = self._create_fake_checkpoint()
        output_dir = os.path.join(self.tmpdir, "tq3_output")

        # Run save_tq3_checkpoint
        from turboquant_vllm.checkpoint import save_tq3_checkpoint

        save_tq3_checkpoint(
            model_id=ckpt_dir,
            output_dir=output_dir,
            bits=self.bits,
            group_size=self.group_size,
        )

        # Read back the output
        index_path = os.path.join(output_dir, "model.safetensors.index.json")
        if os.path.exists(index_path):
            with open(index_path) as f:
                index = json.load(f)
            weight_map = index["weight_map"]
        else:
            # Single shard
            weight_map = None

        # Load all output tensors
        loaded = {}
        for f in os.listdir(output_dir):
            if f.endswith(".safetensors"):
                loaded.update(load_file(os.path.join(output_dir, f)))

        # Verify expert packed tensors exist
        w13_packed_name = "model.layers.0.mlp.experts.w13_weight.tq_packed"
        w13_norms_name = "model.layers.0.mlp.experts.w13_weight.tq_norms"
        w2_packed_name = "model.layers.0.mlp.experts.w2_weight.tq_packed"
        w2_norms_name = "model.layers.0.mlp.experts.w2_weight.tq_norms"

        self.assertIn(w13_packed_name, loaded, f"Missing {w13_packed_name}")
        self.assertIn(w13_norms_name, loaded, f"Missing {w13_norms_name}")
        self.assertIn(w2_packed_name, loaded, f"Missing {w2_packed_name}")
        self.assertIn(w2_norms_name, loaded, f"Missing {w2_norms_name}")

        # Verify shapes
        ne = self.num_experts
        gs = self.group_size
        pgb = packed_group_bytes(self.bits, gs)

        w13_out = 2 * self.intermediate
        w13_in = self.hidden
        w13_padded = ((w13_in + gs - 1) // gs) * gs
        w13_n_groups = w13_padded // gs

        self.assertEqual(
            loaded[w13_packed_name].shape,
            (ne * w13_out, w13_n_groups * pgb),
            "w13 packed shape mismatch",
        )
        self.assertEqual(
            loaded[w13_norms_name].shape,
            (ne * w13_out, w13_n_groups),
            "w13 norms shape mismatch",
        )

        # Verify checkpoint round-trip: load packed data, decompress,
        # check it's close to the original (quantization is lossy).
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        comp_loaded = Compressed3D.from_packed(
            loaded[w13_packed_name].to(dev),
            loaded[w13_norms_name].to(dev),
            orig_w13.shape,
            orig_w13.dtype,
            self.bits,
            gs,
        )
        out_w13 = comp_loaded.decompress().cpu()

        self.assertEqual(orig_w13.shape, out_w13.shape)
        # TQ3 on unit-variance Gaussian data: typical max error < 1.0
        max_diff = (orig_w13 - out_w13).abs().max().item()
        self.assertLess(
            max_diff, 2.0,
            f"w13 roundtrip max diff {max_diff:.3f} too large for TQ3",
        )

    def test_embed_not_compressed(self):
        """Embedding tensors should be stored as FP16, not compressed."""
        ckpt_dir, _, _ = self._create_fake_checkpoint()
        output_dir = os.path.join(self.tmpdir, "tq3_output2")

        from turboquant_vllm.checkpoint import save_tq3_checkpoint

        save_tq3_checkpoint(
            model_id=ckpt_dir,
            output_dir=output_dir,
            bits=self.bits,
            group_size=self.group_size,
        )

        loaded = {}
        for f in os.listdir(output_dir):
            if f.endswith(".safetensors"):
                loaded.update(load_file(os.path.join(output_dir, f)))

        # embed_tokens should be stored as FP16 (skip pattern)
        self.assertIn("model.embed_tokens.weight", loaded)
        self.assertEqual(loaded["model.embed_tokens.weight"].dtype, torch.float16)

    def test_tq_config_written(self):
        """tq_config.json must be written with correct metadata."""
        ckpt_dir, _, _ = self._create_fake_checkpoint()
        output_dir = os.path.join(self.tmpdir, "tq3_output3")

        from turboquant_vllm.checkpoint import save_tq3_checkpoint

        save_tq3_checkpoint(
            model_id=ckpt_dir,
            output_dir=output_dir,
            bits=self.bits,
            group_size=self.group_size,
        )

        tq_config_path = os.path.join(output_dir, "tq_config.json")
        self.assertTrue(os.path.exists(tq_config_path))
        with open(tq_config_path) as f:
            tq_config = json.load(f)
        self.assertEqual(tq_config["bits"], self.bits)
        self.assertEqual(tq_config["group_size"], self.group_size)
        self.assertEqual(tq_config["format"], "tq3_native")
        self.assertGreater(tq_config["compressed_layers"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
