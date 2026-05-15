"""Regression test for save_tq3_checkpoint local-path support.

Historical bug: the docstring advertised 'HuggingFace model ID or local
path' but the implementation unconditionally called HfApi().list_repo_files
and hf_hub_download, so passing a local path raised a hub error.

This test exercises the local-path branch end-to-end on a tiny synthetic
safetensors checkpoint so the bug can't reappear.  We can't cover the full
compression pipeline here (it needs scipy, torch, Lloyd-Max), but we DO
assert that huggingface_hub is never touched and that a local-path call
discovers shards from the filesystem.
"""

import os
import sys
import tempfile
import unittest
import unittest.mock as mock
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestSaveTqCheckpointLocalPath(unittest.TestCase):
    def test_remote_repo_missing_shards_raises(self):
        """Remote repos with no safetensors shards should fail with a clear error."""
        from turboquant_vllm.checkpoint import save_tq3_checkpoint

        dummy = mock.Mock()
        dummy.save_pretrained = mock.Mock()

        with tempfile.TemporaryDirectory() as outdir:
            with (
                mock.patch("transformers.AutoConfig.from_pretrained", return_value=dummy),
                mock.patch("transformers.AutoTokenizer.from_pretrained", return_value=dummy),
                mock.patch("huggingface_hub.HfApi.list_repo_files", return_value=[]),
            ):
                with self.assertRaises(FileNotFoundError) as ctx:
                    save_tq3_checkpoint(
                        model_id="fake-org/fake-model",
                        output_dir=outdir,
                        bits=3,
                        group_size=8,
                    )
        self.assertIn("No .safetensors shards", str(ctx.exception))

    def test_local_path_does_not_touch_hf_hub(self):
        """Pass a local dir and verify HF Hub is not touched and local source shards are not deleted."""
        from turboquant_vllm.checkpoint import save_tq3_checkpoint

        with tempfile.TemporaryDirectory() as srcdir, tempfile.TemporaryDirectory() as outdir:
            source_shard = os.path.join(srcdir, "model-00001-of-00001.safetensors")
            # Create a minimal safetensors shard with one small 2D tensor
            from safetensors.torch import save_file

            weight = torch.randn(8, 8)  # too small to actually compress (< 128)
            save_file({"model.layers.0.mlp.fake.weight": weight}, source_shard)

            # Fake config.json so AutoConfig.from_pretrained works
            import json

            with open(os.path.join(srcdir, "config.json"), "w") as f:
                json.dump(
                    {
                        "model_type": "bert",  # simplest arch with no tokenizer requirement
                        "hidden_size": 8,
                        "num_hidden_layers": 1,
                        "num_attention_heads": 1,
                        "vocab_size": 10,
                        "intermediate_size": 8,
                        "max_position_embeddings": 16,
                    },
                    f,
                )
            # Fake tokenizer file so AutoTokenizer doesn't go online
            with open(os.path.join(srcdir, "tokenizer_config.json"), "w") as f:
                json.dump({"model_type": "bert", "tokenizer_class": "BertTokenizer"}, f)
            with open(os.path.join(srcdir, "vocab.txt"), "w") as f:
                f.write("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nhello\nworld\n")

            # Patch huggingface_hub so any accidental call raises loudly.
            # We do NOT swallow other exceptions here: if save_tq3_checkpoint
            # raises for any reason other than the mock sentinels, this test
            # should fail so we know the shard-preservation code path was not
            # reached.
            with (
                mock.patch(
                    "huggingface_hub.HfApi.list_repo_files",
                    side_effect=AssertionError("save_tq3_checkpoint must not call HfApi for a local path"),
                ),
                mock.patch(
                    "huggingface_hub.hf_hub_download",
                    side_effect=AssertionError("save_tq3_checkpoint must not call hf_hub_download for a local path"),
                ),
            ):
                save_tq3_checkpoint(
                    model_id=srcdir,
                    output_dir=outdir,
                    bits=3,
                    group_size=8,  # match tiny tensor dim
                )

            self.assertTrue(
                os.path.exists(source_shard),
                "save_tq3_checkpoint should not delete local source shards",
            )

    def test_local_path_missing_shards_raises(self):
        """Empty local directory should raise a clear FileNotFoundError."""
        from turboquant_vllm.checkpoint import save_tq3_checkpoint

        with tempfile.TemporaryDirectory() as srcdir, tempfile.TemporaryDirectory() as outdir:
            # config.json but no safetensors
            import json

            with open(os.path.join(srcdir, "config.json"), "w") as f:
                json.dump({"model_type": "bert"}, f)
            with open(os.path.join(srcdir, "tokenizer_config.json"), "w") as f:
                json.dump({"model_type": "bert"}, f)
            with open(os.path.join(srcdir, "vocab.txt"), "w") as f:
                f.write("[PAD]\n[UNK]\n")

            with self.assertRaises(FileNotFoundError) as ctx:
                save_tq3_checkpoint(
                    model_id=srcdir,
                    output_dir=outdir,
                    bits=3,
                    group_size=8,
                )
            self.assertIn("No .safetensors shards", str(ctx.exception))

    def test_non_float_tensor_uses_true_dtype_size_in_ratio(self):
        """Non-float tensors should contribute their real dtype byte size."""
        from turboquant_vllm.checkpoint import save_tq3_checkpoint

        with tempfile.TemporaryDirectory() as srcdir, tempfile.TemporaryDirectory() as outdir:
            from safetensors.torch import save_file

            save_file(
                {
                    "float_tensor": torch.ones(4, dtype=torch.float32),  # 16 -> 8 bytes (fp16)
                    "int_tensor": torch.arange(3, dtype=torch.int64),  # 24 -> 24 bytes
                },
                os.path.join(srcdir, "model-00001-of-00001.safetensors"),
            )

            import json

            with open(os.path.join(srcdir, "config.json"), "w") as f:
                json.dump({"model_type": "bert", "vocab_size": 10}, f)
            with open(os.path.join(srcdir, "tokenizer_config.json"), "w") as f:
                json.dump({"model_type": "bert", "tokenizer_class": "BertTokenizer"}, f)
            with open(os.path.join(srcdir, "vocab.txt"), "w") as f:
                f.write("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nhello\nworld\n")

            with mock.patch("turboquant_vllm.checkpoint.logger.info") as mock_info:
                save_tq3_checkpoint(
                    model_id=srcdir,
                    output_dir=outdir,
                    bits=3,
                    group_size=8,
                )

            final_call = None
            for call in mock_info.call_args_list:
                if call.args and call.args[0].startswith("TQ3 checkpoint saved:"):
                    final_call = call
            self.assertIsNotNone(final_call, "Expected final checkpoint summary log")

            # Args: original_gb, compressed_gb, ratio, compressed_count
            ratio = final_call.args[3]
            self.assertAlmostEqual(ratio, 40 / 32, places=6)

    def test_local_path_copies_additional_json_configs(self):
        """Extra local JSON config files should be preserved in output."""
        from turboquant_vllm.checkpoint import save_tq3_checkpoint

        with tempfile.TemporaryDirectory() as srcdir, tempfile.TemporaryDirectory() as outdir:
            from safetensors.torch import save_file

            save_file(
                {"model.layers.0.mlp.fake.weight": torch.randn(8, 8)},
                os.path.join(srcdir, "model-00001-of-00001.safetensors"),
            )

            import json

            with open(os.path.join(srcdir, "config.json"), "w") as f:
                json.dump({"model_type": "bert", "vocab_size": 10}, f)
            with open(os.path.join(srcdir, "tokenizer_config.json"), "w") as f:
                json.dump({"model_type": "bert", "tokenizer_class": "BertTokenizer"}, f)
            with open(os.path.join(srcdir, "vocab.txt"), "w") as f:
                f.write("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nhello\nworld\n")
            custom_json = {"foo": "bar", "answer": 42}
            with open(os.path.join(srcdir, "custom_config.json"), "w") as f:
                json.dump(custom_json, f)

            save_tq3_checkpoint(
                model_id=srcdir,
                output_dir=outdir,
                bits=3,
                group_size=8,
            )

            copied_path = os.path.join(outdir, "custom_config.json")
            self.assertTrue(os.path.exists(copied_path), "Expected custom local JSON config to be copied")
            with open(copied_path) as f:
                self.assertEqual(json.load(f), custom_json)

    def test_quantized_source_config_with_int_weight_raises_before_float_conversion(self):
        """Quantized non-float source weights must fail without a dequant sidecar."""
        from turboquant_vllm.checkpoint import UnsupportedQuantizedSourceError, save_tq3_checkpoint

        with tempfile.TemporaryDirectory() as srcdir, tempfile.TemporaryDirectory() as outdir:
            from safetensors.torch import save_file

            save_file(
                {"model.layers.0.mlp.fake.weight": torch.randint(-8, 8, (8, 8), dtype=torch.int8)},
                os.path.join(srcdir, "model-00001-of-00001.safetensors"),
            )

            import json

            with open(os.path.join(srcdir, "config.json"), "w") as f:
                json.dump(
                    {
                        "model_type": "bert",
                        "vocab_size": 10,
                        "quantization_config": {
                            "quant_method": "fp8",
                            "fmt": "e4m3",
                            "weight_block_size": [128, 128],
                        },
                    },
                    f,
                )
            with open(os.path.join(srcdir, "tokenizer_config.json"), "w") as f:
                json.dump({"model_type": "bert", "tokenizer_class": "BertTokenizer"}, f)
            with open(os.path.join(srcdir, "vocab.txt"), "w") as f:
                f.write("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nhello\nworld\n")

            with self.assertRaises(UnsupportedQuantizedSourceError) as ctx:
                save_tq3_checkpoint(
                    model_id=srcdir,
                    output_dir=outdir,
                    bits=3,
                    group_size=8,
                )

        self.assertIn("source config advertises quantized weights", str(ctx.exception))

    def test_weight_scale_sidecar_raises_before_float_conversion(self):
        """A .weight + .scale pair is a source-quantized tensor layout."""
        from turboquant_vllm.checkpoint import UnsupportedQuantizedSourceError, save_tq3_checkpoint

        with tempfile.TemporaryDirectory() as srcdir, tempfile.TemporaryDirectory() as outdir:
            from safetensors.torch import save_file

            save_file(
                {
                    "model.layers.0.mlp.fake.weight": torch.randint(-8, 8, (8, 8), dtype=torch.int8),
                    "model.layers.0.mlp.fake.scale": torch.ones(1, 1, dtype=torch.float32),
                },
                os.path.join(srcdir, "model-00001-of-00001.safetensors"),
            )

            import json

            with open(os.path.join(srcdir, "config.json"), "w") as f:
                json.dump({"model_type": "bert", "vocab_size": 10}, f)
            with open(os.path.join(srcdir, "tokenizer_config.json"), "w") as f:
                json.dump({"model_type": "bert", "tokenizer_class": "BertTokenizer"}, f)
            with open(os.path.join(srcdir, "vocab.txt"), "w") as f:
                f.write("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nhello\nworld\n")

            with self.assertRaises(UnsupportedQuantizedSourceError) as ctx:
                save_tq3_checkpoint(
                    model_id=srcdir,
                    output_dir=outdir,
                    bits=3,
                    group_size=8,
                )

        self.assertIn("found sibling .scale tensor", str(ctx.exception))

    def test_fp8_scale_sidecar_is_dequantized_for_skipped_weight(self):
        """Supported FP8 source weights are dequantized before storing skipped tensors."""
        from safetensors.torch import load_file, save_file

        from turboquant_vllm.checkpoint import save_tq3_checkpoint

        with tempfile.TemporaryDirectory() as srcdir, tempfile.TemporaryDirectory() as outdir:
            weight_fp32 = torch.ones(128, 128, dtype=torch.float32)
            scale_fp32 = torch.full((1, 1), 2.0, dtype=torch.float32)
            save_file(
                {
                    "model.layers.0.attn.compressor.wkv.weight": weight_fp32.to(torch.float8_e4m3fn),
                    "model.layers.0.attn.compressor.wkv.scale": scale_fp32.to(torch.float8_e8m0fnu),
                },
                os.path.join(srcdir, "model-00001-of-00001.safetensors"),
            )

            import json

            with open(os.path.join(srcdir, "config.json"), "w") as f:
                json.dump(
                    {
                        "model_type": "bert",
                        "vocab_size": 10,
                        "quantization_config": {
                            "quant_method": "fp8",
                            "fmt": "e4m3",
                            "weight_block_size": [128, 128],
                        },
                    },
                    f,
                )
            with open(os.path.join(srcdir, "tokenizer_config.json"), "w") as f:
                json.dump({"model_type": "bert", "tokenizer_class": "BertTokenizer"}, f)
            with open(os.path.join(srcdir, "vocab.txt"), "w") as f:
                f.write("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nhello\nworld\n")

            save_tq3_checkpoint(
                model_id=srcdir,
                output_dir=outdir,
                bits=3,
                group_size=128,
            )

            loaded = {}
            for fname in os.listdir(outdir):
                if fname.endswith(".safetensors"):
                    loaded.update(load_file(os.path.join(outdir, fname)))

        name = "model.layers.0.attn.compressor.wkv.weight"
        self.assertIn(name, loaded)
        self.assertNotIn("model.layers.0.attn.compressor.wkv.scale", loaded)
        self.assertEqual(loaded[name].dtype, torch.float16)
        self.assertTrue(torch.allclose(loaded[name].float(), torch.full((128, 128), 2.0), atol=0, rtol=0))

    def test_fp8_block_dequant_accepts_raw_ue8m0_uint8_scale(self):
        """Some loaders expose UE8M0 sidecars as raw uint8 exponent bytes."""
        from turboquant_vllm.checkpoint import _dequant_fp8_block_weight

        weight = torch.ones(2, 2, dtype=torch.float32).to(torch.float8_e4m3fn)
        scale = torch.full((1, 1), 128, dtype=torch.uint8)

        out = _dequant_fp8_block_weight("model.layers.0.attn.wq.weight", weight, scale, (128, 128))

        self.assertTrue(torch.allclose(out, torch.full((2, 2), 2.0), atol=0, rtol=0))

    def test_mxfp4_scale_sidecar_is_dequantized_before_storage(self):
        """Supported MXFP4 source expert weights are unpacked before TQ handling."""
        from safetensors.torch import load_file, save_file

        from turboquant_vllm.checkpoint import save_tq3_checkpoint

        with tempfile.TemporaryDirectory() as srcdir, tempfile.TemporaryDirectory() as outdir:
            # Low nibble 0x2 -> 1.0, high nibble 0xB -> -1.5.
            packed = torch.full((4, 16), 0xB2, dtype=torch.uint8)
            # Raw E8M0 exponent 128 means scale 2 ** (128 - 127) == 2.
            scale = torch.full((4, 1), 128, dtype=torch.uint8)
            save_file(
                {
                    "model.layers.0.ffn.experts.0.w1.weight": packed,
                    "model.layers.0.ffn.experts.0.w1.scale": scale,
                },
                os.path.join(srcdir, "model-00001-of-00001.safetensors"),
            )

            import json

            with open(os.path.join(srcdir, "config.json"), "w") as f:
                json.dump(
                    {
                        "model_type": "bert",
                        "vocab_size": 10,
                        "expert_dtype": "fp4",
                        "quantization_config": {
                            "quant_method": "fp8",
                            "fmt": "e4m3",
                            "scale_fmt": "ue8m0",
                            "weight_block_size": [128, 128],
                        },
                    },
                    f,
                )
            with open(os.path.join(srcdir, "tokenizer_config.json"), "w") as f:
                json.dump({"model_type": "bert", "tokenizer_class": "BertTokenizer"}, f)
            with open(os.path.join(srcdir, "vocab.txt"), "w") as f:
                f.write("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nhello\nworld\n")

            save_tq3_checkpoint(
                model_id=srcdir,
                output_dir=outdir,
                bits=3,
                group_size=128,
            )

            loaded = {}
            for fname in os.listdir(outdir):
                if fname.endswith(".safetensors"):
                    loaded.update(load_file(os.path.join(outdir, fname)))

        name = "model.layers.0.ffn.experts.0.w1.weight"
        expected_row = torch.tensor([2.0, -3.0] * 16, dtype=torch.float32)
        expected = expected_row.repeat(4, 1)
        self.assertIn(name, loaded)
        self.assertNotIn("model.layers.0.ffn.experts.0.w1.scale", loaded)
        self.assertEqual(loaded[name].shape, (4, 32))
        self.assertEqual(loaded[name].dtype, torch.float16)
        self.assertTrue(torch.allclose(loaded[name].float(), expected, atol=0, rtol=0))

    def test_checkpoint_and_runtime_skip_patterns_stay_in_sync(self):
        """checkpoint.py and weight_quant.py skip lists must not drift."""
        from turboquant_vllm.checkpoint import _SKIP_PATTERNS as checkpoint_skips
        from turboquant_vllm.weight_quant import _SKIP_PATTERNS as runtime_skips

        self.assertEqual(checkpoint_skips, runtime_skips)
        for pattern in ("mtp", "compressor", "indexer", "ape", "attn_sink", "hc_"):
            self.assertIn(pattern, checkpoint_skips)

    def test_deepseek_v4_control_tensors_are_not_compressed(self):
        """CSA/HCA control tensors should remain full precision in TQ3 checkpoints."""
        from turboquant_vllm.checkpoint import save_tq3_checkpoint

        with tempfile.TemporaryDirectory() as srcdir, tempfile.TemporaryDirectory() as outdir:
            from safetensors.torch import load_file, save_file

            save_file(
                {
                    "model.layers.0.mlp.dense.weight": torch.randn(128, 128),
                    "model.layers.0.attn.compressor.wkv.weight": torch.randn(128, 128),
                },
                os.path.join(srcdir, "model-00001-of-00001.safetensors"),
            )

            import json

            with open(os.path.join(srcdir, "config.json"), "w") as f:
                json.dump({"model_type": "bert", "vocab_size": 10}, f)
            with open(os.path.join(srcdir, "tokenizer_config.json"), "w") as f:
                json.dump({"model_type": "bert", "tokenizer_class": "BertTokenizer"}, f)
            with open(os.path.join(srcdir, "vocab.txt"), "w") as f:
                f.write("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nhello\nworld\n")

            save_tq3_checkpoint(
                model_id=srcdir,
                output_dir=outdir,
                bits=3,
                group_size=128,
            )

            loaded = {}
            for fname in os.listdir(outdir):
                if fname.endswith(".safetensors"):
                    loaded.update(load_file(os.path.join(outdir, fname)))

        self.assertIn("model.layers.0.mlp.dense.weight.tq_packed", loaded)
        self.assertIn("model.layers.0.attn.compressor.wkv.weight", loaded)
        self.assertNotIn("model.layers.0.attn.compressor.wkv.weight.tq_packed", loaded)
        self.assertEqual(loaded["model.layers.0.attn.compressor.wkv.weight"].dtype, torch.float16)


if __name__ == "__main__":
    unittest.main(verbosity=2)
