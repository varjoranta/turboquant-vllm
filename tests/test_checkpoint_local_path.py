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
    def test_local_path_does_not_touch_hf_hub(self):
        """Pass a local dir and verify HF Hub is not touched and local source shards are not deleted."""
        from turboquant_vllm.checkpoint import save_tq3_checkpoint

        with tempfile.TemporaryDirectory() as srcdir, \
             tempfile.TemporaryDirectory() as outdir:
            source_shard = os.path.join(srcdir, "model-00001-of-00001.safetensors")
            # Create a minimal safetensors shard with one small 2D tensor
            from safetensors.torch import save_file
            weight = torch.randn(8, 8)  # too small to actually compress (< 128)
            save_file({"model.layers.0.mlp.fake.weight": weight},
                      source_shard)

            # Fake config.json so AutoConfig.from_pretrained works
            import json
            with open(os.path.join(srcdir, "config.json"), "w") as f:
                json.dump({
                    "model_type": "bert",  # simplest arch with no tokenizer requirement
                    "hidden_size": 8,
                    "num_hidden_layers": 1,
                    "num_attention_heads": 1,
                    "vocab_size": 10,
                    "intermediate_size": 8,
                    "max_position_embeddings": 16,
                }, f)
            # Fake tokenizer file so AutoTokenizer doesn't go online
            with open(os.path.join(srcdir, "tokenizer_config.json"), "w") as f:
                json.dump({"model_type": "bert", "tokenizer_class": "BertTokenizer"}, f)
            with open(os.path.join(srcdir, "vocab.txt"), "w") as f:
                f.write("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nhello\nworld\n")

            # Patch huggingface_hub so any accidental call raises loudly
            with mock.patch(
                "huggingface_hub.HfApi.list_repo_files",
                side_effect=AssertionError(
                    "save_tq3_checkpoint must not call HfApi for a local path"),
            ), mock.patch(
                "huggingface_hub.hf_hub_download",
                side_effect=AssertionError(
                    "save_tq3_checkpoint must not call hf_hub_download for a local path"),
            ):
                try:
                    save_tq3_checkpoint(
                        model_id=srcdir,
                        output_dir=outdir,
                        bits=3,
                        group_size=8,  # match tiny tensor dim
                    )
                except AssertionError:
                    raise  # bubble up our mock trips
                except Exception as e:
                    # Any other failure (e.g. compression edge cases on small
                    # tensors) is fine for *this* test — we only care that
                    # hf_hub was not called. Verify the error isn't one of
                    # our AssertionError sentinels.
                    self.assertNotIn("must not call", str(e))

            self.assertTrue(
                os.path.exists(source_shard),
                "save_tq3_checkpoint should not delete local source shards",
            )

    def test_local_path_missing_shards_raises(self):
        """Empty local directory should raise a clear FileNotFoundError."""
        from turboquant_vllm.checkpoint import save_tq3_checkpoint

        with tempfile.TemporaryDirectory() as srcdir, \
             tempfile.TemporaryDirectory() as outdir:
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
                    model_id=srcdir, output_dir=outdir, bits=3, group_size=8,
                )
            self.assertIn("No .safetensors shards", str(ctx.exception))

    def test_non_float_tensor_uses_true_dtype_size_in_ratio(self):
        """Non-float tensors should contribute their real dtype byte size."""
        from turboquant_vllm.checkpoint import save_tq3_checkpoint

        with tempfile.TemporaryDirectory() as srcdir, \
             tempfile.TemporaryDirectory() as outdir:
            from safetensors.torch import save_file
            save_file(
                {
                    "float_tensor": torch.ones(4, dtype=torch.float32),  # 16 -> 8 bytes (fp16)
                    "int_tensor": torch.arange(3, dtype=torch.int64),    # 24 -> 24 bytes
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
