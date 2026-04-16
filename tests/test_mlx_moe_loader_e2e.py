"""End-to-end TDD goal tests for MLX MoE loading.

These tests define "done" for local MoE testing on Apple Silicon. They
round-trip a synthetic MoE model through compress -> save TQ3 native
-> load via ``load_tq3_model`` -> forward, asserting the reconstructed
output matches a reference built from the dequantised weights.

DO NOT MODIFY these tests unless the overall plan changes. They are the
stable definition of "MLX MoE testing locally is done". When they pass,
the MLX MoE path has been validated through every layer:

  1. ``TurboQuantMLXSwitchLinear`` (unit parity — covered elsewhere)
  2. The mlx_loader's SwitchLinear detection + replacement
  3. Weight layout: 3D expert tensors packed + restored correctly
  4. Forward parity with an uncompressed reference model

A second, slower test gated behind a slow marker does the same with a
small *real* HF MoE checkpoint (``ibm-granite/granite-3.0-1b-a400m-base``)
to catch config/quirks the synthetic model doesn't cover.
"""

from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm.models.qwen2_moe import Model, ModelArgs
    from mlx_lm.models.switch_layers import SwitchLinear

    HAS_MLX = True
except ImportError:
    HAS_MLX = False


def _tiny_qwen2_moe_args() -> "ModelArgs":
    return ModelArgs(
        model_type="qwen2_moe",
        hidden_size=128,
        num_hidden_layers=2,
        intermediate_size=256,
        num_attention_heads=4,
        num_experts_per_tok=2,
        num_experts=4,
        moe_intermediate_size=128,
        shared_expert_intermediate_size=128,
        rms_norm_eps=1e-6,
        vocab_size=100,
        num_key_value_heads=2,
        rope_theta=1000000,
        rope_traditional=False,
        rope_scaling=None,
        tie_word_embeddings=True,
    )


def _qwen2_moe_config_dict(args: "ModelArgs") -> dict:
    return {
        "architectures": ["Qwen2MoeForCausalLM"],
        "model_type": args.model_type,
        "hidden_size": args.hidden_size,
        "num_hidden_layers": args.num_hidden_layers,
        "intermediate_size": args.intermediate_size,
        "num_attention_heads": args.num_attention_heads,
        "num_experts_per_tok": args.num_experts_per_tok,
        "num_experts": args.num_experts,
        "moe_intermediate_size": args.moe_intermediate_size,
        "shared_expert_intermediate_size": args.shared_expert_intermediate_size,
        "num_key_value_heads": args.num_key_value_heads,
        "rms_norm_eps": args.rms_norm_eps,
        "vocab_size": args.vocab_size,
        "rope_theta": args.rope_theta,
        "tie_word_embeddings": args.tie_word_embeddings,
    }


def _compress_weight(w_np: np.ndarray, bits: int, group_size: int, seed: int):
    """Compress a 2D (out, in) or 3D (num_experts, out, in) weight to TQ3."""
    from turboquant_vllm.torch_ops import PolarQuantTorch
    from turboquant_vllm.weight_quant import pack_indices, padded_size

    w_pt = torch.from_numpy(w_np).float()
    orig_shape = w_pt.shape
    in_features = orig_shape[-1]
    padded_in, n_groups = padded_size(in_features, group_size)

    if padded_in > in_features:
        pad = [(0, 0)] * (w_pt.ndim - 1) + [(0, padded_in - in_features)]
        # torch.nn.functional.pad expects a flat list of (before, after) reversed
        padded = torch.zeros(*orig_shape[:-1], padded_in, dtype=w_pt.dtype)
        padded[..., :in_features] = w_pt
    else:
        padded = w_pt

    pq = PolarQuantTorch(dim=group_size, bit_width=bits, seed=seed, device="cpu")
    grouped = padded.reshape(-1, group_size)
    indices, norms_raw = pq.quantize(grouped, norm_correction=True)
    packed = pack_indices(indices, bits)

    # Reconstruct for reference: dequant + strip padding
    w_groups = pq.dequantize(indices, norms_raw)
    w_rec = w_groups.reshape(*orig_shape[:-1], padded_in)[..., :in_features]

    # Norms reshape: for 2D weight -> (out, n_groups); for 3D -> (n_experts*out, n_groups)
    leading = int(np.prod(orig_shape[:-1]))
    norms = norms_raw.reshape(leading, n_groups)

    return packed, norms, w_rec.numpy(), pq


@unittest.skipUnless(HAS_MLX, "MLX not installed (Mac-only)")
class TestMLXMoELoaderEndToEnd(unittest.TestCase):
    """Synthetic-MoE round-trip — the TDD gate for MLX MoE support."""

    def _write_tq3_moe_checkpoint(self, tmp_path: Path, bits: int = 3, group_size: int = 128):
        """Compress + save a tiny qwen2_moe TQ3 checkpoint. Returns reference weights."""
        args = _tiny_qwen2_moe_args()
        mx.random.seed(0)
        model = Model(args)
        mx.eval(model.parameters())

        def flatten(tree, prefix=""):
            out = {}
            if isinstance(tree, dict):
                for k, v in tree.items():
                    out.update(flatten(v, f"{prefix}{k}."))
            elif isinstance(tree, list):
                for i, v in enumerate(tree):
                    out.update(flatten(v, f"{prefix}{i}."))
            elif isinstance(tree, mx.array):
                out[prefix.rstrip(".")] = tree
            return out

        params = flatten(model.parameters())

        linear_paths = {name for name, m in model.named_modules() if isinstance(m, nn.Linear)}
        switch_paths = {name for name, m in model.named_modules() if isinstance(m, SwitchLinear)}

        weights_to_save: dict[str, torch.Tensor] = {}
        reconstructed: dict[str, np.ndarray] = {}
        seed = 42

        for name, arr in params.items():
            if not name.endswith(".weight"):
                weights_to_save[name] = torch.from_numpy(np.array(arr))
                continue

            module_path = name.rsplit(".", 1)[0]
            w_np = np.array(arr)

            if module_path in linear_paths or module_path in switch_paths:
                if w_np.shape[-1] < group_size:
                    # Small layers (routing gates, etc.) stay uncompressed — the
                    # group_size constraint makes compression meaningless for them.
                    weights_to_save[name] = torch.from_numpy(w_np)
                    continue
                packed, norms, w_rec, _ = _compress_weight(w_np, bits, group_size, seed)
                weights_to_save[f"{module_path}.weight.tq_packed"] = packed
                weights_to_save[f"{module_path}.weight.tq_norms"] = norms
                reconstructed[name] = w_rec.astype(np.float32)
            else:
                weights_to_save[name] = torch.from_numpy(w_np)

        from safetensors.torch import save_file

        save_file(weights_to_save, str(tmp_path / "model.safetensors"))
        with open(tmp_path / "config.json", "w") as f:
            json.dump(_qwen2_moe_config_dict(args), f)
        with open(tmp_path / "tq_config.json", "w") as f:
            json.dump(
                {
                    "format": "tq3_native",
                    "bits": bits,
                    "group_size": group_size,
                    "quantizer_seed": seed,
                    "compressed_layers": len(reconstructed),
                },
                f,
            )

        return args, reconstructed

    def _build_reference_model(self, args, reconstructed):
        """Reference model with the dequantised weights patched in."""
        mx.random.seed(0)
        ref_model = Model(args)
        mx.eval(ref_model.parameters())

        for name, w_np in reconstructed.items():
            parent = ref_model
            parts = name.split(".")
            for p in parts[:-1]:
                parent = parent[int(p)] if p.isdigit() else getattr(parent, p)
            setattr(parent, parts[-1], mx.array(w_np.astype(np.float32)))
        mx.eval(ref_model.parameters())
        return ref_model

    def test_load_tq3_moe_round_trip(self):
        """Compress -> save -> load via load_tq3_model -> forward parity."""
        import tempfile

        from turboquant_vllm.mlx_loader import load_tq3_model

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            args, reconstructed = self._write_tq3_moe_checkpoint(tmp_path)

            model_tq, config = load_tq3_model(str(tmp_path))
            self.assertEqual(config["model_type"], "qwen2_moe")

            model_ref = self._build_reference_model(args, reconstructed)

            # Verify at least one SwitchLinear got replaced — otherwise the
            # test is passing by accident with a dense-only path.
            from turboquant_vllm.mlx_model import TurboQuantMLXSwitchLinear

            replaced_switch = sum(1 for _, m in model_tq.named_modules() if isinstance(m, TurboQuantMLXSwitchLinear))
            self.assertGreater(
                replaced_switch,
                0,
                "Loader didn't replace any SwitchLinear modules — MoE path untested.",
            )

            tokens = mx.array([[1, 2, 3, 4, 5]])
            out_ref = model_ref(tokens)
            out_tq = model_tq(tokens)
            mx.eval(out_ref, out_tq)

            np.testing.assert_allclose(
                np.array(out_tq).astype(np.float32),
                np.array(out_ref).astype(np.float32),
                rtol=5e-3,
                atol=5e-3,
            )


@unittest.skipUnless(HAS_MLX, "MLX not installed (Mac-only)")
@unittest.skipUnless(
    os.environ.get("TQ_REAL_MODEL_TEST") == "1",
    "Real-model test disabled (set TQ_REAL_MODEL_TEST=1 to run; needs ~2 GB download + several minutes).",
)
class TestMLXMoELoaderRealModel(unittest.TestCase):
    """Second gate: a real small HF MoE compresses + loads + generates coherent text."""

    def test_granite_3b_a400m_roundtrip(self):
        """Compress ibm-granite/granite-3.0-1b-a400m-base -> TQ3 -> generate."""
        import tempfile

        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler

        from turboquant_vllm.checkpoint import save_tq3_checkpoint
        from turboquant_vllm.mlx_loader import load_tq3

        with tempfile.TemporaryDirectory() as tmp:
            save_tq3_checkpoint(
                "ibm-granite/granite-3.0-1b-a400m-base",
                tmp,
                bits=3,
                group_size=128,
            )
            model, tokenizer = load_tq3(tmp)
            text = generate(
                model,
                tokenizer,
                "The capital of France is",
                max_tokens=16,
                verbose=False,
                sampler=make_sampler(temp=0.0),
            )
            self.assertGreater(len(text.strip()), 5, f"Output too short/empty: {text!r}")


if __name__ == "__main__":
    unittest.main()
