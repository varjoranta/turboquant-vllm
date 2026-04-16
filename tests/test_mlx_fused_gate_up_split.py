"""Unit + integration tests for the fused gate_up_proj packed-tensor split.

HF Qwen3.5-MoE (and architecturally similar MoE models) store each layer's
gate and up projections fused as one 3D tensor
``(n_experts, 2*moe_intermediate, hidden)``. mlx_lm's ``sanitize`` splits
them at load time into two ``SwitchLinear`` modules. Our
``save_tq3_checkpoint`` compresses the fused tensor, so the split has to
run on the packed form instead.

These tests pin down:
  1. Splitting packed + norms halves matches compressing the two halves
     separately.
  2. The post-split keys match the ``switch_mlp.{gate,up}_proj.weight``
     module paths, plus renamed ``.switch_mlp.down_proj.weight``.
  3. End-to-end: a synthetic qwen3_5_moe model with a fused-gate-up
     checkpoint loads via ``load_tq3_model`` with the split applied and
     produces the same forward output as a reference using the
     dequantised halves.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import mlx.core as mx

    HAS_MLX = True
except ImportError:
    HAS_MLX = False


@unittest.skipUnless(HAS_MLX, "MLX not installed (Mac-only)")
class TestFusedGateUpSplit(unittest.TestCase):
    """Splitting packed fused gate_up_proj == compressing the two halves alone."""

    def _compress_fused_vs_separate(self, n_experts: int, out_half: int, in_features: int):
        """Return (fused packed/norms, gate-alone packed/norms, up-alone packed/norms)."""
        from turboquant_vllm.torch_ops import PolarQuantTorch
        from turboquant_vllm.weight_quant import pack_indices, padded_size

        bits, group_size = 3, 128

        torch.manual_seed(0)
        w_fused = torch.randn(n_experts, 2 * out_half, in_features, dtype=torch.float32)

        padded_in, n_groups = padded_size(in_features, group_size)
        if padded_in > in_features:
            padded = torch.zeros(n_experts, 2 * out_half, padded_in, dtype=w_fused.dtype)
            padded[:, :, :in_features] = w_fused
        else:
            padded = w_fused

        pq = PolarQuantTorch(dim=group_size, bit_width=bits, seed=42, device="cpu")

        def compress(w3d: torch.Tensor):
            grouped = w3d.reshape(-1, group_size)
            idx, norms = pq.quantize(grouped, norm_correction=True)
            return pack_indices(idx, bits), norms.reshape(-1, n_groups)

        fused_packed, fused_norms = compress(padded)
        gate_packed, gate_norms = compress(padded[:, :out_half, :])
        up_packed, up_norms = compress(padded[:, out_half:, :])
        return {
            "fused": (fused_packed, fused_norms),
            "gate": (gate_packed, gate_norms),
            "up": (up_packed, up_norms),
            "n_groups": n_groups,
        }

    def test_split_matches_separate_compression(self):
        """Splitting the fused packed + norms tensors recovers the halves bit-for-bit."""
        n_experts, out_half, in_features = 4, 64, 128
        comp = self._compress_fused_vs_separate(n_experts, out_half, in_features)

        fused_packed, fused_norms = comp["fused"]
        gate_packed_ref, gate_norms_ref = comp["gate"]
        up_packed_ref, up_norms_ref = comp["up"]
        n_groups = comp["n_groups"]

        # Replicate the split the loader performs
        bytes_per_group = fused_packed.shape[-1]
        reshaped_p = fused_packed.reshape(n_experts, 2 * out_half, n_groups, bytes_per_group)
        gate_packed = reshaped_p[:, :out_half, :, :].reshape(-1, bytes_per_group)
        up_packed = reshaped_p[:, out_half:, :, :].reshape(-1, bytes_per_group)

        reshaped_n = fused_norms.reshape(n_experts, 2 * out_half, n_groups)
        gate_norms = reshaped_n[:, :out_half, :].reshape(-1, n_groups)
        up_norms = reshaped_n[:, out_half:, :].reshape(-1, n_groups)

        np.testing.assert_array_equal(gate_packed.numpy(), gate_packed_ref.numpy())
        np.testing.assert_array_equal(up_packed.numpy(), up_packed_ref.numpy())
        np.testing.assert_allclose(gate_norms.numpy(), gate_norms_ref.numpy())
        np.testing.assert_allclose(up_norms.numpy(), up_norms_ref.numpy())


@unittest.skipUnless(HAS_MLX, "MLX not installed (Mac-only)")
class TestFusedGateUpSplitLoader(unittest.TestCase):
    """End-to-end: fused-gate-up TQ3 checkpoint loads via the loader."""

    def _write_fused_moe_checkpoint(self, tmp_path: Path):
        """Build a tiny synthetic qwen3_5_moe-style checkpoint with the
        *fused* gate_up_proj layout (mimicking an HF pre-sanitize layout).
        """
        import json

        import mlx.nn as nn
        from mlx_lm.models.qwen2_moe import Model as Qwen2MoEModel
        from mlx_lm.models.qwen2_moe import ModelArgs as Qwen2MoEArgs
        from mlx_lm.models.switch_layers import SwitchLinear
        from safetensors.torch import save_file

        from turboquant_vllm.torch_ops import PolarQuantTorch
        from turboquant_vllm.weight_quant import pack_indices, padded_size

        args = Qwen2MoEArgs(
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
        mx.random.seed(0)
        model = Qwen2MoEModel(args)
        mx.eval(model.parameters())

        bits, group_size, seed = 3, 128, 42
        pq = PolarQuantTorch(dim=group_size, bit_width=bits, seed=seed, device="cpu")

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
        switch_paths = {name: m for name, m in model.named_modules() if isinstance(m, SwitchLinear)}

        weights_to_save: dict[str, torch.Tensor] = {}
        reconstructed: dict[str, np.ndarray] = {}

        def compress_weight(w_np: np.ndarray, path_prefix: str):
            """Return packed + norms for the given weight and store the
            dequantised reference under reconstructed[path_prefix]."""
            w_pt = torch.from_numpy(w_np).float()
            shape = w_pt.shape
            in_features = shape[-1]
            padded_in, n_groups = padded_size(in_features, group_size)
            if padded_in > in_features:
                padded = torch.zeros(*shape[:-1], padded_in, dtype=w_pt.dtype)
                padded[..., :in_features] = w_pt
            else:
                padded = w_pt
            grouped = padded.reshape(-1, group_size)
            idx, norms_raw = pq.quantize(grouped, norm_correction=True)
            packed = pack_indices(idx, bits)
            norms = norms_raw.reshape(int(np.prod(shape[:-1])), n_groups)
            w_groups = pq.dequantize(idx, norms_raw)
            w_rec = w_groups.reshape(*shape[:-1], padded_in)[..., :in_features]
            return packed, norms, w_rec.numpy()

        # For each MoE layer, fuse gate_proj + up_proj into a single 3D tensor
        # whose top half is gate and bottom half is up (matching HF layout).
        fused_sources: dict[str, tuple[str, str]] = {}
        for path in switch_paths:
            if path.endswith(".switch_mlp.gate_proj"):
                prefix = path.rsplit(".switch_mlp.gate_proj", 1)[0]
                up_path = f"{prefix}.switch_mlp.up_proj"
                if up_path in switch_paths:
                    fused_sources[prefix] = (path, up_path)

        skip_paths: set[str] = set()
        for prefix, (gate_path, up_path) in fused_sources.items():
            g = np.array(params[f"{gate_path}.weight"])
            u = np.array(params[f"{up_path}.weight"])
            fused = np.concatenate([g, u], axis=-2)  # (n_experts, 2*out_half, in)
            packed, norms, rec = compress_weight(fused, prefix)
            fused_key = f"{prefix}.experts.gate_up_proj"
            weights_to_save[f"{fused_key}.tq_packed"] = packed
            weights_to_save[f"{fused_key}.tq_norms"] = norms
            # Remember the halves so the reference model knows what to load
            out_half = g.shape[-2]
            reconstructed[f"{gate_path}.weight"] = rec[:, :out_half, :].astype(np.float32)
            reconstructed[f"{up_path}.weight"] = rec[:, out_half:, :].astype(np.float32)
            skip_paths.update([gate_path, up_path])

        # Compress everything else as usual (dense Linears + down_proj SwitchLinear)
        for name, arr in params.items():
            if not name.endswith(".weight"):
                weights_to_save[name] = torch.from_numpy(np.array(arr))
                continue
            path = name.rsplit(".", 1)[0]
            if path in skip_paths:
                continue
            w_np = np.array(arr)
            if (path in linear_paths or path in switch_paths) and w_np.shape[-1] >= group_size:
                if path.endswith(".switch_mlp.down_proj"):
                    # Rename to HF-style experts.down_proj so the loader's split
                    # can also exercise the down_proj rename path.
                    prefix_mlp = path.rsplit(".switch_mlp.down_proj", 1)[0]
                    hf_key = f"{prefix_mlp}.experts.down_proj"
                    packed, norms, rec = compress_weight(w_np, path)
                    weights_to_save[f"{hf_key}.tq_packed"] = packed
                    weights_to_save[f"{hf_key}.tq_norms"] = norms
                    reconstructed[name] = rec.astype(np.float32)
                else:
                    packed, norms, rec = compress_weight(w_np, path)
                    weights_to_save[f"{path}.weight.tq_packed"] = packed
                    weights_to_save[f"{path}.weight.tq_norms"] = norms
                    reconstructed[name] = rec.astype(np.float32)
            else:
                weights_to_save[name] = torch.from_numpy(w_np)

        save_file(weights_to_save, str(tmp_path / "model.safetensors"))
        with open(tmp_path / "config.json", "w") as f:
            json.dump(
                {
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
                },
                f,
            )
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

    def test_fused_gate_up_proj_checkpoint_loads(self):
        """Fused gate_up_proj + renamed down_proj TQ3 checkpoint loads end-to-end."""
        import tempfile

        from mlx_lm.models.qwen2_moe import Model as Qwen2MoEModel

        from turboquant_vllm.mlx_loader import load_tq3_model

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            args, reconstructed = self._write_fused_moe_checkpoint(tmp_path)

            model_tq, _ = load_tq3_model(str(tmp_path))

            # Reference model with the dequantised halves patched in
            mx.random.seed(0)
            model_ref = Qwen2MoEModel(args)
            mx.eval(model_ref.parameters())
            for name, w_np in reconstructed.items():
                parent = model_ref
                parts = name.split(".")
                for p in parts[:-1]:
                    parent = parent[int(p)] if p.isdigit() else getattr(parent, p)
                setattr(parent, parts[-1], mx.array(w_np.astype(np.float32)))
            mx.eval(model_ref.parameters())

            # Verify the loader actually replaced the switch_mlp modules
            from turboquant_vllm.mlx_model import TurboQuantMLXSwitchLinear

            replaced = sum(1 for _, m in model_tq.named_modules() if isinstance(m, TurboQuantMLXSwitchLinear))
            # Qwen2-MoE has 2 layers × 3 switch projections (gate/up/down) = 6
            self.assertGreaterEqual(replaced, 6, f"Expected at least 6 switch replacements, got {replaced}")

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


if __name__ == "__main__":
    unittest.main()
