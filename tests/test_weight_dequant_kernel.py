"""Tests for the fused weight dequant CUDA kernel.

Verifies that the CUDA kernel produces the same output as the PyTorch
dequantization path. Requires GPU.

Run: pytest tests/test_weight_dequant_kernel.py -v
"""

import pytest
import torch

from turboquant_vllm.torch_ops import optimal_centroids
from turboquant_vllm.weight_quant import pack_indices, unpack_indices, _get_quantizer


@pytest.fixture(scope="session")
def cuda_mod():
    """Lazy-build CUDA module. Only triggers JIT when a test needs it."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    from turboquant_vllm.build import build

    mod = build()
    if not hasattr(mod, "weight_dequant"):
        pytest.skip("CUDA weight_dequant kernel not compiled")
    return mod


def _prepare_quantized(out_dim, in_dim, bits, group_size, device="cuda"):
    """Quantize a random weight matrix and return packed data for testing."""
    weight = torch.randn(out_dim, in_dim, device=device, dtype=torch.float32)
    n_groups = in_dim // group_size
    grouped = weight.reshape(-1, group_size)
    quantizer = _get_quantizer(group_size, bits, device)
    indices, norms = quantizer.quantize(grouped)
    packed = pack_indices(indices, bits).contiguous()
    norms_2d = norms.reshape(out_dim, n_groups).contiguous()
    centroids = torch.tensor(optimal_centroids(bits, group_size), device=device, dtype=torch.float32)
    return packed, norms_2d, quantizer, centroids


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestWeightDequantKernel:
    @pytest.mark.parametrize("bits", [2, 4])
    @pytest.mark.parametrize("group_size", [64, 128])
    def test_correctness(self, cuda_mod, bits, group_size):
        out_dim, in_dim = 256, 512
        packed, norms_2d, quantizer, centroids = _prepare_quantized(out_dim, in_dim, bits, group_size)

        # PyTorch reference
        idx = unpack_indices(packed, bits, group_size)
        groups_ref = quantizer.dequantize(idx, norms_2d.reshape(-1))
        w_ref = groups_ref.reshape(out_dim, in_dim)

        # CUDA kernel
        w_cuda = torch.empty(out_dim, in_dim, device="cuda", dtype=torch.float32)
        cuda_mod.weight_dequant(
            packed, norms_2d, quantizer.signs1, quantizer.signs2, centroids, w_cuda, group_size, bits, out_dim, in_dim
        )

        max_diff = (w_ref - w_cuda).abs().max().item()
        assert max_diff < 1e-4, f"Max diff {max_diff} (bits={bits}, gs={group_size})"

    def test_fp16_output(self, cuda_mod):
        out_dim, in_dim, bits, group_size = 256, 256, 4, 128
        packed, norms_2d, quantizer, centroids = _prepare_quantized(out_dim, in_dim, bits, group_size)

        w_f32 = torch.empty(out_dim, in_dim, device="cuda", dtype=torch.float32)
        w_f16 = torch.empty(out_dim, in_dim, device="cuda", dtype=torch.float16)
        args = (
            packed,
            norms_2d,
            quantizer.signs1,
            quantizer.signs2,
            centroids,
            None,
            group_size,
            bits,
            out_dim,
            in_dim,
        )
        cuda_mod.weight_dequant(*args[:5], w_f32, *args[6:])
        cuda_mod.weight_dequant(*args[:5], w_f16, *args[6:])

        max_diff = (w_f32.half().float() - w_f16.float()).abs().max().item()
        assert max_diff < 1e-3, f"FP16 diff {max_diff}"

    def test_3d_moe(self, cuda_mod):
        n_experts, out_dim, in_dim, bits, group_size = 8, 128, 256, 4, 128
        weight = torch.randn(n_experts, out_dim, in_dim, device="cuda")

        n_groups = in_dim // group_size
        grouped = weight.reshape(-1, group_size)
        quantizer = _get_quantizer(group_size, bits, "cuda")
        indices, norms = quantizer.quantize(grouped)
        packed = pack_indices(indices, bits).contiguous()
        norms_2d = norms.reshape(n_experts * out_dim, n_groups).contiguous()
        centroids = torch.tensor(optimal_centroids(bits, group_size), device="cuda", dtype=torch.float32)

        output_3d = torch.empty(n_experts, out_dim, in_dim, device="cuda")
        cuda_mod.weight_dequant_3d(
            packed,
            norms_2d,
            quantizer.signs1,
            quantizer.signs2,
            centroids,
            output_3d,
            group_size,
            bits,
            n_experts,
            out_dim,
            in_dim,
        )

        output_2d = torch.empty(n_experts * out_dim, in_dim, device="cuda")
        cuda_mod.weight_dequant(
            packed,
            norms_2d,
            quantizer.signs1,
            quantizer.signs2,
            centroids,
            output_2d,
            group_size,
            bits,
            n_experts * out_dim,
            in_dim,
        )

        max_diff = (output_3d.reshape(-1, in_dim) - output_2d).abs().max().item()
        assert max_diff < 1e-6, f"3D vs 2D diff {max_diff}"

    def test_speed_vs_pytorch(self, cuda_mod):
        out_dim, in_dim, bits, group_size = 4096, 4096, 4, 128
        packed, norms_2d, quantizer, centroids = _prepare_quantized(out_dim, in_dim, bits, group_size)
        w_out = torch.empty(out_dim, in_dim, device="cuda", dtype=torch.float16)

        # Warmup
        for _ in range(3):
            cuda_mod.weight_dequant(
                packed,
                norms_2d,
                quantizer.signs1,
                quantizer.signs2,
                centroids,
                w_out,
                group_size,
                bits,
                out_dim,
                in_dim,
            )
        torch.cuda.synchronize()

        import time

        n_iters = 100

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_iters):
            cuda_mod.weight_dequant(
                packed,
                norms_2d,
                quantizer.signs1,
                quantizer.signs2,
                centroids,
                w_out,
                group_size,
                bits,
                out_dim,
                in_dim,
            )
        torch.cuda.synchronize()
        cuda_ms = (time.perf_counter() - t0) / n_iters * 1000

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_iters):
            idx = unpack_indices(packed, bits, group_size)
            g = quantizer.dequantize(idx, norms_2d.reshape(-1))
            _ = g.reshape(out_dim, in_dim)
        torch.cuda.synchronize()
        pytorch_ms = (time.perf_counter() - t0) / n_iters * 1000

        speedup = pytorch_ms / cuda_ms
        print(f"\n  CUDA: {cuda_ms:.3f} ms, PyTorch: {pytorch_ms:.3f} ms, Speedup: {speedup:.1f}x")
        assert speedup > 2.0, f"Expected >2x speedup, got {speedup:.1f}x"
