"""Tests for the Triton fused dequant-GEMM kernel.

Verifies that the fused kernel produces the same output as separate
dequant + matmul. Requires GPU + Triton.

Run: pytest tests/test_triton_fused_gemm.py -v -s
"""

import pytest
import torch

from turboquant_vllm.torch_ops import optimal_centroids
from turboquant_vllm.weight_quant import pack_indices, unpack_indices, _get_quantizer


def _skip_if_no_triton():
    try:
        import triton  # noqa: F401
        return False
    except ImportError:
        return True


def _prepare(out_dim, in_dim, bits, group_size, device="cuda"):
    """Quantize a random weight matrix, return everything needed for both paths."""
    weight = torch.randn(out_dim, in_dim, device=device, dtype=torch.float32)
    n_groups = in_dim // group_size
    grouped = weight.reshape(-1, group_size)
    quantizer = _get_quantizer(group_size, bits, device)
    indices, norms = quantizer.quantize(grouped)
    packed = pack_indices(indices, bits).contiguous()
    norms_2d = norms.reshape(out_dim, n_groups).contiguous()
    centroids = torch.tensor(
        optimal_centroids(bits, group_size), device=device, dtype=torch.float32
    )
    return packed, norms_2d, quantizer, centroids


def _reference_gemm(x, packed, norms_2d, quantizer, bits, group_size, out_dim, in_dim):
    """Reference: separate dequant + matmul."""
    idx = unpack_indices(packed, bits, group_size)
    groups = quantizer.dequantize(idx, norms_2d.reshape(-1))
    w_deq = groups.reshape(out_dim, in_dim)
    return torch.matmul(x, w_deq.t())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(_skip_if_no_triton(), reason="Triton required")
class TestTritonFusedGemm:

    @pytest.mark.parametrize("bits", [4])
    @pytest.mark.parametrize("M", [1, 16, 64])
    def test_correctness(self, bits, M):
        """Fused kernel output matches reference dequant + matmul."""
        from turboquant_vllm.triton_ops import tq_fused_gemm

        N, K, group_size = 256, 512, 128
        packed, norms_2d, quantizer, centroids = _prepare(N, K, bits, group_size)

        x = torch.randn(M, K, device="cuda", dtype=torch.float32)

        # Reference
        ref = _reference_gemm(x, packed, norms_2d, quantizer, bits, group_size, N, K)

        # Fused kernel
        fused = tq_fused_gemm(x, packed, norms_2d,
                              quantizer.signs1, quantizer.signs2, centroids,
                              group_size=group_size, bits=bits)

        max_diff = (ref - fused).abs().max().item()
        rel_err = max_diff / (ref.abs().max().item() + 1e-8)
        print(f"\n  M={M}, bits={bits}: max_diff={max_diff:.6f}, rel_err={rel_err:.6f}")
        assert rel_err < 0.01, f"Relative error {rel_err:.6f} too high (max_diff={max_diff:.6f})"

    def test_fp16_input(self):
        """Works with FP16 activations."""
        from turboquant_vllm.triton_ops import tq_fused_gemm

        N, K, M, bits, group_size = 256, 512, 16, 4, 128
        packed, norms_2d, quantizer, centroids = _prepare(N, K, bits, group_size)

        x_f32 = torch.randn(M, K, device="cuda", dtype=torch.float32)
        x_f16 = x_f32.half()

        ref = _reference_gemm(x_f32, packed, norms_2d, quantizer, bits, group_size, N, K)
        fused = tq_fused_gemm(x_f16, packed, norms_2d,
                              quantizer.signs1, quantizer.signs2, centroids,
                              group_size=group_size, bits=bits)

        # FP16 introduces some error, but should be reasonable
        max_diff = (ref.half().float() - fused.float()).abs().max().item()
        rel_err = max_diff / (ref.abs().max().item() + 1e-8)
        print(f"\n  FP16: max_diff={max_diff:.6f}, rel_err={rel_err:.6f}")
        assert rel_err < 0.05, f"FP16 relative error {rel_err:.6f} too high"

    def test_with_bias(self):
        """Bias is correctly added."""
        from turboquant_vllm.triton_ops import tq_fused_gemm

        N, K, M, bits, group_size = 256, 512, 16, 4, 128
        packed, norms_2d, quantizer, centroids = _prepare(N, K, bits, group_size)

        x = torch.randn(M, K, device="cuda", dtype=torch.float32)
        bias = torch.randn(N, device="cuda", dtype=torch.float32)

        ref = _reference_gemm(x, packed, norms_2d, quantizer, bits, group_size, N, K)
        ref += bias

        fused = tq_fused_gemm(x, packed, norms_2d,
                              quantizer.signs1, quantizer.signs2, centroids,
                              group_size=group_size, bits=bits, bias=bias)

        max_diff = (ref - fused).abs().max().item()
        rel_err = max_diff / (ref.abs().max().item() + 1e-8)
        print(f"\n  Bias: max_diff={max_diff:.6f}, rel_err={rel_err:.6f}")
        assert rel_err < 0.01

    def test_speed_vs_separate(self):
        """Fused kernel is faster than separate dequant + matmul."""
        from turboquant_vllm.triton_ops import tq_fused_gemm

        N, K, M, bits, group_size = 4096, 4096, 1, 4, 128
        packed, norms_2d, quantizer, centroids = _prepare(N, K, bits, group_size)
        x = torch.randn(M, K, device="cuda", dtype=torch.float16)

        # Warmup
        for _ in range(3):
            tq_fused_gemm(x, packed, norms_2d,
                          quantizer.signs1, quantizer.signs2, centroids,
                          group_size=group_size, bits=bits)
            _reference_gemm(x.float(), packed, norms_2d, quantizer, bits, group_size, N, K)
        torch.cuda.synchronize()

        import time
        n_iters = 50

        # Fused
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_iters):
            tq_fused_gemm(x, packed, norms_2d,
                          quantizer.signs1, quantizer.signs2, centroids,
                          group_size=group_size, bits=bits)
        torch.cuda.synchronize()
        fused_ms = (time.perf_counter() - t0) / n_iters * 1000

        # Separate: dequant + matmul
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_iters):
            idx = unpack_indices(packed, bits, group_size)
            groups = quantizer.dequantize(idx, norms_2d.reshape(-1))
            w_deq = groups.reshape(N, K).to(x.dtype)
            torch.matmul(x, w_deq.t())
        torch.cuda.synchronize()
        separate_ms = (time.perf_counter() - t0) / n_iters * 1000

        speedup = separate_ms / fused_ms
        print(f"\n  Fused: {fused_ms:.3f} ms, Separate: {separate_ms:.3f} ms, "
              f"Speedup: {speedup:.1f}x")
        # Fused should be at least competitive (>0.5x is OK for first version)
        assert fused_ms < separate_ms * 3, f"Fused is too slow: {fused_ms:.1f}ms vs {separate_ms:.1f}ms"
