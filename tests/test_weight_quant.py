"""Tests for TurboQuant weight quantization.

These tests validate the weight compression pipeline without vLLM.
Can run on any CUDA device.

Run: pytest tests/test_weight_quant.py -v
"""

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA not available", allow_module_level=True)

from turboquant_vllm.weight_quant import (
    quantize_weight,
    dequantize_weight,
    pack_indices,
    unpack_indices,
    TurboQuantLinearMethod,
    TurboQuantWrapper,
)


class TestQuantizeDequantize:
    """Basic weight quantize/dequantize roundtrip."""

    def test_roundtrip_shape(self):
        w = torch.randn(256, 512, device="cuda")
        compressed = quantize_weight(w, bits=4)
        w_hat = dequantize_weight(compressed)
        assert w_hat.shape == w.shape

    def test_4bit_mse(self):
        torch.manual_seed(42)
        w = torch.randn(256, 512, device="cuda")
        compressed = quantize_weight(w, bits=4)
        w_hat = dequantize_weight(compressed)
        mse = ((w - w_hat) ** 2).mean().item()
        assert mse < 0.02, f"4-bit MSE {mse:.6f} too high"

    def test_3bit_higher_mse(self):
        torch.manual_seed(42)
        w = torch.randn(256, 512, device="cuda")
        c4 = quantize_weight(w, bits=4)
        c3 = quantize_weight(w, bits=3)
        mse4 = ((w - dequantize_weight(c4)) ** 2).mean().item()
        mse3 = ((w - dequantize_weight(c3)) ** 2).mean().item()
        assert mse3 > mse4, f"3-bit MSE {mse3:.6f} should be > 4-bit {mse4:.6f}"

    def test_2bit(self):
        torch.manual_seed(42)
        w = torch.randn(128, 256, device="cuda")
        compressed = quantize_weight(w, bits=2)
        w_hat = dequantize_weight(compressed)
        mse = ((w - w_hat) ** 2).mean().item()
        assert mse < 0.2, f"2-bit MSE {mse:.6f} too high"


class TestPacking:
    """Index packing and unpacking."""

    def test_4bit_roundtrip(self):
        indices = torch.randint(0, 16, (32, 128), dtype=torch.int64, device="cuda")
        packed = pack_indices(indices, bits=4)
        assert packed.shape == (32, 64)  # 2 per byte
        unpacked = unpack_indices(packed, bits=4, dim=128)
        assert torch.equal(indices, unpacked)

    def test_2bit_roundtrip(self):
        indices = torch.randint(0, 4, (32, 128), dtype=torch.int64, device="cuda")
        packed = pack_indices(indices, bits=2)
        assert packed.shape == (32, 32)  # 4 per byte
        unpacked = unpack_indices(packed, bits=2, dim=128)
        assert torch.equal(indices, unpacked)

    def test_3bit_roundtrip(self):
        indices = torch.randint(0, 8, (32, 128), dtype=torch.int64, device="cuda")
        packed = pack_indices(indices, bits=3)
        assert packed.shape == (32, 128)  # no packing
        unpacked = unpack_indices(packed, bits=3, dim=128)
        assert torch.equal(indices, unpacked)

    def test_4bit_memory_savings(self):
        indices = torch.randint(0, 16, (1024, 4096), dtype=torch.int64, device="cuda")
        packed = pack_indices(indices, bits=4)
        # int64 = 8 bytes, packed uint8 = 1 byte, 2 per byte
        # Ratio: (1024 * 4096 * 8) / (1024 * 2048 * 1) = 16x
        assert packed.numel() * packed.element_size() < indices.numel() * indices.element_size()


class TestLinearMethod:
    """End-to-end linear layer compression."""

    def test_compress_and_matmul(self):
        torch.manual_seed(42)
        linear = torch.nn.Linear(512, 256, bias=True).cuda()
        x = torch.randn(4, 512, device="cuda")

        # Original output
        with torch.no_grad():
            y_orig = linear(x)

        # Compressed output
        method = TurboQuantLinearMethod(bits=4)
        compressed = method.compress_layer(linear)
        y_comp = method.decompress_and_matmul(compressed, x, linear.bias)

        # Should be close
        rel_error = ((y_orig - y_comp) ** 2).mean() / (y_orig ** 2).mean()
        assert rel_error < 0.05, f"Relative error {rel_error:.4f} too high"

    def test_compression_ratio(self):
        linear = torch.nn.Linear(4096, 4096, bias=False).cuda()
        method = TurboQuantLinearMethod(bits=3)
        compressed = method.compress_layer(linear)
        # 3-bit unpacked: ratio ~= 2 bytes / 1 byte per element (FP16 vs uint8)
        # With norms: slightly less
        assert compressed["ratio"] > 1.5, f"Ratio {compressed['ratio']:.1f} too low"


class TestWrapper:
    """TurboQuantWrapper drop-in replacement."""

    def test_wrapper_forward(self):
        torch.manual_seed(42)
        linear = torch.nn.Linear(512, 256, bias=True).cuda()
        x = torch.randn(4, 512, device="cuda")

        with torch.no_grad():
            y_orig = linear(x)

        wrapper = TurboQuantWrapper(linear, bits=4)
        with torch.no_grad():
            y_wrapped = wrapper(x)

        rel_error = ((y_orig - y_wrapped) ** 2).mean() / (y_orig ** 2).mean()
        assert rel_error < 0.05, f"Wrapper relative error {rel_error:.4f} too high"

    def test_wrapper_no_bias(self):
        linear = torch.nn.Linear(256, 128, bias=False).cuda()
        wrapper = TurboQuantWrapper(linear, bits=3)
        x = torch.randn(2, 256, device="cuda")
        with torch.no_grad():
            y = wrapper(x)
        assert y.shape == (2, 128)

    def test_wrapper_memory_smaller(self):
        linear = torch.nn.Linear(4096, 4096, bias=False).cuda()
        orig_bytes = linear.weight.numel() * linear.weight.element_size()
        wrapper = TurboQuantWrapper(linear, bits=3)
        comp_bytes = wrapper.packed_weight.numel() + wrapper.norms.numel() * 4
        assert comp_bytes < orig_bytes, f"Compressed {comp_bytes} >= original {orig_bytes}"

    def test_wrapper_different_batch_sizes(self):
        linear = torch.nn.Linear(512, 256, bias=True).cuda()
        wrapper = TurboQuantWrapper(linear, bits=4)
        for batch in [1, 4, 16, 64]:
            x = torch.randn(batch, 512, device="cuda")
            with torch.no_grad():
                y = wrapper(x)
            assert y.shape == (batch, 256)
