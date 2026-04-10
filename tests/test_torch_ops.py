"""Tests for TurboQuant+ PyTorch operations.

Verifies the PyTorch fallback path (no CUDA compilation required).
These tests run on any CUDA-capable device.

Run: pytest tests/test_torch_ops.py -v
"""

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA not available", allow_module_level=True)

from turboquant_vllm.torch_ops import PolarQuantTorch, KVCacheCompressorTorch

HEAD_DIM = 128
SEED = 42


@pytest.fixture
def random_vectors():
    torch.manual_seed(123)
    return torch.randn(64, HEAD_DIM, dtype=torch.float16, device="cuda")


class TestPolarQuant:
    """PolarQuant roundtrip and quality."""

    def test_roundtrip_shape(self, random_vectors):
        pq = PolarQuantTorch(HEAD_DIM, bits=4, seed=SEED, device="cuda")
        indices, norms = pq.quantize(random_vectors)
        assert indices.shape == (64, HEAD_DIM)
        assert norms.shape == (64,)

        restored = pq.dequantize(indices, norms)
        assert restored.shape == (64, HEAD_DIM)

    def test_4bit_mse(self, random_vectors):
        pq = PolarQuantTorch(HEAD_DIM, bits=4, seed=SEED, device="cuda")
        indices, norms = pq.quantize(random_vectors)
        restored = pq.dequantize(indices, norms)
        mse = ((random_vectors.float() - restored.float()) ** 2).mean().item()
        assert mse < 0.02, f"4-bit MSE {mse:.6f} too high"

    def test_3bit_higher_mse(self, random_vectors):
        pq4 = PolarQuantTorch(HEAD_DIM, bits=4, seed=SEED, device="cuda")
        pq3 = PolarQuantTorch(HEAD_DIM, bits=3, seed=SEED, device="cuda")

        r4 = pq4.dequantize(*pq4.quantize(random_vectors))
        r3 = pq3.dequantize(*pq3.quantize(random_vectors))

        mse4 = ((random_vectors.float() - r4.float()) ** 2).mean().item()
        mse3 = ((random_vectors.float() - r3.float()) ** 2).mean().item()
        assert mse3 > mse4

    def test_zero_vector(self):
        pq = PolarQuantTorch(HEAD_DIM, bits=4, seed=SEED, device="cuda")
        zero = torch.zeros(1, HEAD_DIM, dtype=torch.float16, device="cuda")
        indices, norms = pq.quantize(zero)
        restored = pq.dequantize(indices, norms)
        assert restored.abs().max().item() < 1e-6

    def test_seed_determinism(self, random_vectors):
        pq1 = PolarQuantTorch(HEAD_DIM, bits=4, seed=SEED, device="cuda")
        pq2 = PolarQuantTorch(HEAD_DIM, bits=4, seed=SEED, device="cuda")
        i1, n1 = pq1.quantize(random_vectors)
        i2, n2 = pq2.quantize(random_vectors)
        assert torch.equal(i1, i2)
        assert torch.equal(n1, n2)


class TestKVCacheCompressor:
    """Full K/V compression with QJL."""

    def test_compress_decompress_k(self, random_vectors):
        comp = KVCacheCompressorTorch(HEAD_DIM, k_bits=4, v_bits=4, seed=SEED, device="cuda")
        ck = comp.compress_k(random_vectors)
        restored = comp.decompress_k(ck)
        mse = ((random_vectors.float() - restored.float()) ** 2).mean().item()
        assert mse < 0.02

    def test_compress_decompress_v(self, random_vectors):
        comp = KVCacheCompressorTorch(HEAD_DIM, k_bits=4, v_bits=4, seed=SEED, device="cuda")
        cv = comp.compress_v(random_vectors)
        restored = comp.decompress_v(cv)
        mse = ((random_vectors.float() - restored.float()) ** 2).mean().item()
        assert mse < 0.02

    def test_asymmetric_k4_v3(self, random_vectors):
        comp = KVCacheCompressorTorch(HEAD_DIM, k_bits=4, v_bits=3, seed=SEED, device="cuda")
        ck = comp.compress_k(random_vectors)
        cv = comp.compress_v(random_vectors)
        k_restored = comp.decompress_k(ck)
        v_restored = comp.decompress_v(cv)
        k_mse = ((random_vectors.float() - k_restored.float()) ** 2).mean().item()
        v_mse = ((random_vectors.float() - v_restored.float()) ** 2).mean().item()
        # V at 3-bit should have higher MSE than K at 4-bit
        assert v_mse > k_mse

    def test_compression_ratio(self):
        comp = KVCacheCompressorTorch(HEAD_DIM, k_bits=4, v_bits=4, seed=SEED, device="cuda")
        stats = comp.memory_stats()
        assert 3.0 < stats["compression_ratio"] < 5.0

    def test_cosine_preservation(self, random_vectors):
        """Top-10 cosine ranking should be mostly preserved."""
        comp = KVCacheCompressorTorch(HEAD_DIM, k_bits=4, v_bits=4, seed=SEED, device="cuda")
        query = random_vectors[0:1].float()
        others = random_vectors[1:].float()

        true_sims = torch.nn.functional.cosine_similarity(query, others)
        true_top = set(torch.argsort(-true_sims)[:10].tolist())

        # Compress and decompress K (inner product preservation matters)
        ck = comp.compress_k(random_vectors)
        restored = comp.decompress_k(ck).float()
        comp_sims = torch.nn.functional.cosine_similarity(restored[0:1], restored[1:])
        comp_top = set(torch.argsort(-comp_sims)[:10].tolist())

        overlap = len(true_top & comp_top)
        assert overlap >= 7, f"Top-10 overlap {overlap}/10 too low"
