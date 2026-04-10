import math

import torch

from turboquant_vllm.native_backend import TurboQuantAttentionImpl
from turboquant_vllm.tq_config import TurboQuantConfig, get_centroids


def _pack_bits_generic(indices: torch.Tensor, bits: int) -> torch.Tensor:
    rows, dim = indices.shape
    nbytes = math.ceil(dim * bits / 8)
    packed = torch.zeros(rows, nbytes, dtype=torch.uint8)
    for j in range(dim):
        bo = j * bits
        bi, si = bo // 8, bo % 8
        packed[:, bi] |= ((indices[:, j].int() << si) & 0xFF).to(torch.uint8)
        if si + bits > 8 and bi + 1 < nbytes:
            packed[:, bi + 1] |= ((indices[:, j].int() >> (8 - si)) & 0xFF).to(torch.uint8)
    return packed


def _unpack_bits_generic(packed: torch.Tensor, bits: int, dim: int) -> torch.Tensor:
    out = torch.zeros(packed.shape[0], dim, dtype=torch.uint8)
    for j in range(dim):
        bo = j * bits
        bi, si = bo // 8, bo % 8
        val = (packed[:, bi].int() >> si) & ((1 << bits) - 1)
        if si + bits > 8 and bi + 1 < packed.shape[1]:
            val |= (packed[:, bi + 1].int() << (8 - si)) & ((1 << bits) - 1)
        out[:, j] = val.to(torch.uint8)
    return out


def test_mse_3bit_roundtrip_random_indices():
    idx = torch.randint(0, 8, (128, 19), dtype=torch.uint8)
    packed = _pack_bits_generic(idx, bits=3)
    unpacked = _unpack_bits_generic(packed, bits=3, dim=idx.shape[1])
    assert torch.equal(unpacked, idx)


@torch.no_grad()
def test_store_kv_uses_true_3bit_layout_for_representative_dims():
    torch.manual_seed(0)

    for d in (10, 16):
        impl = TurboQuantAttentionImpl(
            num_heads=1,
            head_size=d,
            scale=1.0,
            num_kv_heads=1,
            kv_cache_dtype="tq3",
        )
        impl.tq_config = TurboQuantConfig(head_dim=d, total_bits=3, value_quant_bits=8, no_qjl=True)

        layer = torch.nn.Module()
        layer._tq_PiT = torch.eye(d, dtype=torch.float32)
        layer._tq_Pi_S_T = torch.eye(d, dtype=torch.float32)
        centroids = get_centroids(d, 3)
        c_sorted, _ = centroids.sort()
        layer._tq_midpoints = (c_sorted[:-1] + c_sorted[1:]) / 2
        impl._current_layer = layer

        n = 7
        key = torch.randn(n, 1, d)
        value = torch.randn(n, 1, d)

        block_size = 16
        slot_size = impl.tq_config.slot_size
        kv_cache = torch.zeros(1, block_size, 1, slot_size, dtype=torch.uint8)
        slot_mapping = torch.arange(n, dtype=torch.int64)

        impl._store_kv(
            key=key,
            value=value,
            kv_cache=kv_cache,
            slot_mapping=slot_mapping,
            Pi=torch.eye(d),
            S=torch.eye(d),
            centroids=centroids,
        )

        k_flat = key.float().reshape(-1, d)
        x_hat = k_flat / (k_flat.norm(dim=1, keepdim=True) + 1e-8)
        expected_idx = torch.bucketize(x_hat, layer._tq_midpoints).to(torch.uint8)

        mse_bytes = math.ceil(d * 3 / 8)
        stored_mse = kv_cache[0, :n, 0, :mse_bytes].contiguous()
        expected_packed = _pack_bits_generic(expected_idx, bits=3)

        assert torch.equal(stored_mse, expected_packed), f"3-bit packed bytes mismatch for D={d}"

        unpacked = _unpack_bits_generic(stored_mse, bits=3, dim=d)
        assert torch.equal(unpacked, expected_idx), f"3-bit unpack mismatch for D={d}"
