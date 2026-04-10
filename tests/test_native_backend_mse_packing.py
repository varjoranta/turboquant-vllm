import math
from dataclasses import dataclass

import torch

from turboquant_vllm.native_backend import TurboQuantAttentionImpl
from turboquant_vllm.tq_config import TurboQuantConfig
from turboquant_vllm.weight_quant import pack_indices, unpack_indices


def _make_synthetic_centroids_and_midpoints(d: int, bits: int):
    """Return (centroids, midpoints) for testing without expensive Lloyd-Max."""
    n_levels = 2 ** bits
    sigma = 1.0 / math.sqrt(d)
    lo, hi = -3.0 * sigma, 3.0 * sigma
    centroids = torch.linspace(lo, hi, n_levels, dtype=torch.float32)
    midpoints = (centroids[:-1] + centroids[1:]) / 2
    return centroids, midpoints


def test_mse_3bit_roundtrip_random_indices():
    idx = torch.randint(0, 8, (128, 19), dtype=torch.uint8)
    packed = pack_indices(idx, bits=3)
    unpacked = unpack_indices(packed, bits=3, dim=idx.shape[1]).to(torch.uint8)
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
        impl.tq_config = TurboQuantConfig(head_dim=d, total_bits=3, value_quant_bits=4, no_qjl=True)

        centroids, midpoints = _make_synthetic_centroids_and_midpoints(d, 3)

        layer = torch.nn.Module()
        layer._tq_PiT = torch.eye(d, dtype=torch.float32)
        layer._tq_Pi_S_T = torch.eye(d, dtype=torch.float32)
        layer._tq_midpoints = midpoints
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
        expected_idx = torch.bucketize(x_hat, midpoints).to(torch.uint8)

        mse_bytes = math.ceil(d * 3 / 8)
        stored_mse = kv_cache[0, :n, 0, :mse_bytes].contiguous()
        expected_packed = pack_indices(expected_idx, bits=3)

        # pack_indices pads to multiples of 8 indices; compare only the
        # first mse_bytes that the native backend actually writes.
        assert torch.equal(stored_mse, expected_packed[:, :mse_bytes]), (
            f"3-bit packed bytes mismatch for D={d}"
        )

        # To unpack with weight_quant, pad stored bytes to a multiple of 3
        pad_needed = (3 - mse_bytes % 3) % 3
        if pad_needed:
            stored_padded = torch.nn.functional.pad(stored_mse, (0, pad_needed))
        else:
            stored_padded = stored_mse
        unpacked = unpack_indices(stored_padded, bits=3, dim=d).to(torch.uint8)
        assert torch.equal(unpacked, expected_idx), f"3-bit unpack mismatch for D={d}"


@dataclass
class _FakeDecodeMetadata:
    """Minimal metadata for exercising _decode_attention_python."""
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    query_start_loc: torch.Tensor
    slot_mapping: torch.Tensor
    num_actual_tokens: int = 0
    max_query_len: int = 0
    max_seq_len: int = 0
    is_prefill: bool = False


@torch.no_grad()
def test_decode_roundtrip_3bit_mse():
    """Verify store → decode roundtrip for 3-bit MSE with D%8==0 and D%8!=0."""
    torch.manual_seed(42)

    for d in (10, 16):
        impl = TurboQuantAttentionImpl(
            num_heads=1,
            head_size=d,
            scale=1.0,
            num_kv_heads=1,
            kv_cache_dtype="tq3",
        )
        impl.tq_config = TurboQuantConfig(
            head_dim=d, total_bits=3, value_quant_bits=4, no_qjl=True,
        )

        centroids, midpoints = _make_synthetic_centroids_and_midpoints(d, 3)

        layer = torch.nn.Module()
        layer._tq_PiT = torch.eye(d, dtype=torch.float32)
        layer._tq_Pi_S_T = torch.eye(d, dtype=torch.float32)
        layer._tq_midpoints = midpoints
        impl._current_layer = layer

        n = 5
        key = torch.randn(n, 1, d)
        value = torch.randn(n, 1, d)

        block_size = 16
        slot_size = impl.tq_config.slot_size
        kv_cache = torch.zeros(1, block_size, 1, slot_size, dtype=torch.uint8)
        slot_mapping = torch.arange(n, dtype=torch.int64)

        impl._store_kv(
            key=key, value=value, kv_cache=kv_cache,
            slot_mapping=slot_mapping,
            Pi=torch.eye(d), S=torch.eye(d),
            centroids=centroids,
        )

        # Compute expected MSE indices
        k_flat = key.float().reshape(-1, d)
        x_hat = k_flat / (k_flat.norm(dim=1, keepdim=True) + 1e-8)
        expected_idx = torch.bucketize(x_hat, midpoints)

        # Extract packed MSE bytes from cache and unpack with weight_quant
        mse_bytes = math.ceil(d * 3 / 8)
        stored_mse = kv_cache[0, :n, 0, :mse_bytes].contiguous()
        # Pad stored bytes to a multiple of 3 for weight_quant unpack
        pad_needed = (3 - mse_bytes % 3) % 3
        if pad_needed:
            stored_mse_padded = torch.nn.functional.pad(stored_mse, (0, pad_needed))
        else:
            stored_mse_padded = stored_mse
        decoded_idx = unpack_indices(stored_mse_padded, bits=3, dim=d)
        assert torch.equal(decoded_idx, expected_idx), (
            f"weight_quant unpack mismatch for D={d}"
        )

        # Now exercise the actual Python decode path in native_backend
        query = torch.randn(1, 1, d)
        num_blocks_needed = math.ceil(n / block_size)
        meta = _FakeDecodeMetadata(
            seq_lens=torch.tensor([n]),
            block_table=torch.arange(num_blocks_needed, dtype=torch.int64).unsqueeze(0),
            query_start_loc=torch.tensor([0, 1]),
            slot_mapping=torch.zeros(1, dtype=torch.int64),
            num_actual_tokens=1,
            max_query_len=1,
            max_seq_len=n,
        )

        # This should not raise — it exercises the 3-bit unpack inside
        # _decode_attention_python and produces a valid attention output.
        output = impl._decode_attention_python(
            query=query, kv_cache=kv_cache,
            attn_metadata=meta,
            Pi=torch.eye(d), S=torch.eye(d),
            centroids=centroids,
        )
        assert output.shape == (1, 1, d), f"decode output shape mismatch for D={d}"
        assert torch.isfinite(output).all(), f"decode output has non-finite values for D={d}"
