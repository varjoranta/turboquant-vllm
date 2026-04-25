# SPDX-License-Identifier: MIT
"""CPU correctness for ``forward_rotate_batched``.

The new helper lifts the per-group ``D2 @ H @ D1`` rotation that the bs=1
fused GEMV path needs out of ``PolarQuantTorch._rotate``. This test pins
down two properties:

1. It matches ``PolarQuantTorch._rotate`` exactly (same math) for both
   full-width and block-diagonal cases — meaning the fused GEMV path
   doesn't introduce a different rotation than what the offline compression
   already used.
2. The block-diagonal mode does not mix energy across blocks — a signal
   confined to block 0 of the input stays in block 0 after rotation.
"""

from __future__ import annotations

import torch

from turboquant_vllm.torch_ops import PolarQuantTorch, forward_rotate_batched


def test_forward_rotate_matches_polarquant_full_width() -> None:
    torch.manual_seed(0)
    group_size = 128
    bits = 3
    pq = PolarQuantTorch(dim=group_size, bit_width=bits, device="cpu")

    x = torch.randn(4, group_size, dtype=torch.float32)
    expected = pq._rotate(x)
    got = forward_rotate_batched(x, pq.signs1, pq.signs2, group_size, block_size=None)
    assert torch.allclose(got, expected, atol=1e-5), \
        f"max diff {(got - expected).abs().max().item()}"


def test_forward_rotate_matches_polarquant_block_diag() -> None:
    torch.manual_seed(1)
    group_size = 128
    bits = 3
    block_size = 64
    pq = PolarQuantTorch(dim=group_size, bit_width=bits, device="cpu", rotary_dim=block_size)

    x = torch.randn(4, group_size, dtype=torch.float32)
    expected = pq._rotate(x)
    got = forward_rotate_batched(x, pq.signs1, pq.signs2, group_size, block_size=block_size)
    assert torch.allclose(got, expected, atol=1e-5), \
        f"max diff {(got - expected).abs().max().item()}"


def test_forward_rotate_block_diag_isolation() -> None:
    """Signal in block 0 must not bleed into block 1 after block-diag rotation."""
    torch.manual_seed(2)
    group_size = 128
    block_size = 64
    pq = PolarQuantTorch(dim=group_size, bit_width=3, device="cpu", rotary_dim=block_size)

    x = torch.zeros(1, group_size, dtype=torch.float32)
    x[0, :block_size] = torch.randn(block_size)  # only first block populated
    rotated = forward_rotate_batched(x, pq.signs1, pq.signs2, group_size, block_size=block_size)

    block0 = rotated[0, :block_size]
    block1 = rotated[0, block_size:]
    assert block0.abs().max() > 0.1, "block 0 lost energy"
    assert block1.abs().max() < 1e-6, f"block 1 contaminated: {block1.abs().max().item()}"


def test_forward_rotate_preserves_dtype_when_signs_match_x() -> None:
    """If signs are pre-cast to x's dtype, output dtype matches x.

    Regression test: passing fp32 signs against bf16 x auto-promotes the
    result to fp32 (PyTorch type-promotion rules), and downstream kernels
    with strict dtype checks (e.g. tq3_gemv_bs1 requires bf16 input) will
    reject the promoted tensor at runtime. Callers must hand in dtype-
    matched signs.
    """
    torch.manual_seed(4)
    group_size = 128
    pq = PolarQuantTorch(dim=group_size, bit_width=3, device="cpu")

    x_bf16 = torch.randn(1, group_size, dtype=torch.bfloat16)
    s1_bf16 = pq.signs1.to(torch.bfloat16)
    s2_bf16 = pq.signs2.to(torch.bfloat16)
    out = forward_rotate_batched(x_bf16, s1_bf16, s2_bf16, group_size, block_size=None)
    assert out.dtype == torch.bfloat16

    # And the canary: with fp32 signs against bf16 x, output IS promoted to
    # fp32. This is the surprise that broke `_forward_gpu` on first GPU run.
    out_promoted = forward_rotate_batched(x_bf16, pq.signs1, pq.signs2, group_size, block_size=None)
    assert out_promoted.dtype == torch.float32


def test_forward_rotate_preserves_shape_with_extra_groups() -> None:
    """Multi-group input — output shape preserved, each group rotated independently."""
    torch.manual_seed(3)
    group_size = 128
    bits = 3
    pq = PolarQuantTorch(dim=group_size, bit_width=bits, device="cpu")

    x = torch.randn(2, group_size * 4, dtype=torch.float32)  # 2 batch, 4 groups per row
    got = forward_rotate_batched(x, pq.signs1, pq.signs2, group_size, block_size=None)
    assert got.shape == x.shape

    # Each group should match _rotate applied independently
    for b in range(2):
        for g in range(4):
            slice_in = x[b : b + 1, g * group_size : (g + 1) * group_size]
            expected = pq._rotate(slice_in)[0]
            actual = got[b, g * group_size : (g + 1) * group_size]
            assert torch.allclose(actual, expected, atol=1e-5)
