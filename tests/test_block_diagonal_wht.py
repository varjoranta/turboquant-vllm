# SPDX-License-Identifier: MIT
"""CPU-only tests for the block-diagonal WHT port.

Covers:
- ``_fast_wht_batch_blocked`` against explicit ``block_diag(H, H, ...)`` matmul
- ``PolarQuantTorch`` round-trip and semantic-block isolation at block_size < dim
- ``_derive_rotary_dim`` config-walk: partial_rotary_factor direct + aliases
  (rotary_pct, rotary_emb_fraction), inside rope_parameters, and the non-
  partial-rotary cases that must return None
- ``TurboQuantWrapper.rotary_dim`` is honored on the forward path: block-diag
  wrappers go through PolarQuantTorch dequant (skipping Triton / CUDA kernels
  that hardcode a full-width WHT)
"""
from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from turboquant_vllm.torch_ops import (
    PolarQuantTorch,
    _fast_wht_batch,
    _fast_wht_batch_blocked,
)
from turboquant_vllm.weight_quant import (
    TurboQuantWrapper,
    _derive_rotary_dim,
)


def _explicit_hadamard(n: int) -> torch.Tensor:
    assert n & (n - 1) == 0
    H = torch.tensor([[1.0]])
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return H / math.sqrt(n)


class TestFastWHTBlocked:
    """``_fast_wht_batch_blocked(x, block)`` ≡ ``block_diag(H, H, ...) @ x``."""

    @pytest.mark.parametrize("block_size", [32, 64, 128])
    def test_matches_explicit_block_diag(self, block_size):
        batch, n = 4, 256
        torch.manual_seed(0)
        x = torch.randn(batch, n)

        got = _fast_wht_batch_blocked(x.clone(), block_size)
        H = _explicit_hadamard(block_size)
        num_blocks = n // block_size
        ref = (torch.block_diag(*[H] * num_blocks) @ x.T).T
        torch.testing.assert_close(got, ref, rtol=1e-5, atol=1e-5)

    def test_full_block_equals_full_width_wht(self):
        torch.manual_seed(0)
        x = torch.randn(4, 128)
        blocked = _fast_wht_batch_blocked(x.clone(), 128)
        full = _fast_wht_batch(x.clone())
        torch.testing.assert_close(blocked, full)

    def test_rejects_non_power_of_two_block(self):
        x = torch.randn(4, 192)
        with pytest.raises(AssertionError):
            _fast_wht_batch_blocked(x, block_size=96)  # 96 isn't 2^k

    def test_rejects_non_divisible_dim(self):
        x = torch.randn(4, 100)
        with pytest.raises(AssertionError):
            _fast_wht_batch_blocked(x, block_size=64)  # 100 % 64 != 0


class TestPolarQuantTorchBlockDiagonal:
    """Round-trip + semantic isolation of PolarQuantTorch with rotary_dim."""

    @pytest.mark.parametrize(
        "dim,rotary_dim",
        [
            (128, 64),    # MiniMax M2.7 head_dim=128, factor=0.5
            (128, 32),    # hypothetical factor=0.25 on head_dim=128
            (256, 64),    # Qwen3.6-A3B head_dim=256, factor=0.25
            (256, 128),   # factor=0.5 on head_dim=256
        ],
    )
    def test_rotate_inverse_round_trip(self, dim, rotary_dim):
        pq = PolarQuantTorch(dim=dim, bit_width=3, device="cpu", rotary_dim=rotary_dim)
        x = torch.randn(8, dim)
        x_back = pq._rotate_inverse(pq._rotate(x.clone()))
        torch.testing.assert_close(x_back, x, rtol=1e-4, atol=1e-4)

    def test_block_isolation(self):
        """Signal in block 0 must not leak into block 1 after _rotate."""
        dim, rotary_dim = 128, 64
        pq = PolarQuantTorch(dim=dim, bit_width=3, device="cpu", rotary_dim=rotary_dim)

        x = torch.zeros(4, dim)
        x[:, :rotary_dim] = torch.randn(4, rotary_dim)
        y = pq._rotate(x.clone())
        torch.testing.assert_close(
            y[:, rotary_dim:], torch.zeros(4, dim - rotary_dim),
            rtol=0.0, atol=1e-6,
        )

        x2 = torch.zeros(4, dim)
        x2[:, rotary_dim:] = torch.randn(4, dim - rotary_dim)
        y2 = pq._rotate(x2.clone())
        torch.testing.assert_close(
            y2[:, :rotary_dim], torch.zeros(4, rotary_dim),
            rtol=0.0, atol=1e-6,
        )

    def test_rotary_dim_gte_dim_folds_to_none(self):
        """rotary_dim >= dim should produce full-width WHT output."""
        dim = 128
        torch.manual_seed(0)
        x = torch.randn(4, dim)
        full = PolarQuantTorch(dim=dim, bit_width=3, device="cpu", rotary_dim=None)
        boundary = PolarQuantTorch(dim=dim, bit_width=3, device="cpu", rotary_dim=dim)
        torch.testing.assert_close(full._rotate(x.clone()), boundary._rotate(x.clone()))


class TestDeriveRotaryDim:
    """``_derive_rotary_dim(model_config)`` walks the config and returns a
    power-of-two divisor of head_dim, or None for non-partial-rotary cases."""

    def _make_config(self, head_dim, factor_name=None, factor=None, in_rope_params=False):
        cfg = type("T", (), {})()
        cfg.head_dim = head_dim
        if factor_name and not in_rope_params:
            setattr(cfg, factor_name, factor)
        if factor_name and in_rope_params:
            cfg.rope_parameters = {factor_name: factor}
        mc = type("M", (), {})()
        mc.hf_text_config = cfg
        mc.get_head_size = lambda: head_dim
        return mc

    def test_partial_rotary_factor_direct(self):
        mc = self._make_config(head_dim=128, factor_name="partial_rotary_factor", factor=0.5)
        assert _derive_rotary_dim(mc) == 64

    def test_partial_rotary_factor_025_qwen36(self):
        mc = self._make_config(head_dim=256, factor_name="partial_rotary_factor", factor=0.25)
        assert _derive_rotary_dim(mc) == 64

    def test_rotary_pct_alias(self):
        mc = self._make_config(head_dim=128, factor_name="rotary_pct", factor=0.25)
        assert _derive_rotary_dim(mc) == 32

    def test_rotary_emb_fraction_alias(self):
        mc = self._make_config(head_dim=128, factor_name="rotary_emb_fraction", factor=0.5)
        assert _derive_rotary_dim(mc) == 64

    def test_factor_in_rope_parameters_dict(self):
        mc = self._make_config(
            head_dim=128, factor_name="partial_rotary_factor", factor=0.5,
            in_rope_params=True,
        )
        assert _derive_rotary_dim(mc) == 64

    @pytest.mark.parametrize("factor", [1.0, 1.5, 0.0, -0.1, None])
    def test_full_or_invalid_factor_returns_none(self, factor):
        mc = self._make_config(
            head_dim=128,
            factor_name="partial_rotary_factor" if factor is not None else None,
            factor=factor,
        )
        assert _derive_rotary_dim(mc) is None

    def test_none_model_config_returns_none(self):
        assert _derive_rotary_dim(None) is None

    def test_head_dim_none_returns_none(self):
        cfg = type("T", (), {})()
        cfg.partial_rotary_factor = 0.5
        # head_dim missing entirely
        mc = type("M", (), {})()
        mc.hf_text_config = cfg
        assert _derive_rotary_dim(mc) is None


class TestTurboQuantWrapperBlockDiagonal:
    """End-to-end: a TurboQuantWrapper compressed with rotary_dim produces
    output close to (input @ original_weight.T) within TQ3 noise, and routes
    through the PolarQuant dequant path (not Triton/CUDA kernels)."""

    def test_forward_cpu_matches_original_within_noise(self):
        """Wrap a 256-in / 512-out Linear with rotary_dim=64 (Qwen3.6-A3B
        shape) and confirm output tracks bf16 matmul within TQ3 tolerance."""
        torch.manual_seed(0)
        in_features, out_features = 256, 512
        original = nn.Linear(in_features, out_features, bias=False)
        original.weight.data.mul_(0.02)
        w_orig = original.weight.data.clone()

        wrapper = TurboQuantWrapper(
            original, bits=3, group_size=128, rotary_dim=64,
        )
        assert wrapper.rotary_dim == 64

        x = torch.randn(4, in_features) * 0.05
        out = wrapper(x)
        ref = x @ w_orig.T
        rel_err = (
            torch.linalg.norm(out - ref) / torch.linalg.norm(ref)
        ).item()
        assert rel_err < 0.25, (
            f"TQ3 block-diag forward rel_err {rel_err:.3f} — should be ≤ 0.25"
        )

    def test_rotary_dim_routes_through_polarquant(self, monkeypatch):
        """``self._can_use_full_wht_kernels()`` must return False when
        ``rotary_dim`` is set, so the forward path avoids Triton / CUDA
        kernels that hardcode a full-width WHT."""
        torch.manual_seed(0)
        original = nn.Linear(128, 256, bias=False)
        wrapper = TurboQuantWrapper(
            original, bits=3, group_size=128, rotary_dim=64,
        )
        assert not wrapper._full_wht_ok

        wrapper_full = TurboQuantWrapper(original, bits=3, group_size=128)
        assert wrapper_full._full_wht_ok
