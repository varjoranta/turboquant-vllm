#!/usr/bin/env python3
"""Phase B end-to-end microbench: TurboQuantMLXSwitchLinear with/without
the new batched bs=1 Metal GEMV kernel.

Constructs a synthetic SwitchLinear at Qwen3.5-35B-A3B-relevant shape
(256 experts, top-k=8, hidden=4096, intermediate=2560) with random
TQ3 weights, then times one decode-step forward in fp16.

The fast path triggers automatically when fp16 input + bs=1 detected.
We toggle it off by casting to fp32 to fall through to the existing
einsum path for an A/B comparison.
"""

from __future__ import annotations

import time

import mlx.core as mx
import numpy as np
import torch

from turboquant_vllm.mlx_model import TurboQuantMLXSwitchLinear
from turboquant_vllm.mlx_ops import PolarQuantStateMLX
from turboquant_vllm.weight_quant import pack_indices, padded_size


def build_switch_linear(in_f: int, out_f: int, n_experts: int, group_size: int = 128):
    rng = np.random.default_rng(0)
    padded_in, n_groups = padded_size(in_f, group_size)

    indices = rng.integers(0, 8, size=(n_experts * out_f, n_groups, group_size), dtype=np.int64)
    packed_torch = pack_indices(
        torch.from_numpy(indices.reshape(n_experts * out_f, n_groups * group_size)),
        bits=3,
    )
    # Layer expects (n_experts*out_f*n_groups, 48) — reshape flat packed.
    packed_flat = packed_torch.numpy().reshape(n_experts * out_f * n_groups, 48)
    packed = mx.array(packed_flat)

    norms = mx.array(rng.standard_normal((n_experts * out_f, n_groups)).astype(np.float32) * 0.1)

    centroids = mx.array(np.sort(rng.standard_normal(8).astype(np.float32)))
    signs1 = mx.array(rng.choice([-1.0, 1.0], size=group_size).astype(np.float32))
    signs2 = mx.array(rng.choice([-1.0, 1.0], size=group_size).astype(np.float32))
    state = PolarQuantStateMLX(
        signs1=signs1,
        signs2=signs2,
        centroids=centroids,
        dim=group_size,
    )

    return TurboQuantMLXSwitchLinear(
        packed_weight=packed,
        norms=norms,
        state=state,
        in_features=in_f,
        out_features=out_f,
        num_experts=n_experts,
        bias=None,
    )


def time_forward(layer, x, indices, iters: int = 50, warmup: int = 10) -> float:
    for _ in range(warmup):
        y = layer(x, indices)
        mx.eval(y)
    t0 = time.perf_counter()
    for _ in range(iters):
        y = layer(x, indices)
        mx.eval(y)
    return (time.perf_counter() - t0) / iters * 1e3  # ms


def main() -> None:
    # Qwen3.5-35B-A3B-ish: 256 experts, top-8, hidden=4096, intermediate=2560.
    HIDDEN = 4096
    INT_DIM = 2560
    N_EXPERTS = 256
    TOPK = 8

    print(f"Device: {mx.default_device()}")
    print(f"Shapes: hidden={HIDDEN} int={INT_DIM} n_experts={N_EXPERTS} top_k={TOPK}")
    print()

    rng = np.random.default_rng(42)
    # SwitchGLU calling convention: x has shape (*leading, 1, 1, in)
    # For bs=1 decode that's (1, 1, 1, hidden).
    x_fp16 = mx.random.normal(shape=(1, 1, 1, HIDDEN)).astype(mx.float16) * 0.1
    indices = mx.array(rng.integers(0, N_EXPERTS, size=(1, TOPK)).astype(np.int32))

    # Same input, but fp32 to force the fallback (kernel only fires for fp16).
    x_fp32 = x_fp16.astype(mx.float32)

    print(f"{'op':22s} {'fast µs':>10s} {'slow µs':>10s} {'speedup':>8s}")

    for name, in_f, out_f in [("gate/up_proj", HIDDEN, INT_DIM), ("down_proj", INT_DIM, HIDDEN)]:
        layer = build_switch_linear(in_f, out_f, N_EXPERTS)
        if in_f == HIDDEN:
            x_fast, x_slow = x_fp16, x_fp32
        else:
            x_fast = mx.random.normal(shape=(1, TOPK, 1, in_f)).astype(mx.float16) * 0.1
            x_slow = x_fast.astype(mx.float32)
        fast_ms = time_forward(layer, x_fast, indices)
        slow_ms = time_forward(layer, x_slow, indices)
        spd = slow_ms / fast_ms if fast_ms > 0 else 0.0
        print(f"  {name:22s} {fast_ms * 1000:10.1f} {slow_ms * 1000:10.1f} {spd:7.2f}x")

    # Aggregate per-token projection: gate + up + down per layer × num_layers.
    # Qwen3.5-35B-A3B has 28 layers (architecture-typical).
    NUM_LAYERS = 28
    layer_gateup = build_switch_linear(HIDDEN, INT_DIM, N_EXPERTS)
    layer_down = build_switch_linear(INT_DIM, HIDDEN, N_EXPERTS)
    x_gateup = mx.random.normal(shape=(1, 1, 1, HIDDEN)).astype(mx.float16) * 0.1
    x_down = mx.random.normal(shape=(1, TOPK, 1, INT_DIM)).astype(mx.float16) * 0.1

    gateup_ms = time_forward(layer_gateup, x_gateup, indices)
    down_ms = time_forward(layer_down, x_down, indices)
    per_token_ms = NUM_LAYERS * (2 * gateup_ms + down_ms)  # gate + up + down per layer
    print()
    print(f"Per-MoE-layer: {2 * gateup_ms + down_ms:.1f} ms (kernel path, gate+up+down)")
    print(f"Projected per-token MoE-only: {per_token_ms:.0f} ms = {1000 / per_token_ms:.2f} tok/s")
    print("(attention + norms + sampling additional, not measured here)")


if __name__ == "__main__":
    main()
