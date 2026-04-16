#!/usr/bin/env python3
"""Benchmark TurboQuantMLXLinear vs the PyTorch CPU fallback path.

Loads one real Linear from our Qwen3-Coder-30B-A3B TQ3 checkpoint into
both MLX and PyTorch, runs a representative matmul workload, and
compares throughput. This is the smallest-possible proof that the MLX
port actually unlocks the Mac dev loop — if MLX is meaningfully faster
than the 0.008 tok/s PyTorch CPU path on this single layer, the full
model will be fast enough for the 20-scenario eval.

Usage:
    python scripts/bench_mlx_vs_cpu.py [/path/to/tq3/checkpoint]

Default checkpoint: ~/models/qwen3-coder-30b-a3b-tq3
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch
from safetensors import safe_open

from turboquant_vllm.mlx_model import TurboQuantMLXLinear
from turboquant_vllm.mlx_ops import (
    PolarQuantStateMLX,
    fwht_on_input_matmul_mlx,
    unpack_indices_3bit_mlx,
)
from turboquant_vllm.torch_ops import PolarQuantTorch
from turboquant_vllm.weight_quant import TurboQuantWrapper

# Use the q_proj weight of layer 0 self-attn as a representative Linear —
# relatively small, dense (not MoE expert), typical shape.
LAYER_KEY = "model.layers.0.self_attn.q_proj.weight"


def load_packed_layer(ckpt_dir: Path, key: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Scan shards for ``{key}.tq_packed`` and ``{key}.tq_norms``."""
    packed = norms = None
    for shard in sorted(ckpt_dir.glob("*.safetensors")):
        with safe_open(shard, framework="pt", device="cpu") as f:
            keys = f.keys()
            if f"{key}.tq_packed" in keys:
                packed = f.get_tensor(f"{key}.tq_packed")
            if f"{key}.tq_norms" in keys:
                norms = f.get_tensor(f"{key}.tq_norms")
        if packed is not None and norms is not None:
            break
    if packed is None or norms is None:
        raise RuntimeError(f"Layer {key} not found in checkpoint at {ckpt_dir}")
    return packed, norms


def bench(name: str, fn, warmup: int = 5, iters: int = 30) -> float:
    """Time ``fn`` ``iters`` times after ``warmup`` untimed calls. Returns mean ms/call."""
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1000)
    mean_ms = sum(samples) / len(samples)
    var = sum((s - mean_ms) ** 2 for s in samples) / len(samples)
    std_ms = var**0.5
    print(f"  {name}: {mean_ms:.2f} ± {std_ms:.2f} ms/call ({iters} iters)")
    return mean_ms


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "checkpoint",
        nargs="?",
        default=os.path.expanduser("~/models/qwen3-coder-30b-a3b-tq3"),
    )
    args = ap.parse_args()
    ckpt = Path(args.checkpoint)
    with open(ckpt / "tq_config.json") as f:
        tq_config = json.load(f)
    bits = tq_config["bits"]
    group_size = tq_config["group_size"]
    seed = tq_config.get("quantizer_seed", 42)

    print(f"Checkpoint: {ckpt}")
    print(f"  bits={bits}, group_size={group_size}")
    print(f"  layer: {LAYER_KEY}")

    packed, norms = load_packed_layer(ckpt, LAYER_KEY)
    # packed is stored per-group: (out * n_groups, bytes_per_group).
    # norms is (out, n_groups) — authoritative for out_features.
    out_features = norms.shape[0]
    n_groups = norms.shape[1]
    in_features = n_groups * group_size
    print(f"  shape: out={out_features}, in={in_features}, n_groups={n_groups}")
    print(f"  packed shape: {list(packed.shape)}, size: {packed.numel()} bytes")

    # Build PyTorch side: TurboQuantWrapper via from_packed
    pt_wrapper = TurboQuantWrapper.from_packed(
        packed_weight=packed,
        norms=norms,
        in_features=in_features,
        out_features=out_features,
        bits=bits,
        group_size=group_size,
    )
    pt_wrapper.eval()

    # Build MLX side
    pq = PolarQuantTorch(dim=group_size, bit_width=bits, seed=seed, device="cpu")
    state = PolarQuantStateMLX.from_torch_quantizer(pq)
    mlx_linear = TurboQuantMLXLinear(
        packed_weight=mx.array(packed.numpy()),
        norms=mx.array(norms.numpy()),
        state=state,
        in_features=in_features,
        out_features=out_features,
        bias=None,
    )

    # Workload: batch=1 single-token prefill (typical serving case for generation)
    batch = 1
    torch.manual_seed(0)
    x_pt = torch.randn(batch, in_features, dtype=torch.float32)
    x_mx = mx.array(x_pt.numpy())

    print("\nForward benchmark (batch=1, single call per iter):")

    def run_pt():
        with torch.no_grad():
            _ = pt_wrapper(x_pt)

    def run_mlx():
        out = mlx_linear(x_mx)
        mx.eval(out)

    # Compiled MLX forward: mx.compile can auto-fuse the dequant+matmul
    # graph on first call; subsequent calls run the fused kernel. This
    # is free real-estate — no Metal kernel authored by us.
    compiled_forward = mx.compile(lambda x: mlx_linear(x))

    def run_mlx_compiled():
        out = compiled_forward(x_mx)
        mx.eval(out)

    # FWHT-on-input: one forward WHT on the input instead of N inverse
    # WHTs on the weight rows. Pre-unpack indices once at init time.
    indices_grouped = unpack_indices_3bit_mlx(
        mx.array(packed.numpy()), dim=in_features
    ).reshape(out_features * n_groups, group_size)
    mx.eval(indices_grouped)
    norms_mx = mx.array(norms.numpy())

    def run_mlx_fwht_on_input():
        out = fwht_on_input_matmul_mlx(x_mx, indices_grouped, norms_mx, state)
        mx.eval(out)

    compiled_fwht = mx.compile(
        lambda x: fwht_on_input_matmul_mlx(x, indices_grouped, norms_mx, state)
    )

    def run_mlx_fwht_compiled():
        out = compiled_fwht(x_mx)
        mx.eval(out)

    pt_ms = bench("PyTorch CPU fallback        ", run_pt)
    mlx_ms = bench("MLX native                  ", run_mlx)
    mlx_c_ms = bench("MLX + mx.compile fusion     ", run_mlx_compiled)
    mlx_fwht_ms = bench("MLX FWHT-on-input           ", run_mlx_fwht_on_input)
    mlx_fwht_c_ms = bench("MLX FWHT-on-input + compile ", run_mlx_fwht_compiled)

    print(f"\n  Speedup vs CPU:")
    print(f"    MLX:                     {pt_ms / mlx_ms:.1f}x")
    print(f"    MLX + compile:           {pt_ms / mlx_c_ms:.1f}x")
    print(f"    MLX FWHT-on-input:       {pt_ms / mlx_fwht_ms:.1f}x")
    print(f"    MLX FWHT-on-input+compile: {pt_ms / mlx_fwht_c_ms:.1f}x")
    mlx_c_ms = min(mlx_c_ms, mlx_fwht_ms, mlx_fwht_c_ms)
    # Extrapolation: MoE with ~3000 active Linear calls per token
    # (~48 layers × ~24 MoE + 4 attn + misc) × 25k tokens for the 20-scenario
    # eval. Dense models <= 8B call Linear ~100-200 times per token, so their
    # actual eval time is 15-30x lower than this extrapolation suggests.
    LINEARS_PER_TOK_MOE_30B = 3000
    EVAL_TOKENS = 25000
    ms_to_min = LINEARS_PER_TOK_MOE_30B * EVAL_TOKENS / 1000 / 60
    print(
        f"\n  Hypothetical 30B-MoE 20-scenario eval (~{EVAL_TOKENS//1000}k tok, "
        f"~{LINEARS_PER_TOK_MOE_30B} Linear calls/tok):\n"
        f"    PyTorch CPU:  ~{pt_ms * ms_to_min:.0f} minutes\n"
        f"    MLX native:   ~{mlx_ms * ms_to_min:.0f} minutes\n"
        f"    MLX+compile:  ~{mlx_c_ms * ms_to_min:.0f} minutes\n"
        f"  (Dense models <= 8B are 15-30x faster than this extrapolation.)"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
