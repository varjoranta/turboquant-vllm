# SPDX-License-Identifier: MIT
"""Phase B step 6: bit-equivalence + microbench for the bs=1 Metal GEMV.

Reference: full-fp32 lookup + matmul, mirroring fwht_on_input_matmul_mlx
sans the FWHT (we pass pre-rotated x as input).

Run: python tests/test_mlx_gemv_bs1.py
"""

from __future__ import annotations

import time

import mlx.core as mx
import numpy as np

from turboquant_vllm.mlx_metal_kernels import tq3_gemv_bs1_mlx
from turboquant_vllm.weight_quant import pack_indices


# Match Qwen3-8B / Qwen3.5-35B-A3B linear shapes.
SHAPES = [
    ("tiny",         128,    32),
    ("q/k/v/o_proj", 4096,   4096),
    ("gate/up_proj", 4096,   12288),
    ("down_proj",    12288,  4096),
]


def build_case(rng: np.random.Generator, K: int, OC: int):
    n_groups = K // 128
    indices_3d = rng.integers(0, 8, size=(OC, n_groups, 128), dtype=np.int64)
    packed_np = pack_indices(
        # weight_quant.pack_indices is torch-tensor based; flatten to (rows, 128)
        __import__("torch").from_numpy(indices_3d.reshape(OC * n_groups, 128)),
        bits=3,
    ).numpy()
    codebook_np = np.sort(rng.standard_normal(8).astype(np.float32))
    norms_np = (rng.standard_normal((OC, n_groups)).astype(np.float32) * 0.1)
    x_np = (rng.standard_normal(K).astype(np.float32) * 0.5)

    # Reference computed in fp16 throughout to match the kernel's storage.
    cb_fp16 = mx.array(codebook_np).astype(mx.float16).astype(mx.float32)
    norms_fp16 = mx.array(norms_np).astype(mx.float16).astype(mx.float32)
    x_fp16 = mx.array(x_np).astype(mx.float16).astype(mx.float32)
    indices_mx = mx.array(indices_3d.astype(np.int32))

    w_g = cb_fp16[indices_mx] * norms_fp16[:, :, None]   # (OC, n_groups, 128) fp32
    out_ref = (w_g.reshape(OC, K) @ x_fp16.reshape(K, 1)).reshape(OC)
    out_ref_np = np.asarray(out_ref).astype(np.float32)

    return packed_np, codebook_np, norms_np, x_np, out_ref_np


def run_case(name: str, K: int, OC: int) -> dict:
    rng = np.random.default_rng(0)
    packed_np, codebook_np, norms_np, x_np, out_ref_np = build_case(rng, K, OC)

    # Move to MLX in float16 (kernel storage type).
    packed = mx.array(packed_np)                                         # uint8
    codebook = mx.array(codebook_np).astype(mx.float16)
    norms = mx.array(norms_np).astype(mx.float16)
    x_rot = mx.array(x_np).astype(mx.float16)

    out = tq3_gemv_bs1_mlx(x_rot, packed, norms, codebook)
    mx.eval(out)
    out_np = np.asarray(out).astype(np.float32)

    abs_err = np.abs(out_np - out_ref_np)
    ref_scale = float(np.abs(out_ref_np).max())
    large = np.abs(out_ref_np) > 0.05 * ref_scale
    max_abs = float(abs_err.max())
    max_rel = float((abs_err[large] / np.abs(out_ref_np[large])).max()) if large.any() else 0.0

    # Microbench: per-iter eval to defeat MLX's lazy DCE.
    for _ in range(50):
        y = tq3_gemv_bs1_mlx(x_rot, packed, norms, codebook)
        mx.eval(y)
    t0 = time.perf_counter()
    iters = 200
    for _ in range(iters):
        y = tq3_gemv_bs1_mlx(x_rot, packed, norms, codebook)
        mx.eval(y)
    us_per = (time.perf_counter() - t0) / iters * 1e6

    return dict(name=name, K=K, OC=OC, max_abs=max_abs, max_rel=max_rel, us=us_per)


def bench_reference(K: int, OC: int) -> float:
    """Time the existing MLX path: full bf16 dequant + matmul."""
    rng = np.random.default_rng(0)
    n_groups = K // 128
    indices_3d = rng.integers(0, 8, size=(OC, n_groups, 128), dtype=np.int64)
    codebook = mx.array(np.sort(rng.standard_normal(8).astype(np.float32))).astype(mx.float16)
    norms = mx.array(rng.standard_normal((OC, n_groups)).astype(np.float32) * 0.1).astype(mx.float16)
    x = mx.array(rng.standard_normal(K).astype(np.float32) * 0.5).astype(mx.float16)
    indices = mx.array(indices_3d.astype(np.int32))

    def step():
        # Same shape as existing path: dequant → matmul.
        w = codebook[indices]                        # (OC, n_groups, 128) fp16
        w = w * norms[:, :, None]                    # apply norm
        w = w.reshape(OC, K)                         # (OC, K)
        return w @ x.reshape(K, 1)                   # (OC, 1)

    for _ in range(10):
        y = step(); mx.eval(y)
    t0 = time.perf_counter()
    iters = 100
    for _ in range(iters):
        y = step(); mx.eval(y)
    return (time.perf_counter() - t0) / iters * 1e6


def main() -> None:
    print(f"Device: {mx.default_device()}")
    print()
    print(f"{'shape':14s} {'K':>5s} {'OC':>5s}  {'max_abs':>8s} {'max_rel':>8s}  "
          f"{'kernel µs':>10s} {'ref µs':>10s} {'speedup':>8s}")
    for name, K, OC in SHAPES:
        r = run_case(name, K, OC)
        ref_us = bench_reference(K, OC)
        spd = ref_us / r["us"] if r["us"] > 0 else 0
        print(f"  {r['name']:14s} {r['K']:5d} {r['OC']:5d}  "
              f"{r['max_abs']:8.4f} {r['max_rel']:8.4f}  "
              f"{r['us']:10.1f} {ref_us:10.1f} {spd:7.2f}x")
        assert r["max_rel"] < 0.05, f"{name}: max_rel={r['max_rel']} > 5%"
    print("\nPASS")


if __name__ == "__main__":
    main()
