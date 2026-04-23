# SPDX-License-Identifier: MIT
"""GPU correctness + latency test for the bs=1 GEMV kernel.

Requires CUDA (sm_80+). Compares turbo_quant_cuda.tq3_gemv_bs1 against a
bf16-rounded fp32 reference that models the kernel's load-and-promote math.

Run standalone (CUDA required):
    python tests/gpu/test_tq3_gemv_bs1.py
"""

from __future__ import annotations

import time

import numpy as np

from turboquant_vllm.weight_quant import _get_cuda_module, pack_indices


QWEN3_8B_SHAPES = [
    ("q/k/v/o_proj", 4096, 4096),
    ("gate/up_proj", 4096, 12288),
    ("down_proj", 12288, 4096),
]


def _to_bf16(arr: np.ndarray) -> np.ndarray:
    import torch

    return torch.from_numpy(arr).to(torch.bfloat16).float().numpy()


def build_reference(rng, K: int, OC: int):
    import torch

    n_groups = K // 128
    indices_3d = rng.integers(0, 8, size=(OC, n_groups, 128), dtype=np.int64)
    packed_2d = pack_indices(torch.from_numpy(indices_3d.reshape(OC * n_groups, 128)), bits=3).numpy()

    codebook_np = np.sort(rng.standard_normal(8).astype(np.float32))
    norms_np = rng.standard_normal((OC, n_groups)).astype(np.float32) * 0.1
    x_np = rng.standard_normal(K).astype(np.float32) * 0.5

    cb_bf = _to_bf16(codebook_np)
    norms_bf = _to_bf16(norms_np)
    x_bf = _to_bf16(x_np)

    W_ref = cb_bf[indices_3d] * norms_bf[:, :, None]
    out_ref = W_ref.reshape(OC, K) @ x_bf
    return packed_2d, codebook_np, norms_np, x_np, out_ref


def run_case(gemv, torch, name: str, K: int, OC: int, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    packed_np, codebook_np, norms_np, x_np, out_ref = build_reference(rng, K, OC)

    packed = torch.from_numpy(packed_np).cuda().contiguous()
    codebook = torch.from_numpy(codebook_np).to(torch.bfloat16).cuda().contiguous()
    norms = torch.from_numpy(norms_np).to(torch.bfloat16).cuda().contiguous()
    x_rot = torch.from_numpy(x_np).to(torch.bfloat16).cuda().contiguous()

    out = gemv(x_rot, packed, norms, codebook)
    out_np = out.float().cpu().numpy()
    abs_err = np.abs(out_np - out_ref)
    ref_scale = float(np.abs(out_ref).max())
    large = np.abs(out_ref) > 0.05 * ref_scale
    max_abs = float(abs_err.max())
    max_rel = float((abs_err[large] / np.abs(out_ref[large])).max()) if large.any() else 0.0

    for _ in range(50):
        gemv(x_rot, packed, norms, codebook)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    iters = 200
    for _ in range(iters):
        gemv(x_rot, packed, norms, codebook)
    torch.cuda.synchronize()
    us = (time.perf_counter() - t0) / iters * 1e6

    return {"name": name, "K": K, "OC": OC, "max_abs": max_abs, "max_rel": max_rel, "us": us}


def main() -> None:
    import torch

    if not torch.cuda.is_available():
        print("SKIP: CUDA not available")
        return

    mod = _get_cuda_module()
    if mod is None or not hasattr(mod, "tq3_gemv_bs1"):
        print("SKIP: tq3_gemv_bs1 not built into the CUDA module")
        return

    print(f"Device: {torch.cuda.get_device_name(0)}")
    for name, K, OC in [("tiny", 128, 32), *QWEN3_8B_SHAPES]:
        r = run_case(mod.tq3_gemv_bs1, torch, name, K, OC)
        print(
            f"  {r['name']:20s} K={r['K']:5d} OC={r['OC']:5d}  "
            f"max_abs={r['max_abs']:.4f} max_rel={r['max_rel']:.4f}  "
            f"{r['us']:.2f} µs"
        )
        assert r["max_rel"] < 5e-2, f"{name}: max_rel={r['max_rel']} > 5% on non-cancellation entries"
    print("\nPASS")


if __name__ == "__main__":
    main()
