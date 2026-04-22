# SPDX-License-Identifier: MIT
"""GPU correctness test for the fused FWHT-in-SMEM bs=1 GEMV kernel.

Compares the new fused kernel against:
  (a) the existing two-kernel pipeline: Triton rotate_input + tq3_gemv_bs1
  (b) a pure-PyTorch reference

Same max_rel tolerance as test_tq3_gemv_bs1.py (5 %). Timing is a sanity
probe, not a benchmark — the real A/B lives in scripts/gpu/cuda-bs1-ab/.

Run standalone (CUDA required):
    python tests/gpu/test_tq3_gemv_bs1_fwht_smem.py
"""

from __future__ import annotations

import math
import time

import numpy as np
import torch

from turboquant_vllm.weight_quant import _get_cuda_module, pack_indices
from turboquant_vllm.triton_ops import rotate_input


QWEN3_8B_SHAPES = [
    ("q/k/v/o_proj", 4096, 4096),
    ("gate/up_proj", 4096, 12288),
    ("down_proj", 12288, 4096),
]


def _to_bf16_np(arr: np.ndarray) -> np.ndarray:
    return torch.from_numpy(arr).to(torch.bfloat16).float().numpy()


def build_reference(rng, K: int, OC: int):
    """Return packed + norms + codebook + signs + x + expected y = W @ x.

    W here is the *original-space* weight; the kernel must recover it from
    (indices, signs1, signs2, codebook) and dot with original-space x.
    """
    n_groups = K // 128
    indices_3d = rng.integers(0, 8, size=(OC, n_groups, 128), dtype=np.int64)
    packed_2d = pack_indices(
        torch.from_numpy(indices_3d.reshape(OC * n_groups, 128)), bits=3
    ).numpy()

    codebook_np = np.sort(rng.standard_normal(8).astype(np.float32))
    norms_np = rng.standard_normal((OC, n_groups)).astype(np.float32) * 0.1
    signs1_np = rng.choice([-1.0, 1.0], size=128).astype(np.float32)
    signs2_np = rng.choice([-1.0, 1.0], size=128).astype(np.float32)

    # Original-space input.
    x_np = rng.standard_normal(K).astype(np.float32) * 0.5

    cb_bf = _to_bf16_np(codebook_np)
    norms_bf = _to_bf16_np(norms_np)
    signs1_bf = _to_bf16_np(signs1_np)
    signs2_bf = _to_bf16_np(signs2_np)
    x_bf = _to_bf16_np(x_np)

    # Stored weights in *rotated* space: W_rot[oc, g, :] = cb[idx] * norm
    W_rot = cb_bf[indices_3d] * norms_bf[:, :, None]  # (OC, n_groups, 128)

    # Reference: recover original-space W, then y = W @ x. We do the inverse
    # rotation per group (matches PolarQuantTorch._rotate_inverse):
    #   w_orig = signs1 * H(signs2 * w_rot) / sqrt(128)
    def _wht(v: np.ndarray) -> np.ndarray:
        # In-place butterfly on the last axis (length 128).
        out = v.copy()
        h = 1
        n = out.shape[-1]
        while h < n:
            out_view = out.reshape(*out.shape[:-1], n // (h * 2), 2, h)
            a = out_view[..., 0, :].copy()
            b = out_view[..., 1, :].copy()
            out_view[..., 0, :] = a + b
            out_view[..., 1, :] = a - b
            h *= 2
        return out / math.sqrt(n)

    w_s2 = W_rot * signs2_bf.reshape(1, 1, 128)      # (OC, n_groups, 128)
    w_hadamard = _wht(w_s2)
    W_orig = w_hadamard * signs1_bf.reshape(1, 1, 128)
    W_orig = W_orig.reshape(OC, K)

    y_ref = W_orig @ x_bf  # (OC,)
    return packed_2d, codebook_np, norms_np, signs1_np, signs2_np, x_np, y_ref


def run_case(name: str, K: int, OC: int, seed: int = 0):
    mod = _get_cuda_module()
    assert mod is not None and hasattr(mod, "tq3_gemv_bs1_fwht_smem"), (
        "tq3_gemv_bs1_fwht_smem not built into the CUDA module"
    )

    rng = np.random.default_rng(seed)
    packed_np, codebook_np, norms_np, signs1_np, signs2_np, x_np, y_ref = (
        build_reference(rng, K, OC)
    )

    packed = torch.from_numpy(packed_np).cuda().contiguous()
    codebook = torch.from_numpy(codebook_np).to(torch.bfloat16).cuda().contiguous()
    norms = torch.from_numpy(norms_np).to(torch.bfloat16).cuda().contiguous()
    signs1 = torch.from_numpy(signs1_np).to(torch.bfloat16).cuda().contiguous()
    signs2 = torch.from_numpy(signs2_np).to(torch.bfloat16).cuda().contiguous()
    x = torch.from_numpy(x_np).to(torch.bfloat16).cuda().contiguous()

    # Fused kernel.
    out = mod.tq3_gemv_bs1_fwht_smem(x, packed, norms, codebook, signs1, signs2)
    out_np = out.float().cpu().numpy()
    abs_err = np.abs(out_np - y_ref)
    ref_scale = float(np.abs(y_ref).max())
    mask = np.abs(y_ref) > 0.05 * ref_scale
    max_rel_vs_ref = (
        float((abs_err[mask] / np.abs(y_ref[mask])).max()) if mask.any() else 0.0
    )

    # Two-kernel pipeline for comparison (same inputs, different path).
    # rotate_input expects fp32 x + fp32 signs; returns rotated in fp32; then bf16.
    x_rot = rotate_input(
        x.float().unsqueeze(0), signs1.float(), signs2.float(), 128,
    ).to(torch.bfloat16)
    out_pipeline = mod.tq3_gemv_bs1(x_rot.view(-1), packed, norms, codebook)
    out_pipeline_np = out_pipeline.float().cpu().numpy()
    diff_vs_pipeline = np.abs(out_np - out_pipeline_np)
    max_rel_vs_pipeline = (
        float(diff_vs_pipeline[mask].max() / np.abs(out_pipeline_np[mask]).max())
        if mask.any() else 0.0
    )

    # Light timing (not a benchmark).
    for _ in range(20):
        mod.tq3_gemv_bs1_fwht_smem(x, packed, norms, codebook, signs1, signs2)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    iters = 100
    for _ in range(iters):
        mod.tq3_gemv_bs1_fwht_smem(x, packed, norms, codebook, signs1, signs2)
    torch.cuda.synchronize()
    us = (time.perf_counter() - t0) / iters * 1e6

    return {
        "name": name, "K": K, "OC": OC,
        "max_rel_vs_ref": max_rel_vs_ref,
        "max_rel_vs_pipeline": max_rel_vs_pipeline,
        "us": us,
    }


def main():
    if not torch.cuda.is_available():
        print("SKIP: CUDA not available")
        return
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"  {'shape':20s} {'max_rel ref':>12s} {'max_rel pipe':>13s} {'us':>8s}")
    for name, K, OC in [("tiny", 128, 32), *QWEN3_8B_SHAPES]:
        r = run_case(name, K, OC)
        print(
            f"  {r['name']:20s} "
            f"{r['max_rel_vs_ref']:12.4e} "
            f"{r['max_rel_vs_pipeline']:13.4e} "
            f"{r['us']:8.2f}"
        )
        assert r["max_rel_vs_ref"] < 5e-2, (
            f"{name}: max_rel_vs_ref={r['max_rel_vs_ref']} > 5%"
        )
        assert r["max_rel_vs_pipeline"] < 5e-2, (
            f"{name}: max_rel_vs_pipeline={r['max_rel_vs_pipeline']} > 5%"
        )
    print("\nPASS")


if __name__ == "__main__":
    main()
