# SPDX-License-Identifier: MIT
"""Phase A step 1: prove mx.fast.metal_kernel works for our pattern.

Identity kernel + a slightly less trivial 'multiply by codebook[idx]' kernel
that exercises the LUT-in-threadgroup pattern we'll need for the real GEMV.

Run: python -m turboquant_vllm.smoke_metal_kernel
"""

from __future__ import annotations

import sys

import mlx.core as mx


def smoke_identity() -> None:
    src = """
        uint elem = thread_position_in_grid.x;
        if (elem < n[0]) {
            out[elem] = inp[elem];
        }
    """
    kernel = mx.fast.metal_kernel(
        name="identity_bf16",
        input_names=["inp", "n"],
        output_names=["out"],
        source=src,
    )
    a = mx.random.normal(shape=(1024,)).astype(mx.bfloat16)
    n = mx.array([a.size], dtype=mx.uint32)
    out = kernel(
        inputs=[a, n],
        grid=(a.size, 1, 1),
        threadgroup=(64, 1, 1),
        output_shapes=[a.shape],
        output_dtypes=[a.dtype],
    )[0]
    mx.eval(out)
    assert mx.allclose(out, a, rtol=0, atol=0).item(), "identity mismatch"
    print(f"  identity bf16 PASS  ({a.size} elements)")


def smoke_lut_lookup() -> None:
    """LUT lookup: out[i] = codebook[idx[i]] * scale.
    Exercises the codebook-from-threadgroup-mem pattern we'll need.
    """
    header = """
    """
    src = """
        // Stage codebook into threadgroup memory once per threadgroup.
        threadgroup half cb_lut[8];
        if (thread_position_in_threadgroup.x < 8u) {
            cb_lut[thread_position_in_threadgroup.x] = codebook[thread_position_in_threadgroup.x];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint elem = thread_position_in_grid.x;
        if (elem < n[0]) {
            uint i = (uint)idx[elem];
            half cb = cb_lut[i];
            float s = float(scale[0]);
            out[elem] = (half)(float(cb) * s);
        }
    """
    kernel = mx.fast.metal_kernel(
        name="lut_lookup_bf16",
        input_names=["idx", "codebook", "scale", "n"],
        output_names=["out"],
        source=src,
        header=header,
    )

    rng_idx = mx.random.randint(0, 8, shape=(256,)).astype(mx.uint8)
    cb = mx.array([-1.5, -1.0, -0.5, -0.1, 0.1, 0.5, 1.0, 1.5], dtype=mx.float16)
    scale = mx.array([0.25], dtype=mx.float16)
    n = mx.array([rng_idx.size], dtype=mx.uint32)
    out = kernel(
        inputs=[rng_idx, cb, scale, n],
        grid=(rng_idx.size, 1, 1),
        threadgroup=(64, 1, 1),
        output_shapes=[(rng_idx.size,)],
        output_dtypes=[mx.float16],
    )[0]
    mx.eval(out)

    # Reference: gather + scale.
    ref = mx.take(cb, rng_idx.astype(mx.int32)) * scale[0]
    mx.eval(ref)
    if not mx.allclose(out, ref, rtol=1e-3, atol=1e-3).item():
        diff = mx.abs(out - ref).max().item()
        print(f"  LUT max diff: {diff}")
        sys.exit("LUT mismatch")
    print(f"  LUT lookup fp16 PASS  ({rng_idx.size} elements)")


def main() -> None:
    smoke_identity()
    smoke_lut_lookup()
    print("Phase A step 1: mx.fast.metal_kernel works for our pattern.")


if __name__ == "__main__":
    main()
