#!/usr/bin/env python3
"""End-to-end GPU smoke test for TurboQuant online weight compression.

Used to validate the upstream vLLM PR (Linear-only, `--quantization turboquant`)
on real GPU hardware. Two phases:

1. **Triton kernels direct test** — compresses a synthetic Linear layer,
   compares both kernel paths (fused dequant-GEMM for small dims, FWHT-on-input
   for large dims) against a PyTorch dequant reference via cosine similarity.
   Also exercises M=0 (chunked prefill), 3D input, and bias.

2. **vLLM end-to-end** — loads Qwen/Qwen2.5-0.5B with
   ``--quantization turboquant``, generates 32 tokens, checks for coherent
   output.

Runtime: ~2 minutes on a mid-range GPU (RTX 6000 Ada, A100, H100, etc.).
Disk: ~1 GB (Qwen2.5-0.5B bf16).

Usage:
    # Requires vLLM built from the upstream PR branch:
    #   pip install git+https://github.com/varjoranta/vllm-1.git@feat/turboquant-online-weight-quant
    # Or a main-line vLLM after the PR merges.
    python tests/gpu/test_turboquant_online_gpu.py

Exit 0 = all pass. Exit 1 = any failure (details on stderr).
"""

import sys
import time

import torch


def test_triton_kernels() -> bool:
    """Phase 1: validate both Triton kernels against a PyTorch reference."""
    from vllm.model_executor.layers.quantization.online.turboquant import (
        TurboQuantOnlineLinearMethod,
        _get_quantizer,
        _unpack_indices,
        tq_fused_gemm,
    )

    if tq_fused_gemm is None:
        print("  SKIP: Triton unavailable")
        return True

    method = TurboQuantOnlineLinearMethod(bits=3, group_size=128)
    ok = True
    layer = None
    out_dim = 0
    x = None

    # Dispatch between the two kernel paths is driven by out_dim (crossover at
    # 4096 per process_weights_after_loading); exercise both.
    for out_dim, label in [(64, "small/fused"), (4096, "large/fwht")]:
        layer = torch.nn.Module()
        layer.weight = torch.nn.Parameter(
            torch.randn(out_dim, 128, device="cuda", dtype=torch.bfloat16)
        )
        method.process_weights_after_loading(layer)

        # Reference: PyTorch-side unpack + dequantize + matmul
        indices = _unpack_indices(layer.tq_packed_weight, 3, 128)
        q = _get_quantizer(128, 3, "cuda")
        w_deq = q.dequantize(indices, layer.tq_norms.reshape(-1))
        w_deq = w_deq.reshape(out_dim, -1).to(torch.bfloat16)

        x = torch.randn(4, 128, device="cuda", dtype=torch.bfloat16)
        ref = x @ w_deq.t()
        out = method.apply(layer, x, bias=None)

        cos = torch.nn.functional.cosine_similarity(
            ref.float(), out.float(), dim=1
        )
        min_cos = cos.min().item()
        status = "OK" if min_cos >= 0.90 else "FAIL"
        print(f"  {label}: cosine_sim={min_cos:.4f} {status}")
        if min_cos < 0.90:
            ok = False

    assert layer is not None and x is not None  # noqa: S101

    # Chunked prefill can emit M=0 batches; must not crash.
    x_zero = torch.randn(0, 128, device="cuda", dtype=torch.bfloat16)
    out_zero = method.apply(layer, x_zero, bias=None)
    assert out_zero.shape[0] == 0, f"M=0 shape mismatch: {out_zero.shape}"
    print("  M=0 early exit: OK")

    # 3D input (batch, seq, features)
    x_3d = torch.randn(2, 8, 128, device="cuda", dtype=torch.bfloat16)
    out_3d = method.apply(layer, x_3d, bias=None)
    assert out_3d.shape == (2, 8, out_dim), f"3D shape mismatch: {out_3d.shape}"
    print("  3D input: OK")

    # Bias path
    bias = torch.randn(out_dim, device="cuda", dtype=torch.bfloat16)
    out_bias = method.apply(layer, x, bias=bias)
    assert out_bias.shape == (4, out_dim)
    print("  Bias: OK")

    return ok


def test_vllm_generate() -> bool:
    """Phase 2: full ``--quantization turboquant`` path through vLLM.

    Loads a tiny model, compresses it, generates a short completion. Success
    criterion is coherent output (non-empty, non-garbage); specific text is
    not checked because a 0.5B model is unpredictable.
    """
    from vllm import LLM, SamplingParams

    print("  Loading Qwen/Qwen2.5-0.5B with quantization=turboquant ...")
    t0 = time.time()
    llm = LLM(
        model="Qwen/Qwen2.5-0.5B",
        quantization="turboquant",
        enforce_eager=True,
        gpu_memory_utilization=0.5,
        max_model_len=256,
    )
    print(f"  Loaded in {time.time() - t0:.1f}s")

    params = SamplingParams(max_tokens=32, temperature=0)
    outputs = llm.generate(["The capital of France is"], params)
    text = outputs[0].outputs[0].text
    print(f"  Output: {text[:100]!r}")

    if len(text.strip()) < 5:
        print("  FAIL: output too short")
        return False

    print("  Generate: OK")
    return True


def main() -> int:
    if not torch.cuda.is_available():
        print("SKIP: no CUDA")
        return 0

    props = torch.cuda.get_device_properties(0)
    total = getattr(props, "total_memory", None) or getattr(props, "total_mem", 0)
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"VRAM: {total / 1e9:.1f} GB")

    ok = True

    print("\n1. Triton kernels:")
    if not test_triton_kernels():
        ok = False

    print("\n2. vLLM generate:")
    if not test_vllm_generate():
        ok = False

    print("\n" + ("ALL PASSED" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
