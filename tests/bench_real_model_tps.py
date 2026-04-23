#!/usr/bin/env python3
"""End-to-end tok/s on a real TQ3 MoE model: Qwen3-Coder-30B-A3B.

Loads the local TQ3 checkpoint, generates N tokens, reports decode tok/s.
Compares the new Metal GEMV kernel (fp16 fast path) against the existing
fp32 einsum fallback by toggling input dtype.

Usage: python tests/bench_real_model_tps.py [model_path]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import mlx.core as mx

from turboquant_vllm.mlx_loader import load_tq3


def time_generate(model, tokenizer, prompt: str, max_tokens: int, dtype: mx.Dtype) -> dict:
    """Run prefill + decode, return wall, prompt tokens, completion tokens, decode tok/s."""
    from mlx_lm.generate import generate_step

    enc = tokenizer.encode(prompt) if not hasattr(tokenizer, "_tokenizer") else tokenizer.encode(prompt)
    prompt_ids = mx.array(enc)
    prompt_len = prompt_ids.size

    # Warm: build kernels + caches.
    warm_iter = generate_step(prompt_ids, model, max_tokens=2)
    for _ in range(2):
        next(warm_iter)
    mx.eval(mx.zeros(1))

    t_total = time.perf_counter()
    it = generate_step(prompt_ids, model, max_tokens=max_tokens)
    n = 0
    t_first = None
    for tok, _logp in it:
        if t_first is None:
            t_first = time.perf_counter()
        n += 1
        mx.eval(tok)
        if n >= max_tokens:
            break
    t_end = time.perf_counter()

    decode_tps = (n - 1) / (t_end - t_first) if n > 1 and t_first else 0.0
    return {
        "prompt_len": prompt_len,
        "completion_tokens": n,
        "ttft_s": (t_first - t_total) if t_first else 0.0,
        "wall_s": t_end - t_total,
        "decode_tok_per_s": decode_tps,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", nargs="?", default=str(Path.home() / "models/qwen3-coder-30b-a3b-tq3"))
    parser.add_argument("--max_tokens", type=int, default=64)
    parser.add_argument("--prompt", default="Write a short Python function that returns the n-th Fibonacci number.")
    args = parser.parse_args()

    print(f"Device:  {mx.default_device()}")
    print(f"Model:   {args.model}")
    print(f"Prompt:  {args.prompt!r}")
    print(f"Max:     {args.max_tokens} tokens")
    print()

    print("Loading TQ3 model + tokenizer...")
    t0 = time.perf_counter()
    model, tokenizer = load_tq3(args.model)
    print(f"  loaded in {time.perf_counter() - t0:.1f}s")
    print()

    # Run twice: once with fp16 (fast path fires), once with fp32 (fallback).
    # mlx_lm.generate_step doesn't expose a dtype knob, so we monkeypatch
    # the model's input embedding to cast its output.
    embed = model.model.embed_tokens
    orig_call = embed.__class__.__call__

    for label, dtype in [("fp16 (kernel ON)", mx.float16), ("fp32 (fallback)", mx.float32)]:

        def patched(self, x, _dtype=dtype):
            return orig_call(self, x).astype(_dtype)

        embed.__class__.__call__ = patched

        print(f"=== {label} ===")
        try:
            r = time_generate(model, tokenizer, args.prompt, args.max_tokens, dtype)
            print(f"  prompt_len:        {r['prompt_len']}")
            print(f"  completion_tokens: {r['completion_tokens']}")
            print(f"  ttft:              {r['ttft_s']:.3f} s")
            print(f"  wall:              {r['wall_s']:.3f} s")
            print(f"  decode tok/s:      {r['decode_tok_per_s']:.2f}")
        except Exception as e:
            print(f"  FAIL: {e}")
        print()

    embed.__class__.__call__ = orig_call


if __name__ == "__main__":
    main()
