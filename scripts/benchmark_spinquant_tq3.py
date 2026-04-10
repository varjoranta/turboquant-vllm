"""Benchmark: SpinQuant learned rotation + TQ3 compression.

Tests whether learned rotations unlock viable TQ3 (8 centroids) on weights.
Compares: TQ4 fixed WHT vs TQ3 fixed WHT vs TQ3 learned rotation.

Usage: python3 -u scripts/benchmark_spinquant_tq3.py [--model MODEL]
"""

import argparse
import gc
import os
import re
import time
import torch

os.environ["PYTHONUNBUFFERED"] = "1"

PROMPTS = [
    ("Capital", "What is the capital of Finland? One word."),
    ("Math", "What is 17 * 23? Just the number."),
    ("Code", "Write a Python function is_prime(n) that returns True if n is prime."),
    ("Reasoning", "A bat and ball cost $1.10. Bat costs $1 more than ball. Ball cost?"),
]


def mem():
    gc.collect()
    torch.cuda.empty_cache()
    return torch.cuda.memory_allocated() / 1e6


def gen(model, tok, prompt, max_tok=80):
    inp = tok(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=max_tok, temperature=0.0, do_sample=False)
    text = tok.decode(out[0][inp.input_ids.shape[1] :], skip_special_tokens=True)
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def quality(model, tok, label):
    print(f"\n  Quality ({label}):", flush=True)
    for name, prompt in PROMPTS:
        a = gen(model, tok, prompt)
        print(f"    {name}: {a[:100]}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-30B-A3B")
    parser.add_argument("--rotation-steps", type=int, default=200)
    args = parser.parse_args()

    import logging

    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("=" * 60, flush=True)
    print(f"SpinQuant + TQ3 Benchmark: {args.model}", flush=True)
    print("=" * 60, flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)

    # ── Test 1: TQ4 fixed WHT (baseline) ─────────────────────────────
    print("\n── Test 1: TQ4 fixed WHT (baseline) ──", flush=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda")
    m0 = mem()
    print(f"BF16: {m0:.0f} MB", flush=True)
    quality(model, tok, "BF16 baseline")

    from turboquant_vllm.weight_quant import _replace_linear_layers

    t0 = time.time()
    _replace_linear_layers(model, bits=4, group_size=128)
    t1 = time.time()
    m1 = mem()
    print(f"TQ4 fixed WHT: {m1:.0f} MB ({m0 / m1:.1f}x), {t1 - t0:.0f}s", flush=True)
    quality(model, tok, "TQ4 fixed WHT")

    del model
    gc.collect()
    torch.cuda.empty_cache()

    # ── Test 2: TQ3 fixed WHT (expected: quality loss) ───────────────
    print("\n── Test 2: TQ3 fixed WHT ──", flush=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda")
    t0 = time.time()
    _replace_linear_layers(model, bits=3, group_size=128)
    t1 = time.time()
    m2 = mem()
    print(f"TQ3 fixed WHT: {m2:.0f} MB ({m0 / m2:.1f}x), {t1 - t0:.0f}s", flush=True)
    quality(model, tok, "TQ3 fixed WHT")

    del model
    gc.collect()
    torch.cuda.empty_cache()

    # ── Test 3: TQ3 with learned rotation (the wow test) ─────────────
    print("\n── Test 3: TQ3 with learned rotation ──", flush=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda")

    print("Optimizing rotations (this takes a few minutes)...", flush=True)
    from turboquant_vllm.learned_rotation import optimize_all_rotations

    t0 = time.time()
    rotations = optimize_all_rotations(model, bits=3, group_size=128, steps=args.rotation_steps)
    t_rot = time.time() - t0
    print(f"Rotation optimization: {t_rot:.0f}s for {len(rotations)} layers", flush=True)

    t0 = time.time()
    _replace_linear_layers(model, bits=3, group_size=128, learned_rotations=rotations)
    t_comp = time.time() - t0
    m3 = mem()
    print(
        f"TQ3 learned rotation: {m3:.0f} MB ({m0 / m3:.1f}x), rotation={t_rot:.0f}s + compress={t_comp:.0f}s",
        flush=True,
    )
    quality(model, tok, "TQ3 learned rotation")

    del model
    gc.collect()
    torch.cuda.empty_cache()

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n{'=' * 60}", flush=True)
    print("SUMMARY", flush=True)
    print(f"{'=' * 60}", flush=True)
    print(f"BF16:               {m0:.0f} MB", flush=True)
    print(f"TQ4 fixed WHT:      {m1:.0f} MB ({m0 / m1:.1f}x)", flush=True)
    print(f"TQ3 fixed WHT:      {m2:.0f} MB ({m0 / m2:.1f}x)", flush=True)
    print(f"TQ3 learned rotation: {m3:.0f} MB ({m0 / m3:.1f}x)", flush=True)


if __name__ == "__main__":
    main()
