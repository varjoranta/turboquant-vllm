"""Benchmark: REAP pruning + TQ4 compression on MoE models.

Tests the full pipeline: REAP saliency → prune → TQ compress → quality check.
Measures memory at each stage and validates output quality.

Usage: python3 -u scripts/benchmark_reap_tq.py [--model MODEL] [--prune FRACTION]
"""
import argparse
import gc
import json
import os
import re
import time
import torch

os.environ["PYTHONUNBUFFERED"] = "1"

QUALITY_PROMPTS = [
    ("Capital", "What is the capital of Finland? One word."),
    ("Math", "What is 17 * 23? Just the number."),
    ("Code", "Write a Python function is_prime(n) that returns True if n is prime."),
    ("Reasoning", "A bat and ball cost $1.10 together. The bat costs $1 more than the ball. How much does the ball cost? Show reasoning."),
    ("Product", "Write a one-sentence product description for a SaaS analytics platform."),
]


def measure_memory():
    gc.collect()
    torch.cuda.empty_cache()
    return torch.cuda.memory_allocated() / 1e6


def generate_answer(model, tokenizer, prompt, max_tokens=100):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_tokens,
            temperature=0.0, do_sample=False,
        )
    text = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def run_quality_check(model, tokenizer, label=""):
    print(f"\n  Quality check ({label}):")
    answers = {}
    for name, prompt in QUALITY_PROMPTS:
        answer = generate_answer(model, tokenizer, prompt)
        print(f"    {name}: {answer[:120]}")
        answers[name] = answer[:200]
    return answers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-30B-A3B")
    parser.add_argument("--prune", type=float, default=0.5)
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--calibration-samples", type=int, default=1024)
    parser.add_argument("--group-size", type=int, default=128)
    args = parser.parse_args()

    print("=" * 60)
    print(f"REAP + TQ Benchmark: {args.model}")
    print(f"Prune: {args.prune*100:.0f}%, Bits: {args.bits}, Calibration: {args.calibration_samples}")
    print("=" * 60)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    import logging
    logging.basicConfig(level=logging.INFO, format='%(name)s: %(message)s')

    # Step 1: Load model
    print("\n--- Step 1: Load BF16 model ---")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda"
    )
    mem_bf16 = measure_memory()
    print(f"BF16 loaded: {mem_bf16:.0f} MB ({mem_bf16/1000:.1f} GB)")

    # Step 1b: Baseline quality
    answers_baseline = run_quality_check(model, tokenizer, "BF16 baseline")

    # Step 2: REAP expert pruning
    if args.prune > 0:
        print(f"\n--- Step 2: REAP pruning ({args.prune*100:.0f}%) ---")
        t0 = time.time()
        from turboquant_vllm.expert_pruning import reap_prune
        pruned = reap_prune(
            model, tokenizer,
            prune_fraction=args.prune,
            num_samples=args.calibration_samples,
        )
        prune_time = time.time() - t0
        mem_pruned = measure_memory()
        print(f"After REAP: {mem_pruned:.0f} MB ({mem_pruned/1000:.1f} GB), took {prune_time:.1f}s")

        # Quality after pruning (before TQ compression)
        answers_pruned = run_quality_check(model, tokenizer, "after REAP prune")
    else:
        mem_pruned = mem_bf16
        answers_pruned = answers_baseline

    # Step 3: TQ compression
    print(f"\n--- Step 3: TQ{args.bits} compression ---")
    t0 = time.time()
    from turboquant_vllm.weight_quant import _replace_linear_layers
    n_layers = _replace_linear_layers(model, bits=args.bits, group_size=args.group_size)
    compress_time = time.time() - t0
    mem_compressed = measure_memory()
    print(f"After TQ{args.bits}: {mem_compressed:.0f} MB ({mem_compressed/1000:.1f} GB), "
          f"took {compress_time:.1f}s, {n_layers} layers")

    # Quality after full compression
    answers_compressed = run_quality_check(model, tokenizer, "after REAP + TQ")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Model:       {args.model}")
    print(f"BF16:        {mem_bf16:.0f} MB ({mem_bf16/1000:.1f} GB)")
    if args.prune > 0:
        print(f"After REAP:  {mem_pruned:.0f} MB ({mem_pruned/1000:.1f} GB) "
              f"[{args.prune*100:.0f}% experts pruned]")
    print(f"After TQ{args.bits}:   {mem_compressed:.0f} MB ({mem_compressed/1000:.1f} GB) "
          f"[{mem_bf16/max(mem_compressed,1):.1f}x total compression]")
    print(f"Compress time: {compress_time:.1f}s" +
          (f" + {prune_time:.1f}s REAP" if args.prune > 0 else ""))

    # Quality comparison
    print(f"\n{'Prompt':<12} {'BF16':>20} {'Compressed':>20} {'Match?':>8}")
    print("-" * 62)
    for name, _ in QUALITY_PROMPTS:
        bl = answers_baseline.get(name, "")[:30]
        cp = answers_compressed.get(name, "")[:30]
        match = "YES" if bl[:15].lower() == cp[:15].lower() else "COMPARE"
        print(f"{name:<12} {bl:>20} {cp:>20} {match:>8}")

    # Save results
    results = {
        "model": args.model,
        "prune_fraction": args.prune,
        "bits": args.bits,
        "mem_bf16_mb": round(mem_bf16),
        "mem_pruned_mb": round(mem_pruned),
        "mem_compressed_mb": round(mem_compressed),
        "compression_ratio": round(mem_bf16 / max(mem_compressed, 1), 1),
        "compress_time_s": round(compress_time, 1),
        "answers_baseline": answers_baseline,
        "answers_compressed": answers_compressed,
    }
    out_path = "/tmp/reap_tq_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
