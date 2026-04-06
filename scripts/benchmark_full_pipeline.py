"""Full pipeline benchmark: REAP + router finetune + mixed-precision TQ + sparse outliers.

Tests the complete compression stack on Qwen3-30B-A3B:
1. BF16 baseline (quality reference)
2. REAP saliency scoring (1024 calibration samples)
3. Expert pruning (20% and 50%)
4. Router fine-tuning (200 steps)
5. Hessian diagonal collection
6. Mixed-precision TQ compression (shared@TQ5, routed@TQ3/TQ2)
7. Sparse outlier extraction (0.1% FP16)
8. Quality check at each stage

Usage: python3 -u scripts/benchmark_full_pipeline.py [--model MODEL] [--prune FRACTION]
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
        out = model.generate(**inputs, max_new_tokens=max_tokens, temperature=0.0, do_sample=False)
    text = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def quality_check(model, tokenizer, label=""):
    print(f"\n  Quality check ({label}):", flush=True)
    answers = {}
    for name, prompt in QUALITY_PROMPTS:
        answer = generate_answer(model, tokenizer, prompt)
        print(f"    {name}: {answer[:120]}", flush=True)
        answers[name] = answer[:200]
    return answers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-30B-A3B")
    parser.add_argument("--prune", type=float, default=0.5)
    parser.add_argument("--calibration-samples", type=int, default=1024)
    parser.add_argument("--finetune-steps", type=int, default=200)
    parser.add_argument("--skip-finetune", action="store_true")
    parser.add_argument("--skip-sparse", action="store_true")
    args = parser.parse_args()

    import logging
    logging.basicConfig(level=logging.INFO, format='%(name)s: %(message)s')

    print("=" * 70, flush=True)
    print(f"FULL PIPELINE BENCHMARK: {args.model}", flush=True)
    print(f"Prune: {args.prune*100:.0f}%, Calibration: {args.calibration_samples}, "
          f"Finetune: {args.finetune_steps} steps", flush=True)
    print("=" * 70, flush=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    results = {"model": args.model, "stages": {}}

    # ── Stage 1: Load BF16 ────────────────────────────────────────────
    print("\n── Stage 1: Load BF16 model ──", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda")
    mem_bf16 = measure_memory()
    print(f"BF16 loaded: {mem_bf16:.0f} MB ({mem_bf16/1000:.1f} GB)", flush=True)
    answers_baseline = quality_check(model, tokenizer, "BF16 baseline")
    results["stages"]["bf16"] = {"memory_mb": round(mem_bf16), "answers": answers_baseline}

    # ── Stage 2: REAP saliency + pruning ──────────────────────────────
    print(f"\n── Stage 2: REAP pruning ({args.prune*100:.0f}%) ──", flush=True)
    t0 = time.time()
    from turboquant_vllm.expert_pruning import reap_prune, compute_reap_saliency
    saliency = compute_reap_saliency(model, tokenizer, args.calibration_samples)
    pruned = reap_prune(model, tokenizer, prune_fraction=args.prune,
                        num_samples=args.calibration_samples)
    reap_time = time.time() - t0
    mem_pruned = measure_memory()
    print(f"After REAP: {mem_pruned:.0f} MB, took {reap_time:.0f}s", flush=True)
    answers_pruned = quality_check(model, tokenizer, "after REAP prune")
    results["stages"]["reap"] = {
        "memory_mb": round(mem_pruned), "time_s": round(reap_time),
        "answers": answers_pruned,
    }

    # ── Stage 3: Router fine-tuning ───────────────────────────────────
    if not args.skip_finetune:
        print(f"\n── Stage 3: Router fine-tuning ({args.finetune_steps} steps) ──", flush=True)
        t0 = time.time()
        from turboquant_vllm.expert_pruning import finetune_router
        final_loss = finetune_router(model, tokenizer, num_steps=args.finetune_steps)
        ft_time = time.time() - t0
        print(f"Router fine-tune: {ft_time:.0f}s, final loss={final_loss:.4f}", flush=True)
        answers_finetuned = quality_check(model, tokenizer, "after router fine-tune")
        results["stages"]["router_finetune"] = {
            "time_s": round(ft_time), "loss": round(final_loss, 4),
            "answers": answers_finetuned,
        }
    else:
        print("\n── Stage 3: Router fine-tuning SKIPPED ──", flush=True)

    # ── Stage 4: Hessian collection ───────────────────────────────────
    print("\n── Stage 4: Hessian diagonal collection ──", flush=True)
    t0 = time.time()
    from turboquant_vllm.expert_pruning import collect_hessian_diagonal
    hessian = collect_hessian_diagonal(model, tokenizer, num_samples=256)
    hess_time = time.time() - t0
    print(f"Hessian collected: {len(hessian)} layers, {hess_time:.0f}s", flush=True)

    # ── Stage 5: Mixed-precision TQ compression ───────────────────────
    print("\n── Stage 5: Mixed-precision TQ compression ──", flush=True)
    t0 = time.time()
    from turboquant_vllm.expert_pruning import compute_expert_bit_widths
    from turboquant_vllm.weight_quant import _replace_linear_layers

    bit_widths = compute_expert_bit_widths(model, hessian, saliency=saliency)

    # Apply compression with per-module bit widths
    n_layers = _replace_linear_layers(model, bits=4, group_size=128, per_module_bits=bit_widths)
    compress_time = time.time() - t0
    gc.collect()
    torch.cuda.empty_cache()
    mem_compressed = measure_memory()
    print(f"After TQ4: {mem_compressed:.0f} MB ({mem_bf16/max(mem_compressed,1):.1f}x), "
          f"took {compress_time:.0f}s, {n_layers} layers", flush=True)
    answers_compressed = quality_check(model, tokenizer, "after REAP + finetune + TQ4")
    results["stages"]["compressed"] = {
        "memory_mb": round(mem_compressed),
        "compression_ratio": round(mem_bf16 / max(mem_compressed, 1), 1),
        "time_s": round(compress_time),
        "answers": answers_compressed,
    }

    # ── Stage 6: Sparse outlier extraction ────────────────────────────
    if not args.skip_sparse:
        print("\n── Stage 6: Sparse outlier extraction ──", flush=True)
        t0 = time.time()
        from turboquant_vllm.expert_pruning import extract_sparse_outliers
        outliers = extract_sparse_outliers(model, hessian, outlier_fraction=0.001)
        sparse_time = time.time() - t0
        print(f"Sparse outliers: {len(outliers)} layers, {sparse_time:.0f}s", flush=True)
        results["stages"]["sparse_outliers"] = {
            "layers": len(outliers), "time_s": round(sparse_time),
        }
    else:
        print("\n── Stage 6: Sparse outlier extraction SKIPPED ──", flush=True)

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n{'='*70}", flush=True)
    print("SUMMARY", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"Model:       {args.model}", flush=True)
    print(f"BF16:        {mem_bf16:.0f} MB ({mem_bf16/1000:.1f} GB)", flush=True)
    print(f"After REAP:  {mem_pruned:.0f} MB ({args.prune*100:.0f}% experts pruned)", flush=True)
    print(f"After TQ4:   {mem_compressed:.0f} MB ({mem_bf16/max(mem_compressed,1):.1f}x total)", flush=True)

    total_time = reap_time + compress_time
    if not args.skip_finetune:
        total_time += ft_time
    print(f"Total time:  {total_time:.0f}s", flush=True)

    # Quality comparison
    print(f"\n{'Prompt':<12} {'BF16':<25} {'After REAP':<25} {'After TQ4':<25}", flush=True)
    print("-" * 87, flush=True)
    for name, _ in QUALITY_PROMPTS:
        bl = answers_baseline.get(name, "")[:22]
        pr = answers_pruned.get(name, "")[:22]
        cp = answers_compressed.get(name, "")[:22]
        print(f"{name:<12} {bl:<25} {pr:<25} {cp:<25}", flush=True)

    # Save
    out_path = "/tmp/full_pipeline_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
