#!/usr/bin/env python3
"""Benchmark: TQ3 weight compression + DFlash speculative decoding.

Compares four configurations on Qwen3-8B:
1. BF16 baseline (autoregressive)
2. TQ3 only (autoregressive, compressed weights)
3. DFlash only (speculative, BF16 weights)
4. TQ3 + DFlash (speculative, compressed weights)

Usage:
    python scripts/benchmark_dflash.py --target Qwen/Qwen3-8B \
        --draft z-lab/Qwen3-8B-DFlash-b16 --dataset gsm8k
"""

import argparse
import json
import logging
import time

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
logger = logging.getLogger("dflash_bench")


def load_dataset(name: str, max_samples: int = 50) -> list[str]:
    """Load benchmark prompts."""
    from datasets import load_dataset as hf_load

    if name == "gsm8k":
        ds = hf_load("openai/gsm8k", "main", split="test")
        prompts = [
            f"{row['question']}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
            for row in ds
        ]
    elif name == "math500":
        ds = hf_load("HuggingFaceH4/MATH-500", split="test")
        prompts = [
            f"{row['problem']}\nPlease reason step by step, and put your final answer within \\boxed{{}}." for row in ds
        ]
    else:
        # Simple test prompts
        prompts = [
            "What is 2+2? Think step by step.",
            "Explain gravity in one sentence.",
            "Write a Python function to check if a number is prime.",
            "What is the capital of Finland?",
            "List 5 colors of the rainbow.",
        ]

    return prompts[:max_samples]


def benchmark_autoregressive(model, tokenizer, prompts, max_new_tokens=256, label=""):
    """Standard autoregressive generation benchmark."""
    total_tokens = 0
    total_time = 0
    results = []

    for i, prompt in enumerate(prompts):
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        input_ids = tokenizer.encode(text, return_tensors="pt").to(model.device)

        torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            output = model.generate(input_ids, max_new_tokens=max_new_tokens, do_sample=False)
        torch.cuda.synchronize()
        elapsed = time.time() - t0

        new_tokens = output.shape[1] - input_ids.shape[1]
        total_tokens += new_tokens
        total_time += elapsed
        results.append({"tokens": new_tokens, "time": elapsed, "tok_s": new_tokens / elapsed})

        if i < 3:
            resp = tokenizer.decode(output[0][input_ids.shape[1] :], skip_special_tokens=True)
            logger.info("[%s] Q: %s...", label, prompt[:60])
            logger.info("[%s] A: %s...", label, resp[:100])

    avg_tok_s = total_tokens / total_time if total_time > 0 else 0
    peak_mem = torch.cuda.max_memory_allocated() / 1e9
    logger.info(
        "[%s] Total: %d tokens in %.1fs (%.1f tok/s), peak GPU: %.1f GB",
        label,
        total_tokens,
        total_time,
        avg_tok_s,
        peak_mem,
    )
    return {
        "label": label,
        "total_tokens": total_tokens,
        "total_time": total_time,
        "avg_tok_s": avg_tok_s,
        "peak_gpu_gb": peak_mem,
        "results": results,
    }


def benchmark_dflash(draft_model, target, tokenizer, prompts, max_new_tokens=256, temperature=0.0, label=""):
    """DFlash speculative decoding benchmark."""
    total_tokens = 0
    total_time = 0
    results = []
    stop_token_ids = [tokenizer.eos_token_id]
    if hasattr(tokenizer, "additional_special_tokens_ids"):
        stop_token_ids.extend(tokenizer.additional_special_tokens_ids[:5])

    for i, prompt in enumerate(prompts):
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        input_ids = tokenizer.encode(text, return_tensors="pt").to(target.device)

        torch.cuda.synchronize()
        t0 = time.time()
        output_ids = draft_model.spec_generate(
            target=target,
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            stop_token_ids=stop_token_ids,
            temperature=temperature,
        )
        torch.cuda.synchronize()
        elapsed = time.time() - t0

        new_tokens = output_ids.shape[1] - input_ids.shape[1]
        total_tokens += new_tokens
        total_time += elapsed
        results.append({"tokens": new_tokens, "time": elapsed, "tok_s": new_tokens / elapsed})

        if i < 3:
            resp = tokenizer.decode(output_ids[0][input_ids.shape[1] :], skip_special_tokens=True)
            logger.info("[%s] Q: %s...", label, prompt[:60])
            logger.info("[%s] A: %s...", label, resp[:100])

    avg_tok_s = total_tokens / total_time if total_time > 0 else 0
    peak_mem = torch.cuda.max_memory_allocated() / 1e9
    logger.info(
        "[%s] Total: %d tokens in %.1fs (%.1f tok/s), peak GPU: %.1f GB",
        label,
        total_tokens,
        total_time,
        avg_tok_s,
        peak_mem,
    )
    return {
        "label": label,
        "total_tokens": total_tokens,
        "total_time": total_time,
        "avg_tok_s": avg_tok_s,
        "peak_gpu_gb": peak_mem,
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(description="DFlash + TQ3 benchmark")
    parser.add_argument("--target", default="Qwen/Qwen3-8B")
    parser.add_argument("--draft", default="z-lab/Qwen3-8B-DFlash-b16")
    parser.add_argument("--dataset", default="simple", choices=["gsm8k", "math500", "simple"])
    parser.add_argument("--max-samples", type=int, default=20)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--skip-bf16", action="store_true", help="Skip BF16 baseline")
    parser.add_argument("--skip-tq3-only", action="store_true", help="Skip TQ3-only test")
    parser.add_argument("--skip-dflash-only", action="store_true", help="Skip DFlash-only test")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    prompts = load_dataset(args.dataset, args.max_samples)
    logger.info("Loaded %d prompts from %s", len(prompts), args.dataset)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.target)

    all_results = {}

    # ── 1. BF16 baseline ──
    if not args.skip_bf16:
        logger.info("=== BF16 Baseline ===")
        torch.cuda.reset_peak_memory_stats()
        model_bf16 = AutoModelForCausalLM.from_pretrained(args.target, dtype=torch.bfloat16).to(device).eval()
        mem_bf16 = torch.cuda.memory_allocated() / 1e9
        logger.info("BF16 model loaded: %.1f GB", mem_bf16)

        all_results["bf16"] = benchmark_autoregressive(
            model_bf16, tokenizer, prompts, args.max_new_tokens, label="BF16"
        )
        del model_bf16
        torch.cuda.empty_cache()

    # ── 2. TQ3 only (autoregressive) ──
    if not args.skip_tq3_only:
        logger.info("=== TQ3 Only ===")
        torch.cuda.reset_peak_memory_stats()

        # Create TQ3 checkpoint if not exists
        import os

        tq3_dir = f"/tmp/tq3-{args.target.replace('/', '-')}"
        if not os.path.exists(tq3_dir):
            logger.info("Creating TQ3 checkpoint at %s...", tq3_dir)
            from turboquant_vllm.checkpoint import save_tq3_checkpoint

            save_tq3_checkpoint(args.target, tq3_dir, bits=3, group_size=128)

        from turboquant_vllm.checkpoint import load_tq3_model

        model_tq3, _ = load_tq3_model(tq3_dir, device=device)
        # Cast non-quantized params to bfloat16 for DFlash compatibility
        for name, param in model_tq3.named_parameters():
            if param.dtype == torch.float16 and "tq_packed" not in name:
                param.data = param.data.to(torch.bfloat16)
        for name, buf in model_tq3.named_buffers():
            if buf.dtype == torch.float16:
                buf.data = buf.data.to(torch.bfloat16)
        mem_tq3 = torch.cuda.memory_allocated() / 1e9
        logger.info("TQ3 model loaded: %.1f GB", mem_tq3)

        all_results["tq3"] = benchmark_autoregressive(model_tq3, tokenizer, prompts, args.max_new_tokens, label="TQ3")
        # Keep for DFlash+TQ3 test
        target_tq3 = model_tq3

    # ── 3. DFlash only (BF16 target) ──
    if not args.skip_dflash_only:
        logger.info("=== DFlash Only (BF16 target) ===")
        torch.cuda.reset_peak_memory_stats()
        from dflash.model import DFlashDraftModel

        target_bf16 = AutoModelForCausalLM.from_pretrained(args.target, dtype=torch.bfloat16).to(device).eval()

        draft = (
            DFlashDraftModel.from_pretrained(args.draft, attn_implementation="sdpa", dtype=torch.bfloat16)
            .to(device)
            .eval()
        )
        mem_dflash = torch.cuda.memory_allocated() / 1e9
        logger.info("DFlash loaded (BF16 target + draft): %.1f GB", mem_dflash)

        all_results["dflash_bf16"] = benchmark_dflash(
            draft, target_bf16, tokenizer, prompts, args.max_new_tokens, label="DFlash+BF16"
        )
        del target_bf16, draft
        torch.cuda.empty_cache()

    # ── 4. TQ3 + DFlash ──
    logger.info("=== TQ3 + DFlash ===")
    torch.cuda.reset_peak_memory_stats()
    from dflash.model import DFlashDraftModel

    if "target_tq3" not in dir():
        # Load TQ3 if not already loaded
        import os

        tq3_dir = f"/tmp/tq3-{args.target.replace('/', '-')}"
        if not os.path.exists(tq3_dir):
            from turboquant_vllm.checkpoint import save_tq3_checkpoint

            save_tq3_checkpoint(args.target, tq3_dir, bits=3, group_size=128)
        from turboquant_vllm.checkpoint import load_tq3_model

        target_tq3, _ = load_tq3_model(tq3_dir, device=device)
        # Cast non-quantized params to bfloat16 for DFlash compatibility
        for name, param in target_tq3.named_parameters():
            if param.dtype == torch.float16 and "tq_packed" not in name:
                param.data = param.data.to(torch.bfloat16)
        for name, buf in target_tq3.named_buffers():
            if buf.dtype == torch.float16:
                buf.data = buf.data.to(torch.bfloat16)

    draft = (
        DFlashDraftModel.from_pretrained(args.draft, attn_implementation="sdpa", dtype=torch.bfloat16).to(device).eval()
    )
    mem_combined = torch.cuda.memory_allocated() / 1e9
    logger.info("TQ3 + DFlash loaded: %.1f GB", mem_combined)

    all_results["tq3_dflash"] = benchmark_dflash(
        draft, target_tq3, tokenizer, prompts, args.max_new_tokens, label="TQ3+DFlash"
    )

    # ── Summary ──
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Config':<20} {'Tok/s':>8} {'Memory':>10} {'Tokens':>8} {'Time':>8}")
    print("-" * 70)
    for key, r in all_results.items():
        print(
            f"{r['label']:<20} {r['avg_tok_s']:>8.1f} {r['peak_gpu_gb']:>8.1f} GB "
            f"{r['total_tokens']:>8d} {r['total_time']:>7.1f}s"
        )
    print("=" * 70)

    # Save results
    with open("/tmp/dflash_tq3_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to /tmp/dflash_tq3_results.json")


if __name__ == "__main__":
    main()
