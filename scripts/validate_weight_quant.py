#!/usr/bin/env python3
"""Validate TurboQuant weight quantization with proper metrics.

Measures:
1. Real GPU memory usage (before/after compression)
2. Output quality (perplexity on wikitext sample)
3. Generation coherence (multiple prompts)
4. Compression ratio (actual stored bytes)
5. Inference speed (tokens/sec)

Usage:
    python scripts/validate_weight_quant.py [--model MODEL] [--bits BITS] [--group-size GS]

Default: Qwen/Qwen3-0.6B at 4-bit, group_size=128.
"""

import argparse
import gc
import json
import logging
import time
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
logger = logging.getLogger("validate")


def gpu_memory_mb():
    """Current GPU memory allocated in MB."""
    torch.cuda.synchronize()
    return torch.cuda.memory_allocated() / 1e6


def gpu_memory_reserved_mb():
    """Current GPU memory reserved by allocator in MB."""
    torch.cuda.synchronize()
    return torch.cuda.memory_reserved() / 1e6


def measure_perplexity(model, tokenizer, text, max_length=512):
    """Compute perplexity on a text sample."""
    encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    input_ids = encodings.input_ids.to("cuda")

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss

    return torch.exp(loss).item()


def measure_generation(model, tokenizer, prompt, max_new_tokens=100):
    """Generate text and measure speed."""
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    input_len = inputs.input_ids.shape[1]

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    output_len = output.shape[1] - input_len
    tokens_per_sec = output_len / elapsed if elapsed > 0 else 0
    text = tokenizer.decode(output[0], skip_special_tokens=True)

    return text, tokens_per_sec, elapsed


def count_parameters(model):
    """Count total and compressible parameters."""
    total = 0
    compressible_2d = 0
    compressible_3d = 0
    skip = ("lm_head", "embed", "norm", "head")

    for name, param in model.named_parameters():
        total += param.numel()
        if any(p in name.lower() for p in skip):
            continue
        if param.dim() == 2 and (param.shape[0] >= 1024 or param.shape[1] >= 1024):
            compressible_2d += param.numel()
        elif param.dim() == 3 and (param.shape[-1] >= 1024 or param.shape[-2] >= 1024):
            compressible_3d += param.numel()

    return total, compressible_2d, compressible_3d


EVAL_PROMPTS = [
    "The capital of Finland is",
    "Explain what a neural network is in one sentence.",
    "Write a Python function that returns the factorial of a number.",
    "What are the three laws of thermodynamics?",
    "Translate 'good morning' to Japanese, French, and Finnish.",
]

PERPLEXITY_TEXT = """The transformer architecture has become the dominant approach in natural
language processing. At its core, the transformer uses a mechanism called attention to weigh
the importance of different parts of the input when producing each element of the output.
Unlike recurrent neural networks, transformers process all positions in parallel, making them
significantly faster to train on modern hardware. The key innovation is the self-attention
mechanism, which computes attention scores between all pairs of positions in the input sequence.
This allows the model to capture long-range dependencies without the vanishing gradient problem
that plagues recurrent architectures. Modern large language models like GPT and Llama are built
entirely from transformer blocks, stacking dozens or hundreds of layers to achieve remarkable
performance on a wide range of tasks."""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--bits", type=int, default=4, choices=[2, 3, 4])
    parser.add_argument("--group-size", type=int, default=128, choices=[64, 128, 256])
    parser.add_argument("--output", type=str, default=None, help="Save results JSON")
    args = parser.parse_args()

    results = {"model": args.model, "bits": args.bits, "group_size": args.group_size}

    # ============================================================
    # Phase 1: Baseline
    # ============================================================
    logger.info("=" * 60)
    logger.info("BASELINE: %s", args.model)
    logger.info("=" * 60)

    torch.cuda.reset_peak_memory_stats()
    gc.collect()
    torch.cuda.empty_cache()
    mem_before_load = gpu_memory_mb()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.float16,
        device_map="cuda",
        trust_remote_code=True,
    )
    model.eval()

    mem_after_load = gpu_memory_mb()
    model_memory_mb = mem_after_load - mem_before_load
    logger.info("Model memory: %.0f MB (%.2f GB)", model_memory_mb, model_memory_mb / 1024)

    total_params, comp_2d, comp_3d = count_parameters(model)
    logger.info(
        "Parameters: %.0fM total, %.0fM compressible 2D, %.0fM compressible 3D",
        total_params / 1e6,
        comp_2d / 1e6,
        comp_3d / 1e6,
    )

    # Perplexity
    baseline_ppl = measure_perplexity(model, tokenizer, PERPLEXITY_TEXT)
    logger.info("Baseline perplexity: %.4f", baseline_ppl)

    # Generation
    baseline_outputs = []
    baseline_speeds = []
    for prompt in EVAL_PROMPTS:
        text, tps, elapsed = measure_generation(model, tokenizer, prompt, max_new_tokens=50)
        baseline_outputs.append(text)
        baseline_speeds.append(tps)
        logger.info("  [%.1f tok/s] %s", tps, text[:80])

    avg_baseline_speed = sum(baseline_speeds) / len(baseline_speeds)
    logger.info("Baseline avg speed: %.1f tok/s", avg_baseline_speed)

    results["baseline"] = {
        "memory_mb": model_memory_mb,
        "perplexity": baseline_ppl,
        "avg_tokens_per_sec": avg_baseline_speed,
        "outputs": baseline_outputs,
    }

    # ============================================================
    # Phase 2: Compressed
    # ============================================================
    logger.info("")
    logger.info("=" * 60)
    logger.info("COMPRESSED: TQ%d-g%d", args.bits, args.group_size)
    logger.info("=" * 60)

    gc.collect()
    torch.cuda.empty_cache()

    from turboquant_vllm.weight_quant import _replace_linear_layers

    t0 = time.perf_counter()
    num_replaced = _replace_linear_layers(model, bits=args.bits, group_size=args.group_size, min_size=1024)
    compress_time = time.perf_counter() - t0

    gc.collect()
    torch.cuda.empty_cache()
    mem_after_compress = gpu_memory_mb()
    compressed_memory_mb = mem_after_compress - mem_before_load
    memory_saved_mb = model_memory_mb - compressed_memory_mb

    logger.info("Compression time: %.1fs", compress_time)
    logger.info("Layers compressed: %d", num_replaced)
    logger.info(
        "Memory: %.0f MB -> %.0f MB (saved %.0f MB, %.1f%%)",
        model_memory_mb,
        compressed_memory_mb,
        memory_saved_mb,
        memory_saved_mb / model_memory_mb * 100 if model_memory_mb > 0 else 0,
    )

    # Perplexity
    compressed_ppl = measure_perplexity(model, tokenizer, PERPLEXITY_TEXT)
    ppl_delta = (compressed_ppl - baseline_ppl) / baseline_ppl * 100
    logger.info("Compressed perplexity: %.4f (%+.2f%%)", compressed_ppl, ppl_delta)

    # Generation
    compressed_outputs = []
    compressed_speeds = []
    for i, prompt in enumerate(EVAL_PROMPTS):
        text, tps, elapsed = measure_generation(model, tokenizer, prompt, max_new_tokens=50)
        compressed_outputs.append(text)
        compressed_speeds.append(tps)
        logger.info("  [%.1f tok/s] %s", tps, text[:80])

    avg_compressed_speed = sum(compressed_speeds) / len(compressed_speeds)
    logger.info("Compressed avg speed: %.1f tok/s", avg_compressed_speed)

    results["compressed"] = {
        "memory_mb": compressed_memory_mb,
        "memory_saved_mb": memory_saved_mb,
        "memory_saved_pct": memory_saved_mb / model_memory_mb * 100 if model_memory_mb > 0 else 0,
        "perplexity": compressed_ppl,
        "perplexity_delta_pct": ppl_delta,
        "avg_tokens_per_sec": avg_compressed_speed,
        "compression_time_s": compress_time,
        "layers_compressed": num_replaced,
        "outputs": compressed_outputs,
    }

    # ============================================================
    # Summary
    # ============================================================
    logger.info("")
    logger.info("=" * 60)
    logger.info("SUMMARY: %s TQ%d-g%d", args.model, args.bits, args.group_size)
    logger.info("=" * 60)
    logger.info(
        "Memory:     %.0f MB -> %.0f MB (%.1f%% saved)",
        model_memory_mb,
        compressed_memory_mb,
        memory_saved_mb / model_memory_mb * 100 if model_memory_mb > 0 else 0,
    )
    logger.info("Perplexity: %.4f -> %.4f (%+.2f%%)", baseline_ppl, compressed_ppl, ppl_delta)
    logger.info("Speed:      %.1f -> %.1f tok/s", avg_baseline_speed, avg_compressed_speed)
    logger.info("Compression time: %.1fs (%d layers)", compress_time, num_replaced)
    logger.info("")

    # Output comparison
    logger.info("Output comparison:")
    for i, prompt in enumerate(EVAL_PROMPTS):
        logger.info("  Prompt: %s", prompt[:50])
        logger.info("  Base:   %s", baseline_outputs[i][len(prompt) : len(prompt) + 80].strip())
        logger.info("  TQ:     %s", compressed_outputs[i][len(prompt) : len(prompt) + 80].strip())
        logger.info("")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info("Results saved to %s", args.output)


if __name__ == "__main__":
    main()
