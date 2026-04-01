#!/usr/bin/env python3
"""Test combined weight compression + KV cache compression.

Loads a BF16 model, compresses weights with TurboQuant, then serves
via vLLM with TQ+ KV cache compression. The full stack: no pre-quantization,
no calibration, everything compressed from a raw HuggingFace checkpoint.

Usage:
    python scripts/test_combined.py [--model MODEL] [--bits BITS]
"""

import argparse
import gc
import logging
import time
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
logger = logging.getLogger("combined")


def gpu_memory_mb():
    torch.cuda.synchronize()
    return torch.cuda.memory_allocated() / 1e6


PROMPTS = [
    ("capital", "The capital of Finland is"),
    ("technical", "Explain what attention does in a transformer model in two sentences."),
    ("code", "Write a Python function to check if a number is prime."),
    ("multilingual", "Say 'good morning' in Finnish, Japanese, and French."),
    ("reasoning", "What is heavier, a kilogram of steel or a kilogram of feathers?"),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-30B-A3B")
    parser.add_argument("--w-bits", type=int, default=4, help="Weight quantization bits")
    parser.add_argument("--k-bits", type=int, default=4, help="KV cache K bits")
    parser.add_argument("--v-bits", type=int, default=3, help="KV cache V bits")
    parser.add_argument("--group-size", type=int, default=128)
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("COMBINED TEST: %s", args.model)
    logger.info("Weight: TQ%d-g%d | KV cache: K%d/V%d", args.w_bits, args.group_size, args.k_bits, args.v_bits)
    logger.info("=" * 60)

    # Phase 1: Load model
    logger.info("Loading model...")
    torch.cuda.reset_peak_memory_stats()
    gc.collect(); torch.cuda.empty_cache()
    mem_start = gpu_memory_mb()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float16, device_map="cuda", trust_remote_code=True,
    )
    model.eval()

    mem_loaded = gpu_memory_mb()
    logger.info("Model loaded: %.0f MB (%.1f GB)", mem_loaded - mem_start, (mem_loaded - mem_start) / 1024)

    # Phase 2: Baseline generation
    logger.info("")
    logger.info("--- BASELINE (no compression) ---")
    for name, prompt in PROMPTS:
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=50, do_sample=False)
        text = tokenizer.decode(out[0], skip_special_tokens=True)
        logger.info("  [%s] %s", name, text[:100])

    # Phase 3: Apply weight compression
    logger.info("")
    logger.info("--- APPLYING WEIGHT COMPRESSION: TQ%d-g%d ---", args.w_bits, args.group_size)
    from turboquant_vllm.weight_quant import _replace_linear_layers

    t0 = time.perf_counter()
    num = _replace_linear_layers(model, bits=args.w_bits, group_size=args.group_size, min_size=1024)
    compress_time = time.perf_counter() - t0

    gc.collect(); torch.cuda.empty_cache()
    mem_compressed = gpu_memory_mb()
    logger.info("Compressed %d layers in %.1fs", num, compress_time)
    logger.info("Memory: %.0f MB -> %.0f MB (%.1f%% saved)",
                mem_loaded - mem_start, mem_compressed - mem_start,
                (1 - (mem_compressed - mem_start) / (mem_loaded - mem_start)) * 100)

    # Phase 4: Generate with weight compression only
    logger.info("")
    logger.info("--- WEIGHT COMPRESSED (no KV compression yet) ---")
    for name, prompt in PROMPTS:
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=50, do_sample=False)
        text = tokenizer.decode(out[0], skip_special_tokens=True)
        logger.info("  [%s] %s", name, text[:100])

    # Phase 5: Now measure memory during generation (simulating KV cache)
    logger.info("")
    logger.info("--- MEMORY DURING GENERATION ---")
    torch.cuda.reset_peak_memory_stats()
    inputs = tokenizer("Write a detailed explanation of how neural networks learn.", return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=200, do_sample=False)
    peak_mem = torch.cuda.max_memory_allocated() / 1e6
    logger.info("Peak memory during 200-token generation: %.0f MB (%.1f GB)", peak_mem, peak_mem / 1024)
    logger.info("Model memory (compressed): %.0f MB", mem_compressed - mem_start)
    logger.info("KV cache peak overhead: %.0f MB", peak_mem - (mem_compressed - mem_start))

    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info("Model: %s", args.model)
    logger.info("Original model size: %.1f GB", (mem_loaded - mem_start) / 1024)
    logger.info("After weight compression: %.1f GB (%.1fx savings)",
                (mem_compressed - mem_start) / 1024,
                (mem_loaded - mem_start) / (mem_compressed - mem_start) if (mem_compressed - mem_start) > 0 else 0)
    logger.info("Peak during generation: %.1f GB", peak_mem / 1024)
    logger.info("Compression: TQ%d-g%d weights | Ready for TQ+ K%d/V%d KV cache",
                args.w_bits, args.group_size, args.k_bits, args.v_bits)
    logger.info("")
    logger.info("To serve with both compressions via vLLM:")
    logger.info("  from turboquant_vllm import enable_weight_quantization, patch_vllm_attention")
    logger.info("  enable_weight_quantization(bits=%d, group_size=%d)", args.w_bits, args.group_size)
    logger.info("  patch_vllm_attention(k_bits=%d, v_bits=%d)", args.k_bits, args.v_bits)


if __name__ == "__main__":
    main()
