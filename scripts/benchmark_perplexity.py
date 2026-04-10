"""Perplexity benchmark: BF16 vs TQ4 vs TQ3 on WikiText-2.

The standard quantization quality metric. Lower = better.

Usage: python3 -u scripts/benchmark_perplexity.py [--model MODEL]
"""

import argparse
import gc
import math
import os
import time
import torch

os.environ["PYTHONUNBUFFERED"] = "1"


def compute_perplexity(model, tokenizer, dataset_text, max_length=2048, stride=512):
    """Compute perplexity on a text dataset using sliding window.

    Standard method from HuggingFace perplexity docs.
    """
    encodings = tokenizer(dataset_text, return_tensors="pt")
    input_ids = encodings.input_ids.to(model.device)
    seq_len = input_ids.size(1)

    nlls = []
    prev_end_loc = 0

    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc

        input_chunk = input_ids[:, begin_loc:end_loc]
        target_ids = input_chunk.clone()
        target_ids[:, :-trg_len] = -100  # mask non-target tokens

        with torch.no_grad():
            outputs = model(input_chunk, labels=target_ids)
            neg_log_likelihood = outputs.loss

        nlls.append(neg_log_likelihood.item())
        prev_end_loc = end_loc

        if end_loc >= seq_len:
            break

    ppl = math.exp(sum(nlls) / len(nlls))
    return ppl


def load_wikitext2(tokenizer):
    """Load WikiText-2 test set."""
    try:
        from datasets import load_dataset

        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        text = "\n\n".join(dataset["text"])
    except Exception as e:
        print(f"Warning: Could not load wikitext: {e}. Using fallback.", flush=True)
        # Fallback: generate some text
        text = "The quick brown fox " * 10000
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/gemma-4-26B-A4B-it")
    parser.add_argument(
        "--native-checkpoint", default=None, help="Path to native TQ3 checkpoint (also runs native PPL)"
    )
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--stride", type=int, default=512)
    args = parser.parse_args()

    import logging

    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("=" * 60, flush=True)
    print(f"Perplexity Benchmark: {args.model}", flush=True)
    print(f"WikiText-2, max_length={args.max_length}, stride={args.stride}", flush=True)
    print("=" * 60, flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    text = load_wikitext2(tokenizer)
    print(f"WikiText-2 loaded: {len(text):,} chars", flush=True)

    results = {}

    configs = [
        ("BF16", 16, None),
        ("TQ4", 4, None),
        ("TQ3", 3, None),
    ]

    # Add native checkpoint if available
    native_dir = args.native_checkpoint
    if native_dir:
        configs.append(("TQ3-native", 3, native_dir))

    for label, bits, native_path in configs:
        print(f"\n── {label} ──", flush=True)

        if native_path:
            from turboquant_vllm.checkpoint import load_tq3_model

            t0 = time.time()
            model, _ = load_tq3_model(native_path, device="cuda")
            load_time = time.time() - t0
            mem = torch.cuda.memory_allocated() / 1e6
            print(f"Native loaded: {mem:.0f} MB ({load_time:.0f}s)", flush=True)
        else:
            model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda")
            mem_bf16 = torch.cuda.memory_allocated() / 1e6

            if bits < 16:
                from turboquant_vllm.weight_quant import _replace_linear_layers

                t0 = time.time()
                _replace_linear_layers(model, bits=bits, group_size=128)
                gc.collect()
                torch.cuda.empty_cache()
                compress_time = time.time() - t0
                mem = torch.cuda.memory_allocated() / 1e6
                print(
                    f"Compressed: {mem_bf16:.0f} → {mem:.0f} MB ({mem_bf16 / mem:.1f}x), {compress_time:.0f}s",
                    flush=True,
                )
            else:
                mem = mem_bf16
                print(f"Loaded: {mem:.0f} MB", flush=True)

        model.eval()
        t0 = time.time()
        ppl = compute_perplexity(model, tokenizer, text, max_length=args.max_length, stride=args.stride)
        ppl_time = time.time() - t0
        print(f"Perplexity: {ppl:.2f} ({ppl_time:.0f}s)", flush=True)

        results[label] = {"ppl": round(ppl, 2), "memory_mb": round(mem)}

        del model
        gc.collect()
        torch.cuda.empty_cache()

    # Summary
    print(f"\n{'=' * 60}", flush=True)
    print("SUMMARY", flush=True)
    print(f"{'=' * 60}", flush=True)
    print(f"Model: {args.model}", flush=True)
    print(f"{'Config':<12} {'PPL':>8} {'Memory':>10} {'vs BF16':>10}", flush=True)
    print("-" * 44, flush=True)
    bf16_ppl = results.get("BF16", {}).get("ppl", 0)
    for label in [l for l, _, _ in configs]:
        r = results.get(label, {})
        ppl = r.get("ppl", 0)
        mem = r.get("memory_mb", 0)
        delta = f"+{((ppl / bf16_ppl) - 1) * 100:.1f}%" if bf16_ppl > 0 and label != "BF16" else ""
        print(f"{label:<12} {ppl:>8.2f} {mem:>8} MB {delta:>10}", flush=True)


if __name__ == "__main__":
    main()
