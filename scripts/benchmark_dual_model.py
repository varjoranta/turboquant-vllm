"""Dual-model benchmark: TQ3/TQ4 + REAP on Qwen3-30B and Gemma 4 26B.

Tests compression at multiple levels on both models, measuring memory and quality.

Usage: python3 -u scripts/benchmark_dual_model.py
"""

import gc
import json
import os
import re
import time
import torch

os.environ["PYTHONUNBUFFERED"] = "1"

PROMPTS = [
    ("Capital", "What is the capital of Finland? One word."),
    ("Math", "What is 17 * 23? Just the number."),
    (
        "Reasoning",
        "A bat and ball cost $1.10 together. The bat costs $1 more than the ball. How much does the ball cost?",
    ),
    ("Code", "Write a Python function is_prime(n) that returns True if n is prime."),
]

MODELS = [
    "Qwen/Qwen3-30B-A3B",
    "google/gemma-4-26B-A4B-it",
]


def mem():
    gc.collect()
    torch.cuda.empty_cache()
    return torch.cuda.memory_allocated() / 1e6


def gen(model, tok, prompt, max_tok=80):
    # Use chat template for instruction-tuned models (Gemma 4 -it, etc.)
    if hasattr(tok, "chat_template") and tok.chat_template:
        messages = [{"role": "user", "content": prompt}]
        text_input = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inp = tok(text_input, return_tensors="pt").to("cuda")
    else:
        inp = tok(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=max_tok, temperature=0.0, do_sample=False)
    text = tok.decode(out[0][inp.input_ids.shape[1] :], skip_special_tokens=True)
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def quality(model, tok, label):
    print(f"\n  Quality ({label}):", flush=True)
    answers = {}
    for name, prompt in PROMPTS:
        a = gen(model, tok, prompt)
        print(f"    {name}: {a[:100]}", flush=True)
        answers[name] = a[:200]
    return answers


def test_model(model_id, configs):
    """Test a model with multiple compression configs."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import logging

    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")

    print(f"\n{'=' * 70}", flush=True)
    print(f"MODEL: {model_id}", flush=True)
    print(f"{'=' * 70}", flush=True)

    results = {"model": model_id, "configs": {}}
    tok = AutoTokenizer.from_pretrained(model_id)

    for cfg in configs:
        name = cfg["name"]
        bits = cfg["bits"]
        prune = cfg.get("prune", 0.0)
        cal_samples = cfg.get("cal_samples", 512)

        print(f"\n── {name} ──", flush=True)

        # Load fresh model
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map="cuda")
        m0 = mem()
        print(f"BF16: {m0:.0f} MB ({m0 / 1000:.1f} GB)", flush=True)

        if bits == 16:
            # Baseline — just quality check
            answers = quality(model, tok, "BF16 baseline")
            results["configs"][name] = {"memory_mb": round(m0), "ratio": 1.0, "answers": answers}
            del model
            gc.collect()
            torch.cuda.empty_cache()
            continue

        # REAP prune if requested
        if prune > 0:
            t0 = time.time()
            from turboquant_vllm.expert_pruning import reap_prune

            reap_prune(model, tok, prune_fraction=prune, num_samples=cal_samples)
            print(f"REAP {prune * 100:.0f}%: {time.time() - t0:.0f}s", flush=True)

        # TQ compress
        t0 = time.time()
        from turboquant_vllm.weight_quant import _replace_linear_layers

        _replace_linear_layers(model, bits=bits, group_size=128)
        compress_time = time.time() - t0
        m1 = mem()
        ratio = m0 / m1
        print(f"TQ{bits}: {m1:.0f} MB ({ratio:.1f}x), {compress_time:.0f}s", flush=True)

        # Quality
        try:
            answers = quality(model, tok, name)
        except Exception as e:
            print(f"  Quality check failed: {e}", flush=True)
            answers = {"error": str(e)}

        results["configs"][name] = {
            "memory_mb": round(m1),
            "ratio": round(ratio, 1),
            "time_s": round(compress_time),
            "answers": answers,
        }

        del model
        gc.collect()
        torch.cuda.empty_cache()

    return results


def main():
    print("=" * 70, flush=True)
    print("DUAL MODEL BENCHMARK: TQ3/TQ4 + REAP", flush=True)
    print("=" * 70, flush=True)

    all_results = {}

    # Qwen3-30B configs
    qwen_configs = [
        {"name": "BF16 baseline", "bits": 16},
        {"name": "TQ4", "bits": 4},
        {"name": "TQ3", "bits": 3},
        {"name": "TQ3 + REAP 20%", "bits": 3, "prune": 0.2, "cal_samples": 512},
    ]
    all_results["qwen"] = test_model("Qwen/Qwen3-30B-A3B", qwen_configs)

    # Gemma 4 26B configs
    gemma_configs = [
        {"name": "BF16 baseline", "bits": 16},
        {"name": "TQ4", "bits": 4},
        {"name": "TQ3", "bits": 3},
        {"name": "TQ3 + REAP 20%", "bits": 3, "prune": 0.2, "cal_samples": 512},
    ]
    all_results["gemma"] = test_model("google/gemma-4-26B-A4B-it", gemma_configs)

    # Summary
    print(f"\n{'=' * 70}", flush=True)
    print("FINAL SUMMARY", flush=True)
    print(f"{'=' * 70}", flush=True)
    for model_key, result in all_results.items():
        print(f"\n{result['model']}:", flush=True)
        print(f"  {'Config':<25} {'Memory':>10} {'Ratio':>8}", flush=True)
        print(f"  {'-' * 45}", flush=True)
        for name, cfg in result["configs"].items():
            print(f"  {name:<25} {cfg['memory_mb']:>8} MB {cfg['ratio']:>7.1f}x", flush=True)

    with open("/tmp/dual_model_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nResults saved to /tmp/dual_model_results.json", flush=True)


if __name__ == "__main__":
    main()
