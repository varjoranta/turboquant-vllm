#!/usr/bin/env python3
"""Test loading a native TQ3 checkpoint with compressed weights on GPU.

Usage:
    # Download checkpoint first:
    huggingface-cli download varjosoft/gemma-4-26B-A4B-it-TQ3-native --local-dir ./gemma4-tq3

    # Run test:
    python3 scripts/test_tq3_load.py ./gemma4-tq3
"""
import argparse
import logging
import time
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_dir", help="Path to native TQ3 checkpoint")
    parser.add_argument("--device", default="cuda", help="Device (default: cuda)")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    args = parser.parse_args()

    # Show GPU info
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1e9
        logger.info("GPU: %s (%.1f GB)", gpu_name, gpu_mem)
    else:
        logger.warning("No CUDA GPU available, using CPU")
        args.device = "cpu"

    # Load model with compressed weights
    t0 = time.time()
    from turboquant_vllm.checkpoint import load_tq3_model
    model, tokenizer = load_tq3_model(args.checkpoint_dir, device=args.device)
    load_time = time.time() - t0
    logger.info("Model loaded in %.1f seconds", load_time)

    if torch.cuda.is_available():
        logger.info("GPU memory after load: %.1f GB", torch.cuda.memory_allocated() / 1e9)

    # Test generation
    prompts = [
        "What is the capital of Finland?",
        "Explain quantum computing in one paragraph.",
        "Write a haiku about machine learning.",
    ]

    for prompt in prompts:
        logger.info("--- Prompt: %s", prompt)

        chat = [{"role": "user", "content": prompt}]
        input_text = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(input_text, return_tensors="pt").to(args.device)

        t0 = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
            )
        gen_time = time.time() - t0

        new_tokens = outputs.shape[1] - inputs["input_ids"].shape[1]
        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        tok_per_sec = new_tokens / gen_time

        logger.info("Response (%d tokens, %.1f tok/s, %.1fs):", new_tokens, tok_per_sec, gen_time)
        print(response)
        print()

    logger.info("All tests passed!")

    if torch.cuda.is_available():
        logger.info("Peak GPU memory: %.1f GB", torch.cuda.max_memory_allocated() / 1e9)


if __name__ == "__main__":
    main()
