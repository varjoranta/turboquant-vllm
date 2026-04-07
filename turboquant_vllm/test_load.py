"""Quick test: load native TQ3 checkpoint, generate text, report quality.

Usage: python3 -m turboquant_vllm.test_load <checkpoint_path_or_hf_repo>
"""
import logging
import sys
import time
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "varjosoft/gemma-4-26B-A4B-it-TQ3-native"

    # If it's a HF repo, download it first
    if "/" in ckpt and not ckpt.startswith("/"):
        import os
        if not os.path.isdir(ckpt):
            from huggingface_hub import snapshot_download
            ckpt = snapshot_download(ckpt, local_dir="/tmp/tq3_ckpt")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("GPU: %s", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

    t0 = time.time()
    from turboquant_vllm.checkpoint import load_tq3_model
    model, tokenizer = load_tq3_model(ckpt, device=device)
    logger.info("Loaded in %.1f s, GPU: %.1f GB", time.time() - t0,
                torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0)

    prompts = [
        "What is 2+2? Answer with just the number.",
        "What is the capital of Finland?",
        "Write one sentence about machine learning.",
    ]

    for prompt in prompts:
        chat = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(device)

        t0 = time.time()
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=50, do_sample=False)
        gen_time = time.time() - t0
        new_tok = out.shape[1] - inputs["input_ids"].shape[1]
        resp = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        print(f"\nQ: {prompt}")
        print(f"A: {resp}")
        print(f"   [{new_tok} tok, {new_tok/gen_time:.1f} tok/s]")

    if torch.cuda.is_available():
        print(f"\nPeak GPU: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
    print("\nDONE")


if __name__ == "__main__":
    main()
