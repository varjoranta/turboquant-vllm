"""Test AWQ export + vLLM Marlin serving.

1. Load Gemma 4 26B BF16
2. TQ compress → decompress → repack as AWQ
3. Save checkpoint
4. Load in vLLM with --quantization awq
5. Measure tok/s

Usage: python3 -u scripts/test_awq_export.py
"""
import gc
import json
import os
import re
import subprocess
import sys
import time
import torch

os.environ["PYTHONUNBUFFERED"] = "1"

MODEL = "google/gemma-4-26B-A4B-it"
EXPORT_DIR = "/tmp/gemma4-tq-awq"
PORT = 8000

PROMPTS = [
    ("Capital", "What is the capital of Finland? One word."),
    ("Math", "What is 17 * 23? Just the number."),
    ("Reasoning", "A bat and ball cost $1.10. Bat costs $1 more. Ball cost?"),
]


def main():
    import logging
    logging.basicConfig(level=logging.INFO, format='%(name)s: %(message)s')

    print("=" * 60, flush=True)
    print("AWQ Export + Marlin Serving Test", flush=True)
    print("=" * 60, flush=True)

    # Step 1: Export
    print("\n── Step 1: TQ compress + AWQ export ──", flush=True)
    t0 = time.time()
    from turboquant_vllm.export import compress_and_export
    compress_and_export(
        model_id=MODEL,
        output_dir=EXPORT_DIR,
        bits=4,  # AWQ is 4-bit
        group_size=128,
    )
    export_time = time.time() - t0
    print(f"Export complete: {export_time:.0f}s", flush=True)

    # Check exported files
    files = os.listdir(EXPORT_DIR)
    print(f"Files: {files}", flush=True)
    total_size = sum(os.path.getsize(os.path.join(EXPORT_DIR, f)) for f in files)
    print(f"Total size: {total_size / 1e9:.1f} GB", flush=True)

    # Free GPU memory before vLLM
    gc.collect()
    torch.cuda.empty_cache()
    subprocess.run("nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9",
                    shell=True, capture_output=True)
    time.sleep(5)

    # Step 2: Start vLLM with the exported checkpoint
    print("\n── Step 2: Start vLLM with AWQ checkpoint ──", flush=True)
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", EXPORT_DIR,
        "--quantization", "awq",
        "--max-model-len", "4096",
        "--gpu-memory-utilization", "0.9",
        "--enforce-eager",
        "--port", str(PORT),
        "--host", "0.0.0.0",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    # Wait for health check
    import urllib.request
    print("Waiting for server...", flush=True)
    for i in range(300):
        try:
            urllib.request.urlopen(f"http://localhost:{PORT}/health", timeout=2)
            print(f"Server ready in {i}s", flush=True)
            break
        except Exception:
            pass
        time.sleep(1)
    else:
        print("Server failed to start!", flush=True)
        out = proc.stdout.read(5000).decode(errors='replace')
        print(f"Output: {out[-2000:]}", flush=True)
        proc.terminate()
        return

    # Step 3: Query and measure
    print("\n── Step 3: Quality + speed check ──", flush=True)

    # Get model name
    try:
        resp = urllib.request.urlopen(f"http://localhost:{PORT}/v1/models", timeout=10)
        data = json.loads(resp.read())
        model_name = data["data"][0]["id"]
        print(f"Model: {model_name}", flush=True)
    except Exception:
        model_name = EXPORT_DIR

    for label, prompt in PROMPTS:
        payload = json.dumps({
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 100,
            "temperature": 0.0,
        }).encode()
        req = urllib.request.Request(
            f"http://localhost:{PORT}/v1/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        t0 = time.perf_counter()
        resp = urllib.request.urlopen(req, timeout=60)
        elapsed = time.perf_counter() - t0
        data = json.loads(resp.read())
        content = data["choices"][0]["message"]["content"]
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        tokens = data.get("usage", {}).get("completion_tokens", 0)
        tok_s = tokens / elapsed if elapsed > 0 else 0
        print(f"  {label}: [{tokens} tok, {elapsed:.2f}s, {tok_s:.0f} tok/s] {content[:100]}", flush=True)

    # Cleanup
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except Exception:
        proc.kill()
    subprocess.run(["pkill", "-9", "-f", "vllm"], capture_output=True)

    print("\nDone!", flush=True)


if __name__ == "__main__":
    main()
