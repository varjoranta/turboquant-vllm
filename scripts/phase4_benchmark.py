"""Phase 4 benchmark: Expert pruning + TQ2 on Qwen3-30B-A3B.

Target: 59.7 GB BF16 → ~5 GB compressed (12x).
Tests multiple configurations and measures memory, quality, and speed.

Usage: python3 -u scripts/phase4_benchmark.py
"""
import json
import os
import re
import subprocess
import sys
import time

os.environ["PYTHONUNBUFFERED"] = "1"

MODEL = "Qwen/Qwen3-30B-A3B"
PORT = 8000
MAX_MODEL_LEN = 4096

PROMPTS = [
    ("Capital", "What is the capital of Finland? One sentence."),
    ("Math", "What is 17 * 23?"),
    ("Product", "Write a one-sentence product description for a SaaS analytics platform."),
    ("Code", "Write a Python function is_prime(n) that returns True if n is prime."),
    ("Reasoning", "A bat and ball cost $1.10 together. The bat costs $1 more than the ball. How much does the ball cost? Show your reasoning."),
]

CONFIGS = [
    {
        "name": "TQ4 baseline (no pruning)",
        "bits": 4, "prune": 0.0, "routed_bits": None,
        "kurtosis": False,
    },
    {
        "name": "TQ4 + kurtosis-aware",
        "bits": 4, "prune": 0.0, "routed_bits": None,
        "kurtosis": True,
    },
    {
        "name": "TQ4 + 50% pruning",
        "bits": 4, "prune": 0.5, "routed_bits": None,
        "kurtosis": False,
    },
    {
        "name": "TQ4 + 50% pruning + TQ2 routed",
        "bits": 4, "prune": 0.5, "routed_bits": 2,
        "kurtosis": False,
    },
    {
        "name": "TQ4 + kurtosis + 50% pruning + TQ2 routed",
        "bits": 4, "prune": 0.5, "routed_bits": 2,
        "kurtosis": True,
    },
]


def get_gpu_memory():
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True
    )
    return int(result.stdout.strip())


def start_server(cfg):
    """Start vLLM server with TQ+ weight compression."""
    config_name = cfg["name"]
    prune = cfg["prune"]
    routed_bits = cfg["routed_bits"]
    kurtosis = cfg["kurtosis"]
    bits = cfg["bits"]

    routed_arg = f", routed_expert_bits={routed_bits}" if routed_bits else ""
    cmd = [
        sys.executable, "-u", "-c",
        f"""
import os, sys, asyncio
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

from turboquant_vllm.weight_quant import enable_weight_quantization
enable_weight_quantization(bits={bits}, group_size=128, kurtosis_aware={kurtosis},
                           prune_experts={prune}{routed_arg})
print("Weight compression configured: {config_name}", flush=True)

from vllm.entrypoints.openai.api_server import FlexibleArgumentParser, make_arg_parser, validate_parsed_serve_args, run_server
parser = FlexibleArgumentParser(description="TQ+ server")
parser = make_arg_parser(parser)
args = parser.parse_args(['--model', '{MODEL}',
    '--max-model-len', '{MAX_MODEL_LEN}',
    '--gpu-memory-utilization', '0.95',
    '--enforce-eager',
    '--port', '{PORT}',
    '--host', '0.0.0.0'])
validate_parsed_serve_args(args)
asyncio.run(run_server(args))
"""
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return proc


def wait_for_server(timeout=600):
    import urllib.request
    for _ in range(timeout):
        try:
            urllib.request.urlopen(f"http://localhost:{PORT}/health", timeout=2)
            return True
        except Exception:
            pass
        time.sleep(1)
    return False


def get_model_name():
    import urllib.request
    try:
        resp = urllib.request.urlopen(f"http://localhost:{PORT}/v1/models", timeout=10)
        data = json.loads(resp.read())
        models = data.get("data", [])
        if models:
            return models[0]["id"]
    except Exception:
        pass
    return MODEL


def query(prompt, model_name=None, max_tokens=150):
    import urllib.request
    payload = json.dumps({
        "model": model_name or MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }).encode()
    req = urllib.request.Request(
        f"http://localhost:{PORT}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    resp = urllib.request.urlopen(req, timeout=120)
    elapsed = time.perf_counter() - t0
    data = json.loads(resp.read())
    content = data["choices"][0]["message"]["content"]
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
    usage = data.get("usage", {})
    return content, usage.get("completion_tokens", 0), elapsed


def kill_server(proc):
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except Exception:
        proc.kill()
    subprocess.run(["pkill", "-9", "-f", "vllm"], capture_output=True)
    subprocess.run(
        "nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9",
        shell=True, capture_output=True
    )
    time.sleep(5)


def run_benchmark(cfg):
    name = cfg["name"]
    print(f"\n{'='*60}")
    print(f"CONFIG: {name}")
    print(f"{'='*60}")

    mem_before = get_gpu_memory()
    print(f"GPU memory before: {mem_before} MiB")

    proc = start_server(cfg)
    print("Waiting for server (model loading + compression)...")

    if not wait_for_server(timeout=600):
        print("FAILED: server did not start")
        # Dump last output for debugging
        try:
            out = proc.stdout.read(10000).decode(errors='replace')
            print(f"Server output:\n{out[-2000:]}")
        except Exception:
            pass
        kill_server(proc)
        return None

    mem_loaded = get_gpu_memory()
    print(f"GPU memory after load: {mem_loaded} MiB ({mem_loaded - mem_before} MiB used)")

    results = {"name": name, "memory_mib": mem_loaded, "prompts": []}
    model_name = get_model_name()
    print(f"  Model: {model_name}")

    for label, prompt in PROMPTS:
        try:
            content, tokens, elapsed = query(prompt, model_name=model_name)
            tok_s = tokens / elapsed if elapsed > 0 else 0
            print(f"\n  {label}: [{tokens} tok, {elapsed:.2f}s, {tok_s:.0f} tok/s]")
            print(f"    {content[:200]}")
            results["prompts"].append({
                "label": label, "tokens": tokens,
                "time_s": round(elapsed, 3), "tok_s": round(tok_s, 1),
                "response": content[:300],
            })
        except Exception as e:
            print(f"\n  {label}: ERROR - {e}")
            results["prompts"].append({"label": label, "error": str(e)})

    kill_server(proc)
    return results


if __name__ == "__main__":
    print("=" * 60)
    print("Phase 4 Benchmark: Expert Pruning + TQ2 on Qwen3-30B")
    print("=" * 60)

    all_results = {}
    for cfg in CONFIGS:
        result = run_benchmark(cfg)
        if result:
            all_results[cfg["name"]] = result

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Config':<45} {'Memory':>10} {'Avg tok/s':>10}")
    print("-" * 65)
    for name, r in all_results.items():
        avg_toks = sum(p.get("tok_s", 0) for p in r["prompts"]) / max(1, len(r["prompts"]))
        print(f"{name:<45} {r['memory_mib']:>8} MiB {avg_toks:>8.1f}")

    # Save
    with open("/tmp/phase4_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nResults saved to /tmp/phase4_results.json")
