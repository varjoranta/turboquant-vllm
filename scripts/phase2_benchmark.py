"""Phase 2 TQ+ benchmark: Gemma 4 26B MoE on A100.

Measures memory, quality, and speed with and without TQ+ Phase 2 features.
Run on a Verda A100 80GB instance.

Usage: python3 -u scripts/phase2_benchmark.py
"""
import json
import os
import re
import subprocess
import sys
import time

# Force unbuffered output
os.environ["PYTHONUNBUFFERED"] = "1"

MODEL = "google/gemma-4-26B-A4B-it"
PORT = 8000
MAX_MODEL_LEN = 4096

PROMPTS = [
    ("Capital", "What is the capital of Finland? One sentence."),
    ("Math", "What is 17 * 23?"),
    ("Product", "Write a one-sentence product description for a SaaS analytics platform that helps sales teams track pipeline metrics."),
    ("Code", "Write a Python function is_prime(n) that returns True if n is prime."),
    ("Reasoning", "A bat and ball cost $1.10 together. The bat costs $1 more than the ball. How much does the ball cost? Show your reasoning."),
]


def get_gpu_memory():
    """Get GPU memory used in MiB."""
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True
    )
    return int(result.stdout.strip())


def start_server(tq_enabled=False, k_bits=4, v_bits=3):
    """Start vLLM server, return process."""
    env = os.environ.copy()

    if tq_enabled:
        # Start with TQ+ patch, then use vLLM v0.19.0 API:
        # 1. patch_vllm_attention() before any vLLM engine init
        # 2. Parse args via make_arg_parser (cli_env_setup just sets env vars)
        # 3. run_server(args) is async, run via asyncio
        cmd = [
            sys.executable, "-u", "-c",
            f"""
import os, sys, asyncio
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

from turboquant_vllm import patch_vllm_attention
patch_vllm_attention(k_bits={k_bits}, v_bits={v_bits}, norm_correction=True,
                     sink_tokens=4, boundary_layers=5)
print("TQ+ Phase 2 applied: K{k_bits}/V{v_bits}, NC, sinks=4, boundary=5", flush=True)

from vllm.entrypoints.openai.api_server import FlexibleArgumentParser, make_arg_parser, run_server, validate_parsed_serve_args

parser = FlexibleArgumentParser(description="vLLM TQ+ server")
parser = make_arg_parser(parser)
args = parser.parse_args(['--model', '{MODEL}',
    '--max-model-len', '{MAX_MODEL_LEN}',
    '--gpu-memory-utilization', '0.9',
    '--enforce-eager',
    '--port', '{PORT}',
    '--host', '0.0.0.0'])
validate_parsed_serve_args(args)
print("Starting vLLM with TQ+ patches...", flush=True)
asyncio.run(run_server(args))
"""
        ]
    else:
        cmd = [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", MODEL,
            "--max-model-len", str(MAX_MODEL_LEN),
            "--gpu-memory-utilization", "0.9",
            "--enforce-eager",
            "--port", str(PORT),
            "--host", "0.0.0.0",
        ]

    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return proc


def wait_for_server(timeout=300):
    """Wait for health check to pass."""
    import urllib.request
    for i in range(timeout):
        try:
            urllib.request.urlopen(f"http://localhost:{PORT}/health", timeout=2)
            return True
        except Exception:
            pass
        time.sleep(1)
    return False


def get_model_name():
    """Get the actual model name from the server's /v1/models endpoint."""
    import urllib.request
    try:
        resp = urllib.request.urlopen(f"http://localhost:{PORT}/v1/models", timeout=10)
        data = json.loads(resp.read())
        models = data.get("data", [])
        if models:
            name = models[0]["id"]
            print(f"  Server model name: {name}", flush=True)
            return name
    except Exception as e:
        print(f"  Warning: could not get model name: {e}", flush=True)
    return MODEL


def query(prompt, model_name=None, max_tokens=150):
    """Send a chat completion request, return (content, tokens, time)."""
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
    resp = urllib.request.urlopen(req, timeout=60)
    elapsed = time.perf_counter() - t0
    data = json.loads(resp.read())
    content = data["choices"][0]["message"]["content"]
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
    usage = data.get("usage", {})
    return content, usage.get("completion_tokens", 0), elapsed


def kill_server(proc):
    """Kill server and free GPU."""
    proc.terminate()
    proc.wait(timeout=10)
    subprocess.run(["pkill", "-9", "-f", "vllm"], capture_output=True)
    subprocess.run(
        "nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9",
        shell=True, capture_output=True
    )
    time.sleep(5)


def run_benchmark(name, tq_enabled=False, k_bits=4, v_bits=3):
    """Run full benchmark for one configuration."""
    print(f"\n{'='*60}")
    print(f"CONFIG: {name}")
    print(f"{'='*60}")

    mem_before = get_gpu_memory()
    print(f"GPU memory before: {mem_before} MiB")

    proc = start_server(tq_enabled=tq_enabled, k_bits=k_bits, v_bits=v_bits)

    print("Waiting for server...")
    if not wait_for_server():
        print("FAILED: server did not start")
        kill_server(proc)
        return None

    mem_loaded = get_gpu_memory()
    print(f"GPU memory after load: {mem_loaded} MiB ({mem_loaded - mem_before} MiB used)")

    results = {"name": name, "memory_mib": mem_loaded, "prompts": []}

    model_name = get_model_name()

    for label, prompt in PROMPTS:
        content, tokens, elapsed = query(prompt, model_name=model_name)
        tok_s = tokens / elapsed if elapsed > 0 else 0
        print(f"\n  {label}: [{tokens} tok, {elapsed:.2f}s, {tok_s:.0f} tok/s]")
        print(f"    {content[:200]}")
        results["prompts"].append({
            "label": label,
            "tokens": tokens,
            "time_s": round(elapsed, 3),
            "tok_s": round(tok_s, 1),
            "response": content[:300],
        })

    kill_server(proc)
    return results


if __name__ == "__main__":
    tq_only = "--tq-only" in sys.argv

    print("=" * 60)
    print("TQ+ Phase 2 Benchmark: Gemma 4 26B MoE on A100 80GB")
    print("=" * 60)

    if tq_only:
        # Use cached baseline results
        baseline = {
            "name": "Baseline (FP16 KV)", "memory_mib": 73699,
            "prompts": [
                {"label": "Capital", "tokens": 8, "time_s": 0.56, "tok_s": 14.0,
                 "response": "The capital of Finland is Helsinki."},
                {"label": "Math", "tokens": 13, "time_s": 0.79, "tok_s": 16.0,
                 "response": "17 * 23 = **391**"},
                {"label": "Product", "tokens": 22, "time_s": 1.32, "tok_s": 17.0,
                 "response": "Empower your sales team with real-time visibility into pipeline health through actionable metrics and automated forecasting insights."},
                {"label": "Code", "tokens": 150, "time_s": 8.40, "tok_s": 18.0,
                 "response": "Here is a clean, efficient implementation of the `is_prime` function in Python."},
                {"label": "Reasoning", "tokens": 150, "time_s": 8.50, "tok_s": 18.0,
                 "response": "The ball costs **$0.05** (5 cents)."},
            ]
        }
        print("Using cached baseline (73699 MiB)")
    else:
        baseline = run_benchmark("Baseline (FP16 KV)", tq_enabled=False)

    # Test 2: TQ+ Phase 2
    tqplus = run_benchmark("TQ+ Phase 2 (K4/V3, NC, sinks, boundary)",
                           tq_enabled=True, k_bits=4, v_bits=3)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    if baseline and tqplus:
        mem_saved = baseline["memory_mib"] - tqplus["memory_mib"]
        print(f"Memory: {baseline['memory_mib']} MiB → {tqplus['memory_mib']} MiB "
              f"(saved {mem_saved} MiB, {mem_saved/baseline['memory_mib']*100:.0f}%)")
        print()
        print(f"{'Prompt':<12} {'Baseline tok/s':>15} {'TQ+ tok/s':>12} {'Quality match?':>15}")
        for bp, tp in zip(baseline["prompts"], tqplus["prompts"]):
            bl = bp["response"][:50].lower()
            tl = tp["response"][:50].lower()
            match = "YES" if bl[:20] == tl[:20] else "COMPARE"
            print(f"{bp['label']:<12} {bp['tok_s']:>13.1f} {tp['tok_s']:>12.1f} {match:>15}")

    # Save results
    with open("/tmp/phase2_results.json", "w") as f:
        json.dump({"baseline": baseline, "tqplus": tqplus}, f, indent=2)
    print("\nResults saved to /tmp/phase2_results.json")
