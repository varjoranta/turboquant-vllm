#!/usr/bin/env python3
"""Concurrent throughput benchmark: TQ3 vs BF16.

Sends parallel requests at different concurrency levels and measures
throughput (tok/s), latency (p50/p95/p99), and QPS.

Usage:
    python3 scripts/benchmark_throughput.py http://localhost:8000/v1

Expects a vLLM server running at the given URL.
"""

import argparse
import asyncio
import json
import statistics
import time

import httpx


PROMPTS = [
    "Explain the difference between TCP and UDP in two sentences.",
    "What causes seasons on Earth?",
    "Write a short product description for noise-canceling headphones.",
    "What is the capital of Japan and what is it known for?",
    "Explain recursion to a five-year-old.",
    "Mikä on koneoppiminen yhdellä lauseella?",
    "Write a haiku about cloud computing.",
    "What are the three laws of thermodynamics?",
    "Compare Python and Rust in three bullet points.",
    "Explain why the ocean is salty.",
    "What is the time complexity of quicksort?",
    "Write a one-paragraph pitch for a SaaS analytics tool.",
    "How does a transistor work?",
    "What is the difference between AI and machine learning?",
    "Explain photosynthesis briefly.",
    "Name three renewable energy sources and their advantages.",
]


async def send_request(
    client: httpx.AsyncClient,
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int = 128,
) -> dict:
    """Send one chat completion request and return timing stats."""
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": False,
    }

    t0 = time.perf_counter()
    resp = await client.post(
        f"{base_url}/chat/completions",
        json=body,
        headers={"Content-Type": "application/json"},
    )
    latency = time.perf_counter() - t0

    if resp.status_code != 200:
        return {"error": True, "latency": latency, "tokens": 0}

    data = resp.json()
    tokens = data.get("usage", {}).get("completion_tokens", 0)
    return {"error": False, "latency": latency, "tokens": tokens}


async def run_concurrency_level(
    base_url: str,
    model: str,
    concurrency: int,
    num_requests: int,
    max_tokens: int = 128,
) -> dict:
    """Run num_requests at a given concurrency level."""
    semaphore = asyncio.Semaphore(concurrency)
    results = []

    async def bounded_request(prompt):
        async with semaphore:
            async with httpx.AsyncClient(timeout=120.0) as client:
                return await send_request(client, base_url, model, prompt, max_tokens)

    prompts = [PROMPTS[i % len(PROMPTS)] for i in range(num_requests)]

    t0 = time.perf_counter()
    results = await asyncio.gather(*[bounded_request(p) for p in prompts])
    wall_time = time.perf_counter() - t0

    successful = [r for r in results if not r["error"]]
    errors = len(results) - len(successful)

    if not successful:
        return {"concurrency": concurrency, "error": "all requests failed"}

    latencies = [r["latency"] for r in successful]
    total_tokens = sum(r["tokens"] for r in successful)

    return {
        "concurrency": concurrency,
        "requests": num_requests,
        "errors": errors,
        "wall_time_s": round(wall_time, 2),
        "total_tokens": total_tokens,
        "throughput_tok_s": round(total_tokens / wall_time, 1),
        "qps": round(len(successful) / wall_time, 2),
        "latency_p50_s": round(statistics.median(latencies), 2),
        "latency_p95_s": round(sorted(latencies)[int(len(latencies) * 0.95)], 2),
        "latency_p99_s": round(sorted(latencies)[int(len(latencies) * 0.99)], 2),
        "latency_avg_s": round(statistics.mean(latencies), 2),
    }


async def run_benchmark(base_url: str, model: str, max_tokens: int = 128):
    """Run benchmark at multiple concurrency levels."""
    levels = [1, 2, 4, 8, 16]
    requests_per_level = 32

    print(f"Benchmark: {model} @ {base_url}")
    print(f"Max tokens: {max_tokens}, requests per level: {requests_per_level}")
    print()
    print(f"{'Conc':>4} | {'QPS':>6} | {'Tok/s':>7} | {'P50':>6} | {'P95':>6} | {'P99':>6} | {'Errors':>6}")
    print("-" * 60)

    results = []
    for c in levels:
        r = await run_concurrency_level(base_url, model, c, requests_per_level, max_tokens)
        results.append(r)
        if "error" in r and isinstance(r["error"], str):
            print(f"{c:>4} | {'FAIL':>6}")
        else:
            print(
                f"{c:>4} | {r['qps']:>6} | {r['throughput_tok_s']:>7} | {r['latency_p50_s']:>5}s | {r['latency_p95_s']:>5}s | {r['latency_p99_s']:>5}s | {r['errors']:>6}"
            )

    print()
    return results


def main():
    parser = argparse.ArgumentParser(description="Concurrent throughput benchmark")
    parser.add_argument("base_url", help="vLLM API base URL (e.g., http://localhost:8000/v1)")
    parser.add_argument("--model", default=None, help="Model name (auto-detected if not set)")
    parser.add_argument("--max-tokens", type=int, default=128, help="Max tokens per response")
    parser.add_argument("--output", "-o", help="Save results JSON")
    args = parser.parse_args()

    # Auto-detect model
    model = args.model
    if not model:
        resp = httpx.get(f"{args.base_url}/models", timeout=10)
        models = resp.json().get("data", [])
        if models:
            model = models[0]["id"]
            print(f"Auto-detected model: {model}")
        else:
            print("Could not detect model. Use --model flag.")
            return

    results = asyncio.run(run_benchmark(args.base_url, model, args.max_tokens))

    if args.output:
        with open(args.output, "w") as f:
            json.dump({"model": model, "base_url": args.base_url, "results": results}, f, indent=2)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
