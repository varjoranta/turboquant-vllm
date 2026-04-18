#!/usr/bin/env python3
"""Serve a TQ3 native checkpoint through mlx_lm's OpenAI-compatible server.

Monkey-patches ``mlx_lm.server.load`` so any path containing ``tq_config.json``
routes through ``turboquant_vllm.mlx_loader.load_tq3`` (v0.10.0 Metal GEMV path);
anything else falls back to the stock mlx_lm loader.

Also serializes requests with a single lock and caps MLX memory so the server
survives agent clients (e.g. opencode) that fire parallel requests with
multi-thousand-token system prompts on a 48 GiB Mac.

Usage:
    python examples/mac-serve-tq3.py --model ~/models/qwen3-coder-30b-a3b-tq3 \
        --port 8080
"""

from __future__ import annotations

import threading
from pathlib import Path

import mlx.core as mx
from mlx_lm import server as _server
from turboquant_vllm.mlx_loader import load_tq3

# Cap Metal allocations so a runaway prefill can't starve the rest of macOS.
# 14 GiB wired fits the ~11 GB pinned weight state + headroom; 22 GiB total
# caps attention scratch; relaxed=True lets excess spill to swap instead of
# aborting with kIOGPUCommandBufferCallbackErrorOutOfMemory.
for _call, _arg in (
    (getattr(mx, "set_wired_limit", None), 14 * 1024**3),
    (getattr(mx, "set_memory_limit", None), 22 * 1024**3),
    (getattr(mx, "set_cache_limit", None), 2 * 1024**3),
):
    if _call is not None:
        try:
            _call(_arg, relaxed=True) if _call is mx.set_memory_limit else _call(_arg)
        except (TypeError, ValueError):
            pass

# Serialize POST handlers. ThreadingHTTPServer dispatches each request on its
# own thread; two concurrent agent calls blow past the Metal command buffer
# budget on a 48 GiB Mac. Also clear the Metal cache between requests so
# attention scratch from a big prefill is released.
_inference_lock = threading.Lock()
_orig_do_POST = _server.APIHandler.do_POST


def _locked_do_POST(self):
    with _inference_lock:
        try:
            return _orig_do_POST(self)
        finally:
            if hasattr(mx, "clear_cache"):
                mx.clear_cache()


_server.APIHandler.do_POST = _locked_do_POST

_orig_load = _server.load  # type: ignore[attr-defined]


def _patched_load(path_or_hf_repo, *args, **kwargs):
    path = Path(path_or_hf_repo).expanduser()
    if path.is_dir() and (path / "tq_config.json").exists():
        return load_tq3(str(path), tokenizer_config=kwargs.get("tokenizer_config"))
    return _orig_load(path_or_hf_repo, *args, **kwargs)


_server.load = _patched_load  # type: ignore[attr-defined]


if __name__ == "__main__":
    _server.main()
