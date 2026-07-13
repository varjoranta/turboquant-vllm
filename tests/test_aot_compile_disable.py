"""The plugin must default vLLM's AOT compile off.

vLLM 0.25 (torch >= 2.10) turns ``VLLM_USE_AOT_COMPILE`` on by default. Its AOT
path loads a compiled forward and binds the model's parameters by name —
including ``weight``. This plugin replaces each Linear's ``.weight`` with
compressed buffers, so that binding raises ``KeyError: 'weight'`` at engine
init (dense TQ models under vLLM 0.25, e.g. Qwen3-8B). Regression guard for the
fix in ``_vllm_plugin._disable_incompatible_aot_compile``; runs on CPU CI
(no vLLM install required).
"""

import os

from turboquant_vllm._vllm_plugin import _disable_incompatible_aot_compile


def test_defaults_aot_compile_off(monkeypatch):
    monkeypatch.delenv("VLLM_USE_AOT_COMPILE", raising=False)
    _disable_incompatible_aot_compile()
    assert os.environ["VLLM_USE_AOT_COMPILE"] == "0"


def test_respects_explicit_user_override(monkeypatch):
    # setdefault must not clobber a user who deliberately opted into AOT.
    monkeypatch.setenv("VLLM_USE_AOT_COMPILE", "1")
    _disable_incompatible_aot_compile()
    assert os.environ["VLLM_USE_AOT_COMPILE"] == "1"


def test_register_sets_it(monkeypatch):
    # The public entry point vLLM calls must also disable AOT (it runs in every
    # process, including the engine-core subprocess that compiles).
    monkeypatch.delenv("VLLM_USE_AOT_COMPILE", raising=False)
    from turboquant_vllm._vllm_plugin import register

    register()
    assert os.environ["VLLM_USE_AOT_COMPILE"] == "0"
