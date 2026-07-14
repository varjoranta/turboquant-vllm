"""TurboQuantFusedMoEMethod.apply must absorb + forward new vLLM apply kwargs.

vLLM 0.25 added `shared_experts=` (and `shared_experts_input=`) to the call
`RoutedExperts.forward_modular` makes into `quant_method.apply(...)`. If the
plugin's apply doesn't accept the new keyword, the engine crashes at forward
with `TypeError: apply() got an unexpected keyword argument 'shared_experts'`.
Signature-level regression guard; runs on CPU CI (no vLLM install).
"""

import inspect

import pytest

moe_quant = pytest.importorskip("turboquant_vllm.moe_quant")


def _has_var_keyword(fn) -> bool:
    return any(p.kind is inspect.Parameter.VAR_KEYWORD for p in inspect.signature(fn).parameters.values())


def test_apply_absorbs_extra_kwargs():
    method = getattr(moe_quant, "TurboQuantFusedMoEMethod", None)
    if method is None:
        pytest.skip("TurboQuantFusedMoEMethod unavailable without vLLM")
    assert _has_var_keyword(method.apply), (
        "apply() must accept **kwargs so vLLM version additions (e.g. shared_experts) don't crash the engine"
    )


def test_apply_monolithic_absorbs_extra_kwargs():
    method = getattr(moe_quant, "TurboQuantFusedMoEMethod", None)
    if method is None:
        pytest.skip("TurboQuantFusedMoEMethod unavailable without vLLM")
    assert _has_var_keyword(method.apply_monolithic)
