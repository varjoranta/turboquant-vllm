"""`_moe_module_types` must match the right MoE module class across vLLM versions.

vLLM 0.25 refactored `FusedMoE` from a class into a factory **function**, and moved
the expert-weight module (w13/w2, quant_method, _replace_quant_method) to
`RoutedExperts`. `isinstance(module, FusedMoE)` therefore raises `TypeError` on 0.25,
and matching the function would never find the MoE modules. These tests mock both
vLLM shapes so the matching logic is validated on CPU CI (no vLLM install).
"""

import sys
import types as _types


def _install_fake_fused_moe(monkeypatch, **attrs):
    """Put a fake `vllm.model_executor.layers.fused_moe` (and its parent packages)
    in sys.modules so `from vllm.model_executor.layers import fused_moe` resolves."""
    for pkg in ("vllm", "vllm.model_executor", "vllm.model_executor.layers"):
        m = _types.ModuleType(pkg)
        m.__path__ = []  # mark as a package
        monkeypatch.setitem(sys.modules, pkg, m)
    fm = _types.ModuleType("vllm.model_executor.layers.fused_moe")
    for k, v in attrs.items():
        setattr(fm, k, v)
    monkeypatch.setitem(sys.modules, "vllm.model_executor.layers.fused_moe", fm)
    sys.modules["vllm.model_executor.layers"].fused_moe = fm


def test_matches_routed_experts_class_not_fusedmoe_factory(monkeypatch):
    class RoutedExperts:  # 0.25: the expert-weight module is a class
        pass

    def FusedMoE(*a, **k):  # 0.25: FusedMoE is now a factory FUNCTION
        pass

    _install_fake_fused_moe(monkeypatch, FusedMoE=FusedMoE, RoutedExperts=RoutedExperts)
    from turboquant_vllm.weight_quant import _moe_module_types

    types = _moe_module_types()
    assert RoutedExperts in types
    assert FusedMoE not in types  # a function must never be an isinstance target


def test_matches_fusedmoe_class_on_old_vllm(monkeypatch):
    class FusedMoE:  # <= 0.20: FusedMoE is the module class, no RoutedExperts
        pass

    _install_fake_fused_moe(monkeypatch, FusedMoE=FusedMoE)
    from turboquant_vllm.weight_quant import _moe_module_types

    assert _moe_module_types() == (FusedMoE,)


def test_empty_without_vllm(monkeypatch):
    monkeypatch.setitem(sys.modules, "vllm.model_executor.layers.fused_moe", None)
    from turboquant_vllm.weight_quant import _moe_module_types

    assert _moe_module_types() == ()
