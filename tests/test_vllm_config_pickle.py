"""Regression test for issue #39: TurboQuantConfig must be picklable.

Bug: TurboQuantConfig was defined inside register() as a closure.
cloudpickle pickles closure-defined classes 'by value' — serializing the
class object plus its closure cells. The closure transitively references
torch.ops.turboquant.*, an _OpNamespace that pickle cannot handle, which
crashed every vLLM multi-worker startup with:

    TypeError: cannot pickle '_OpNamespace' object

Fix: TurboQuantConfig is now defined at module top level. cloudpickle
uses 'by reference' (qualified-name) pickling for top-level classes,
which avoids walking the closure and never touches torch.ops.

Two tests, one each side of the vLLM-availability fence:

* Source-level (always runs): the class definition isn't nested in a
  function. Catches anyone re-introducing a closure later, even on a
  Mac CI that can't install vLLM.
* cloudpickle round-trip (vLLM only): exercises the actual code path
  that crashed in production.
"""

import inspect
import re

import pytest

from turboquant_vllm import vllm_quant


def test_turboquant_config_defined_at_module_level():
    """The class definition isn't nested in a function (regression for #39)."""
    source = inspect.getsource(vllm_quant)
    matches = re.findall(r"^([ \t]*)class TurboQuantConfig\b", source, re.MULTILINE)
    assert matches, "TurboQuantConfig class definition not found"
    for indent in matches:
        # 0 = truly top-level; 4 = inside `if LinearBase is not None:` (still
        # module-level scope for cloudpickle). Anything deeper means a function.
        assert len(indent) <= 4, (
            f"TurboQuantConfig defined at indent={len(indent)}; "
            "must be at module scope so cloudpickle uses by-reference pickling. "
            "See issue #39."
        )


@pytest.mark.skipif(
    vllm_quant.TurboQuantConfig is None,
    reason="vLLM not installed; runtime pickle check requires the class to exist",
)
def test_turboquant_config_cloudpickle_roundtrip():
    """An instance survives cloudpickle — the path vLLM actually uses."""
    cloudpickle = pytest.importorskip("cloudpickle")
    cfg = vllm_quant.TurboQuantConfig(bits=4, group_size=64, sensitive_bits=3)
    blob = cloudpickle.dumps(cfg)
    restored = cloudpickle.loads(blob)
    assert restored.bits == 4
    assert restored.group_size == 64
    assert restored.sensitive_bits == 3
    assert isinstance(restored, vllm_quant.TurboQuantConfig)
