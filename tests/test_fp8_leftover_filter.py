"""Regression test for issue #39 follow-up: TQ3-native loader must strip
FP8 leftover scale tensors when loading a checkpoint that was re-quantized
from FP8 to TQ3.

Bug: `fno2010/MiniMax-M2.7-TQ3` started life as FP8-block-quantized, then was
re-quantized to TQ3. The TQ3 packing replaced the FP8 weights with `.weight.tq_packed`
+ `.weight.tq_norms`, but the FP8 metadata (`.weight_scale_inv`) is still in
the safetensors index. Our loader rewrote the TQ3 pair → bf16 `.weight` and
let the scale tensor through unchanged. vLLM's MiniMaxM2 loader then saw the
scale tensor, inferred "FP8 fused MoE", tried to bind `experts.w13_weight_scale_inv`,
and crashed with `KeyError`.

Fix: drop tensors matching `_FP8_LEFTOVER_SCALE_SUFFIXES` in TQ3-native mode.
"""

import pytest

from turboquant_vllm.vllm_quant import (
    _FP8_LEFTOVER_SCALE_SUFFIXES,
    _is_fp8_leftover_scale,
)


@pytest.mark.parametrize(
    "name",
    [
        "model.layers.0.block_sparse_moe.experts.0.w1.weight_scale_inv",
        "model.layers.0.block_sparse_moe.experts.5.w3.weight_scale_inv",
        "model.layers.0.self_attn.q_proj.weight_scale",
        "model.layers.5.mlp.gate_proj.input_scale",
    ],
)
def test_predicate_matches_fp8_leftover_names(name):
    assert _is_fp8_leftover_scale(name), f"expected to drop {name}"


@pytest.mark.parametrize(
    "name",
    [
        # TQ3 tensors — the loader's primary input
        "model.layers.0.block_sparse_moe.experts.0.w1.weight.tq_packed",
        "model.layers.0.block_sparse_moe.experts.0.w1.weight.tq_norms",
        # Decompressed bf16 weights
        "model.layers.0.block_sparse_moe.experts.0.w1.weight",
        "model.layers.0.self_attn.q_proj.weight",
        # Layer norms and embeddings — never quantized
        "model.norm.weight",
        "model.layers.0.input_layernorm.weight",
        "model.embed_tokens.weight",
        "lm_head.weight",
        # Defensive: names that contain "scale" but aren't the FP8 patterns
        "model.layers.0.scale_factor",
    ],
)
def test_predicate_rejects_legitimate_names(name):
    assert not _is_fp8_leftover_scale(name), f"unexpectedly dropped {name}"


def test_known_suffixes_present():
    """If new FP8 metadata patterns surface in the wild, add them here."""
    assert ".weight_scale_inv" in _FP8_LEFTOVER_SCALE_SUFFIXES
    assert ".weight_scale" in _FP8_LEFTOVER_SCALE_SUFFIXES
    assert ".input_scale" in _FP8_LEFTOVER_SCALE_SUFFIXES
