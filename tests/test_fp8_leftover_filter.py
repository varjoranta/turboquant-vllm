"""TQ3-native loader strips FP8 leftover scale tensors (issue #39 follow-up)."""

import pytest

from turboquant_vllm.vllm_quant import _FP8_LEFTOVER_SCALE_SUFFIXES


@pytest.mark.parametrize(
    "name",
    [
        "model.layers.0.block_sparse_moe.experts.0.w1.weight_scale_inv",
        "model.layers.0.self_attn.q_proj.weight_scale",
        "model.layers.5.mlp.gate_proj.input_scale",
    ],
)
def test_predicate_matches_fp8_leftover_names(name):
    assert name.endswith(_FP8_LEFTOVER_SCALE_SUFFIXES)


@pytest.mark.parametrize(
    "name",
    [
        "model.layers.0.block_sparse_moe.experts.0.w1.weight.tq_packed",
        "model.layers.0.block_sparse_moe.experts.0.w1.weight.tq_norms",
        "model.layers.0.block_sparse_moe.experts.0.w1.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.norm.weight",
        "model.layers.0.input_layernorm.weight",
        "model.embed_tokens.weight",
        "lm_head.weight",
        "model.layers.0.scale_factor",
    ],
)
def test_predicate_rejects_legitimate_names(name):
    assert not name.endswith(_FP8_LEFTOVER_SCALE_SUFFIXES)
