"""Load a TurboQuant (TQ3) native checkpoint through mlx_lm.

Replaces each ``nn.Linear`` whose packed form is on disk with a
``TurboQuantMLXLinear`` and each ``SwitchLinear`` with a
``TurboQuantMLXSwitchLinear``, then loads the remaining tensors
(embeddings, norms, biases, scales) through the standard
``model.load_weights`` path. Uses mlx_lm's architecture registry to
build the model skeleton, so any architecture mlx_lm supports works
without extra glue.

Entry points:
    load_tq3_model(path_or_hf_repo) -> (nn.Module, config_dict)
    load_tq3(path_or_hf_repo) -> (nn.Module, TokenizerWrapper)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from turboquant_vllm.mlx_model import TurboQuantMLXLinear, TurboQuantMLXSwitchLinear
from turboquant_vllm.mlx_ops import PolarQuantStateMLX
from turboquant_vllm.torch_ops import PolarQuantTorch

try:
    from mlx_lm.models.switch_layers import SwitchLinear

    _HAS_SWITCH_LINEAR = True
except ImportError:
    SwitchLinear = None  # type: ignore[assignment,misc]
    _HAS_SWITCH_LINEAR = False

logger = logging.getLogger(__name__)

# Layers that benefit from higher precision. Mirrors
# weight_quant._SENSITIVE_PATTERNS so sensitive_bits in tq_config
# routes the right layers to the higher-bit quantizer state.
_SENSITIVE_PATTERNS = ("o_proj", "down_proj")


def _build_state(group_size: int, bits: int, seed: int) -> PolarQuantStateMLX:
    """Recreate the PolarQuantStateMLX from the deterministic seed."""
    pq = PolarQuantTorch(dim=group_size, bit_width=bits, seed=seed, device="cpu")
    return PolarQuantStateMLX.from_torch_quantizer(pq)


def _set_by_path(root: nn.Module, dotted: str, new_module: nn.Module) -> None:
    """Navigate ``root.a.b.c`` (with numeric parts as list indices) and
    set the final attribute to ``new_module``."""
    parent: Any = root
    parts = dotted.split(".")
    for p in parts[:-1]:
        parent = parent[int(p)] if p.isdigit() else getattr(parent, p)
    setattr(parent, parts[-1], new_module)


def _replace_linears_with_tq(
    model: nn.Module,
    weights: dict[str, mx.array],
    default_state: PolarQuantStateMLX,
    sensitive_state: PolarQuantStateMLX | None,
) -> set[str]:
    """Replace each Linear whose packed form is present in ``weights``.

    Returns the set of weight keys consumed (so the caller can skip them
    when calling ``model.load_weights``).
    """
    consumed: set[str] = set()

    targets: list[tuple[str, nn.Linear]] = [(name, m) for name, m in model.named_modules() if isinstance(m, nn.Linear)]

    for name, module in targets:
        packed_key = f"{name}.weight.tq_packed"
        norms_key = f"{name}.weight.tq_norms"
        if packed_key not in weights:
            continue

        out_features, in_features = module.weight.shape

        use_sensitive = sensitive_state is not None and any(p in name for p in _SENSITIVE_PATTERNS)
        state = sensitive_state if use_sensitive else default_state

        bias_key = f"{name}.bias"
        bias = weights.get(bias_key)

        new_layer = TurboQuantMLXLinear(
            packed_weight=weights[packed_key],
            norms=weights[norms_key],
            state=state,
            in_features=in_features,
            out_features=out_features,
            bias=bias,
        )
        _set_by_path(model, name, new_layer)

        consumed.add(packed_key)
        consumed.add(norms_key)
        if bias is not None:
            consumed.add(bias_key)

    return consumed


def _replace_switch_linears_with_tq(
    model: nn.Module,
    weights: dict[str, mx.array],
    default_state: PolarQuantStateMLX,
    sensitive_state: PolarQuantStateMLX | None,
) -> set[str]:
    """Replace each SwitchLinear whose packed form is in ``weights``.

    SwitchLinear stores a 3D weight ``(num_experts, out, in)`` at ``.weight``;
    the compressed form follows the same naming as for Linear:
    ``{path}.weight.tq_packed`` + ``{path}.weight.tq_norms``.

    Returns the consumed weight keys.
    """
    consumed: set[str] = set()
    if not _HAS_SWITCH_LINEAR:
        return consumed

    targets = [(name, m) for name, m in model.named_modules() if isinstance(m, SwitchLinear)]

    for name, module in targets:
        packed_key = f"{name}.weight.tq_packed"
        norms_key = f"{name}.weight.tq_norms"
        if packed_key not in weights:
            continue

        num_experts, out_features, in_features = module.weight.shape

        use_sensitive = sensitive_state is not None and any(p in name for p in _SENSITIVE_PATTERNS)
        state = sensitive_state if use_sensitive else default_state

        bias_key = f"{name}.bias"
        bias = weights.get(bias_key)

        new_layer = TurboQuantMLXSwitchLinear(
            packed_weight=weights[packed_key],
            norms=weights[norms_key],
            state=state,
            in_features=in_features,
            out_features=out_features,
            num_experts=num_experts,
            bias=bias,
        )
        _set_by_path(model, name, new_layer)

        consumed.add(packed_key)
        consumed.add(norms_key)
        if bias is not None:
            consumed.add(bias_key)

    return consumed


def _load_safetensor_shards(model_path: Path) -> dict[str, mx.array]:
    weights: dict[str, mx.array] = {}
    shards = sorted(model_path.glob("model*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"No safetensors shards found in {model_path}")
    for shard in shards:
        weights.update(mx.load(str(shard)))
    return weights


def load_tq3_model(path_or_hf_repo: str) -> tuple[nn.Module, dict[str, Any]]:
    """Load a TQ3 native checkpoint and return (model, config)."""
    # Local import keeps the optional mlx_lm dependency hidden from the
    # module's top-level import path (the plugin ships without mlx_lm).
    from mlx_lm.utils import _download, _get_classes, load_config

    model_path = Path(_download(path_or_hf_repo))

    tq_config_path = model_path / "tq_config.json"
    if not tq_config_path.exists():
        raise ValueError(f"Not a TQ3 checkpoint (missing tq_config.json): {model_path}")

    with open(tq_config_path) as f:
        tq_config = json.load(f)

    fmt = tq_config.get("format")
    if fmt != "tq3_native":
        raise ValueError(f"Unsupported TQ format: {fmt!r}. Expected 'tq3_native'.")

    bits = tq_config["bits"]
    group_size = tq_config["group_size"]
    seed = tq_config.get("quantizer_seed", 42)
    sensitive_bits = tq_config.get("sensitive_bits")

    config = load_config(model_path)
    model_class, model_args_class = _get_classes(config=config)
    model_args = model_args_class.from_dict(config)
    model = model_class(model_args)

    weights = _load_safetensor_shards(model_path)
    if hasattr(model, "sanitize"):
        weights = model.sanitize(weights)

    default_state = _build_state(group_size, bits, seed)
    sensitive_state = (
        _build_state(group_size, sensitive_bits, seed)
        if sensitive_bits is not None and sensitive_bits != bits
        else None
    )

    consumed = _replace_linears_with_tq(model, weights, default_state, sensitive_state)
    consumed |= _replace_switch_linears_with_tq(model, weights, default_state, sensitive_state)

    # Remaining keys: uncompressed tensors (embeddings, norms, biases that
    # weren't part of a replacement, scales). Any leftover ``.tq_packed`` /
    # ``.tq_norms`` keys are packed tensors we couldn't match to a module —
    # typically HF-layout expert stacks where the checkpoint was compressed
    # before the mlx_lm ``sanitize`` split (e.g. Qwen3.5 ``experts.gate_up_proj``
    # → ``switch_mlp.gate_proj/up_proj``). Those need a checkpoint-side fix
    # and are skipped here with a warning.
    remaining: list[tuple[str, mx.array]] = []
    unresolved_packed = 0
    for key, value in weights.items():
        if key in consumed:
            continue
        if ".tq_packed" in key or ".tq_norms" in key:
            unresolved_packed += 1
            continue
        remaining.append((key, value))

    if unresolved_packed:
        logger.warning(
            "Skipped %d packed TQ tensor(s) not matched to a Linear or SwitchLinear; "
            "likely a pre-sanitize HF MoE layout that needs a split-before-compress fix.",
            unresolved_packed,
        )

    model.eval()
    model.load_weights(remaining, strict=False)
    mx.eval(model.parameters())

    return model, config


def load_tq3(
    path_or_hf_repo: str,
    tokenizer_config: dict[str, Any] | None = None,
) -> tuple[nn.Module, Any]:
    """Load model + tokenizer for a TQ3 checkpoint."""
    from mlx_lm.utils import _download, load_tokenizer

    model_path = Path(_download(path_or_hf_repo))
    model, config = load_tq3_model(path_or_hf_repo)
    tokenizer = load_tokenizer(
        str(model_path),
        tokenizer_config or {},
        eos_token_ids=config.get("eos_token_id"),
    )
    return model, tokenizer
