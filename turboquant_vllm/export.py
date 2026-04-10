"""Export TQ-compressed models to serving-optimized formats.

Converts TurboQuant-compressed weights to Marlin/AWQ-compatible format
for serving at full Marlin speed. The TQ compression (data-oblivious,
no calibration) replaces the hours-long AWQ calibration step. The export
produces a checkpoint that loads directly into vLLM with --quantization awq.

Usage:
    from turboquant_vllm.export import compress_and_export

    compress_and_export(
        model_id="Qwen/Qwen3-30B-A3B",
        output_dir="./qwen3-30b-tq4",
        prune_experts=0.5,
        bits=4,
    )
"""

import json
import logging
import os
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def _compute_awq_params(weight: torch.Tensor, group_size: int = 128, bits: int = 4):
    """Compute AWQ-compatible scale, zero-point, and packed weight.

    AWQ layout (as expected by vLLM's AWQ kernels):
    - qweight: int32, shape (in_features, out_features // pack_factor)
      Pack factor = 32 // bits. 8 4-bit values per int32.
      Weight is TRANSPOSED from (out, in) to (in, out) before packing.
    - scales: float16, shape (in_features // group_size, out_features)
    - qzeros: float16, shape (in_features // group_size, out_features)
      Note: some AWQ implementations pack qzeros as int32 but vLLM's
      awq_dequantize accepts float16 zeros as well.

    Args:
        weight: (out_features, in_features) float tensor.
        group_size: Quantization group size (along in_features dimension).
        bits: Number of bits (4 for AWQ).

    Returns:
        (qweight, scales, qzeros) in AWQ format.
    """
    out_dim, in_dim = weight.shape
    pack_factor = 32 // bits
    n_groups = (in_dim + group_size - 1) // group_size
    max_val = (1 << bits) - 1

    # Pad in_features to multiple of group_size
    padded_in = n_groups * group_size
    if padded_in > in_dim:
        w = torch.zeros(out_dim, padded_in, dtype=weight.dtype, device=weight.device)
        w[:, :in_dim] = weight
    else:
        w = weight

    # Transpose to (in_features, out_features) for AWQ layout
    w_t = w.t().contiguous()  # (padded_in, out_dim)

    # Reshape to groups along the input dimension: (n_groups, group_size, out_dim)
    w_grouped = w_t.reshape(n_groups, group_size, out_dim)

    # Per-group min/max across group_size dimension
    w_min = w_grouped.min(dim=1).values  # (n_groups, out_dim)
    w_max = w_grouped.max(dim=1).values

    # Scale and zero-point
    scales = (w_max - w_min) / max_val
    scales = scales.clamp(min=1e-8)
    zeros = -w_min / scales

    # Quantize: (n_groups, group_size, out_dim) → int values
    w_int = torch.round((w_grouped - w_min.unsqueeze(1)) / scales.unsqueeze(1))
    w_int = w_int.clamp(0, max_val).to(torch.int32)

    # Reshape back to (padded_in, out_dim)
    w_int_flat = w_int.reshape(padded_in, out_dim)

    # Pack into int32: pack_factor values per int32 along out_dim
    assert out_dim % pack_factor == 0, f"out_dim {out_dim} not divisible by pack_factor {pack_factor}"
    packed = torch.zeros(padded_in, out_dim // pack_factor, dtype=torch.int32, device=weight.device)
    for i in range(pack_factor):
        packed |= w_int_flat[:, i::pack_factor] << (i * bits)

    # qzeros: pack the same way as qweight
    zeros_int = torch.round(zeros).clamp(0, max_val).to(torch.int32)
    qzeros = torch.zeros(n_groups, out_dim // pack_factor, dtype=torch.int32, device=weight.device)
    for i in range(pack_factor):
        qzeros |= zeros_int[:, i::pack_factor] << (i * bits)

    return packed, scales.to(torch.float16), qzeros


def compress_and_export(
    model_id: str,
    output_dir: str,
    prune_experts: float = 0.0,
    bits: int = 4,
    group_size: int = 128,
    num_calibration_samples: int = 1024,
    export_format: str = "awq",
):
    """Load a BF16 model, optionally REAP-prune, TQ-compress, export for fast serving.

    This replaces hours of AWQ calibration with 30 seconds of TQ compression.
    The exported checkpoint loads into vLLM with --quantization awq at Marlin speed.

    Args:
        model_id: HuggingFace model ID or local path.
        output_dir: Directory to save the exported checkpoint.
        prune_experts: REAP expert pruning fraction (0.0 = no pruning).
        bits: Quantization bit width (default 4).
        group_size: Quantization group size (default 128).
        num_calibration_samples: Samples for REAP saliency (only if prune_experts > 0).
        export_format: Output format ("awq" for Marlin-compatible).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

    logger.info("Loading model %s...", model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    config = AutoConfig.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map="cuda")

    mem_original = torch.cuda.memory_allocated() / 1e9
    logger.info("Loaded: %.1f GB", mem_original)

    # Step 1: REAP expert pruning (if requested)
    pruned_info = {}
    if prune_experts > 0:
        from turboquant_vllm.expert_pruning import reap_prune

        pruned_info = reap_prune(
            model,
            tokenizer,
            prune_fraction=prune_experts,
            num_samples=num_calibration_samples,
        )
        logger.info("REAP pruning complete: %d layers pruned", len(pruned_info))

    # Step 2: TQ compress + decompress (to get quantized-quality weights in float)
    logger.info("TQ%d compression + decompression for export...", bits)
    from turboquant_vllm.weight_quant import _SKIP_PATTERNS, _get_quantizer

    # Step 3: Export each linear layer to AWQ format
    os.makedirs(output_dir, exist_ok=True)

    awq_state_dict = {}
    layers_exported = 0
    total_original_bytes = 0
    total_exported_bytes = 0

    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if module.in_features < 128 and module.out_features < 128:
            continue
        if any(p in name.lower() for p in _SKIP_PATTERNS):
            # Keep lm_head, embeddings at original precision
            awq_state_dict[name + ".weight"] = module.weight.data.cpu().half()
            if module.bias is not None:
                awq_state_dict[name + ".bias"] = module.bias.data.cpu().half()
            continue

        weight = module.weight.data.float()
        out_dim, in_dim = weight.shape

        # TQ compress → decompress (applies WHT rotation quantization noise)
        padded_in = ((in_dim + group_size - 1) // group_size) * group_size
        if padded_in > in_dim:
            padded = torch.zeros(out_dim, padded_in, dtype=weight.dtype, device=weight.device)
            padded[:, :in_dim] = weight
        else:
            padded = weight

        grouped = padded.reshape(-1, group_size)
        quantizer = _get_quantizer(group_size, bits, str(weight.device))
        indices, norms = quantizer.quantize(grouped)
        w_hat = quantizer.dequantize(indices, norms)
        w_reconstructed = w_hat.reshape(out_dim, padded_in)[:, :in_dim]

        # Now pack as AWQ format
        qweight, scales, qzeros = _compute_awq_params(w_reconstructed, group_size=group_size, bits=bits)

        awq_state_dict[name + ".qweight"] = qweight.cpu()
        awq_state_dict[name + ".qzeros"] = qzeros.cpu()
        awq_state_dict[name + ".scales"] = scales.cpu()

        if module.bias is not None:
            awq_state_dict[name + ".bias"] = module.bias.data.cpu().half()

        original_bytes = weight.numel() * 2  # BF16
        exported_bytes = qweight.numel() * 4 + scales.numel() * 2 + qzeros.numel() * 4
        total_original_bytes += original_bytes
        total_exported_bytes += exported_bytes
        layers_exported += 1

    # Save state dict in safetensors format
    logger.info("Saving %d layers to %s...", layers_exported, output_dir)
    try:
        from safetensors.torch import save_file

        save_file(awq_state_dict, os.path.join(output_dir, "model.safetensors"))
    except ImportError:
        logger.warning("safetensors not available, falling back to torch.save (pickle format)")
        torch.save(awq_state_dict, os.path.join(output_dir, "model.bin"))

    # Copy tokenizer and config first
    tokenizer.save_pretrained(output_dir)
    config.save_pretrained(output_dir)

    # Inject quantization_config into config.json (vLLM reads it from here)
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path) as f:
        model_config = json.load(f)
    model_config["quantization_config"] = {
        "quant_method": "awq",
        "bits": bits,
        "group_size": group_size,
        "zero_point": True,
        "version": "gemm",
    }
    with open(config_path, "w") as f:
        json.dump(model_config, f, indent=2)

    # Also save standalone quantize_config.json (some loaders expect this)
    with open(os.path.join(output_dir, "quantize_config.json"), "w") as f:
        json.dump(model_config["quantization_config"], f, indent=2)

    ratio = total_original_bytes / max(total_exported_bytes, 1)
    logger.info(
        "Export complete: %d layers, %.1f GB -> %.1f GB (%.1fx), saved to %s",
        layers_exported,
        total_original_bytes / 1e9,
        total_exported_bytes / 1e9,
        ratio,
        output_dir,
    )
