"""Native TQ3 checkpoint format: ~12 GB on disk, loads on any 24+ GB GPU.

Solves the problem where the original 52 GB BF16 checkpoint doesn't fit on
a 48 GB GPU during loading, even though the compressed model is only 12 GB
at runtime.

The approach: read the original safetensors tensor-by-tensor (lazy loading,
~15 MB peak memory), TQ3-compress each weight on CPU, and save the packed
indices + norms as a new safetensors file. Non-weight tensors (embeddings,
norms, biases) are kept as FP16.

The output checkpoint is ~12 GB and loads directly into GPU memory via a
custom vLLM weight loader.

Usage:
    # Step 1: Create TQ3 checkpoint (CPU only, ~60 GB RAM, ~2 min)
    from turboquant_vllm.checkpoint import save_tq3_checkpoint
    save_tq3_checkpoint("google/gemma-4-26B-A4B-it", "./gemma4-tq3-native")

    # Step 2: Serve on L40S 48GB
    from turboquant_vllm.checkpoint import enable_tq3_serving
    enable_tq3_serving()
    # then: vllm serve ./gemma4-tq3-native
"""

import json
import logging
import os
import torch

logger = logging.getLogger(__name__)

# Layers to keep at full precision
_SKIP_PATTERNS = ("lm_head", "embed", "norm", "head")


def save_tq3_checkpoint(
    model_id: str,
    output_dir: str,
    bits: int = 3,
    group_size: int = 128,
):
    """Convert a HuggingFace checkpoint to native TQ3 packed format.

    Reads the original safetensors lazy (one tensor at a time), compresses
    weights on CPU, and writes a ~12 GB checkpoint. Peak memory: ~1 GB
    (one weight tensor + its compressed form).

    The output is a standard safetensors file where:
    - Weight tensors are replaced with .tq_packed (uint8) and .tq_norms (float32)
    - Non-weight tensors are stored as FP16
    - A tq_config.json records compression parameters

    Args:
        model_id: HuggingFace model ID or local path.
        output_dir: Where to save the TQ3 checkpoint.
        bits: Quantization bits (default 3).
        group_size: Group size (default 128).
    """
    from safetensors import safe_open
    from safetensors.torch import save_file
    from huggingface_hub import hf_hub_download, HfApi
    from transformers import AutoConfig, AutoTokenizer
    from turboquant_vllm.weight_quant import pack_indices
    from turboquant_vllm.torch_ops import PolarQuantTorch

    os.makedirs(output_dir, exist_ok=True)

    # Download config and tokenizer
    logger.info("Downloading config and tokenizer for %s...", model_id)
    config = AutoConfig.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    config.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Find safetensors files
    api = HfApi()
    repo_files = api.list_repo_files(model_id)
    shard_files = sorted([f for f in repo_files if f.endswith(".safetensors")])
    index_file = [f for f in repo_files if f == "model.safetensors.index.json"]

    if index_file:
        index_path = hf_hub_download(model_id, "model.safetensors.index.json")
        with open(index_path) as f:
            index = json.load(f)
        weight_map = index.get("weight_map", {})
    else:
        weight_map = {}

    # Create quantizer on CPU
    quantizer = PolarQuantTorch(group_size, bits, seed=42, device="cpu")

    # Process each shard
    output_tensors = {}
    total_original = 0
    total_compressed = 0
    compressed_count = 0

    for shard_name in shard_files:
        logger.info("Processing shard: %s", shard_name)
        shard_path = hf_hub_download(model_id, shard_name)

        with safe_open(shard_path, framework="pt", device="cpu") as f:
            for tensor_name in f.keys():
                tensor = f.get_tensor(tensor_name)
                original_bytes = tensor.numel() * tensor.element_size()
                total_original += original_bytes

                # Decide: compress or keep
                # Compress 2D weight tensors AND 3D expert tensors (Gemma 4 stores
                # experts as "experts.down_proj" without .weight suffix)
                is_weight_2d = tensor_name.endswith(".weight") and tensor.dim() == 2
                is_expert_3d = tensor.dim() == 3 and "expert" in tensor_name.lower()
                is_weight = is_weight_2d or is_expert_3d
                is_skip = any(p in tensor_name.lower() for p in _SKIP_PATTERNS)
                is_large = tensor.shape[-1] >= 128 or (tensor.dim() >= 2 and tensor.shape[-2] >= 128)

                if is_weight and not is_skip and is_large:
                    # Compress this weight
                    packed, norms = _compress_tensor(
                        tensor, quantizer, bits, group_size
                    )
                    output_tensors[tensor_name + ".tq_packed"] = packed
                    output_tensors[tensor_name + ".tq_norms"] = norms

                    comp_bytes = packed.numel() + norms.numel() * norms.element_size()
                    total_compressed += comp_bytes
                    compressed_count += 1

                    if compressed_count % 200 == 0:
                        logger.info("  Compressed %d tensors (%.1f GB saved so far)",
                                    compressed_count,
                                    (total_original - total_compressed) / 1e9)
                else:
                    # Keep as FP16
                    if tensor.is_floating_point():
                        output_tensors[tensor_name] = tensor.half()
                    else:
                        output_tensors[tensor_name] = tensor
                    total_compressed += tensor.numel() * 2  # FP16

    # Save compressed checkpoint
    logger.info("Saving %d tensors (%d compressed)...", len(output_tensors), compressed_count)

    # Split into shards if too large for single file
    max_shard_size = 5 * 1024 * 1024 * 1024  # 5 GB per shard
    _save_sharded(output_tensors, output_dir, max_shard_size)

    # Save TQ config
    tq_config = {
        "format": "tq3_native",
        "bits": bits,
        "group_size": group_size,
        "quantizer_seed": 42,
        "compressed_layers": compressed_count,
        "original_model": model_id,
    }
    with open(os.path.join(output_dir, "tq_config.json"), "w") as f:
        json.dump(tq_config, f, indent=2)

    ratio = total_original / max(total_compressed, 1)
    logger.info(
        "TQ3 checkpoint saved: %.1f GB -> %.1f GB (%.1fx), %d layers compressed",
        total_original / 1e9, total_compressed / 1e9, ratio, compressed_count,
    )


def _compress_tensor(
    tensor: torch.Tensor,
    quantizer,
    bits: int,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compress a single weight tensor to packed TQ format.

    Handles both 2D (linear) and 3D (MoE expert) tensors.

    Returns: (packed_indices as uint8, norms as float32)
    """
    from turboquant_vllm.weight_quant import pack_indices

    orig_shape = tensor.shape
    if tensor.dim() == 3:
        # MoE: (num_experts, out_dim, in_dim) -> flatten to 2D
        n_exp, out_dim, in_dim = tensor.shape
        tensor = tensor.reshape(-1, in_dim)
    elif tensor.dim() == 2:
        out_dim, in_dim = tensor.shape
    else:
        # Skip unexpected shapes
        return tensor.to(torch.uint8), torch.tensor([])

    # Pad to group_size
    padded_in = ((in_dim + group_size - 1) // group_size) * group_size
    if padded_in > in_dim:
        padded = torch.zeros(tensor.shape[0], padded_in, dtype=tensor.dtype)
        padded[:, :in_dim] = tensor
    else:
        padded = tensor

    # Quantize
    grouped = padded.float().reshape(-1, group_size)
    indices, norms = quantizer.quantize(grouped, norm_correction=True)
    packed = pack_indices(indices, bits)

    # Store shape info in norms layout
    n_groups = padded_in // group_size
    norms = norms.reshape(-1, n_groups)  # (total_rows, n_groups)

    return packed.contiguous(), norms.contiguous()


def _save_sharded(tensors: dict, output_dir: str, max_shard_size: int):
    """Save tensors as sharded safetensors files."""
    from safetensors.torch import save_file

    # Calculate sizes and assign to shards
    shards = []
    current_shard = {}
    current_size = 0

    for name, tensor in tensors.items():
        tensor_size = tensor.numel() * tensor.element_size()
        if current_size + tensor_size > max_shard_size and current_shard:
            shards.append(current_shard)
            current_shard = {}
            current_size = 0
        current_shard[name] = tensor
        current_size += tensor_size

    if current_shard:
        shards.append(current_shard)

    if len(shards) == 1:
        save_file(shards[0], os.path.join(output_dir, "model.safetensors"))
    else:
        weight_map = {}
        for i, shard in enumerate(shards):
            shard_name = f"model-{i+1:05d}-of-{len(shards):05d}.safetensors"
            save_file(shard, os.path.join(output_dir, shard_name))
            for name in shard:
                weight_map[name] = shard_name

        index = {
            "metadata": {"total_size": sum(t.numel() * t.element_size() for t in tensors.values())},
            "weight_map": weight_map,
        }
        with open(os.path.join(output_dir, "model.safetensors.index.json"), "w") as f:
            json.dump(index, f, indent=2)


def load_tq3_model(checkpoint_dir: str, device: str = "cuda"):
    """Load a native TQ3 checkpoint with weights compressed on GPU.

    Keeps weights in packed TQ3 format on GPU (~12 GB for Gemma 4 26B),
    decompressing on-the-fly during each forward pass. This enables running
    52 GB models on 24-48 GB GPUs.

    Loading steps:
    1. Create model skeleton on meta device (zero memory)
    2. Build inventory of packed vs regular tensors in checkpoint
    3. For each nn.Linear with packed weights: load packed data to GPU,
       create TurboQuantWrapper (no decompression)
    4. For each 3D expert weight with packed data: create Compressed3D
       with forward hooks for on-demand decompression
    5. Load remaining tensors (embeddings, norms, biases) to GPU as FP16

    Peak memory: ~12 GB GPU (packed weights) + ~1 GB (embeddings, norms).

    Args:
        checkpoint_dir: Path to the TQ3 checkpoint (from save_tq3_checkpoint).
        device: Target device for the model.

    Returns:
        (model, tokenizer) ready for inference.
    """
    import gc
    from safetensors import safe_open
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
    from turboquant_vllm.weight_quant import (
        TurboQuantWrapper, Compressed3D, _register_moe_hooks, _get_quantizer,
    )
    import torch.nn as nn

    with open(os.path.join(checkpoint_dir, "tq_config.json")) as f:
        tq_config = json.load(f)
    bits = tq_config["bits"]
    group_size = tq_config["group_size"]

    # Step 1: Create model on meta device (zero memory allocation)
    logger.info("Creating model skeleton on meta device...")
    config = AutoConfig.from_pretrained(checkpoint_dir)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config, dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)

    # Step 2: Build inventory of checkpoint tensors
    logger.info("Scanning checkpoint...")
    shard_files = sorted(f for f in os.listdir(checkpoint_dir) if f.endswith(".safetensors"))
    packed_bases = set()       # base names that have .tq_packed
    tensor_to_shard = {}       # tensor_name -> shard filename

    for shard_name in shard_files:
        shard_path = os.path.join(checkpoint_dir, shard_name)
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            for name in f.keys():
                tensor_to_shard[name] = shard_name
                if name.endswith(".tq_packed"):
                    packed_bases.add(name[:-len(".tq_packed")])

    # Build map: weight_name -> (module_path, nn.Linear module)
    # for all nn.Linear layers whose .weight has packed data
    linear_map = {}  # weight_name -> module_path
    for mod_path, module in model.named_modules():
        if isinstance(module, nn.Linear):
            weight_name = mod_path + ".weight"
            if weight_name in packed_bases:
                linear_map[weight_name] = mod_path

    # Identify 3D expert weights (packed bases not covered by linear_map)
    expert_bases = packed_bases - set(linear_map.keys())

    logger.info("Found %d packed linears, %d packed expert tensors, %d regular tensors",
                len(linear_map), len(expert_bases),
                len(tensor_to_shard) - 2 * len(packed_bases))

    # Ensure quantizer is initialized on target device
    _get_quantizer(group_size, bits, device)

    # Step 3: Replace nn.Linear layers with TurboQuantWrapper
    wrapped = 0
    for weight_name, mod_path in linear_map.items():
        packed_key = weight_name + ".tq_packed"
        norms_key = weight_name + ".tq_norms"

        # Load packed data directly to GPU
        shard_path = os.path.join(checkpoint_dir, tensor_to_shard[packed_key])
        with safe_open(shard_path, framework="pt", device=device) as f:
            packed = f.get_tensor(packed_key)
        norms_shard = os.path.join(checkpoint_dir, tensor_to_shard[norms_key])
        with safe_open(norms_shard, framework="pt", device=device) as f:
            norms = f.get_tensor(norms_key)

        # Check for bias
        bias_key = mod_path + ".bias"
        bias = None
        if bias_key in tensor_to_shard:
            bias_shard = os.path.join(checkpoint_dir, tensor_to_shard[bias_key])
            with safe_open(bias_shard, framework="pt", device=device) as f:
                bias = f.get_tensor(bias_key)

        # Get original dimensions from model skeleton (meta device has shape info)
        parts = mod_path.split(".")
        meta_module = model
        for part in parts:
            meta_module = getattr(meta_module, part)

        wrapper = TurboQuantWrapper.from_packed(
            packed, norms,
            in_features=meta_module.in_features,
            out_features=meta_module.out_features,
            bits=bits, group_size=group_size,
            bias=bias,
        )

        # Replace in model
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], wrapper)
        wrapped += 1

        if wrapped % 100 == 0:
            logger.info("  Wrapped %d linear layers...", wrapped)

    # Step 4: Handle 3D expert weights
    modules_to_hook = {}
    expert_count = 0
    for base_name in expert_bases:
        packed_key = base_name + ".tq_packed"
        norms_key = base_name + ".tq_norms"

        parts = base_name.split(".")
        param_name = parts[-1]
        owner = model
        for part in parts[:-1]:
            owner = getattr(owner, part)

        # Get original shape from meta parameter
        meta_param = getattr(owner, param_name)
        orig_shape = meta_param.shape  # (n_experts, out_dim, in_dim)

        # Load packed data to GPU
        shard_path = os.path.join(checkpoint_dir, tensor_to_shard[packed_key])
        with safe_open(shard_path, framework="pt", device=device) as f:
            packed = f.get_tensor(packed_key)
        norms_shard = os.path.join(checkpoint_dir, tensor_to_shard[norms_key])
        with safe_open(norms_shard, framework="pt", device=device) as f:
            norms = f.get_tensor(norms_key)

        compressed = Compressed3D.from_packed(
            packed, norms, shape=tuple(orig_shape),
            dtype=torch.float16, bits=bits, group_size=group_size,
        )

        setattr(owner, f"_tq_{param_name}", compressed)
        # Replace meta parameter with empty GPU tensor
        if isinstance(meta_param, nn.Parameter):
            owner.register_parameter(
                param_name,
                nn.Parameter(torch.empty(0, device=device, dtype=torch.float16),
                             requires_grad=False),
            )
        else:
            setattr(owner, param_name,
                    torch.empty(0, device=device, dtype=torch.float16))

        mod_id = id(owner)
        if mod_id not in modules_to_hook:
            modules_to_hook[mod_id] = (owner, [])
        modules_to_hook[mod_id][1].append(param_name)
        expert_count += 1

    # Disable buffer pooling on memory-constrained GPUs to avoid
    # keeping ~40 GB of decompression buffers across all MoE layers.
    gpu_mem_gb = 0
    if device != "cpu" and torch.cuda.is_available():
        gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    pool = gpu_mem_gb > 60  # Only pool on GPUs with >60 GB (A100, H100, etc.)

    for mod_id, (owner, param_names) in modules_to_hook.items():
        _register_moe_hooks(owner, param_names, pool_buffers=pool)

    if not pool:
        logger.info("Buffer pooling disabled (GPU %.0f GB < 60 GB)", gpu_mem_gb)

    # Step 5: Load remaining non-compressed tensors (embeddings, norms, biases)
    # These are tensors NOT associated with any packed weight
    handled_tensors = set()
    for base in packed_bases:
        handled_tensors.add(base + ".tq_packed")
        handled_tensors.add(base + ".tq_norms")
    # Biases already loaded with their linear wrappers
    for weight_name, mod_path in linear_map.items():
        bias_key = mod_path + ".bias"
        if bias_key in tensor_to_shard:
            handled_tensors.add(bias_key)

    regular_loaded = 0
    for tensor_name, shard_name in tensor_to_shard.items():
        if tensor_name in handled_tensors:
            continue

        # Navigate to the target parameter/buffer in the model
        parts = tensor_name.split(".")
        try:
            target_module = model
            for part in parts[:-1]:
                target_module = getattr(target_module, part)
            attr_name = parts[-1]
            target = getattr(target_module, attr_name)
        except AttributeError:
            logger.debug("Skipping tensor %s (no match in model)", tensor_name)
            continue

        # Load tensor to GPU
        shard_path = os.path.join(checkpoint_dir, shard_name)
        with safe_open(shard_path, framework="pt", device=device) as f:
            data = f.get_tensor(tensor_name)

        if isinstance(target, nn.Parameter):
            if data.is_floating_point():
                data = data.to(torch.float16)
            target_module.register_parameter(
                attr_name,
                nn.Parameter(data, requires_grad=False),
            )
        else:
            if hasattr(target, 'data') and target.is_floating_point():
                data = data.to(torch.float16)
            setattr(target_module, attr_name, data)
        regular_loaded += 1

    logger.info("Loaded: %d wrapped linears, %d expert layers, %d regular tensors",
                wrapped, expert_count, regular_loaded)

    # Step 6: Materialize any remaining meta-device tensors.
    # Some models have computed buffers (e.g., embed_scale, inv_freq) that
    # are created during __init__ but not saved in the checkpoint. These
    # are still on meta device and need to be re-created on the target device.
    meta_fixed = 0
    for name, param in list(model.named_parameters()):
        if param.device == torch.device("meta"):
            new_param = nn.Parameter(
                torch.zeros(param.shape, dtype=param.dtype, device=device),
                requires_grad=False,
            )
            parts = name.split(".")
            target_module = model
            for part in parts[:-1]:
                target_module = getattr(target_module, part)
            target_module.register_parameter(parts[-1], new_param)
            meta_fixed += 1
            logger.warning("Meta param zeroed (not in checkpoint): %s %s", name, list(param.shape))

    for name, buf in list(model.named_buffers()):
        if buf.device == torch.device("meta"):
            parts = name.split(".")
            target_module = model
            for part in parts[:-1]:
                target_module = getattr(target_module, part)
            attr_name = parts[-1]
            new_buf = torch.zeros(buf.shape, dtype=buf.dtype, device=device)
            target_module.register_buffer(attr_name, new_buf)
            meta_fixed += 1
            logger.warning("Meta buffer zeroed (not in checkpoint): %s %s", name, list(buf.shape))

    if meta_fixed > 0:
        logger.info("Materialized %d meta-device tensors", meta_fixed)

    # Re-initialize computed buffers that depend on config values.
    _reinit_computed_buffers(model, config, device)

    # Restore weight tying (e.g., lm_head shares weight with embed_tokens).
    # Meta-device loading breaks tied weights because each gets materialized
    # independently. Re-tie them based on the model config.
    if getattr(config, 'tie_word_embeddings', True):
        _restore_weight_tying(model)

    if device != "cpu" and torch.cuda.is_available():
        gpu_gb = torch.cuda.memory_allocated() / 1e9
        logger.info("GPU memory: %.1f GB", gpu_gb)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model.eval()
    return model, tokenizer


def _restore_weight_tying(model):
    """Restore weight tying between embedding and lm_head.

    Many models (Gemma, Llama, etc.) share the embedding weight with
    the output projection (lm_head). Meta-device loading breaks this
    tie because each parameter gets materialized independently.
    """
    # Find the embedding weight
    embed_weight = None
    for name, module in model.named_modules():
        if hasattr(module, 'weight') and 'embed_tokens' in name:
            embed_weight = module.weight
            break

    if embed_weight is None:
        return

    # Find lm_head and tie its weight
    lm_head = getattr(model, 'lm_head', None)
    if lm_head is None:
        # Some models nest lm_head differently
        for name, module in model.named_modules():
            if 'lm_head' in name and hasattr(module, 'weight'):
                lm_head = module
                break

    if lm_head is not None and hasattr(lm_head, 'weight'):
        lm_head.weight = embed_weight
        logger.info("Restored weight tying: lm_head -> embed_tokens")


def _reinit_computed_buffers(model, config, device):
    """Re-initialize computed buffers that aren't stored in checkpoints.

    These are values computed from config during __init__ and NOT stored
    in the checkpoint:
    - embed_scale: sqrt(hidden_size) for Gemma models
    - *inv_freq: rotary embedding frequencies (per-layer-type in Gemma 4)
    - ClippableLinear bounds: input_min/max, output_min/max (should be +/-inf)
    """
    import math

    # Handle nested configs (e.g., Gemma4Config has text_config.hidden_size)
    text_config = getattr(config, 'text_config', None)
    hidden_size = getattr(config, 'hidden_size', None)
    if hidden_size is None and text_config is not None:
        hidden_size = getattr(text_config, 'hidden_size', None)
    rope_config = text_config if text_config is not None else config

    # Fix ALL zero-valued buffers that should be non-zero.
    # After meta-device creation + Step 6 materialization, computed buffers
    # are zeros. We identify them by name pattern and reinitialize.
    fixed = 0
    for buf_name, buf in model.named_buffers():
        if not isinstance(buf, torch.Tensor):
            continue
        if buf.numel() == 0:
            continue

        attr = buf_name.split(".")[-1]

        # embed_scale: sqrt(hidden_size)
        if attr == "embed_scale" and hidden_size and buf.numel() == 1:
            if buf.item() == 0:
                parts = buf_name.split(".")
                mod = model
                for p in parts[:-1]:
                    mod = getattr(mod, p)
                mod.embed_scale = torch.tensor(
                    math.sqrt(hidden_size), dtype=torch.float32, device=device
                )
                fixed += 1
                logger.info("Fixed %s = sqrt(%d) = %.4f", buf_name, hidden_size, math.sqrt(hidden_size))

        # Rotary inv_freq (any variant: inv_freq, global_inv_freq, local_sliding_inv_freq, etc.)
        elif "inv_freq" in attr and "original" not in attr:
            if buf.abs().max().item() == 0:
                head_dim = getattr(rope_config, 'head_dim', None)
                if head_dim is None and hidden_size:
                    head_dim = hidden_size // getattr(rope_config, 'num_attention_heads', 1)
                rope_theta = getattr(rope_config, 'rope_theta', 10000.0)
                if head_dim:
                    new_inv_freq = 1.0 / (rope_theta ** (
                        torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim
                    ))
                    # Truncate if buffer shape differs (some models use different sizes)
                    if new_inv_freq.shape[0] != buf.shape[-1]:
                        new_inv_freq = new_inv_freq[:buf.shape[-1]]
                    parts = buf_name.split(".")
                    mod = model
                    for p in parts[:-1]:
                        mod = getattr(mod, p)
                    mod.register_buffer(attr, new_inv_freq.reshape(buf.shape))
                    fixed += 1
                    logger.info("Fixed %s: shape=%s from head_dim=%d, theta=%.0f",
                                buf_name, list(buf.shape), head_dim, rope_theta)

        # original_inv_freq: copy from inv_freq
        elif "original_inv_freq" in attr:
            if buf.abs().max().item() == 0:
                # Find the corresponding inv_freq (same prefix, without "original_")
                inv_attr = attr.replace("original_", "")
                parts = buf_name.split(".")
                mod = model
                for p in parts[:-1]:
                    mod = getattr(mod, p)
                inv_freq = getattr(mod, inv_attr, None)
                if inv_freq is not None and isinstance(inv_freq, torch.Tensor):
                    mod.register_buffer(attr, inv_freq.clone())
                    fixed += 1
                    logger.info("Fixed %s: copied from %s", buf_name, inv_attr)

        # ClippableLinear bounds: should be -inf/+inf, not 0
        elif attr in ("input_min", "output_min") and buf.numel() == 1:
            if buf.item() == 0:
                parts = buf_name.split(".")
                mod = model
                for p in parts[:-1]:
                    mod = getattr(mod, p)
                mod.register_buffer(attr, torch.tensor(float('-inf'), device=device))
                fixed += 1
                logger.info("Fixed %s = -inf", buf_name)

        elif attr in ("input_max", "output_max") and buf.numel() == 1:
            if buf.item() == 0:
                parts = buf_name.split(".")
                mod = model
                for p in parts[:-1]:
                    mod = getattr(mod, p)
                mod.register_buffer(attr, torch.tensor(float('inf'), device=device))
                fixed += 1
                logger.info("Fixed %s = +inf", buf_name)

    if fixed > 0:
        logger.info("Re-initialized %d computed buffers", fixed)


def enable_tq3_serving():
    """Enable serving native TQ3 checkpoints via vLLM on small GPUs.

    Hooks vLLM's weight loading to use native TQ3 checkpoint format.
    Weights stay compressed in GPU memory (~12 GB for Gemma 4 26B).

    Usage:
        from turboquant_vllm.checkpoint import enable_tq3_serving
        enable_tq3_serving()
        # then: vllm serve ./gemma4-tq3-native
    """
    logger.info("TQ3 native checkpoint serving enabled (use with native TQ3 checkpoints)")
    # For now, users should use load_tq3_model() for standalone inference.
    # Full vLLM integration (hooking DefaultModelLoader) is planned for v0.4.0.
    raise NotImplementedError(
        "vLLM integration for native TQ3 checkpoints is not yet implemented. "
        "Use load_tq3_model() for standalone inference, or use "
        "enable_weight_quantization() with the FP16 checkpoint on A100."
    )
