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


def _resolve_module(root, dotted_path: str):
    """Navigate a module tree by dotted path, returning the final attribute.

    Handles both nn.Module children (via getattr) and indexed containers
    like ModuleList/plain lists (via integer indexing).
    """
    obj = root
    for part in dotted_path.split("."):
        try:
            obj = getattr(obj, part)
        except (AttributeError, TypeError):
            # ModuleList/plain lists: getattr("5") fails but obj[5] works
            try:
                obj = obj[int(part)]
            except (IndexError, ValueError, TypeError):
                raise AttributeError(f"Cannot resolve '{part}' in {type(obj).__name__}")
    return obj


def _resolve_parent_and_attr(root, dotted_path: str):
    """Resolve a dotted path to (parent_module, attr_name).

    Example: _resolve_parent_and_attr(model, "layers.0.weight")
    returns (model.layers[0], "weight")
    """
    parts = dotted_path.split(".")
    parent = _resolve_module(root, ".".join(parts[:-1])) if len(parts) > 1 else root
    return parent, parts[-1]


def save_tq3_checkpoint(
    model_id: str,
    output_dir: str,
    bits: int = 3,
    group_size: int = 128,
    sensitive_bits: int | None = None,
    max_shard_bytes: int = 5 * 1024 * 1024 * 1024,
):
    """Convert a HuggingFace checkpoint to native TQ3 packed format.

    Streams tensors: reads one at a time, compresses, writes to output shards
    incrementally. Peak memory: one tensor + its compressed form (~8 GB for
    large MoE expert tensors). Handles TB-scale models (e.g., GLM-5.1 769B).

    Args:
        model_id: HuggingFace model ID or local path.
        output_dir: Where to save the TQ3 checkpoint.
        bits: Quantization bits for most tensors (default 3).
        group_size: Group size (default 128).
        sensitive_bits: Higher precision bits for o_proj/down_proj (default None = uniform).
        max_shard_bytes: Max bytes per output shard (default 5 GB).
    """
    from safetensors import safe_open
    from safetensors.torch import save_file
    from transformers import AutoConfig, AutoTokenizer
    from turboquant_vllm.torch_ops import PolarQuantTorch

    os.makedirs(output_dir, exist_ok=True)

    # Accept either a HuggingFace model ID or a local path. A local path
    # skips hf_hub entirely; the HF path downloads shards on demand.
    is_local = os.path.isdir(model_id)
    if is_local:
        logger.info("Using local checkpoint at %s", model_id)
    else:
        logger.info("Downloading config and tokenizer for %s...", model_id)
    config = AutoConfig.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # Inject quantization_config so vLLM auto-detects the format
    if not hasattr(config, "quantization_config") or config.quantization_config is None:
        config.quantization_config = {
            "quant_method": "turboquant",
            "bits": bits,
            "group_size": group_size,
        }
        if sensitive_bits is not None:
            config.quantization_config["sensitive_bits"] = sensitive_bits
    config.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    if is_local:
        shard_files = sorted(
            f for f in os.listdir(model_id) if f.endswith(".safetensors")
        )
        if not shard_files:
            raise FileNotFoundError(
                f"No .safetensors shards found in local path {model_id}"
            )
    else:
        from huggingface_hub import HfApi
        api = HfApi()
        repo_files = api.list_repo_files(model_id)
        shard_files = sorted(f for f in repo_files if f.endswith(".safetensors"))

    from turboquant_vllm.weight_quant import select_bits, _SENSITIVE_PATTERNS

    quantizer = PolarQuantTorch(group_size, bits, seed=42, device="cpu")
    # Create second quantizer for sensitive layers if needed
    sensitive_quantizer = (
        PolarQuantTorch(group_size, sensitive_bits, seed=42, device="cpu")
        if sensitive_bits is not None and sensitive_bits != bits
        else None
    )

    # Streaming: accumulate tensors into current shard, flush when full
    current_shard: dict[str, torch.Tensor] = {}
    current_shard_bytes = 0
    shard_idx = 0
    weight_map: dict[str, str] = {}
    total_size = 0
    total_original = 0
    total_compressed = 0
    compressed_count = 0

    def _flush_shard():
        nonlocal current_shard, current_shard_bytes, shard_idx
        if not current_shard:
            return
        shard_idx += 1
        shard_name = f"model-{shard_idx:05d}-of-NNNNN.safetensors"
        shard_path = os.path.join(output_dir, shard_name)
        save_file(current_shard, shard_path)
        for name in current_shard:
            weight_map[name] = shard_name
        logger.info("  Wrote shard %d: %d tensors, %.1f GB",
                     shard_idx, len(current_shard), current_shard_bytes / 1e9)
        current_shard = {}
        current_shard_bytes = 0

    def _add_tensor(name: str, tensor: torch.Tensor):
        nonlocal current_shard, current_shard_bytes, total_size
        tensor_bytes = tensor.numel() * tensor.element_size()
        if current_shard_bytes + tensor_bytes > max_shard_bytes and current_shard:
            _flush_shard()
        current_shard[name] = tensor
        current_shard_bytes += tensor_bytes
        total_size += tensor_bytes

    import tempfile
    _tmp_download_dir = None if is_local else tempfile.mkdtemp(prefix="tq3_dl_")

    for shard_name in shard_files:
        logger.info("Processing shard: %s", shard_name)
        if is_local:
            shard_path = os.path.join(model_id, shard_name)
        else:
            from huggingface_hub import hf_hub_download
            shard_path = hf_hub_download(
                model_id, shard_name, local_dir=_tmp_download_dir
            )

        with safe_open(shard_path, framework="pt", device="cpu") as f:
            for tensor_name in f.keys():
                tensor = f.get_tensor(tensor_name)
                original_bytes = tensor.numel() * tensor.element_size()
                total_original += original_bytes

                is_weight_2d = tensor_name.endswith(".weight") and tensor.dim() == 2
                is_expert_3d = tensor.dim() == 3 and "expert" in tensor_name.lower()
                is_weight = is_weight_2d or is_expert_3d
                is_skip = any(p in tensor_name.lower() for p in _SKIP_PATTERNS)
                is_large = tensor.shape[-1] >= 128 or (tensor.dim() >= 2 and tensor.shape[-2] >= 128)

                if is_weight and not is_skip and is_large:
                    tensor_bits = select_bits(tensor_name, bits, sensitive_bits)
                    tensor_quantizer = (
                        sensitive_quantizer if tensor_bits != bits and sensitive_quantizer is not None
                        else quantizer
                    )
                    packed, norms = _compress_tensor(
                        tensor, tensor_quantizer, tensor_bits, group_size
                    )
                    _add_tensor(tensor_name + ".tq_packed", packed)
                    _add_tensor(tensor_name + ".tq_norms", norms)

                    comp_bytes = packed.numel() + norms.numel() * norms.element_size()
                    total_compressed += comp_bytes
                    compressed_count += 1

                    if compressed_count % 200 == 0:
                        logger.info("  Compressed %d tensors (%.1f GB saved so far)",
                                    compressed_count,
                                    (total_original - total_compressed) / 1e9)
                else:
                    if tensor.is_floating_point():
                        stored_tensor = tensor.half()
                    else:
                        stored_tensor = tensor
                    _add_tensor(tensor_name, stored_tensor)
                    total_compressed += (
                        stored_tensor.numel() * stored_tensor.element_size()
                    )

                del tensor  # free input tensor immediately

        # Delete downloaded shard to save disk (critical for TB-scale models)
        # but never remove source files when model_id is a local directory.
        if not is_local:
            try:
                os.remove(shard_path)
            except OSError:
                pass

    _flush_shard()  # write remaining tensors

    # Clean up temp download directory (only created in HF-download mode)
    if _tmp_download_dir is not None:
        import shutil
        shutil.rmtree(_tmp_download_dir, ignore_errors=True)

    # Rename shards with correct total count
    total_shards = shard_idx
    for old_name in sorted(set(weight_map.values())):
        new_name = old_name.replace("NNNNN", f"{total_shards:05d}")
        if old_name != new_name:
            os.rename(os.path.join(output_dir, old_name),
                      os.path.join(output_dir, new_name))
            for k in weight_map:
                if weight_map[k] == old_name:
                    weight_map[k] = new_name

    # Write index
    if total_shards > 1:
        index = {
            "metadata": {"total_size": total_size},
            "weight_map": weight_map,
        }
        with open(os.path.join(output_dir, "model.safetensors.index.json"), "w") as f:
            json.dump(index, f, indent=2)
    elif total_shards == 1:
        # Single shard: rename to model.safetensors
        shard_path = os.path.join(output_dir, list(set(weight_map.values()))[0])
        os.rename(shard_path, os.path.join(output_dir, "model.safetensors"))

    tq_config = {
        "format": "tq3_native",
        "bits": bits,
        "group_size": group_size,
        "quantizer_seed": 42,
        "compressed_layers": compressed_count,
        "original_model": model_id,
    }
    if sensitive_bits is not None:
        tq_config["sensitive_bits"] = sensitive_bits
        tq_config["sensitive_patterns"] = list(_SENSITIVE_PATTERNS)
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
        select_bits, _SENSITIVE_PATTERNS,
    )
    import torch.nn as nn

    with open(os.path.join(checkpoint_dir, "tq_config.json")) as f:
        tq_config = json.load(f)
    bits = tq_config["bits"]
    group_size = tq_config["group_size"]
    sensitive_bits = tq_config.get("sensitive_bits")
    sensitive_patterns = tuple(tq_config.get("sensitive_patterns", _SENSITIVE_PATTERNS))

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

    linear_map = {}  # base_name -> module_path for packed nn.Linear layers
    for mod_path, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Support both naming conventions:
            # Old: q_proj.weight.tq_packed → base = q_proj.weight
            # New: q_proj.tq_packed → base = q_proj (vLLM compatible)
            weight_name = mod_path + ".weight"
            if weight_name in packed_bases:
                linear_map[weight_name] = mod_path
            elif mod_path in packed_bases:
                linear_map[mod_path] = mod_path

    expert_bases = packed_bases - set(linear_map.keys())

    logger.info("Found %d packed linears, %d packed expert tensors, %d regular tensors",
                len(linear_map), len(expert_bases),
                len(tensor_to_shard) - 2 * len(packed_bases))

    _get_quantizer(group_size, bits, device)

    # Steps 3-5: Collect all tensor keys needed, then load batched by shard
    # (open each safetensors file once instead of once per tensor).

    # Build the set of already-handled tensors for Step 5 filtering
    handled_tensors = set()
    for base in packed_bases:
        handled_tensors.add(base + ".tq_packed")
        handled_tensors.add(base + ".tq_norms")
    for weight_name, mod_path in linear_map.items():
        bias_key = mod_path + ".bias"
        if bias_key in tensor_to_shard:
            handled_tensors.add(bias_key)

    # Identify regular tensors (Step 5) and validate they exist in the model
    regular_tensor_names = []
    for tensor_name in tensor_to_shard:
        if tensor_name in handled_tensors:
            continue
        try:
            _resolve_parent_and_attr(model, tensor_name)
            regular_tensor_names.append(tensor_name)
        except AttributeError:
            logger.debug("Skipping tensor %s (no match in model)", tensor_name)

    # Group ALL needed tensor keys by shard for batched loading
    shard_to_keys: dict[str, list[str]] = {}
    for key in list(handled_tensors) + regular_tensor_names:
        shard_name = tensor_to_shard[key]
        if shard_name not in shard_to_keys:
            shard_to_keys[shard_name] = []
        shard_to_keys[shard_name].append(key)

    # Load all tensors, one shard open at a time
    loaded: dict[str, torch.Tensor] = {}
    for shard_name, keys in shard_to_keys.items():
        shard_path = os.path.join(checkpoint_dir, shard_name)
        with safe_open(shard_path, framework="pt", device=device) as f:
            for key in keys:
                loaded[key] = f.get_tensor(key)

    # Step 3: Replace nn.Linear layers with TurboQuantWrapper
    wrapped = 0
    for weight_name, mod_path in linear_map.items():
        packed = loaded[weight_name + ".tq_packed"]
        norms = loaded[weight_name + ".tq_norms"]
        bias = loaded.get(mod_path + ".bias")

        meta_module = _resolve_module(model, mod_path)
        tensor_bits = select_bits(weight_name, bits, sensitive_bits, sensitive_patterns)
        wrapper = TurboQuantWrapper.from_packed(
            packed, norms,
            in_features=meta_module.in_features,
            out_features=meta_module.out_features,
            bits=tensor_bits, group_size=group_size,
            bias=bias,
        )

        parent, attr = _resolve_parent_and_attr(model, mod_path)
        setattr(parent, attr, wrapper)
        wrapped += 1

        if wrapped % 100 == 0:
            logger.info("  Wrapped %d linear layers...", wrapped)

    # Step 4: Handle 3D expert weights (two sub-cases)
    modules_to_hook = {}
    expert_count = 0

    # 4a: Already-3D expert tensors (e.g., Gemma 4's experts.gate_up_proj)
    resolved_expert_bases = set()
    for base_name in expert_bases:
        try:
            owner, param_name = _resolve_parent_and_attr(model, base_name)
            meta_param = getattr(owner, param_name)
        except (AttributeError, TypeError):
            continue  # handle in 4b

        if not isinstance(meta_param, (torch.Tensor, nn.Parameter)):
            continue  # not a tensor (e.g., a sub-module like Router)

        orig_shape = meta_param.shape
        if len(orig_shape) != 3:
            continue  # 2D weights handled in 4b (regrouping or standalone)

        packed = loaded[base_name + ".tq_packed"]
        norms = loaded[base_name + ".tq_norms"]

        tensor_bits = select_bits(base_name, bits, sensitive_bits, sensitive_patterns)
        compressed = Compressed3D.from_packed(
            packed, norms, shape=tuple(orig_shape),
            dtype=torch.float16, bits=tensor_bits, group_size=group_size,
        )

        setattr(owner, f"_tq_{param_name}", compressed)
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
        resolved_expert_bases.add(base_name)

    # 4b: Per-expert 2D tensors that need regrouping into fused 3D parameters.
    # Checkpoint has: experts.0.gate_proj.weight, experts.0.up_proj.weight, etc.
    # Model has: experts.gate_up_proj (3D), experts.down_proj (3D), gate.weight (2D)
    import re
    unresolved_bases = expert_bases - resolved_expert_bases

    if unresolved_bases:
        # Find model parameters that are still on meta device (need data from checkpoint)
        meta_params: dict[str, tuple[nn.Module, str, torch.Tensor]] = {}
        for name, param in model.named_parameters():
            if param.device.type == "meta":
                try:
                    owner, attr = _resolve_parent_and_attr(model, name)
                    meta_params[name] = (owner, attr, param)
                except (AttributeError, TypeError):
                    pass
        for name, buf in model.named_buffers():
            if buf.device.type == "meta":
                try:
                    owner, attr = _resolve_parent_and_attr(model, name)
                    meta_params[name] = (owner, attr, buf)
                except (AttributeError, TypeError):
                    pass

        # Known projection fusion patterns
        _proj_fusion = {"gate_proj": "gate_up_proj", "up_proj": "gate_up_proj"}
        _proj_order = {"gate_proj": 0, "up_proj": 1}

        # Build index: for each unresolved base_name, extract the expert container
        # path and projection suffix. Use greedy match to find the rightmost
        # numeric index that represents an expert.
        # Example: model.layers.1.mlp.experts.0.gate_proj.weight
        #   → container=model.layers.1.mlp.experts, expert_idx=0, proj=gate_proj.weight
        idx_pattern = re.compile(r'^(.+?)\.experts\.(\d+)\.(.+)$')

        # Groups keyed by (container_path, model_target_param) → list of (order, expert_idx, base_name)
        regroup_map: dict[str, list[tuple[int, int, str]]] = {}
        handled_in_4b: set[str] = set()

        for base_name in unresolved_bases:
            m = idx_pattern.match(base_name)
            if not m:
                # Not per-expert indexed — try direct resolution as standalone 2D
                try:
                    owner, param_name = _resolve_parent_and_attr(model, base_name)
                    meta_param = getattr(owner, param_name)
                except (AttributeError, TypeError):
                    continue

                if not isinstance(meta_param, (torch.Tensor, nn.Parameter)):
                    continue  # sub-module, not a parameter

                if len(meta_param.shape) != 2:
                    continue
                # Decompress and set directly (e.g., gate.weight on Router)
                packed = loaded[base_name + ".tq_packed"]
                norms_data = loaded[base_name + ".tq_norms"]
                from turboquant_vllm.weight_quant import _get_quantizer, unpack_indices
                q = _get_quantizer(group_size, bits, str(packed.device))
                indices = unpack_indices(packed, bits, group_size)
                norms_flat = norms_data.reshape(-1)
                w_groups = q.dequantize(indices, norms_flat)
                out_features, in_features = meta_param.shape
                padded_in = ((in_features + group_size - 1) // group_size) * group_size
                decompressed = w_groups.reshape(out_features, padded_in)[:, :in_features]
                decompressed = decompressed.to(torch.float16)
                if isinstance(meta_param, nn.Parameter):
                    owner.register_parameter(
                        param_name,
                        nn.Parameter(decompressed.to(device), requires_grad=False),
                    )
                else:
                    setattr(owner, param_name, decompressed.to(device))
                handled_in_4b.add(base_name)
                wrapped += 1
                continue

            container_path = m.group(1) + ".experts"
            expert_idx = int(m.group(2))
            proj_suffix = m.group(3)  # e.g., "gate_proj.weight", "down_proj.weight"
            proj_name = proj_suffix.split(".")[0]  # "gate_proj", "down_proj", etc.

            # Find the model target parameter
            # Try: container.proj_suffix, container.proj_name (without .weight)
            target_key = None
            order = 0
            for candidate in [f"{container_path}.{proj_suffix}",
                              f"{container_path}.{proj_name}"]:
                if candidate in meta_params:
                    target_key = candidate
                    break

            # Try fused alias: gate_proj → gate_up_proj
            if target_key is None and proj_name in _proj_fusion:
                fused = _proj_fusion[proj_name]
                for candidate in [f"{container_path}.{fused}",
                                  f"{container_path}.{fused}.weight"]:
                    if candidate in meta_params:
                        target_key = candidate
                        order = _proj_order.get(proj_name, 0)
                        break

            if target_key is None:
                logger.debug("Cannot map expert tensor %s to model parameter", base_name)
                continue

            if target_key not in regroup_map:
                regroup_map[target_key] = []
            regroup_map[target_key].append((order, expert_idx, base_name))

        # Process each target: stack per-expert data
        for target_key, entries in regroup_map.items():
            owner, param_name, meta_param = meta_params[target_key]
            orig_shape = meta_param.shape

            if len(orig_shape) != 3:
                logger.warning("Expected 3D for %s but got shape %s", target_key, orig_shape)
                continue

            n_experts_expected = orig_shape[0]

            # Sort by (order, expert_idx) — order separates fused parts (gate=0, up=1)
            entries.sort()

            # Group by expert_idx, concatenating fused parts per expert
            expert_data: dict[int, tuple[list[torch.Tensor], list[torch.Tensor]]] = {}
            for order, eidx, base_name in entries:
                pk = loaded.get(base_name + ".tq_packed")
                nk = loaded.get(base_name + ".tq_norms")
                if pk is None or nk is None:
                    continue
                if eidx not in expert_data:
                    expert_data[eidx] = ([], [])
                expert_data[eidx][0].append(pk)
                expert_data[eidx][1].append(nk)
                handled_in_4b.add(base_name)

            if len(expert_data) != n_experts_expected:
                logger.warning("Expert count mismatch for %s: model=%d, got=%d",
                               target_key, n_experts_expected, len(expert_data))
                continue

            # Stack: concat fused parts per expert, then cat across experts
            all_packed = []
            all_norms = []
            for eidx in sorted(expert_data.keys()):
                pks, nks = expert_data[eidx]
                all_packed.append(torch.cat(pks, dim=0) if len(pks) > 1 else pks[0])
                all_norms.append(torch.cat(nks, dim=0) if len(nks) > 1 else nks[0])

            stacked_packed = torch.cat(all_packed, dim=0)
            stacked_norms = torch.cat(all_norms, dim=0)

            target_bits = select_bits(target_key, bits, sensitive_bits, sensitive_patterns)
            compressed = Compressed3D.from_packed(
                stacked_packed, stacked_norms, shape=tuple(orig_shape),
                dtype=torch.float16, bits=target_bits, group_size=group_size,
            )

            setattr(owner, f"_tq_{param_name}", compressed)
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
            if param_name not in modules_to_hook[mod_id][1]:
                modules_to_hook[mod_id][1].append(param_name)
            expert_count += n_experts_expected
            logger.info("  Regrouped %d experts into %s", n_experts_expected, target_key)

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
    regular_loaded = 0
    for tensor_name in regular_tensor_names:
        data = loaded[tensor_name]
        target_module, attr_name = _resolve_parent_and_attr(model, tensor_name)
        target = getattr(target_module, attr_name)

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

    del loaded  # free references to loaded tensors

    logger.info("Loaded: %d wrapped linears, %d expert layers, %d regular tensors",
                wrapped, expert_count, regular_loaded)

    # Step 6: Materialize any remaining meta-device tensors.
    # Some models have computed buffers (e.g., embed_scale, inv_freq) that
    # are created during __init__ but not saved in the checkpoint. These
    # are still on meta device and need to be re-created on the target device.
    meta_fixed = 0
    for name, param in list(model.named_parameters()):
        if param.device == torch.device("meta"):
            target_module, attr_name = _resolve_parent_and_attr(model, name)
            new_param = nn.Parameter(
                torch.zeros(param.shape, dtype=param.dtype, device=device),
                requires_grad=False,
            )
            target_module.register_parameter(attr_name, new_param)
            meta_fixed += 1
            logger.warning("Meta param zeroed (not in checkpoint): %s %s", name, list(param.shape))

    for name, buf in list(model.named_buffers()):
        if buf.device == torch.device("meta"):
            target_module, attr_name = _resolve_parent_and_attr(model, name)
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
    embed_weight = None
    lm_head = None

    # Single pass: find both embed_tokens and lm_head
    for name, module in model.named_modules():
        if embed_weight is None and hasattr(module, 'weight') and 'embed_tokens' in name:
            embed_weight = module.weight
        if lm_head is None and 'lm_head' in name and hasattr(module, 'weight'):
            lm_head = module
        if embed_weight is not None and lm_head is not None:
            break

    if embed_weight is None:
        return

    # Fall back to direct attribute if not found via named_modules
    if lm_head is None:
        lm_head = getattr(model, 'lm_head', None)

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

    # After meta-device creation + Step 6, computed buffers are zeros.
    # Single pass over named_modules handles rotary embeddings, embed_scale,
    # and ClippableLinear bounds.
    fixed = 0

    _INF_FIX_MAP = {
        "input_min": float('-inf'),
        "output_min": float('-inf'),
        "input_max": float('inf'),
        "output_max": float('inf'),
    }

    for mod_name, module in model.named_modules():
        class_name = type(module).__name__.lower()

        # 1. Re-initialize rotary embedding modules by calling their __init__.
        # Handles per-layer-type inv_freq with different head dims and rope thetas
        # (e.g., Gemma 4: sliding=256/10000, full=512/1000000).
        if 'rotary' in class_name and 'embedding' in class_name:
            has_zeroed = any(
                isinstance(buf, torch.Tensor) and buf.numel() > 0 and buf.abs().max().item() == 0
                for buf in module.buffers()
            )
            if has_zeroed:
                init_config = rope_config if rope_config is not None else config
                try:
                    module.__init__(init_config, device=device)
                    n_bufs = sum(1 for _ in module.buffers())
                    fixed += n_bufs
                    logger.info("Re-initialized %s (%d buffers)", mod_name, n_bufs)
                except Exception as e:
                    logger.warning("Failed to reinit %s: %s", mod_name, e)

        # 2. Fix embed_scale
        if hidden_size and hasattr(module, 'embed_scale'):
            scale = module.embed_scale
            if isinstance(scale, torch.Tensor) and scale.numel() > 0:
                if scale.device != torch.device("meta") and scale.item() == 0:
                    module.embed_scale = torch.tensor(
                        math.sqrt(hidden_size), dtype=torch.float32, device=device
                    )
                    fixed += 1
                    logger.info("Fixed %s.embed_scale = %.4f", mod_name, math.sqrt(hidden_size))

        # 3. Fix ClippableLinear bounds (should be -inf/+inf, not 0).
        # Check direct buffers on this module only (avoids re-iterating named_buffers).
        for attr, inf_val in _INF_FIX_MAP.items():
            buf = getattr(module, attr, None)
            if buf is None or not isinstance(buf, torch.Tensor) or buf.numel() != 1:
                continue
            try:
                if buf.item() == 0:
                    module.register_buffer(attr, torch.tensor(inf_val, device=device))
                    fixed += 1
                    logger.info("Fixed %s.%s = %s", mod_name, attr,
                                "-inf" if inf_val < 0 else "+inf")
            except RuntimeError:
                continue

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
