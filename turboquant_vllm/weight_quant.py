"""TurboQuant weight quantization for vLLM.

Online quantization: load any BF16/FP16 model, compress weights to TQ3/TQ4
at load time using WHT rotation + Lloyd-Max codebook. Weights stay compressed
in GPU memory, dequantized per forward pass.

Uses group quantization: each weight row is split into groups of group_size
(default 128), and each group gets its own L2 norm and rotation. This prevents
error accumulation across the full row dimension, matching how GPTQ/AWQ handle
weight quantization (per-group scales).

Inspired by @coffeecup2020's TQ3_1S llama.cpp implementation (dual half-block
scales) showing TurboQuant weight compression achieves near-Q4_0 quality.
"""

import logging
import torch
import torch.nn as nn

from turboquant_vllm.torch_ops import PolarQuantTorch

logger = logging.getLogger(__name__)

_quantizers: dict[tuple[int, int], PolarQuantTorch] = {}
_cuda_mod = None
_cuda_available = None


def _get_cuda_module():
    """Lazy-load the CUDA weight dequant kernel."""
    global _cuda_mod, _cuda_available
    if _cuda_available is not None:
        return _cuda_mod if _cuda_available else None
    try:
        from turboquant_vllm.build import build
        _cuda_mod = build()
        if hasattr(_cuda_mod, 'weight_dequant'):
            _cuda_available = True
            logger.info("CUDA weight dequant kernel loaded")
            return _cuda_mod
        else:
            _cuda_available = False
            logger.info("CUDA module missing weight_dequant, using PyTorch")
            return None
    except Exception as e:
        _cuda_available = False
        logger.warning("CUDA weight dequant not available: %s", e)
        return None


def _get_quantizer(group_size: int, bits: int, device: str) -> PolarQuantTorch:
    """Get or create a PolarQuant quantizer for a given group size and bit width."""
    key = (group_size, bits)
    if key not in _quantizers:
        _quantizers[key] = PolarQuantTorch(group_size, bits, seed=42, device=device)
    return _quantizers[key]


def pack_indices(indices: torch.Tensor, bits: int) -> torch.Tensor:
    """Pack quantization indices into uint8.

    4-bit: 2 per byte. 3-bit: 1 per byte (no sub-byte packing). 2-bit: 4 per byte.
    """
    if bits == 4:
        assert indices.shape[-1] % 2 == 0
        flat = indices.reshape(-1, indices.shape[-1])
        lo = flat[:, 0::2].to(torch.uint8)
        hi = flat[:, 1::2].to(torch.uint8)
        return (lo | (hi << 4)).reshape(indices.shape[0], -1)
    elif bits == 2:
        assert indices.shape[-1] % 4 == 0
        flat = indices.reshape(-1, indices.shape[-1])
        shifts = torch.tensor([0, 2, 4, 6], device=indices.device, dtype=torch.uint8)
        parts = torch.stack([flat[:, i::4].to(torch.uint8) for i in range(4)], dim=-1)
        return (parts << shifts).sum(dim=-1).to(torch.uint8).reshape(indices.shape[0], -1)
    else:
        return indices.to(torch.uint8)


def unpack_indices(packed: torch.Tensor, bits: int, dim: int) -> torch.Tensor:
    """Unpack uint8 packed indices back to int64."""
    if bits == 4:
        flat = packed.reshape(-1, packed.shape[-1])
        lo = (flat & 0x0F).to(torch.int64)
        hi = ((flat >> 4) & 0x0F).to(torch.int64)
        unpacked = torch.zeros(flat.shape[0], flat.shape[1] * 2, dtype=torch.int64, device=packed.device)
        unpacked[:, 0::2] = lo
        unpacked[:, 1::2] = hi
        return unpacked.reshape(packed.shape[0], -1)[:, :dim]
    elif bits == 2:
        flat = packed.reshape(-1, packed.shape[-1])
        unpacked = torch.zeros(flat.shape[0], flat.shape[1] * 4, dtype=torch.int64, device=packed.device)
        for i in range(4):
            unpacked[:, i::4] = ((flat >> (i * 2)) & 0x03).to(torch.int64)
        return unpacked.reshape(packed.shape[0], -1)[:, :dim]
    else:
        return packed.to(torch.int64)


class TurboQuantWrapper(nn.Module):
    """Drop-in replacement for nn.Linear with TQ group-quantized weights.

    Each weight row is split into groups of group_size. Each group is
    independently normalized, rotated (WHT), and quantized against the
    Lloyd-Max codebook. This matches how GPTQ/AWQ use per-group scales.
    """

    def __init__(self, original: nn.Linear, bits: int = 3, group_size: int = 128):
        super().__init__()
        self.bits = bits
        self.group_size = group_size
        self.in_features = original.in_features
        self.out_features = original.out_features

        weight = original.weight.data  # (out_features, in_features)
        out_dim, in_dim = weight.shape

        # Pad in_features to multiple of group_size
        self.padded_in = ((in_dim + group_size - 1) // group_size) * group_size
        self.n_groups = self.padded_in // group_size

        if self.padded_in > in_dim:
            padded = torch.zeros(out_dim, self.padded_in, dtype=weight.dtype, device=weight.device)
            padded[:, :in_dim] = weight
        else:
            padded = weight

        # Reshape to (out_dim * n_groups, group_size) for batch quantization
        grouped = padded.reshape(-1, group_size)

        quantizer = _get_quantizer(group_size, bits, str(weight.device))
        indices, norms = quantizer.quantize(grouped)
        packed = pack_indices(indices, bits)

        # norms shape: (out_dim * n_groups,) -> store as (out_dim, n_groups)
        norms = norms.reshape(out_dim, self.n_groups)

        # Memory stats
        original_bytes = weight.numel() * weight.element_size()
        compressed_bytes = packed.numel() + norms.numel() * 4
        self._ratio = original_bytes / compressed_bytes

        self.register_buffer("packed_weight", packed)
        self.register_buffer("norms", norms)

        if original.bias is not None:
            self.bias = nn.Parameter(original.bias.data.clone())
        else:
            self.bias = None

        logger.debug(
            "TQ%d-g%d compressed %dx%d (%.1fx)",
            bits, group_size, out_dim, in_dim, self._ratio,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cuda_mod = _get_cuda_module()

        if cuda_mod is not None:
            # Fast path: CUDA fused dequant kernel
            output_dtype = torch.float16 if x.dtype == torch.float16 else torch.float32
            w_deq = torch.empty(self.out_features, self.in_features,
                                dtype=output_dtype, device=x.device)

            quantizer = _get_quantizer(self.group_size, self.bits, str(x.device))
            cuda_mod.weight_dequant(
                self.packed_weight, self.norms,
                quantizer.signs1, quantizer.signs2,
                quantizer.centroids,
                w_deq,
                self.group_size, self.bits,
                self.out_features, self.in_features,
            )
        else:
            # Fallback: PyTorch dequant
            indices = unpack_indices(self.packed_weight, self.bits, self.group_size)
            norms_flat = self.norms.reshape(-1)

            quantizer = _get_quantizer(self.group_size, self.bits, str(x.device))
            w_groups = quantizer.dequantize(indices, norms_flat)

            w_deq = w_groups.reshape(self.out_features, self.padded_in)[:, :self.in_features]
            w_deq = w_deq.to(x.dtype)

        output = torch.matmul(x, w_deq.t())
        if self.bias is not None:
            output = output + self.bias
        return output

    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, out_features={self.out_features}, "
                f"bits={self.bits}, group_size={self.group_size}, bias={self.bias is not None}")


# Layers to never quantize
_SKIP_PATTERNS = ("lm_head", "embed", "norm", "head")


class Compressed3D:
    """Stores a compressed 3D expert weight tensor (num_experts, out_dim, in_dim).

    Weights are group-quantized and packed into uint8. Decompression restores
    the original shape and dtype. Used with forward hooks to decompress
    one layer at a time during inference.
    """

    def __init__(self, data: torch.Tensor, bits: int, group_size: int):
        n_experts, out_dim, in_dim = data.shape
        self.shape = data.shape
        self.dtype = data.dtype
        self.device = data.device
        self.bits = bits
        self.group_size = group_size
        self.in_dim = in_dim
        self.padded_in = ((in_dim + group_size - 1) // group_size) * group_size
        self.n_groups = self.padded_in // group_size

        flat = data.reshape(-1, in_dim)
        if self.padded_in > in_dim:
            padded = torch.zeros(flat.shape[0], self.padded_in, dtype=flat.dtype, device=flat.device)
            padded[:, :in_dim] = flat
        else:
            padded = flat

        grouped = padded.reshape(-1, group_size)
        quantizer = _get_quantizer(group_size, bits, str(data.device))
        indices, norms = quantizer.quantize(grouped)

        self.packed = pack_indices(indices, bits)
        self.norms = norms.reshape(n_experts * out_dim, self.n_groups)

        self.original_bytes = data.numel() * data.element_size()
        self.compressed_bytes = self.packed.numel() + self.norms.numel() * 4

    def decompress(self) -> torch.Tensor:
        """Decompress back to (num_experts, out_dim, in_dim) at original dtype."""
        cuda_mod = _get_cuda_module()
        n_experts, out_dim, in_dim = self.shape

        if cuda_mod is not None:
            output_dtype = torch.float16 if self.dtype == torch.float16 else torch.float32
            output = torch.empty(n_experts, out_dim, in_dim,
                                 dtype=output_dtype, device=self.packed.device)
            quantizer = _get_quantizer(self.group_size, self.bits, str(self.packed.device))
            cuda_mod.weight_dequant_3d(
                self.packed, self.norms,
                quantizer.signs1, quantizer.signs2,
                quantizer.centroids,
                output,
                self.group_size, self.bits,
                n_experts, out_dim, in_dim,
            )
            return output.to(self.dtype)
        else:
            indices = unpack_indices(self.packed, self.bits, self.group_size)
            norms_flat = self.norms.reshape(-1)
            quantizer = _get_quantizer(self.group_size, self.bits, str(self.packed.device))
            groups = quantizer.dequantize(indices, norms_flat)
            result = groups.reshape(-1, self.padded_in)[:, :self.in_dim]
            return result.reshape(self.shape).to(self.dtype)

    @property
    def ratio(self) -> float:
        return self.original_bytes / self.compressed_bytes if self.compressed_bytes > 0 else 0


def _compress_3d_param(module: nn.Module, param_name: str, bits: int, group_size: int) -> int:
    """Compress a 3D parameter in-place with real memory savings.

    Stores Compressed3D as module attribute, replaces parameter data with empty tensor.
    Registers forward hooks to decompress before use and free after.

    Returns original size in bytes.
    """
    param = getattr(module, param_name)
    compressed = Compressed3D(param.data, bits, group_size)

    # Store compressed data on the module
    setattr(module, f"_tq_{param_name}", compressed)

    # Replace parameter with empty to free memory
    param.data = torch.empty(0, device=param.device, dtype=param.dtype)

    return compressed.original_bytes, compressed.compressed_bytes


def _register_moe_hooks(module: nn.Module, param_names: list[str]):
    """Register pre/post forward hooks that decompress expert weights on demand."""

    def pre_hook(mod, args):
        for pname in param_names:
            compressed = getattr(mod, f"_tq_{pname}", None)
            if compressed is not None:
                getattr(mod, pname).data = compressed.decompress()

    def post_hook(mod, args, output):
        for pname in param_names:
            compressed = getattr(mod, f"_tq_{pname}", None)
            if compressed is not None:
                getattr(mod, pname).data = torch.empty(0, device=compressed.device, dtype=compressed.dtype)
        return output

    module.register_forward_pre_hook(pre_hook)
    module.register_forward_hook(post_hook)


def _select_bits(param: torch.Tensor, default_bits: int, kurtosis_aware: bool = False) -> int:
    """Select quantization bits based on tensor statistics.

    Heavy-tailed distributions (high kurtosis, e.g. shared MoE experts)
    need more bits. Near-Gaussian distributions (low kurtosis, e.g. routed
    MoE experts) tolerate aggressive compression.

    Based on APEX finding: shared expert kurtosis ~13.1, routed ~3.4.
    """
    if not kurtosis_aware:
        return default_bits

    flat = param.float().reshape(-1)
    mean = flat.mean()
    std = flat.std()
    if std < 1e-8:
        return default_bits
    normalized = (flat - mean) / std
    kurt = (normalized ** 4).mean().item()  # excess kurtosis + 3

    if kurt > 8:
        # Heavy-tailed (shared experts, attention projections): more bits
        return min(default_bits + 2, 8)
    elif kurt > 5:
        return min(default_bits + 1, 8)
    elif kurt < 3.5:
        # Near-Gaussian (routed experts): fewer bits OK
        return max(default_bits - 1, 2)
    return default_bits


def _replace_linear_layers(model: nn.Module, bits: int, group_size: int = 128,
                            min_size: int = 1024, kurtosis_aware: bool = False):
    """Compress model weights: nn.Linear layers AND MoE expert weights."""
    replacements = 0
    total_original = 0
    total_compressed = 0
    expert_layers = 0

    # Phase 1: Replace nn.Linear layers with TurboQuantWrapper
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if module.in_features < min_size and module.out_features < min_size:
            continue
        if any(p in name.lower() for p in _SKIP_PATTERNS):
            continue

        parts = name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)

        original_bytes = module.weight.numel() * module.weight.element_size()
        tensor_bits = _select_bits(module.weight.data, bits, kurtosis_aware)
        wrapper = TurboQuantWrapper(module, bits=tensor_bits, group_size=group_size)
        setattr(parent, parts[-1], wrapper)

        compressed_bytes = wrapper.packed_weight.numel() + wrapper.norms.numel() * 4
        total_original += original_bytes
        total_compressed += compressed_bytes
        replacements += 1

    # Phase 2: Compress MoE expert weights (3D tensors) with real memory savings.
    # Store packed indices + norms, replace parameter with empty tensor.
    # Forward hooks decompress one layer at a time during inference.
    modules_to_hook: dict[int, tuple[nn.Module, list[str]]] = {}

    for name, param in list(model.named_parameters()):
        if param.dim() != 3:
            continue
        if param.shape[-1] < min_size and param.shape[-2] < min_size:
            continue
        if any(p in name.lower() for p in _SKIP_PATTERNS):
            continue

        # Find the module that owns this parameter
        parts = name.split(".")
        param_name = parts[-1]
        owner = model
        for part in parts[:-1]:
            owner = getattr(owner, part)

        tensor_bits = _select_bits(param.data, bits, kurtosis_aware)
        orig_bytes, comp_bytes = _compress_3d_param(owner, param_name, tensor_bits, group_size)
        total_original += orig_bytes
        total_compressed += comp_bytes
        expert_layers += 1

        # Track modules that need hooks (group by module identity)
        mod_id = id(owner)
        if mod_id not in modules_to_hook:
            modules_to_hook[mod_id] = (owner, [])
        modules_to_hook[mod_id][1].append(param_name)

    # Register hooks once per module (not per parameter)
    for mod_id, (owner, param_names) in modules_to_hook.items():
        _register_moe_hooks(owner, param_names)

    total = replacements + expert_layers
    if total > 0:
        ratio = total_original / total_compressed if total_compressed > 0 else 0
        logger.info(
            "TQ%d-g%d weight compression: %d linear + %d expert layers, %.1f GB -> %.1f GB (%.1fx)",
            bits, group_size, replacements, expert_layers,
            total_original / 1e9, total_compressed / 1e9, ratio,
        )
        torch.cuda.empty_cache()

    return total


def enable_weight_quantization(bits: int = 3, group_size: int = 128,
                                min_layer_size: int = 1024, kurtosis_aware: bool = False):
    """Monkey-patch vLLM to apply TurboQuant weight compression at model load time.

    Args:
        bits: default quantization bits (2-8). Default 3 for best size/quality.
        group_size: elements per quantization group. Default 128 (matches head_dim).
        min_layer_size: minimum dimension to compress (skip small layers).
        kurtosis_aware: Auto-select bits per tensor based on kurtosis.
            Heavy-tailed tensors (shared MoE experts) get more bits.
            Near-Gaussian tensors (routed experts) get fewer bits.
    """
    assert 2 <= bits <= 8, f"bits must be 2-8, got {bits}"
    assert group_size in (64, 128, 256), f"group_size must be 64/128/256, got {group_size}"

    try:
        from vllm.model_executor.model_loader.utils import process_weights_after_loading as _original_process
    except ImportError:
        logger.error("Cannot import vLLM model loader. Is vLLM installed?")
        raise

    def patched_process_weights(model, model_config, target_device):
        _original_process(model, model_config, target_device)
        _replace_linear_layers(model, bits=bits, group_size=group_size,
                                min_size=min_layer_size, kurtosis_aware=kurtosis_aware)

    import vllm.model_executor.model_loader.utils as loader_utils
    loader_utils.process_weights_after_loading = patched_process_weights

    logger.info("TurboQuant TQ%d-g%d weight compression enabled", bits, group_size)
