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
_triton_available = None
_tq_fused_gemm_fn = None


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

    4-bit: 2 per byte (nibble packing).
    3-bit: 8 indices per 3 bytes (24 bits). For group_size=128: 48 bytes per group.
    2-bit: 4 per byte.
    """
    if bits == 4:
        assert indices.shape[-1] % 2 == 0
        flat = indices.reshape(-1, indices.shape[-1])
        lo = flat[:, 0::2].to(torch.uint8)
        hi = flat[:, 1::2].to(torch.uint8)
        return (lo | (hi << 4)).reshape(indices.shape[0], -1)
    elif bits == 3:
        # Pack 8 3-bit values into 3 bytes (24 bits).
        # Layout per 3-byte group:
        #   byte0 = idx0 | (idx1 << 3) | (idx2[0:2] << 6)
        #   byte1 = idx2[2] | (idx3 << 1) | (idx4 << 4) | (idx5[0:1] << 7)
        #   byte2 = idx5[1:3] | (idx6 << 2) | (idx7 << 5)
        n_rows, n_cols = indices.shape[0], indices.shape[-1]
        flat = indices.reshape(n_rows, -1).to(torch.uint8)
        # Pad to multiple of 8
        pad = (8 - n_cols % 8) % 8
        if pad > 0:
            flat = torch.nn.functional.pad(flat, (0, pad))
        n_packed_cols = flat.shape[1] // 8 * 3
        packed = torch.zeros(n_rows, n_packed_cols, dtype=torch.uint8, device=indices.device)
        for i in range(flat.shape[1] // 8):
            v = flat[:, i*8:(i+1)*8]  # (n_rows, 8) uint8, values 0-7
            b0 = v[:, 0] | (v[:, 1] << 3) | ((v[:, 2] & 0x3) << 6)
            b1 = (v[:, 2] >> 2) | (v[:, 3] << 1) | (v[:, 4] << 4) | ((v[:, 5] & 0x1) << 7)
            b2 = (v[:, 5] >> 1) | (v[:, 6] << 2) | (v[:, 7] << 5)
            packed[:, i*3] = b0
            packed[:, i*3+1] = b1
            packed[:, i*3+2] = b2
        return packed
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
    elif bits == 3:
        # Unpack 3 bytes → 8 3-bit values
        flat = packed.reshape(-1, packed.shape[-1])
        n_rows = flat.shape[0]
        n_groups_of_3 = flat.shape[1] // 3
        unpacked = torch.zeros(n_rows, n_groups_of_3 * 8, dtype=torch.int64, device=packed.device)
        for i in range(n_groups_of_3):
            b0 = flat[:, i*3].to(torch.int64)
            b1 = flat[:, i*3+1].to(torch.int64)
            b2 = flat[:, i*3+2].to(torch.int64)
            unpacked[:, i*8+0] = b0 & 0x7
            unpacked[:, i*8+1] = (b0 >> 3) & 0x7
            unpacked[:, i*8+2] = ((b0 >> 6) | (b1 << 2)) & 0x7
            unpacked[:, i*8+3] = (b1 >> 1) & 0x7
            unpacked[:, i*8+4] = (b1 >> 4) & 0x7
            unpacked[:, i*8+5] = ((b1 >> 7) | (b2 << 1)) & 0x7
            unpacked[:, i*8+6] = (b2 >> 2) & 0x7
            unpacked[:, i*8+7] = (b2 >> 5) & 0x7
        return unpacked[:, :dim]
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

    def __init__(self, original: nn.Linear, bits: int = 3, group_size: int = 128,
                 rotation: torch.Tensor | None = None):
        super().__init__()
        self.bits = bits
        self.group_size = group_size
        self.in_features = original.in_features
        self.out_features = original.out_features
        self._has_learned_rotation = rotation is not None

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

        if rotation is not None:
            # Use learned rotation (SpinQuant-style)
            from turboquant_vllm.learned_rotation import quantize_with_learned_rotation
            packed, norms, _ = quantize_with_learned_rotation(
                weight, rotation, bits=bits, group_size=group_size
            )
            self.register_buffer("rotation", rotation)
        else:
            # Use fixed WHT rotation (default) with norm correction
            quantizer = _get_quantizer(group_size, bits, str(weight.device))
            indices, norms_raw = quantizer.quantize(grouped, norm_correction=True)
            packed = pack_indices(indices, bits)
            norms = norms_raw.reshape(out_dim, self.n_groups)

        # Memory stats (norms stored as FP16 when learned rotation, FP32 otherwise)
        original_bytes = weight.numel() * weight.element_size()
        norm_bytes = norms.numel() * norms.element_size()
        compressed_bytes = packed.numel() + norm_bytes
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

    @classmethod
    def from_packed(cls, packed_weight: torch.Tensor, norms: torch.Tensor,
                    in_features: int, out_features: int,
                    bits: int = 3, group_size: int = 128,
                    bias: torch.Tensor | None = None):
        """Create a TurboQuantWrapper from pre-packed data (native TQ3 checkpoint).

        Skips compression — the packed indices and norms are used directly.
        This enables loading native TQ3 checkpoints without decompressing to FP16.
        """
        wrapper = object.__new__(cls)
        nn.Module.__init__(wrapper)
        wrapper.bits = bits
        wrapper.group_size = group_size
        wrapper.in_features = in_features
        wrapper.out_features = out_features
        wrapper._has_learned_rotation = False
        wrapper.padded_in = ((in_features + group_size - 1) // group_size) * group_size
        wrapper.n_groups = wrapper.padded_in // group_size

        wrapper.register_buffer("packed_weight", packed_weight)
        wrapper.register_buffer("norms", norms)
        wrapper.bias = nn.Parameter(bias) if bias is not None else None

        original_bytes = out_features * in_features * 2  # FP16 equivalent
        compressed_bytes = packed_weight.numel() + norms.numel() * norms.element_size()
        wrapper._ratio = original_bytes / max(compressed_bytes, 1)

        return wrapper

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        global _triton_available, _tq_fused_gemm_fn

        # Fastest path: Triton fused dequant-GEMM (no intermediate buffer)
        if _triton_available is None:
            try:
                from turboquant_vllm.triton_ops import tq_fused_gemm
                _tq_fused_gemm_fn = tq_fused_gemm
                _triton_available = True
            except (ImportError, Exception):
                _triton_available = False

        if _triton_available and x.is_cuda and self.bits != 3:
            # Triton kernel doesn't support 3-bit sub-byte packing yet
            try:
                quantizer = _get_quantizer(self.group_size, self.bits, str(x.device))
                return _tq_fused_gemm_fn(
                    x, self.packed_weight, self.norms,
                    quantizer.signs1, quantizer.signs2, quantizer.centroids,
                    group_size=self.group_size, bits=self.bits, bias=self.bias,
                )
            except (ValueError, RuntimeError):
                pass  # Fall through to CUDA/PyTorch path for incompatible shapes

        cuda_mod = _get_cuda_module()

        if cuda_mod is not None:
            # CUDA dequant kernel + cuBLAS GEMM (intermediate buffer)
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

        if w_deq.dtype != x.dtype:
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

    @classmethod
    def from_packed(cls, packed: torch.Tensor, norms: torch.Tensor,
                    shape: tuple[int, int, int], dtype: torch.dtype,
                    bits: int = 3, group_size: int = 128):
        """Create a Compressed3D from pre-packed data (native TQ3 checkpoint).

        Skips compression — the packed indices and norms are used directly.
        """
        obj = object.__new__(cls)
        n_experts, out_dim, in_dim = shape
        obj.shape = shape
        obj.dtype = dtype
        obj.device = packed.device
        obj.bits = bits
        obj.group_size = group_size
        obj.in_dim = in_dim
        obj.padded_in = ((in_dim + group_size - 1) // group_size) * group_size
        obj.n_groups = obj.padded_in // group_size
        obj.packed = packed
        obj.norms = norms
        obj.original_bytes = n_experts * out_dim * in_dim * 2  # FP16
        obj.compressed_bytes = packed.numel() + norms.numel() * 4
        return obj

    def decompress(self, buf: torch.Tensor | None = None) -> torch.Tensor:
        """Decompress back to (num_experts, out_dim, in_dim) at original dtype.

        Args:
            buf: Optional pre-allocated output buffer to avoid repeated allocation.
                 Must match shape and be on the same device. Dtype can differ
                 (will be used as-is for CUDA path, cast for PyTorch path).
        """
        cuda_mod = _get_cuda_module()
        n_experts, out_dim, in_dim = self.shape

        if cuda_mod is not None:
            output_dtype = torch.float16 if self.dtype == torch.float16 else torch.float32
            if buf is not None and buf.shape == (n_experts, out_dim, in_dim) and buf.dtype == output_dtype:
                output = buf
            else:
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
    """Register pre/post forward hooks that decompress expert weights on demand.

    Uses per-parameter buffer pooling: a decompression buffer is allocated once
    and reused across forward passes, eliminating the allocation/deallocation
    overhead that was the main bottleneck (96 expert tensors × 48 layers per token).

    Buffer attributes are stored as _tq_buf_{param_name} on the module.
    """
    # Pre-compute attribute names to avoid f-string formatting on hot path
    comp_names = [f"_tq_{pname}" for pname in param_names]
    buf_names = [f"_tq_buf_{pname}" for pname in param_names]

    def pre_hook(mod, args):
        for pname, comp_name, buf_name in zip(param_names, comp_names, buf_names):
            compressed = getattr(mod, comp_name, None)
            if compressed is None:
                continue
            buf = getattr(mod, buf_name, None)
            decompressed = compressed.decompress(buf=buf)
            if buf is None or buf.data_ptr() != decompressed.data_ptr():
                setattr(mod, buf_name, decompressed)
            getattr(mod, pname).data = decompressed

    def post_hook(mod, args, output):
        for pname, comp_name in zip(param_names, comp_names):
            compressed = getattr(mod, comp_name, None)
            if compressed is not None:
                # Clear parameter reference but keep buffer allocated.
                # The buffer (_tq_buf_*) stays in memory for reuse next forward.
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


def _find_router_weights(model: nn.Module) -> dict[str, torch.Tensor]:
    """Find MoE router/gate weights for expert importance ranking.

    Returns: {module_path: gate_weight (num_experts, hidden_size)}
    """
    gates = {}
    for name, module in model.named_modules():
        # vLLM MoE blocks have a 'gate' attribute (ReplicatedLinear)
        gate = getattr(module, 'gate', None)
        if gate is not None and hasattr(gate, 'weight'):
            w = gate.weight.data
            if w.dim() == 2:  # (num_experts, hidden_size)
                gates[name] = w
    return gates


def _rank_experts_by_importance(gate_weight: torch.Tensor) -> torch.Tensor:
    """Rank experts by router weight L2 norm (proxy for importance).

    Higher norm = more likely to be selected = more important.
    Returns sorted indices (most important first).
    """
    norms = gate_weight.float().norm(dim=1)  # (num_experts,)
    return norms.argsort(descending=True)


def _prune_expert_weights(
    param: torch.Tensor,
    keep_mask: torch.Tensor,
) -> None:
    """Zero out pruned expert rows in-place in a 3D expert weight tensor.

    Args:
        param: (num_experts, out_dim, in_dim) expert weight tensor, modified in-place
        keep_mask: (num_experts,) bool mask — True for experts to keep
    """
    param[~keep_mask] = 0.0


def _replace_linear_layers(model: nn.Module, bits: int, group_size: int = 128,
                            min_size: int = 1024, kurtosis_aware: bool = False,
                            prune_experts: float = 0.0, routed_expert_bits: int | None = None,
                            per_module_bits: dict[str, int] | None = None,
                            learned_rotations: dict[str, torch.Tensor] | None = None):
    """Compress model weights: nn.Linear layers AND MoE expert weights.

    Args:
        prune_experts: Fraction of routed experts to prune (0.0 = none, 0.5 = half).
        routed_expert_bits: Override bit width for routed expert weights.
        per_module_bits: Dict mapping module name → bit width.
        learned_rotations: Dict mapping module name → rotation matrix.
            From optimize_all_rotations(). When present, uses learned rotation
            instead of fixed WHT for quantization. Enables viable TQ3.
    """
    replacements = 0
    total_original = 0
    total_compressed = 0
    expert_layers = 0
    pruned_count = 0

    # Build set of expert modules to prune (by name) for 2D expert layout.
    # HuggingFace stores experts as individual nn.Linear: experts.{N}.gate_proj
    # vLLM FusedMoE stores as 3D tensors. We handle both.
    pruned_expert_modules: set[str] = set()
    if prune_experts > 0:
        import re
        gates = _find_router_weights(model)
        for gate_path, gate_weight in gates.items():
            num_experts = gate_weight.shape[0]
            n_keep = max(1, int(num_experts * (1.0 - prune_experts)))
            ranked = _rank_experts_by_importance(gate_weight)
            prune_indices = set(ranked[n_keep:].tolist())

            # Find 2D expert modules by name pattern: {gate_path}.experts.{N}.*
            # gate_path is the MoE block (e.g., model.layers.0.mlp) that contains
            # both 'gate' and 'experts' as children.
            pattern = re.compile(rf"^{re.escape(gate_path)}\.experts\.(\d+)\b")
            for mod_name, _ in model.named_modules():
                m = pattern.match(mod_name)
                if m and int(m.group(1)) in prune_indices:
                    pruned_expert_modules.add(mod_name)

            n_marked = len([m for m in pruned_expert_modules if m.startswith(gate_path)])
            logger.info(
                "Expert pruning %s: keeping %d/%d experts (%.0f%% pruned, %d modules marked)",
                gate_path, n_keep, num_experts, prune_experts * 100, n_marked,
            )

    # Phase 1: Replace nn.Linear layers with TurboQuantWrapper
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if module.in_features < min_size and module.out_features < min_size:
            continue
        if any(p in name.lower() for p in _SKIP_PATTERNS):
            continue

        # Check if this is a pruned expert — skip compression, zero it
        is_pruned = name in pruned_expert_modules or any(
            name.startswith(pm + ".") or name == pm for pm in pruned_expert_modules
        )
        if is_pruned:
            # Zero the weight and compress at minimum bits
            module.weight.data.zero_()
            pruned_count += module.weight.numel()

        parts = name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)

        original_bytes = module.weight.numel() * module.weight.element_size()

        # Select bits: per_module_bits > pruned > routed override > kurtosis > default
        if per_module_bits and name in per_module_bits:
            tensor_bits = per_module_bits[name]
        elif is_pruned:
            tensor_bits = 2  # minimum for zeroed weights
        elif "expert" in name.lower() and routed_expert_bits is not None:
            tensor_bits = routed_expert_bits
        else:
            tensor_bits = _select_bits(module.weight.data, bits, kurtosis_aware)

        rotation = learned_rotations.get(name) if learned_rotations else None
        wrapper = TurboQuantWrapper(module, bits=tensor_bits, group_size=group_size,
                                    rotation=rotation)
        setattr(parent, parts[-1], wrapper)

        compressed_bytes = wrapper.packed_weight.numel() + wrapper.norms.numel() * wrapper.norms.element_size()
        total_original += original_bytes
        total_compressed += compressed_bytes
        replacements += 1

    # Phase 2: Compress MoE expert weights (3D tensors) with real memory savings.
    # Store packed indices + norms, replace parameter with empty tensor.
    # Forward hooks decompress one layer at a time during inference.
    #
    # Expert pruning: if prune_experts > 0, find router gate weights and zero out
    # the least-important experts before compression. Zeroed rows produce norms ≈ 0
    # and uniform indices — storage isn't eliminated but norms become negligible.
    modules_to_hook: dict[int, tuple[nn.Module, list[str]]] = {}

    # Build param_name → keep_mask mapping for expert pruning (3D layout).
    # Uses router gate weight norms to rank experts by importance.
    param_to_mask: dict[str, torch.Tensor] = {}  # param_name → bool mask
    if prune_experts > 0:
        gates = _find_router_weights(model)
        # For each gate, find sibling 3D parameters (expert weights in same MoE block)
        for gate_path, gate_weight in gates.items():
            num_experts = gate_weight.shape[0]
            n_keep = max(1, int(num_experts * (1.0 - prune_experts)))
            ranked = _rank_experts_by_importance(gate_weight)
            mask = torch.zeros(num_experts, dtype=torch.bool, device=gate_weight.device)
            mask[ranked[:n_keep]] = True

            # Map 3D params under this MoE block (and parent) to the mask.
            # Expert weights can be children of the gate's module or siblings.
            gate_parts = gate_path.split(".")
            prefixes = [gate_path + "."]
            if len(gate_parts) > 1:
                prefixes.append(".".join(gate_parts[:-1]) + ".")
            for pname, p in model.named_parameters():
                if p.dim() == 3 and p.shape[0] == num_experts:
                    if any(pname.startswith(pfx) for pfx in prefixes):
                        param_to_mask[pname] = mask

            logger.info(
                "Expert pruning %s: keeping %d/%d experts (%.0f%% pruned)",
                gate_path, n_keep, num_experts, prune_experts * 100,
            )

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

        # Apply expert pruning in-place if mask available
        if prune_experts > 0 and name in param_to_mask:
            mask = param_to_mask[name]
            _prune_expert_weights(param.data, mask)
            pruned_count += (~mask).sum().item() * param.shape[1] * param.shape[2]

        # Select bit width: routed_expert_bits override, or kurtosis, or default
        if routed_expert_bits is not None:
            tensor_bits = routed_expert_bits
        else:
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
        prune_msg = f", {pruned_count:,} expert params pruned" if pruned_count > 0 else ""
        expert_bits_msg = f" (routed@TQ{routed_expert_bits})" if routed_expert_bits else ""
        logger.info(
            "TQ%d-g%d%s weight compression: %d linear + %d expert layers, "
            "%.1f GB -> %.1f GB (%.1fx)%s",
            bits, group_size, expert_bits_msg, replacements, expert_layers,
            total_original / 1e9, total_compressed / 1e9, ratio, prune_msg,
        )
        torch.cuda.empty_cache()

    return total


def enable_weight_quantization(bits: int = 3, group_size: int = 128,
                                min_layer_size: int = 1024, kurtosis_aware: bool = False,
                                prune_experts: float = 0.0, routed_expert_bits: int | None = None):
    """Monkey-patch vLLM to apply TurboQuant weight compression at model load time.

    Args:
        bits: default quantization bits (2-8). Default 3 for best size/quality.
        group_size: elements per quantization group. Default 128 (matches head_dim).
        min_layer_size: minimum dimension to compress (skip small layers).
        kurtosis_aware: Auto-select bits per tensor based on kurtosis.
            Heavy-tailed tensors (shared MoE experts) get more bits.
            Near-Gaussian tensors (routed experts) get fewer bits.
        prune_experts: Fraction of routed MoE experts to prune (0.0-1.0).
            Uses router weight norms to rank experts by importance.
            REAP (ICLR 2026): 50% pruning retains 97.6% quality on MoE.
        routed_expert_bits: Override bit width for routed expert weights.
            Set to 2 for aggressive compression (viable for routed experts).
    """
    assert 2 <= bits <= 8, f"bits must be 2-8, got {bits}"
    assert group_size in (64, 128, 256), f"group_size must be 64/128/256, got {group_size}"
    assert 0.0 <= prune_experts < 1.0, f"prune_experts must be 0.0-1.0, got {prune_experts}"

    try:
        from vllm.model_executor.model_loader.utils import process_weights_after_loading as _original_process
    except ImportError:
        logger.error("Cannot import vLLM model loader. Is vLLM installed?")
        raise

    def patched_process_weights(model, model_config, target_device):
        _original_process(model, model_config, target_device)
        _replace_linear_layers(model, bits=bits, group_size=group_size,
                                min_size=min_layer_size, kurtosis_aware=kurtosis_aware,
                                prune_experts=prune_experts,
                                routed_expert_bits=routed_expert_bits)

    import vllm.model_executor.model_loader.utils as loader_utils
    loader_utils.process_weights_after_loading = patched_process_weights

    prune_msg = f", {prune_experts*100:.0f}% expert pruning" if prune_experts > 0 else ""
    expert_msg = f", routed@TQ{routed_expert_bits}" if routed_expert_bits else ""
    logger.info("TurboQuant TQ%d-g%d weight compression enabled%s%s",
                bits, group_size, prune_msg, expert_msg)


def save_compressed_checkpoint(
    model_id: str,
    output_dir: str,
    bits: int = 3,
    group_size: int = 128,
):
    """Save a TQ-compressed checkpoint as FP16 that loads on smaller GPUs.

    Problem: Gemma 4 26B is 52 GB in BF16, which doesn't fit on a 48 GB L40S
    during loading, even though the compressed model is only 12 GB in memory.

    Solution: load BF16 to CPU (needs ~60 GB RAM), compress with TQ on CPU,
    save the decompressed FP16 weights (~26 GB checkpoint). This checkpoint
    loads directly on a 48 GB GPU. Use enable_weight_quantization(bits=3) on
    top to compress in memory to 12 GB at runtime.

    CPU compression takes about 1-2 minutes instead of 9 seconds on GPU.

    Args:
        model_id: HuggingFace model ID.
        output_dir: Where to save the FP16 checkpoint.
        bits: Quantization bits (3 or 4).
        group_size: Group size (128).
    """
    import os

    logger.info("Loading %s to CPU for compression...", model_id)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Compress each linear layer on CPU
    quantizer = _get_quantizer(group_size, bits, "cpu")
    compressed_count = 0

    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if module.in_features < 128 and module.out_features < 128:
            continue
        if any(p in name.lower() for p in _SKIP_PATTERNS):
            continue

        weight = module.weight.data.float()
        out_dim, in_dim = weight.shape

        padded_in = ((in_dim + group_size - 1) // group_size) * group_size
        if padded_in > in_dim:
            padded = torch.zeros(out_dim, padded_in, dtype=weight.dtype)
            padded[:, :in_dim] = weight
        else:
            padded = weight

        grouped = padded.reshape(-1, group_size)
        indices, norms = quantizer.quantize(grouped, norm_correction=True)
        w_hat = quantizer.dequantize(indices, norms)
        w_reconstructed = w_hat.reshape(out_dim, padded_in)[:, :in_dim]

        module.weight.data = w_reconstructed.to(torch.float16)
        compressed_count += 1

        if compressed_count % 500 == 0:
            logger.info("  Compressed %d layers...", compressed_count)

    logger.info("Compressed %d layers on CPU", compressed_count)

    # Convert to FP16 (halves checkpoint size, fits on 48 GB GPUs)
    for param in model.parameters():
        param.data = param.data.half()
    for buf_name, buf in model.named_buffers():
        if buf.is_floating_point():
            buf.data = buf.data.half()
    model.config.torch_dtype = "float16"

    # Save
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)

    # Calculate size
    total_size = sum(
        os.path.getsize(os.path.join(output_dir, f))
        for f in os.listdir(output_dir)
    )
    logger.info("Saved to %s (%.1f GB)", output_dir, total_size / 1e9)
