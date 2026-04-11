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
import os

import torch
import torch.nn as nn

from turboquant_vllm.torch_ops import PolarQuantTorch

try:
    from vllm.logger import init_logger

    logger = init_logger(__name__)
except ImportError:
    logger = logging.getLogger(__name__)

_quantizers: dict[tuple[int, int, str], PolarQuantTorch] = {}
_cuda_mod = None
_cuda_available = None
_triton_available = None
_tq_fused_gemm_fn = None
_tq_fwht_input_fn = None


def _ensure_triton_backends() -> bool:
    """Lazy-load Triton kernel functions on first use.

    Called from ``TurboQuantWrapper.__init__``, which runs once per layer
    during model loading — before any ``forward`` is traced. Must NOT be
    called from inside ``forward`` because vLLM 0.19 compiles forward in
    fullgraph mode and rejects any call to a ``@torch._dynamo.disable``'d
    function with "Skip calling `torch.compiler.disable()`d function".

    Returns True if both Triton kernels loaded; False otherwise.
    """
    global _triton_available, _tq_fused_gemm_fn, _tq_fwht_input_fn
    if _triton_available is not None:
        return _triton_available
    try:
        from turboquant_vllm.triton_ops import tq_fused_gemm, tq_fwht_input_gemm

        _tq_fused_gemm_fn = tq_fused_gemm
        _tq_fwht_input_fn = tq_fwht_input_gemm
        _triton_available = True
        logger.info("Triton kernels available (FWHT-on-input + dequant-GEMM)")
    except (ImportError, Exception):
        _triton_available = False
        logger.info("Triton not available, using CUDA/PyTorch fallback")
    return _triton_available


def _resolve_module(root, dotted_path: str):
    """Navigate a module tree by dotted path, returning the final attribute."""
    obj = root
    for part in dotted_path.split("."):
        obj = getattr(obj, part)
    return obj


def _resolve_parent_and_attr(root, dotted_path: str):
    """Resolve a dotted path to (parent_module, attr_name)."""
    parts = dotted_path.split(".")
    parent = _resolve_module(root, ".".join(parts[:-1])) if len(parts) > 1 else root
    return parent, parts[-1]


def _get_cuda_module():
    """Lazy-load the CUDA weight dequant kernel.

    Must be called from ``TurboQuantWrapper.__init__`` (pre-compile), not
    ``forward`` (which vLLM 0.19 compiles in fullgraph mode).
    """
    global _cuda_mod, _cuda_available
    if _cuda_available is not None:
        return _cuda_mod if _cuda_available else None
    try:
        from turboquant_vllm.build import build

        _cuda_mod = build()
        if hasattr(_cuda_mod, "weight_dequant"):
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
    dev = torch.device(device)
    if dev.type == "cuda" and dev.index is None:
        dev = torch.device("cuda", torch.cuda.current_device())
    normalized_device = str(dev)
    key = (group_size, bits, normalized_device)
    quantizer = _quantizers.get(key)
    if quantizer is None:
        quantizer = PolarQuantTorch(group_size, bits, seed=42, device=normalized_device)
        _quantizers[key] = quantizer

    # Defensive fallback in case a stale/mutated cache entry has mismatched device.
    if str(quantizer.device) != normalized_device:
        quantizer = PolarQuantTorch(group_size, bits, seed=42, device=normalized_device)
        _quantizers[key] = quantizer

    return quantizer


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
            v = flat[:, i * 8 : (i + 1) * 8]  # (n_rows, 8) uint8, values 0-7
            b0 = v[:, 0] | (v[:, 1] << 3) | ((v[:, 2] & 0x3) << 6)
            b1 = (v[:, 2] >> 2) | (v[:, 3] << 1) | (v[:, 4] << 4) | ((v[:, 5] & 0x1) << 7)
            b2 = (v[:, 5] >> 1) | (v[:, 6] << 2) | (v[:, 7] << 5)
            packed[:, i * 3] = b0
            packed[:, i * 3 + 1] = b1
            packed[:, i * 3 + 2] = b2
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
            b0 = flat[:, i * 3].to(torch.int64)
            b1 = flat[:, i * 3 + 1].to(torch.int64)
            b2 = flat[:, i * 3 + 2].to(torch.int64)
            unpacked[:, i * 8 + 0] = b0 & 0x7
            unpacked[:, i * 8 + 1] = (b0 >> 3) & 0x7
            unpacked[:, i * 8 + 2] = ((b0 >> 6) | (b1 << 2)) & 0x7
            unpacked[:, i * 8 + 3] = (b1 >> 1) & 0x7
            unpacked[:, i * 8 + 4] = (b1 >> 4) & 0x7
            unpacked[:, i * 8 + 5] = ((b1 >> 7) | (b2 << 1)) & 0x7
            unpacked[:, i * 8 + 6] = (b2 >> 2) & 0x7
            unpacked[:, i * 8 + 7] = (b2 >> 5) & 0x7
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

    def __init__(self, original: nn.Linear, bits: int = 3, group_size: int = 128, rotation: torch.Tensor | None = None):
        super().__init__()
        self.bits = bits
        self.group_size = group_size
        self.in_features = original.in_features
        self.out_features = original.out_features
        self._has_learned_rotation = rotation is not None

        # vLLM's Linear subclasses return either `output` or a
        # `(output, output_bias)` tuple depending on the `return_bias`
        # flag. Downstream model code (e.g. Qwen2/Gemma4 qkv projection
        # handling) does `qkv, _ = self.qkv_proj(x)`, which crashes
        # under vLLM 0.19 fullgraph dynamo as "Can't unpack a tensor
        # of 2048 rows into a tuple of 2 elements" if we return only
        # a tensor. Mirror the original's return_bias flag — default
        # to False when wrapping a plain nn.Linear (returns a plain
        # tensor) vs True when wrapping a vLLM Linear subclass.
        self.return_bias = getattr(original, "return_bias", False)

        # Probe Triton + CUDA backends during construction, NOT in forward().
        # vLLM 0.19 AOT-compiles forward() in fullgraph mode and can't trace
        # logger calls or @torch._dynamo.disable helpers. Doing the probes
        # here populates the module globals before any forward is traced.
        _ensure_triton_backends()
        _get_cuda_module()

        # Cache the PolarQuant rotation tensors (signs1, signs2, centroids)
        # as buffers on the wrapper. forward() will read them directly as
        # tensor attributes rather than calling _get_quantizer() — that
        # helper does a dict lookup and Python object construction that
        # vLLM 0.19's fullgraph dynamo compile cannot trace.
        _pq = _get_quantizer(group_size, bits, str(original.weight.device))
        self.register_buffer("tq_signs1", _pq.signs1)
        self.register_buffer("tq_signs2", _pq.signs2)
        self.register_buffer("tq_centroids", _pq.centroids)

        # Eagerly populate the module-level rotation matrix cache so the
        # first forward (which may be the warmup pass before CUDA graph
        # capture) does not hit a cache miss and allocate / run a
        # butterfly WHT inside the custom_op body at capture time. The
        # rotation matrix is a pure function of (signs1, signs2,
        # group_size) and fits in 64 KB for group_size=128, so building
        # it once per wrapper instance at construction time is cheap and
        # removes an implicit "warmup must run before capture" coupling.
        # See turboquant_vllm.triton_ops._rotation_matrix_cache comment.
        if _triton_available:
            from turboquant_vllm.triton_ops import _get_cached_rotation_matrix

            _get_cached_rotation_matrix(self.tq_signs1, self.tq_signs2, group_size)

        weight = original.weight.data  # (out_features, in_features)
        # Flatten to 2D when weight has extra leading dimensions (e.g. vLLM
        # parallel linears or MoE expert tensors passed directly).
        if weight.ndim > 2:
            weight = weight.reshape(-1, weight.shape[-1])
            self.in_features = weight.shape[-1]
            self.out_features = weight.shape[0]
        out_dim, in_dim = weight.shape

        # Pad in_features to multiple of group_size
        self.padded_in = ((in_dim + group_size - 1) // group_size) * group_size
        self.n_groups = self.padded_in // group_size

        if self.padded_in > in_dim:
            padded = torch.zeros(out_dim, self.padded_in, dtype=weight.dtype, device=weight.device)
            padded[:, :in_dim] = weight
        else:
            padded = weight

        grouped = padded.reshape(-1, group_size)

        if rotation is not None:
            # Use learned rotation (SpinQuant-style)
            from turboquant_vllm.learned_rotation import quantize_with_learned_rotation

            packed, norms, _ = quantize_with_learned_rotation(weight, rotation, bits=bits, group_size=group_size)
            self.register_buffer("rotation", rotation)
        else:
            # Use fixed WHT rotation (default) with norm correction
            quantizer = _get_quantizer(group_size, bits, str(weight.device))
            indices, norms_raw = quantizer.quantize(grouped, norm_correction=True)
            packed = pack_indices(indices, bits)
            norms = norms_raw.reshape(out_dim, self.n_groups)

        original_bytes = weight.numel() * weight.element_size()
        norm_bytes = norms.numel() * norms.element_size()
        compressed_bytes = packed.numel() + norm_bytes
        self._ratio = original_bytes / compressed_bytes

        self.register_buffer("packed_weight", packed)
        self.register_buffer("norms", norms)

        if original.bias is not None:
            bias_data = original.bias.data
            if bias_data.ndim > 1:
                bias_data = bias_data.reshape(-1)
            if bias_data.numel() != self.out_features:
                msg = (
                    "Bias size does not match flattened out_features: "
                    f"flattened_bias.numel()={bias_data.numel()} vs out_features={self.out_features}"
                )
                raise ValueError(msg)
            self.bias = nn.Parameter(bias_data.clone())
        else:
            self.bias = None

        logger.debug(
            "TQ%d-g%d compressed %dx%d (%.1fx)",
            bits,
            group_size,
            out_dim,
            in_dim,
            self._ratio,
        )

    @classmethod
    def from_packed(
        cls,
        packed_weight: torch.Tensor,
        norms: torch.Tensor,
        in_features: int,
        out_features: int,
        bits: int = 3,
        group_size: int = 128,
        bias: torch.Tensor | None = None,
    ):
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

    def forward(self, x: torch.Tensor):
        # Trivial dispatch that dynamo can specialize on x.is_cuda when
        # vLLM 0.19 fullgraph-compiles the CUDA forward path. The CPU
        # branch is still physically present for pytest / dev machines
        # without a CUDA extension.
        output = self._forward_gpu(x) if x.is_cuda else self._forward_cpu(x)
        # Match vLLM Linear's return_bias contract. See __init__.
        if self.return_bias:
            return output, None
        return output

    def _forward_gpu(self, x: torch.Tensor) -> torch.Tensor:
        """Fullgraph-dynamo-clean forward for GPU (Triton or CUDA backends).

        Must NOT contain logger calls, calls to @torch._dynamo.disable'd
        helpers, or method calls on non-tensor Python objects that do dict
        lookups / string conversions. vLLM 0.19 AOT-compiles this path.
        """
        if _triton_available and x.is_cuda:
            args = (x, self.packed_weight, self.norms, self.tq_signs1, self.tq_signs2, self.tq_centroids)
            kwargs = dict(group_size=self.group_size, bits=self.bits, bias=self.bias)

            # FWHT-on-input wins for large output dims (saves N inverse rotations).
            # Dequant-GEMM wins for small layers (lower fixed overhead).
            # Crossover ~4K output features on H100.
            primary = _tq_fwht_input_fn if self.out_features >= 4096 else _tq_fused_gemm_fn
            fallback = _tq_fused_gemm_fn if self.out_features >= 4096 else _tq_fwht_input_fn
            try:
                return primary(*args, **kwargs)
            except (ValueError, RuntimeError):
                return fallback(*args, **kwargs)

        # CUDA C++ extension path — called via the compiled .so, which
        # dynamo treats as opaque. Built once at plugin init by build.py.
        output_dtype = torch.float16 if x.dtype == torch.float16 else torch.float32
        w_deq = torch.empty(self.out_features, self.in_features, dtype=output_dtype, device=x.device)
        _cuda_mod.weight_dequant(
            self.packed_weight,
            self.norms,
            self.tq_signs1,
            self.tq_signs2,
            self.tq_centroids,
            w_deq,
            self.group_size,
            self.bits,
            self.out_features,
            self.in_features,
        )

        if w_deq.dtype != x.dtype:
            w_deq = w_deq.to(x.dtype)
        output = torch.matmul(x, w_deq.t())
        if self.bias is not None:
            output = output + self.bias
        return output

    def _forward_cpu(self, x: torch.Tensor) -> torch.Tensor:
        """CPU fallback using the PolarQuant reference implementation.

        Used on systems without Triton and without a built CUDA extension
        (typically developer machines, tests, and CI). Not dynamo-friendly
        because _fast_wht_batch has a data-dependent while loop and
        PolarQuantTorch.dequantize is a method call — but that's fine
        because this path only runs in eager mode.
        """
        indices = unpack_indices(self.packed_weight, self.bits, self.group_size)
        norms_flat = self.norms.reshape(-1)
        quantizer = _get_quantizer(self.group_size, self.bits, str(x.device))
        w_groups = quantizer.dequantize(indices, norms_flat)
        w_deq = w_groups.reshape(self.out_features, self.padded_in)[:, : self.in_features]
        w_deq = w_deq.to(x.dtype)
        output = torch.matmul(x, w_deq.t())
        if self.bias is not None:
            output = output + self.bias
        return output

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bits={self.bits}, group_size={self.group_size}, bias={self.bias is not None}"
        )


# Layers to never quantize
_SKIP_PATTERNS = ("lm_head", "embed", "norm", "head")

# Layers that benefit from higher precision (attention output + MLP output)
_SENSITIVE_PATTERNS = ("o_proj", "down_proj")


def select_bits(
    tensor_name: str,
    default_bits: int,
    sensitive_bits: int | None = None,
    sensitive_patterns: tuple[str, ...] = _SENSITIVE_PATTERNS,
) -> int:
    """Return bits for this tensor. Sensitive layers get higher precision."""
    if sensitive_bits is None:
        return default_bits
    if any(p in tensor_name for p in sensitive_patterns):
        return sensitive_bits
    return default_bits


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
    def from_packed(
        cls,
        packed: torch.Tensor,
        norms: torch.Tensor,
        shape: tuple[int, int, int],
        dtype: torch.dtype,
        bits: int = 3,
        group_size: int = 128,
    ):
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

        Uses CUDA kernel when available (fast), falls back to chunked PyTorch
        (8 experts at a time) to limit GPU memory.
        """
        cuda_mod = _get_cuda_module()
        n_experts, out_dim, in_dim = self.shape

        if cuda_mod is not None:
            output_dtype = torch.float16 if self.dtype == torch.float16 else torch.float32
            if buf is not None and buf.shape == (n_experts, out_dim, in_dim) and buf.dtype == output_dtype:
                output = buf
            else:
                output = torch.empty(n_experts, out_dim, in_dim, dtype=output_dtype, device=self.packed.device)
            quantizer = _get_quantizer(self.group_size, self.bits, str(self.packed.device))
            cuda_mod.weight_dequant_3d(
                self.packed,
                self.norms,
                quantizer.signs1,
                quantizer.signs2,
                quantizer.centroids,
                output,
                self.group_size,
                self.bits,
                n_experts,
                out_dim,
                in_dim,
            )
            return output.to(self.dtype)

        # Fallback: chunked PyTorch (8 experts at a time)
        output_dtype = self.dtype
        if buf is not None and buf.shape == (n_experts, out_dim, in_dim):
            output = buf
        else:
            output = torch.empty(n_experts, out_dim, in_dim, dtype=output_dtype, device=self.packed.device)

        quantizer = _get_quantizer(self.group_size, self.bits, str(self.packed.device))
        chunk_experts = max(1, min(8, n_experts))
        groups_per_expert = out_dim * self.n_groups

        for start in range(0, n_experts, chunk_experts):
            end = min(start + chunk_experts, n_experts)

            start_group = start * groups_per_expert
            end_group = end * groups_per_expert
            chunk_packed = self.packed[start_group:end_group]

            start_row = start * out_dim
            end_row = end * out_dim
            chunk_norms = self.norms[start_row:end_row]

            chunk_idx = unpack_indices(chunk_packed, self.bits, self.group_size)
            chunk_groups = quantizer.dequantize(chunk_idx, chunk_norms.reshape(-1))
            del chunk_idx

            chunk_result = chunk_groups.reshape(-1, self.padded_in)[:, : self.in_dim]
            output[start:end] = chunk_result.reshape(end - start, out_dim, in_dim).to(output_dtype)
            del chunk_groups, chunk_result

        return output

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

    setattr(module, f"_tq_{param_name}", compressed)
    param.data = torch.empty(0, device=param.device, dtype=param.dtype)

    return compressed.original_bytes, compressed.compressed_bytes


def _register_moe_hooks(module: nn.Module, param_names: list[str], pool_buffers: bool = True):
    """Register pre/post forward hooks that decompress expert weights on demand.

    Args:
        pool_buffers: If True, keep decompression buffers allocated between
            forward passes for reuse (faster, uses more memory). If False,
            free buffers after each forward (slower, saves ~40 GB for large MoE).
            Use False on memory-constrained GPUs (e.g., L40S 48GB).
    """
    comp_names = [f"_tq_{pname}" for pname in param_names]
    buf_names = [f"_tq_buf_{pname}" for pname in param_names]

    def pre_hook(mod, args):
        for pname, comp_name, buf_name in zip(param_names, comp_names, buf_names):
            compressed = getattr(mod, comp_name, None)
            if compressed is None:
                continue
            buf = getattr(mod, buf_name, None) if pool_buffers else None
            decompressed = compressed.decompress(buf=buf)
            if pool_buffers and (buf is None or buf.data_ptr() != decompressed.data_ptr()):
                setattr(mod, buf_name, decompressed)
            getattr(mod, pname).data = decompressed

    def post_hook(mod, args, output):
        for pname, comp_name, buf_name in zip(param_names, comp_names, buf_names):
            compressed = getattr(mod, comp_name, None)
            if compressed is not None:
                getattr(mod, pname).data = torch.empty(0, device=compressed.device, dtype=compressed.dtype)
                if not pool_buffers:
                    # Free the decompression buffer to save GPU memory
                    if hasattr(mod, buf_name):
                        delattr(mod, buf_name)
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
    kurt = (normalized**4).mean().item()  # excess kurtosis + 3

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
        gate = getattr(module, "gate", None)
        if gate is not None and hasattr(gate, "weight"):
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


def _replace_linear_layers(
    model: nn.Module,
    bits: int,
    group_size: int = 128,
    min_size: int = 1024,
    kurtosis_aware: bool = False,
    prune_experts: float = 0.0,
    routed_expert_bits: int | None = None,
    per_module_bits: dict[str, int] | None = None,
    learned_rotations: dict[str, torch.Tensor] | None = None,
):
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
                gate_path,
                n_keep,
                num_experts,
                prune_experts * 100,
                n_marked,
            )

    # Phase 1: Replace nn.Linear layers with TurboQuantWrapper
    # Also match vLLM's parallel linear layers (ColumnParallelLinear, etc.)
    # which have .weight but inherit from LinearBase, not nn.Linear
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            # Check for vLLM parallel linears: have weight + input_size/output_size
            if not (
                hasattr(module, "weight")
                and isinstance(getattr(module, "weight", None), (torch.Tensor, nn.Parameter))
                and (hasattr(module, "input_size") or hasattr(module, "in_features"))
            ):
                continue
            # Normalize attribute names for vLLM compatibility
            if not hasattr(module, "in_features") and hasattr(module, "input_size"):
                module.in_features = module.input_size
            if not hasattr(module, "out_features") and hasattr(module, "output_size_per_partition"):
                module.out_features = module.output_size_per_partition
            elif not hasattr(module, "out_features") and hasattr(module, "output_size"):
                module.out_features = module.output_size
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

        parent, attr_name = _resolve_parent_and_attr(model, name)

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
        wrapper = TurboQuantWrapper(module, bits=tensor_bits, group_size=group_size, rotation=rotation)
        setattr(parent, attr_name, wrapper)

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
                gate_path,
                n_keep,
                num_experts,
                prune_experts * 100,
            )

    for name, param in list(model.named_parameters()):
        if param.dim() != 3:
            continue
        if param.shape[-1] < min_size and param.shape[-2] < min_size:
            continue
        if any(p in name.lower() for p in _SKIP_PATTERNS):
            continue

        owner, param_name = _resolve_parent_and_attr(model, name)

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
            "TQ%d-g%d%s weight compression: %d linear + %d expert layers, %.1f GB -> %.1f GB (%.1fx)%s",
            bits,
            group_size,
            expert_bits_msg,
            replacements,
            expert_layers,
            total_original / 1e9,
            total_compressed / 1e9,
            ratio,
            prune_msg,
        )
        # Guard: on CPU-only PyTorch builds (Mac, CI, etc.) calling
        # torch.cuda.empty_cache() raises AssertionError.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return total


def patch_vllm_loader(**replace_kwargs) -> None:
    """Monkey-patch vLLM's process_weights_after_loading to call _replace_linear_layers.

    Wraps the original function so that after normal weight loading,
    _replace_linear_layers is called with the given keyword arguments.

    Raises ImportError if vLLM is not installed.
    """
    import vllm.model_executor.model_loader.utils as loader_utils
    import vllm.model_executor.model_loader.base_loader as base_loader

    _original = loader_utils.process_weights_after_loading

    def patched_process_weights(model, model_config, target_device):
        _original(model, model_config, target_device)
        logger.info("Applying TurboQuant weight compression...")
        count = _replace_linear_layers(model, **replace_kwargs)
        if count > 0:
            mem_gb = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
            logger.info("TurboQuant: %d layers compressed, GPU memory: %.1f GB", count, mem_gb)

    # Patch both the module-level function AND all imported references.
    # Python's `from X import Y` binds a local name that doesn't see
    # later module-level replacements, so we must patch every importer.
    loader_utils.process_weights_after_loading = patched_process_weights
    base_loader.process_weights_after_loading = patched_process_weights
    try:
        import vllm.model_executor.model_loader.gguf_loader as gguf_loader

        gguf_loader.process_weights_after_loading = patched_process_weights
    except (ImportError, AttributeError):
        pass


def enable_weight_quantization(
    bits: int = 3,
    group_size: int = 128,
    min_layer_size: int = 1024,
    kurtosis_aware: bool = False,
    prune_experts: float = 0.0,
    routed_expert_bits: int | None = None,
):
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

    os.environ["TQ_WEIGHT_BITS"] = str(bits)
    os.environ["TQ_WEIGHT_GROUP_SIZE"] = str(group_size)

    try:
        patch_vllm_loader(
            bits=bits,
            group_size=group_size,
            min_size=min_layer_size,
            kurtosis_aware=kurtosis_aware,
            prune_experts=prune_experts,
            routed_expert_bits=routed_expert_bits,
        )
    except ImportError:
        logger.error("Cannot import vLLM model loader. Is vLLM installed?")
        raise

    prune_msg = f", {prune_experts * 100:.0f}% expert pruning" if prune_experts > 0 else ""
    expert_msg = f", routed@TQ{routed_expert_bits}" if routed_expert_bits else ""
    logger.info("TurboQuant TQ%d-g%d weight compression enabled%s%s", bits, group_size, prune_msg, expert_msg)


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
    logger.info("Loading %s to CPU for compression...", model_id)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

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

    for param in model.parameters():
        param.data = param.data.half()
    for buf_name, buf in model.named_buffers():
        if buf.is_floating_point():
            buf.data = buf.data.half()
    model.config.torch_dtype = "float16"

    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)

    total_size = sum(os.path.getsize(os.path.join(output_dir, f)) for f in os.listdir(output_dir))
    logger.info("Saved to %s (%.1f GB)", output_dir, total_size / 1e9)
