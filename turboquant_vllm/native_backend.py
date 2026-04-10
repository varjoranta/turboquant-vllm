"""TurboQuant native vLLM attention backend.

This is the proper fix for the vLLM V1 subprocess issue. Instead of
monkey-patching FlashAttentionImpl (which breaks in V1 because CUDA-graph
capture skips non-CUDA-graphable Python operations), this backend is
registered as a first-class vLLM attention backend.

vLLM selects backends at model initialization time (per-process), so there
is no subprocess ordering issue.  All store/decode operations go through
tensor indexing that CUDA graphs can capture.

Cache layout (no leading 2 dimension):
  (num_blocks, block_size, num_kv_heads, slot_size)

Per-head per-position slot layout:
  [key_packed (kps bytes) | value (vps bytes)]
  For tq3 head_dim=128 no_qjl: [50 bytes key | 128 bytes value] = 178 total

Activation:
  vllm serve <model> --kv-cache-dtype tq3
  (or tq4, tq_k4v3)

Registration happens inside turboquant_vllm._vllm_plugin.register() which
patches the vLLM backend selector and CacheDType validator at plugin load.
"""

import math
import os
from dataclasses import dataclass
from typing import ClassVar, Optional

import torch
import torch.nn.functional as F

from turboquant_vllm.tq_config import TurboQuantConfig

_USE_STREAM_OVERLAP = os.environ.get("TQ_STREAM_OVERLAP", "0") == "1"
_TQ_NO_QJL = os.environ.get("TQ_NO_QJL", "1") == "1"

_store_stream: Optional[torch.cuda.Stream] = None

# ---------------------------------------------------------------------------
# Triton fast paths — try to import from vllm fork, then from our own ops,
# then fall back to Python.
# ---------------------------------------------------------------------------
_USE_TRITON_STORE = False
_triton_tq_store = None
if os.environ.get("TQ_PYTHON_STORE", "0") != "1":
    for _store_src in [
        "vllm.v1.attention.ops.triton_tq_store",
        "turboquant_vllm.ops.triton_tq_store",
    ]:
        try:
            import importlib
            _mod = importlib.import_module(_store_src)
            _triton_tq_store = _mod.triton_tq_store
            _USE_TRITON_STORE = True
            break
        except (ImportError, AttributeError):
            pass

_USE_TRITON_DECODE = False
_triton_tq_decode = None
if os.environ.get("TQ_PYTHON_DECODE", "0") != "1":
    for _decode_src in [
        "vllm.v1.attention.ops.triton_tq_decode",
        "turboquant_vllm.ops.triton_tq_decode",
    ]:
        try:
            import importlib
            _mod = importlib.import_module(_decode_src)
            _triton_tq_decode = _mod.triton_tq_decode_attention
            _USE_TRITON_DECODE = True
            break
        except (ImportError, AttributeError):
            pass

# Flash attention for prefill
try:
    from vllm.v1.attention.backends.fa_utils import (
        is_flash_attn_varlen_func_available,
        flash_attn_varlen_func,
    )
    _HAS_FLASH_ATTN = is_flash_attn_varlen_func_available()
except ImportError:
    _HAS_FLASH_ATTN = False

try:
    from vllm.logger import init_logger
    logger = init_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Backend class
# ---------------------------------------------------------------------------

class TurboQuantAttentionBackend:
    """TurboQuant attention backend for vLLM.

    Registered as the CUSTOM backend when kv_cache_dtype starts with 'tq'.
    """

    accept_output_buffer: bool = True
    forward_includes_kv_cache_update: bool = False

    supported_kv_cache_dtypes: ClassVar[list[str]] = ["tq3", "tq4", "tq_k4v3"]

    @staticmethod
    def get_name() -> str:
        return "TURBOQUANT"

    @staticmethod
    def get_supported_kernel_block_sizes() -> list:
        return [16, 32, 64, 128]

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        try:
            from vllm.v1.attention.backend import AttentionType
            return attn_type == AttentionType.DECODER
        except ImportError:
            return attn_type == "decoder"

    @classmethod
    def supports_per_head_quant_scales(cls) -> bool:
        return False

    @staticmethod
    def get_impl_cls() -> "type[TurboQuantAttentionImpl]":
        return TurboQuantAttentionImpl

    @staticmethod
    def get_builder_cls() -> "type[TurboQuantMetadataBuilder]":
        return TurboQuantMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "tq3",
    ) -> tuple:
        """Cache shape: (num_blocks, block_size, num_kv_heads, padded_slot_size).

        head_size here is effective_head_size (padded_slot // 2) from vLLM spec,
        not the model's actual head_dim. padded_slot = head_size * 2.
        """
        return (num_blocks, block_size, num_kv_heads, head_size * 2)

    @classmethod
    def supports_kv_cache_dtype(cls, kv_cache_dtype) -> bool:
        if kv_cache_dtype is None:
            return False
        return str(kv_cache_dtype) in ("tq3", "tq4", "tq_k4v3")

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        # head_size from spec = effective_head_size (padded_slot//2)
        return head_size > 0

    @classmethod
    def validate_configuration(cls, **kwargs) -> list[str]:
        """Return list of reasons this backend is invalid (empty = valid)."""
        kv_cache_dtype = kwargs.get("kv_cache_dtype")
        if not cls.supports_kv_cache_dtype(kv_cache_dtype):
            return [f"kv_cache_dtype must be tq3/tq4/tq_k4v3, got {kv_cache_dtype}"]
        return []


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

@dataclass
class TurboQuantMetadata:
    seq_lens: torch.Tensor
    slot_mapping: torch.Tensor
    block_table: torch.Tensor
    query_start_loc: torch.Tensor
    num_actual_tokens: int = 0
    max_query_len: int = 0
    max_seq_len: int = 0
    is_prefill: bool = False


class TurboQuantMetadataBuilder:
    """Builds TurboQuantMetadata from scheduler output."""

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        # Don't call super().__init__ — AttentionMetadataBuilder base class
        # may require specific vLLM-internal state we don't need.
        self.kv_cache_spec = kv_cache_spec
        self.layer_names = layer_names
        self.vllm_config = vllm_config
        self.device = device

    def reorder_batch(self, input_batch, scheduler_output):
        return False

    def build_for_cudagraph_capture(self, common_attn_metadata) -> TurboQuantMetadata:
        attn_metadata = self.build(0, common_attn_metadata)
        attn_metadata.seq_lens.fill_(1)
        return attn_metadata

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        cam = common_attn_metadata
        return TurboQuantMetadata(
            seq_lens=cam.seq_lens,
            slot_mapping=cam.slot_mapping,
            block_table=cam.block_table_tensor,
            query_start_loc=cam.query_start_loc,
            num_actual_tokens=cam.num_actual_tokens,
            max_query_len=cam.max_query_len,
            max_seq_len=cam.max_seq_len,
            is_prefill=(cam.max_query_len > 1),
        )


# ---------------------------------------------------------------------------
# Impl
# ---------------------------------------------------------------------------

class TurboQuantAttentionImpl:
    """TurboQuant attention implementation.

    Stores compressed K/V into vLLM's kv_cache tensor (no sidecar dict),
    so all operations are CUDA-graph-compatible.
    """

    supports_quant_query_input: bool = False

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int | None = None,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype: str = "auto",
        logits_soft_cap=None,
        attn_type: str = "decoder",
        kv_sharing_target_layer_name=None,
        **kwargs,
    ):
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = scale
        self.num_kv_heads = num_kv_heads or num_heads
        self.num_kv_groups = num_heads // self.num_kv_heads
        self.kv_cache_dtype = kv_cache_dtype

        self._tq_cache_dtype = kv_cache_dtype
        self.tq_config: Optional[TurboQuantConfig] = None

        self._shift_2bit = torch.tensor([0, 2, 4, 6], dtype=torch.int32)
        self._shift_4bit = torch.tensor([0, 4], dtype=torch.int32)
        self._shift_8bit = torch.arange(8, dtype=torch.int32)
        self._shifts_on_device = False
        self._use_triton_decode = _USE_TRITON_DECODE

    def _ensure_on_device(self, layer: torch.nn.Module, device: torch.device) -> None:
        """One-time migration of TQ buffers to GPU and cache precomputed matrices."""
        Pi = layer._tq_Pi

        if self.tq_config is None:
            actual_head_dim = Pi.shape[0]
            self.tq_config = TurboQuantConfig.from_cache_dtype(
                self._tq_cache_dtype, actual_head_dim)

        if Pi.device != device:
            layer._tq_Pi = Pi.to(device)
            layer._tq_S = layer._tq_S.to(device)
            layer._tq_centroids = layer._tq_centroids.to(device)

        if not self._shifts_on_device:
            self._shift_2bit = self._shift_2bit.to(device)
            self._shift_4bit = self._shift_4bit.to(device)
            self._shift_8bit = self._shift_8bit.to(device)
            self._shifts_on_device = True

        if not hasattr(layer, '_tq_cached'):
            Pi_f = layer._tq_Pi.float().contiguous()
            S_f = layer._tq_S.float().contiguous()
            c = layer._tq_centroids.float()
            layer._tq_PiT = Pi_f.T.contiguous()
            layer._tq_Pi_S_T = (Pi_f @ S_f.T).contiguous()
            c_sorted, _ = c.sort()
            layer._tq_midpoints = (c_sorted[:-1] + c_sorted[1:]) / 2
            layer._tq_cached = True

    # ------------------------------------------------------------------ #
    #  vLLM interface                                                      #
    # ------------------------------------------------------------------ #

    def do_kv_cache_update(
        self,
        layer: torch.nn.Module,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        global _store_stream

        N = slot_mapping.shape[0]
        if N <= 0:
            return

        device = key.device
        self._ensure_on_device(layer, device)

        k = key[:N].view(N, self.num_kv_heads, self.head_size)
        v = value[:N].view(N, self.num_kv_heads, self.head_size)
        self._current_layer = layer

        use_overlap = (
            _USE_STREAM_OVERLAP
            and _USE_TRITON_STORE
            and not torch.cuda.is_current_stream_capturing()
        )
        if use_overlap:
            if _store_stream is None:
                _store_stream = torch.cuda.Stream(device=device)
            torch.cuda.current_stream(device).wait_stream(_store_stream)
            with torch.cuda.stream(_store_stream):
                self._store_kv(k, v, kv_cache, slot_mapping,
                               layer._tq_Pi, layer._tq_S, layer._tq_centroids)
        else:
            self._store_kv(k, v, kv_cache, slot_mapping,
                           layer._tq_Pi, layer._tq_S, layer._tq_centroids)

    def forward(
        self,
        layer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: TurboQuantMetadata,
        output: Optional[torch.Tensor] = None,
        output_scale=None,
        output_block_scale=None,
    ) -> torch.Tensor:
        num_tokens = query.shape[0]
        if output is None:
            output = torch.zeros(
                num_tokens, self.num_heads * self.head_size,
                dtype=query.dtype, device=query.device)
        assert output is not None

        if attn_metadata is None:
            return output.fill_(0)

        N = attn_metadata.num_actual_tokens
        if N <= 0:
            return output.fill_(0)

        q = query[:N].view(N, self.num_heads, self.head_size)
        device = q.device
        self._ensure_on_device(layer, device)
        assert self.tq_config is not None
        Pi = layer._tq_Pi
        S = layer._tq_S
        centroids = layer._tq_centroids

        if (_store_stream is not None
                and not attn_metadata.is_prefill
                and not torch.cuda.is_current_stream_capturing()):
            torch.cuda.current_stream(device).wait_stream(_store_stream)

        if not attn_metadata.is_prefill:
            attn_out = self._decode_attention(q, kv_cache, attn_metadata, Pi, S, centroids)
        else:
            query_start_loc = attn_metadata.query_start_loc
            num_reqs = query_start_loc.shape[0] - 1
            if attn_metadata.max_query_len == attn_metadata.max_seq_len:
                has_decodes = False
                q_lens = None
            else:
                q_lens = query_start_loc[1:] - query_start_loc[:num_reqs]
                has_decodes = bool((q_lens == 1).any().item())

            if not has_decodes:
                k = key[:N].view(N, self.num_kv_heads, self.head_size)
                v = value[:N].view(N, self.num_kv_heads, self.head_size)
                attn_out = self._prefill_attention(q, k, v, attn_metadata)
            else:
                assert q_lens is not None
                attn_out = self._mixed_batch_attention(
                    q, key[:N].view(N, self.num_kv_heads, self.head_size),
                    value[:N].view(N, self.num_kv_heads, self.head_size),
                    kv_cache, attn_metadata, Pi, S, centroids,
                    query_start_loc, q_lens, num_reqs,
                )

        if output.ndim == 3:
            output[:N] = attn_out.to(output.dtype)
        else:
            output[:N] = attn_out.reshape(N, -1).to(output.dtype)
        return output

    # ------------------------------------------------------------------ #
    #  Store K/V into combined cache                                       #
    # ------------------------------------------------------------------ #

    def _store_kv(
        self,
        key: torch.Tensor,      # (N, Hk, D)
        value: torch.Tensor,    # (N, Hk, D)
        kv_cache: torch.Tensor, # (num_blocks, block_size, Hk, slot_size)
        slot_mapping: torch.Tensor,
        Pi: torch.Tensor,
        S: torch.Tensor,
        centroids: torch.Tensor,
    ) -> None:
        N, H, D = key.shape
        layer = self._current_layer
        assert self.tq_config is not None

        if _USE_TRITON_STORE and _triton_tq_store is not None:
            _triton_tq_store(
                key, value, kv_cache, slot_mapping,
                layer._tq_PiT, layer._tq_Pi_S_T, centroids, layer._tq_midpoints,
                mse_bits=self.tq_config.mse_bits,
                key_packed_size=self.tq_config.key_packed_size,
                value_quant_bits=self.tq_config.effective_value_quant_bits,
                value_packed_size=self.tq_config.value_packed_size,
                no_qjl=_TQ_NO_QJL,
            )
            return

        mse_bits = self.tq_config.mse_bits
        kps = self.tq_config.key_packed_size
        mse_bytes_n = math.ceil(D * mse_bits / 8)
        qjl_bytes_n = math.ceil(D / 8)
        block_size = kv_cache.shape[1]
        device = key.device

        PiT = layer._tq_PiT
        Pi_S_T = layer._tq_Pi_S_T
        midpoints = layer._tq_midpoints

        k_flat = key.float().reshape(-1, D)
        norms = k_flat.norm(dim=1, keepdim=True)
        x_hat = k_flat / (norms + 1e-8)
        y = x_hat @ PiT
        idx = torch.bucketize(y, midpoints).to(torch.uint8)

        signs: Optional[torch.Tensor] = None
        gamma: Optional[torch.Tensor] = None
        if not self.tq_config.no_qjl:
            y_hat = centroids[idx.long()]
            r_rot = y - y_hat
            gamma = r_rot.norm(dim=1, keepdim=True)
            projected = r_rot @ Pi_S_T
            signs = (projected >= 0).to(torch.uint8)

        # Pack MSE indices.
        # 4-bit: nibble pack (D//2 bytes)
        # 3-bit: tight 8-into-3-bytes pack when D % 8 == 0 (3D/8 bytes)
        # 2-bit: 4-per-byte (D//4 bytes)
        # other: tight byte-spanning pack (ceil(D*mse_bits/8) bytes)
        if mse_bits == 4 and D % 2 == 0:
            idx_r = idx.reshape(-1, D // 2, 2)
            packed_mse = (idx_r[:, :, 0].int() | (idx_r[:, :, 1].int() << 4)).to(torch.uint8)
        elif mse_bits == 3 and D % 8 == 0:
            # True 3-bit packing: 8 indices → 3 bytes (Gaby PR #4)
            idx_r = idx.reshape(-1, D // 8, 8).int()
            packed_mse = torch.empty(N * H, D // 8 * 3, dtype=torch.uint8, device=device)
            packed_mse[:, 0::3] = (
                (idx_r[:, :, 0] & 0x7)
                | ((idx_r[:, :, 1] & 0x7) << 3)
                | ((idx_r[:, :, 2] & 0x3) << 6)
            ).to(torch.uint8)
            packed_mse[:, 1::3] = (
                ((idx_r[:, :, 2] >> 2) & 0x1)
                | ((idx_r[:, :, 3] & 0x7) << 1)
                | ((idx_r[:, :, 4] & 0x7) << 4)
                | ((idx_r[:, :, 5] & 0x1) << 7)
            ).to(torch.uint8)
            packed_mse[:, 2::3] = (
                ((idx_r[:, :, 5] >> 1) & 0x3)
                | ((idx_r[:, :, 6] & 0x7) << 2)
                | ((idx_r[:, :, 7] & 0x7) << 5)
            ).to(torch.uint8)
        elif mse_bits == 2 and D % 4 == 0:
            idx_r = idx.reshape(-1, D // 4, 4)
            packed_mse = (idx_r.int() << self._shift_2bit).sum(-1).to(torch.uint8)
        else:
            packed_mse = torch.zeros(N * H, mse_bytes_n, dtype=torch.uint8, device=device)
            idx_u8 = idx.to(torch.uint8)
            for j in range(D):
                bo = j * mse_bits
                bi, si = bo // 8, bo % 8
                packed_mse[:, bi] |= ((idx_u8[:, j].int() << si) & 0xFF).to(torch.uint8)
                if si + mse_bits > 8 and bi + 1 < mse_bytes_n:
                    packed_mse[:, bi + 1] |= (
                        (idx_u8[:, j].int() >> (8 - si)) & 0xFF).to(torch.uint8)

        norm_b = norms.squeeze(-1).half().contiguous().view(torch.uint8).reshape(-1, 2)
        no_qjl = self.tq_config.no_qjl

        if no_qjl:
            packed_key = torch.cat([packed_mse, norm_b], dim=1)
        else:
            assert signs is not None and gamma is not None
            if D % 8 == 0:
                signs_r = signs.reshape(-1, D // 8, 8)
                packed_signs = (signs_r.int() << self._shift_8bit).sum(-1).to(torch.uint8)
            else:
                packed_signs = torch.zeros(N * H, qjl_bytes_n, dtype=torch.uint8, device=device)
                for j in range(D):
                    packed_signs[:, j // 8] |= (signs[:, j] << (j % 8))
            gamma_b = gamma.squeeze(-1).half().contiguous().view(torch.uint8).reshape(-1, 2)
            packed_key = torch.cat([packed_mse, packed_signs, norm_b, gamma_b], dim=1)

        # Pack values
        vps = self.tq_config.value_packed_size
        if self.tq_config.value_fp8:
            v_flat = value.reshape(-1, D)
            packed_value = v_flat.to(torch.float8_e4m3fn).view(torch.uint8)
        else:
            vqb = self.tq_config.effective_value_quant_bits
            val_data_bytes = math.ceil(D * vqb / 8)
            qmax = (1 << vqb) - 1
            v_flat = value.float().reshape(-1, D)
            vmin = v_flat.min(dim=1, keepdim=True).values
            vmax = v_flat.max(dim=1, keepdim=True).values
            v_scale = ((vmax - vmin) / qmax).clamp(min=1e-8)
            v_idx = ((v_flat - vmin) / v_scale).round().clamp(0, qmax).to(torch.uint8)
            if vqb == 2 and D % 4 == 0:
                v_idx_r = v_idx.reshape(-1, D // 4, 4)
                packed_val = (v_idx_r.int() << self._shift_2bit).sum(-1).to(torch.uint8)
            elif vqb == 4 and D % 2 == 0:
                v_idx_r = v_idx.reshape(-1, D // 2, 2)
                packed_val = (v_idx_r.int() << self._shift_4bit).sum(-1).to(torch.uint8)
            else:
                packed_val = torch.zeros(N * H, val_data_bytes, dtype=torch.uint8, device=device)
                v_u8 = v_idx
                for j in range(D):
                    bo = j * vqb
                    bi, si = bo // 8, bo % 8
                    packed_val[:, bi] |= ((v_u8[:, j].int() << si) & 0xFF).to(torch.uint8)
                    if si + vqb > 8 and bi + 1 < val_data_bytes:
                        packed_val[:, bi + 1] |= (
                            (v_u8[:, j].int() >> (8 - si)) & 0xFF).to(torch.uint8)
            v_scale_b = v_scale.squeeze(-1).half().contiguous().view(torch.uint8).reshape(-1, 2)
            v_zero_b = vmin.squeeze(-1).half().contiguous().view(torch.uint8).reshape(-1, 2)
            packed_value = torch.cat([packed_val, v_scale_b, v_zero_b], dim=1)

        # Scatter to cache
        packed_key = packed_key.reshape(N, H, kps)
        packed_value = packed_value.reshape(N, H, vps)
        valid = slot_mapping >= 0
        safe_slots = torch.where(valid, slot_mapping, torch.zeros_like(slot_mapping))
        blk_idx = (safe_slots // block_size).long()
        blk_off = (safe_slots % block_size).long()
        existing_key = kv_cache[blk_idx, blk_off, :, :kps]
        existing_value = kv_cache[blk_idx, blk_off, :, kps:kps + vps]
        valid_mask = valid.view(N, 1, 1)
        packed_key = torch.where(valid_mask, packed_key, existing_key)
        packed_value = torch.where(valid_mask, packed_value, existing_value)
        kv_cache[blk_idx, blk_off, :, :kps] = packed_key
        kv_cache[blk_idx, blk_off, :, kps:kps + vps] = packed_value

    # ------------------------------------------------------------------ #
    #  Prefill: SDPA on raw Q/K/V                                         #
    # ------------------------------------------------------------------ #

    def _prefill_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: TurboQuantMetadata,
    ) -> torch.Tensor:
        N, Hq, D = query.shape
        if _HAS_FLASH_ATTN and attn_metadata.max_query_len == attn_metadata.max_seq_len:
            output = torch.empty(N, Hq, D, device=query.device, dtype=query.dtype)
            flash_attn_varlen_func(
                q=query, k=key, v=value,
                cu_seqlens_q=attn_metadata.query_start_loc,
                cu_seqlens_k=attn_metadata.query_start_loc,
                max_seqlen_q=attn_metadata.max_query_len,
                max_seqlen_k=attn_metadata.max_query_len,
                softmax_scale=self.scale,
                causal=True, out=output,
            )
            return output

        Hk = key.shape[1]
        use_gqa = Hk < Hq
        query_start_loc = attn_metadata.query_start_loc
        num_reqs = query_start_loc.shape[0] - 1
        output = torch.zeros(N, Hq, D, device=query.device, dtype=query.dtype)
        for i in range(num_reqs):
            q_start = query_start_loc[i].item()
            q_end = query_start_loc[i + 1].item()
            if q_end <= q_start:
                continue
            q_t = query[q_start:q_end].transpose(0, 1).contiguous()
            k_t = key[q_start:q_end].transpose(0, 1).contiguous()
            v_t = value[q_start:q_end].transpose(0, 1).contiguous()
            out = F.scaled_dot_product_attention(
                q_t, k_t, v_t, is_causal=True, scale=self.scale, enable_gqa=use_gqa)
            output[q_start:q_end] = out.transpose(0, 1).to(query.dtype)
        return output

    # ------------------------------------------------------------------ #
    #  Decode: dispatch Triton or Python                                   #
    # ------------------------------------------------------------------ #

    def _decode_attention(
        self,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: TurboQuantMetadata,
        Pi: torch.Tensor,
        S: torch.Tensor,
        centroids: torch.Tensor,
    ) -> torch.Tensor:
        assert self.tq_config is not None
        if self._use_triton_decode and _triton_tq_decode is not None:
            try:
                return _triton_tq_decode(
                    query=query, kv_cache=kv_cache,
                    block_table=attn_metadata.block_table,
                    seq_lens=attn_metadata.seq_lens,
                    Pi=Pi, S=S, centroids=centroids,
                    scale=self.scale,
                    mse_bits=self.tq_config.mse_bits,
                    key_packed_size=self.tq_config.key_packed_size,
                    value_quant_bits=self.tq_config.effective_value_quant_bits,
                    value_packed_size=self.tq_config.value_packed_size,
                    max_seq_len=attn_metadata.max_seq_len,
                )
            except Exception as e:
                logger.warning("Triton TQ decode failed (%s), falling back to Python", e)
                self._use_triton_decode = False

        return self._decode_attention_python(query, kv_cache, attn_metadata, Pi, S, centroids)

    def _decode_attention_python(
        self,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: TurboQuantMetadata,
        Pi: torch.Tensor,
        S: torch.Tensor,
        centroids: torch.Tensor,
    ) -> torch.Tensor:
        assert self.tq_config is not None
        B, Hq, D = query.shape
        Hk = self.num_kv_heads
        kps = self.tq_config.key_packed_size
        mse_bits = self.tq_config.mse_bits
        mse_bytes_n = math.ceil(D * mse_bits / 8)
        no_qjl = self.tq_config.no_qjl
        block_size = kv_cache.shape[1]
        device = query.device

        q_float = query.float()
        q_rot = q_float @ Pi.T
        q_proj: Optional[torch.Tensor] = None
        correction = 0.0
        sign_sh: Optional[torch.Tensor] = None
        if not no_qjl:
            q_proj = q_float @ S.T
            correction = math.sqrt(math.pi / 2) / D
            sign_sh = torch.arange(8, device=device, dtype=torch.int32)

        if mse_bits == 2:
            mse_sh: Optional[torch.Tensor] = torch.tensor([0, 2, 4, 6], device=device, dtype=torch.int32)
            mse_mask = 0x3
        elif mse_bits == 4:
            mse_sh = torch.tensor([0, 4], device=device, dtype=torch.int32)
            mse_mask = 0xF
        else:
            # 3-bit uses its own 8-into-3-bytes unpack below — no shift tensor
            mse_sh = None
            mse_mask = (1 << mse_bits) - 1

        outputs = []
        for i in range(B):
            seq_len = attn_metadata.seq_lens[i].item()
            if seq_len <= 0:
                outputs.append(torch.zeros(Hq, D, device=device, dtype=query.dtype))
                continue

            pos = torch.arange(seq_len, device=device)
            blk_idx = attn_metadata.block_table[i, pos // block_size].long()
            blk_off = (pos % block_size).long()
            slots = kv_cache[blk_idx, blk_off]  # (S, Hk, slot_size)

            mse_raw = slots[:, :, :mse_bytes_n]
            if mse_bits == 2 and D % 4 == 0:
                expanded = mse_raw.unsqueeze(-1).int() >> mse_sh
                idx = (expanded & mse_mask).reshape(seq_len, Hk, -1)[:, :, :D]
            elif mse_bits == 4 and D % 2 == 0:
                # Nibble unpack (4-bit only; 3-bit has its own tight path below)
                expanded = mse_raw.unsqueeze(-1).int() >> mse_sh
                idx = (expanded & mse_mask).reshape(seq_len, Hk, -1)[:, :, :D]
            elif mse_bits == 3 and D % 8 == 0:
                b0 = mse_raw[:, :, 0::3].int()
                b1 = mse_raw[:, :, 1::3].int()
                b2 = mse_raw[:, :, 2::3].int()
                idx = torch.stack([
                    b0 & 7,
                    (b0 >> 3) & 7,
                    ((b0 >> 6) | (b1 << 2)) & 7,
                    (b1 >> 1) & 7,
                    (b1 >> 4) & 7,
                    ((b1 >> 7) | (b2 << 1)) & 7,
                    (b2 >> 2) & 7,
                    (b2 >> 5) & 7,
                ], dim=-1).reshape(seq_len, Hk, D)
            else:
                j = torch.arange(D, device=device)
                bo = j * mse_bits
                bi = (bo // 8).long()
                si = (bo % 8).int()
                mse_raw_padded = F.pad(mse_raw, (0, 1))
                b0 = mse_raw_padded[:, :, bi].int()
                val = (b0 >> si) & ((1 << mse_bits) - 1)
                need_next = si + mse_bits > 8
                bi_next = bi + 1
                b1 = mse_raw_padded[:, :, bi_next].int()
                high = (b1 << (8 - si)) & ((1 << mse_bits) - 1)
                idx = torch.where(need_next, val | high, val)

            c_vals = centroids[idx.long()]

            signs_f: Optional[torch.Tensor] = None
            gammas: Optional[torch.Tensor] = None
            if no_qjl:
                noff = mse_bytes_n
                nd = slots[:, :, noff:noff + 2].contiguous()
                vec_norms = nd.view(torch.float16).squeeze(-1).float()
            else:
                assert sign_sh is not None
                qjl_bytes_n_local = math.ceil(D / 8)
                sign_raw = slots[:, :, mse_bytes_n:mse_bytes_n + qjl_bytes_n_local]
                if D % 8 == 0:
                    s_exp = sign_raw.unsqueeze(-1).int() >> sign_sh
                    signs_01 = (s_exp & 1).reshape(seq_len, Hk, -1)[:, :, :D]
                else:
                    j = torch.arange(D, device=device)
                    s_byte = sign_raw[:, :, j // 8]
                    signs_01 = (s_byte.int() >> (j % 8).int()) & 1
                signs_f = signs_01.float() * 2.0 - 1.0
                noff = mse_bytes_n + qjl_bytes_n_local
                nd = slots[:, :, noff:noff + 2].contiguous()
                gd = slots[:, :, noff + 2:noff + 4].contiguous()
                vec_norms = nd.view(torch.float16).squeeze(-1).float()
                gammas = gd.view(torch.float16).squeeze(-1).float()

            if Hk < Hq:
                g = self.num_kv_groups
                c_vals = c_vals.repeat_interleave(g, dim=1)
                vec_norms = vec_norms.repeat_interleave(g, dim=1)
                if not no_qjl:
                    assert signs_f is not None and gammas is not None
                    signs_f = signs_f.repeat_interleave(g, dim=1)
                    gammas = gammas.repeat_interleave(g, dim=1)

            q_rot_i = q_rot[i]
            term1 = torch.einsum('hd,shd->sh', q_rot_i, c_vals)
            if no_qjl:
                scores = vec_norms * term1 * self.scale
            else:
                assert q_proj is not None and signs_f is not None and gammas is not None
                q_proj_i = q_proj[i]
                term2 = torch.einsum('hd,shd->sh', q_proj_i, signs_f)
                scores = vec_norms * (term1 + correction * gammas * term2) * self.scale

            attn_w = torch.softmax(scores.T, dim=-1)  # (Hq, S)

            if self.tq_config.value_fp8:
                val_raw = slots[:, :, kps:kps + D]
                values = val_raw.view(torch.float8_e4m3fn).float()
            else:
                vqb = self.tq_config.effective_value_quant_bits
                val_data_bytes = math.ceil(D * vqb / 8)
                qmax = (1 << vqb) - 1
                val_raw = slots[:, :, kps:kps + val_data_bytes]
                if vqb == 2 and D % 4 == 0:
                    v_sh = torch.tensor([0, 2, 4, 6], device=device, dtype=torch.int32)
                    v_exp = val_raw.unsqueeze(-1).int() >> v_sh
                    v_idx = (v_exp & 0x3).reshape(seq_len, Hk, -1)[:, :, :D].float()
                elif vqb == 4 and D % 2 == 0:
                    v_sh = torch.tensor([0, 4], device=device, dtype=torch.int32)
                    v_exp = val_raw.unsqueeze(-1).int() >> v_sh
                    v_idx = (v_exp & 0xF).reshape(seq_len, Hk, -1)[:, :, :D].float()
                else:
                    j = torch.arange(D, device=device)
                    bo = j * vqb
                    bi = (bo // 8).long()
                    si = (bo % 8).int()
                    val_raw_padded = F.pad(val_raw, (0, 1))
                    b0 = val_raw_padded[:, :, bi].int()
                    low = (b0 >> si) & qmax
                    need_next = si + vqb > 8
                    b1 = val_raw_padded[:, :, bi + 1].int()
                    high = (b1 << (8 - si)) & qmax
                    v_idx = torch.where(need_next, low | high, low).float()
                sc_off = kps + val_data_bytes
                v_scale = slots[:, :, sc_off:sc_off + 2].contiguous().view(torch.float16).squeeze(-1).float()
                v_zero = slots[:, :, sc_off + 2:sc_off + 4].contiguous().view(torch.float16).squeeze(-1).float()
                values = v_idx * v_scale.unsqueeze(-1) + v_zero.unsqueeze(-1)

            if Hk < Hq:
                values = values.repeat_interleave(self.num_kv_groups, dim=1)

            out = torch.einsum('hs,shd->hd', attn_w, values)
            outputs.append(out.to(query.dtype))

        return torch.stack(outputs, dim=0)

    # ------------------------------------------------------------------ #
    #  Mixed batch: split prefill + decode                                 #
    # ------------------------------------------------------------------ #

    def _mixed_batch_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: TurboQuantMetadata,
        Pi: torch.Tensor,
        S: torch.Tensor,
        centroids: torch.Tensor,
        query_start_loc: torch.Tensor,
        q_lens: torch.Tensor,
        num_reqs: int,
    ) -> torch.Tensor:
        N, Hq, D = query.shape
        device = query.device
        output = torch.zeros(N, Hq, D, device=device, dtype=query.dtype)

        decode_mask = (q_lens == 1)
        prefill_mask = ~decode_mask

        if prefill_mask.any():
            for i in range(num_reqs):
                if decode_mask[i]:
                    continue
                q_start = query_start_loc[i].item()
                q_end = query_start_loc[i + 1].item()
                q_len = q_end - q_start
                if q_len <= 0:
                    continue
                Hk = key.shape[1]
                use_gqa = Hk < Hq
                q_t = query[q_start:q_end].transpose(0, 1).contiguous()
                k_t = key[q_start:q_end].transpose(0, 1).contiguous()
                v_t = value[q_start:q_end].transpose(0, 1).contiguous()
                out = F.scaled_dot_product_attention(
                    q_t, k_t, v_t, is_causal=True, scale=self.scale, enable_gqa=use_gqa)
                output[q_start:q_end] = out.transpose(0, 1)

        if decode_mask.any():
            decode_indices = decode_mask.nonzero(as_tuple=True)[0]
            num_decodes = decode_indices.shape[0]
            decode_token_offsets = query_start_loc[decode_indices]
            decode_q = query[decode_token_offsets]
            decode_meta = TurboQuantMetadata(
                seq_lens=attn_metadata.seq_lens[decode_indices],
                slot_mapping=attn_metadata.slot_mapping,
                block_table=attn_metadata.block_table[decode_indices],
                query_start_loc=torch.arange(num_decodes + 1, device=device, dtype=torch.int32),
                num_actual_tokens=num_decodes,
                max_query_len=1,
                max_seq_len=attn_metadata.seq_lens[decode_indices].max().item(),
                is_prefill=False,
            )
            decode_out = self._decode_attention(decode_q, kv_cache, decode_meta, Pi, S, centroids)
            for j in range(num_decodes):
                output[decode_token_offsets[j].item()] = decode_out[j]

        return output
