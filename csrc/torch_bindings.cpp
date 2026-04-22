/**
 * PyTorch C++ extension bindings for TurboQuant+ CUDA kernels.
 *
 * Registers operations under torch.ops.turbo_quant namespace:
 *   turbo_quant::init(centroids, boundaries, signs1, signs2, head_dim, bit_width)
 *   turbo_quant::init_qjl(qjl_matrix)
 *   turbo_quant::quantize(input, indices, norms)
 *   turbo_quant::dequantize(indices, norms, output)
 *   turbo_quant::reshape_and_cache(key, value, key_cache, value_cache,
 *                                   k_norms, v_norms, slot_mapping)
 *   turbo_quant::dequant_paged_cache(cache, norms, output, block_table, seq_len)
 */

#include "turbo_quant.h"
#include "tq_weight_dequant.h"
#include "tq_weight_gemv_bs1.h"
#include "tq3_gemv_bs1_fwht_smem.h"
#include <torch/extension.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "TurboQuant+ CUDA kernels for KV cache compression";

    m.def("init", &turbo_quant::init,
          "Initialize both K and V with same codebook (symmetric mode)",
          py::arg("centroids"), py::arg("boundaries"),
          py::arg("signs1"), py::arg("signs2"),
          py::arg("head_dim"), py::arg("bit_width"));

    m.def("init_k", &turbo_quant::init_k,
          "Initialize K cache codebook and rotation independently",
          py::arg("centroids"), py::arg("boundaries"),
          py::arg("signs1"), py::arg("signs2"),
          py::arg("head_dim"), py::arg("bit_width"));

    m.def("init_v", &turbo_quant::init_v,
          "Initialize V cache codebook and rotation independently",
          py::arg("centroids"), py::arg("boundaries"),
          py::arg("signs1"), py::arg("signs2"),
          py::arg("head_dim"), py::arg("bit_width"));

    m.def("init_qjl", &turbo_quant::init_qjl,
          "Upload QJL projection matrix for K cache",
          py::arg("qjl_matrix"));

    m.def("quantize", &turbo_quant::quantize,
          "Quantize fp16 vectors to TurboQuant indices + norms",
          py::arg("input"), py::arg("indices"), py::arg("norms"));

    m.def("dequantize", &turbo_quant::dequantize,
          "Dequantize TurboQuant indices + norms to fp16 vectors",
          py::arg("indices"), py::arg("norms"), py::arg("output"));

    m.def("reshape_and_cache", &turbo_quant::reshape_and_cache,
          "Fused quantize + pack into vLLM paged KV cache",
          py::arg("key"), py::arg("value"),
          py::arg("key_cache"), py::arg("value_cache"),
          py::arg("k_norms"), py::arg("v_norms"),
          py::arg("slot_mapping"));

    m.def("dequant_paged_cache", &turbo_quant::dequant_paged_cache,
          "Dequantize from paged cache to contiguous fp16 buffer",
          py::arg("cache"), py::arg("norms"), py::arg("output"),
          py::arg("block_table"), py::arg("seq_len"));

    // Weight dequantization kernels
    m.def("weight_dequant", &tq_weight_dequant,
          "Fused weight dequant: packed indices + norms + codebook → full matrix",
          py::arg("packed_weight"), py::arg("norms"),
          py::arg("signs1"), py::arg("signs2"),
          py::arg("centroids"), py::arg("output"),
          py::arg("group_size"), py::arg("bits"),
          py::arg("out_dim"), py::arg("in_dim"));

    m.def("weight_dequant_3d", &tq_weight_dequant_3d,
          "Fused weight dequant for MoE expert tensors (3D)",
          py::arg("packed_weight"), py::arg("norms"),
          py::arg("signs1"), py::arg("signs2"),
          py::arg("centroids"), py::arg("output"),
          py::arg("group_size"), py::arg("bits"),
          py::arg("n_experts"), py::arg("out_dim"), py::arg("in_dim"));

    m.def("tq3_gemv_bs1", &tq3_gemv_bs1,
          "bs=1 GEMV for 3-bit weights (warp-per-OC, sm_80+, bf16)",
          py::arg("x_rot"), py::arg("packed"),
          py::arg("norms"), py::arg("codebook"));

    m.def("tq3_gemv_bs1_fwht_smem", &tq3_gemv_bs1_fwht_smem,
          "Fused bs=1 GEMV with inverse randomized WHT in SMEM (sm_80+, bf16)",
          py::arg("x"), py::arg("packed"),
          py::arg("norms"), py::arg("codebook"),
          py::arg("signs1"), py::arg("signs2"));
}
