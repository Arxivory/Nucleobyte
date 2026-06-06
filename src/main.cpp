#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "tokenizer/tokenizer.hpp"
#include "sparse_attn.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_core, m) {
    m.doc() = "NucleoByte internal C++/CUDA acceleration engine";

    py::class_<TokenizerMetrics>(m, "TokenizerMetrics")
        .def_readonly("processing_time_ms", &TokenizerMetrics::processing_time_ms)
        .def_readonly("total_tokens_generated", &TokenizerMetrics::total_tokens_generated);

    m.def("launch_fused_tokenizer", [](const std::string& fasta_sequence, int kmer_size, int stride) {
        TokenizerMetrics metrics;
        std::vector<int> tokens = launch_fused_tokenizer(fasta_sequence, kmer_size, stride, metrics);
        return py::make_tuple(tokens, metrics);
        }, "Executes the 2-bit bitpack tokenizer on the GPU.",
        py::arg("fasta_sequence"), py::arg("kmer_size"), py::arg("stride"));

    m.def("launch_block_sparse_attn", &launch_block_sparse_attn,
        "Executes custom block-sparse local/global hybrid attention on the GPU.",
        py::arg("h_Q"), py::arg("h_K"), py::arg("h_V"),
        py::arg("num_tokens"), py::arg("head_dim"), py::arg("window_radius"));

    m.def("launch_block_sparse_attn_device", [](
        uintptr_t q_ptr,
        uintptr_t k_ptr,
        uintptr_t v_ptr,
        uintptr_t o_ptr,
        int num_tokens,
        int head_dim,
        int window_radius
        ) {
            const float* d_Q = reinterpret_cast<const float*>(q_ptr);
            const float* d_K = reinterpret_cast<const float*>(k_ptr);
            const float* d_V = reinterpret_cast<const float*>(v_ptr);
            float* d_O = reinterpret_cast<float*>(o_ptr);

            launch_block_sparse_attn_device(d_Q, d_K, d_V, d_O, num_tokens, head_dim, window_radius);
        }, "Executes true zero-copy VRAM-resident block-sparse attention.",
        py::arg("q_ptr"), py::arg("k_ptr"), py::arg("v_ptr"), py::arg("o_ptr"),
            py::arg("num_tokens"), py::arg("head_dim"), py::arg("window_radius"));
}