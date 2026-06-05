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

    m.def("launch_block_sparse_attn_ptr", [](
        py::array_t<float> Q_arr,
        py::array_t<float> K_arr,
        py::array_t<float> V_arr,
        py::array_t<float> O_arr,
        int num_tokens,
        int head_dim,
        int window_radius
        ) {
            py::buffer_info q_info = Q_arr.request();
            py::buffer_info k_info = K_arr.request();
            py::buffer_info v_info = V_arr.request();
            py::buffer_info o_info = O_arr.request();

            const float* h_Q = static_cast<const float*>(q_info.ptr);
            const float* h_K = static_cast<const float*>(k_info.ptr);
            const float* h_V = static_cast<const float*>(v_info.ptr);
            float* h_O = static_cast<float*>(o_info.ptr);

            launch_block_sparse_attn_ptr(h_Q, h_K, h_V, h_O, num_tokens, head_dim, window_radius);
        }, "Executes zero-copy pointer-passing block-sparse attention on the GPU.",
        py::arg("Q_arr"), py::arg("K_arr"), py::arg("V_arr"), py::arg("O_arr"),
            py::arg("num_tokens"), py::arg("head_dim"), py::arg("window_radius"));
}