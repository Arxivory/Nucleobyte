#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
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
}