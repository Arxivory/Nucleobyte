#pragma once
#include <string>
#include <vector>

struct TokenizerMetrics {
    float processing_time_ms;
    size_t total_tokens_generated;
};

std::vector<int> launch_fused_tokenizer(
    const std::string& fasta_sequence,
    int kmer_size,
    int stride,
    TokenizerMetrics& metrics
);