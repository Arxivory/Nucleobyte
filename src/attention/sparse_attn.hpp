#pragma once
#include <vector>

std::vector<float> launch_block_sparse_attn(
    const std::vector<float>& h_Q,
    const std::vector<float>& h_K,
    const std::vector<float>& h_V,
    int num_tokens,
    int head_dim,
    int window_radius
);

void launch_block_sparse_attn_ptr(
    const float* h_Q,
    const float* h_K,
    const float* h_V,
    float* h_O,
    int num_tokens,
    int head_dim,
    int window_radius
);