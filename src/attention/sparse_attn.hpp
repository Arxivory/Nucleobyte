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

void launch_block_sparse_attn_device(
    const float* d_Q,
    const float* d_K,
    const float* d_V,
    float* d_O,
    float* d_LSE,
    int num_tokens,
    int head_dim,
    int window_radius
);

void launch_block_sparse_attn_backward_device(
    const float* d_grad_O, const float* d_Q, const float* d_K, const float* d_V,
    const float* d_LSE, float* d_grad_Q, float* d_grad_K, float* d_grad_V,
    int num_tokens, int head_dim, int window_radius
);