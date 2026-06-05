#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <vector>
#include <cmath>

// Silence IDE IntelliSense noise for native GPU hardware symbols
#ifdef __INTELLISENSE__
#define __syncthreads()
#define __shfl_xor_sync(mask, val, offset) (val)
#define CUDART_INF_F 3.40282347e+38f
#else
#include <math_constants.h>
#endif

#define BLOCK_SIZE 32
#define HEAD_DIM 64

__device__ inline float warp_reduce_max(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, offset));
    }
    return val;
}

__device__ inline float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void block_sparse_attn_kernel(
    const float* __restrict__ d_Q,
    const float* __restrict__ d_K,
    const float* __restrict__ d_V,
    float* __restrict__ d_O,
    int num_tokens,
    int num_blocks,
    int window_radius,
    float scale
) {
    int block_row = blockIdx.x;
    int thread_x = threadIdx.x;
    int thread_y = threadIdx.y;

    int global_row = block_row * BLOCK_SIZE + thread_y;

    __shared__ float s_K[BLOCK_SIZE][HEAD_DIM];
    __shared__ float s_V[BLOCK_SIZE][HEAD_DIM];
    __shared__ float s_W[BLOCK_SIZE][BLOCK_SIZE];

    float q_local[2] = { 0.0f, 0.0f };
    float o_local[2] = { 0.0f, 0.0f };

    float m_prev = -CUDART_INF_F;
    float d_prev = 0.0f;

    if (global_row < num_tokens) {
        q_local[0] = d_Q[global_row * HEAD_DIM + thread_x];
        q_local[1] = d_Q[global_row * HEAD_DIM + 32 + thread_x];
    }

    for (int block_col = 0; block_col < num_blocks; ++block_col) {
        bool is_global_anchor = (block_col == 0);
        bool is_local_window = (abs(block_row - block_col) <= window_radius);

        if (!is_global_anchor && !is_local_window) {
            continue;
        }

        int load_row_k = (thread_y * 32 + thread_x) / HEAD_DIM;
        int load_col_k = (thread_y * 32 + thread_x) % HEAD_DIM;
        int global_token_k = block_col * BLOCK_SIZE + load_row_k;

        if (global_token_k < num_tokens) {
            s_K[load_row_k][load_col_k] = d_K[global_token_k * HEAD_DIM + load_col_k];
            s_V[load_row_k][load_col_k] = d_V[global_token_k * HEAD_DIM + load_col_k];

            s_K[load_row_k + 16][load_col_k] = d_K[(global_token_k + 16) * HEAD_DIM + load_col_k];
            s_V[load_row_k + 16][load_col_k] = d_V[(global_token_k + 16) * HEAD_DIM + load_col_k];
        }
        else {
            s_K[load_row_k][load_col_k] = 0.0f;
            s_V[load_row_k][load_col_k] = 0.0f;
            s_K[load_row_k + 16][load_col_k] = 0.0f;
            s_V[load_row_k + 16][load_col_k] = 0.0f;
        }

        __syncthreads();

        float score = -CUDART_INF_F;
        if (global_row < num_tokens) {
            for (int j = 0; j < BLOCK_SIZE; ++j) {
                float dot_product = q_local[0] * s_K[j][thread_x] +
                    q_local[1] * s_K[j][32 + thread_x];

                float total_score = warp_reduce_sum(dot_product) * scale;

                if (thread_x == j) {
                    score = total_score;
                }
            }
        }

        float b_max = warp_reduce_max(score);
        float exp_score = (global_row < num_tokens) ? expf(score - b_max) : 0.0f;
        float b_sum = warp_reduce_sum(exp_score);

        if (global_row < num_tokens) {
            s_W[thread_y][thread_x] = exp_score;
        }
        __syncthreads();

        if (global_row < num_tokens) {
            float m_new = fmaxf(m_prev, b_max);
            float alpha = expf(m_prev - m_new);
            float beta = expf(b_max - m_new);
            float d_new = d_prev * alpha + b_sum * beta;

            float v_part1 = 0.0f;
            float v_part2 = 0.0f;

            for (int j = 0; j < BLOCK_SIZE; ++j) {
                float w = s_W[thread_y][j];
                v_part1 += w * s_V[j][thread_x];
                v_part2 += w * s_V[j][32 + thread_x];
            }

            o_local[0] = o_local[0] * alpha + v_part1 * beta;
            o_local[1] = o_local[1] * alpha + v_part2 * beta;

            m_prev = m_new;
            d_prev = d_new;
        }

        __syncthreads();
    }

    if (global_row < num_tokens && d_prev > 0.0f) {
        d_O[global_row * HEAD_DIM + thread_x] = o_local[0] / d_prev;
        d_O[global_row * HEAD_DIM + 32 + thread_x] = o_local[1] / d_prev;
    }
}

std::vector<float> launch_block_sparse_attn(
    const std::vector<float>& h_Q,
    const std::vector<float>& h_K,
    const std::vector<float>& h_V,
    int num_tokens,
    int head_dim,
    int window_radius
) {
    int num_blocks = num_tokens / BLOCK_SIZE;
    size_t matrix_size = num_tokens * head_dim * sizeof(float);
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    std::vector<float> h_O(num_tokens * head_dim, 0.0f);

    float* d_Q = nullptr, * d_K = nullptr, * d_V = nullptr, * d_O = nullptr;
    cudaMalloc(&d_Q, matrix_size);
    cudaMalloc(&d_K, matrix_size);
    cudaMalloc(&d_V, matrix_size);
    cudaMalloc(&d_O, matrix_size);

    cudaMemcpy(d_Q, h_Q.data(), matrix_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K.data(), matrix_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V.data(), matrix_size, cudaMemcpyHostToDevice);

    dim3 grid_size(num_blocks);
    dim3 block_size(BLOCK_SIZE, BLOCK_SIZE);

    block_sparse_attn_kernel << <grid_size, block_size >> > (
        d_Q, d_K, d_V, d_O, num_tokens, num_blocks, window_radius, scale
        );

    cudaDeviceSynchronize();
    cudaMemcpy(h_O.data(), d_O, matrix_size, cudaMemcpyDeviceToHost);

    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_O);

    return h_O;
}

void launch_block_sparse_attn_ptr(
    const float* h_Q,
    const float* h_K,
    const float* h_V,
    float* h_O,
    int num_tokens,
    int head_dim,
    int window_radius
) {
    int num_blocks = num_tokens / BLOCK_SIZE;
    size_t matrix_size = num_tokens * head_dim * sizeof(float);
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    float* d_Q = nullptr, * d_K = nullptr, * d_V = nullptr, * d_O = nullptr;
    cudaMalloc(&d_Q, matrix_size);
    cudaMalloc(&d_K, matrix_size);
    cudaMalloc(&d_V, matrix_size);
    cudaMalloc(&d_O, matrix_size);

    cudaMemcpy(d_Q, h_Q, matrix_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K, matrix_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V, matrix_size, cudaMemcpyHostToDevice);

    dim3 grid_size(num_blocks);
    dim3 block_size(BLOCK_SIZE, BLOCK_SIZE);

    block_sparse_attn_kernel << <grid_size, block_size >> > (
        d_Q, d_K, d_V, d_O, num_tokens, num_blocks, window_radius, scale
        );

    cudaDeviceSynchronize();
    cudaMemcpy(h_O, d_O, matrix_size, cudaMemcpyDeviceToHost);

    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_O);
}