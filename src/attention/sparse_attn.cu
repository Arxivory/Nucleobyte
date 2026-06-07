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
    float* __restrict__ d_LSE,
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
        if (thread_x == 0) {
            d_LSE[global_row] = m_prev + logf(d_prev);
        }
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
    int num_blocks = (num_tokens + BLOCK_SIZE - 1) / BLOCK_SIZE;
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
        d_Q, d_K, d_V, d_O, nullptr, num_tokens, num_blocks, window_radius, scale
        );

    cudaDeviceSynchronize();
    cudaMemcpy(h_O.data(), d_O, matrix_size, cudaMemcpyDeviceToHost);

    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_O);

    return h_O;
}

void launch_block_sparse_attn_device(
    const float* d_Q,
    const float* d_K,
    const float* d_V,
    float* d_O,
    float* d_LSE,
    int num_tokens,
    int head_dim,
    int window_radius
) {
    int num_blocks = (num_tokens + BLOCK_SIZE - 1) / BLOCK_SIZE;
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    dim3 grid_size(num_blocks);
    dim3 block_size(BLOCK_SIZE, BLOCK_SIZE);

    block_sparse_attn_kernel << <grid_size, block_size >> > (
        d_Q, d_K, d_V, d_O, d_LSE, num_tokens, num_blocks, window_radius, scale
        );
}

__global__ void block_sparse_attn_backward_kernel(
    const float* __restrict__ d_grad_O,
    const float* __restrict__ d_Q,
    const float* __restrict__ d_K,
    const float* __restrict__ d_V,
    const float* __restrict__ d_O,
    const float* __restrict__ d_LSE,
    float* __restrict__ d_grad_Q,
    float* __restrict__ d_grad_K,
    float* __restrict__ d_grad_V,
    int num_tokens,
    int num_blocks,
    int window_radius,
    float scale
) {
    int block_row = blockIdx.x;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int global_row_idx = block_row * BLOCK_SIZE + tx;

    float Di = 0.0f;
    if (global_row_idx < num_tokens) {
        for (int d = 0; d < 64; ++d) {
            Di += d_grad_O[global_row_idx * 64 + d] * d_O[global_row_idx * 64 + d];
        }
    }

    __shared__ float s_Q[BLOCK_SIZE][64];
    __shared__ float s_dO[BLOCK_SIZE][64];
    __shared__ float s_K[BLOCK_SIZE][64];
    __shared__ float s_V[BLOCK_SIZE][64];
    __shared__ float s_S[BLOCK_SIZE][BLOCK_SIZE];

    float reg_grad_Q_acc1 = 0.0f;
    float reg_grad_Q_acc2 = 0.0f;

    if (global_row_idx < num_tokens) {
        s_Q[tx][ty] = d_Q[global_row_idx * 64 + ty];
        s_Q[tx][ty + 32] = d_Q[global_row_idx * 64 + ty + 32];

        s_dO[tx][ty] = d_grad_O[global_row_idx * 64 + ty];
        s_dO[tx][ty + 32] = d_grad_O[global_row_idx * 64 + ty + 32];
    }
    __syncthreads();

    for (int block_col = 0; block_col < num_blocks; ++block_col) {
        bool is_global_anchor = (block_col == 0);
        bool is_local_window = (abs(block_row - block_col) <= window_radius);

        if (!is_global_anchor && !is_local_window) {
            continue;
        }

        int global_col_idx = block_col * BLOCK_SIZE + tx;

        if (global_col_idx < num_tokens) {
            s_K[tx][ty] = d_K[global_col_idx * 64 + ty];
            s_K[tx][ty + 32] = d_K[global_col_idx * 64 + ty + 32];

            s_V[tx][ty] = d_V[global_col_idx * 64 + ty];
            s_V[tx][ty + 32] = d_V[global_col_idx * 64 + ty + 32];
        }
        __syncthreads();

        float raw_score = 0.0f;
        if (global_row_idx < num_tokens) {
            for (int d = 0; d < 64; ++d) {
                raw_score += s_Q[tx][d] * s_K[ty][d];
            }
            raw_score *= scale;

            float token_lse = d_LSE[global_row_idx];
            s_S[tx][ty] = expf(raw_score - token_lse);
        }
        else {
            s_S[tx][ty] = 0.0f;
        }
        __syncthreads();

        int global_col_token = block_col * BLOCK_SIZE + ty;
        if (global_col_token < num_tokens) {
            float gv_acc1 = 0.0f;
            float gv_acc2 = 0.0f;
            for (int k = 0; k < BLOCK_SIZE; ++k) {
                float p_ij = s_S[k][ty];
                gv_acc1 += p_ij * s_dO[k][tx];
                gv_acc2 += p_ij * s_dO[k][tx + 32];
            }
            atomicAdd(&d_grad_V[global_col_token * 64 + tx], gv_acc1);
            atomicAdd(&d_grad_V[global_col_token * 64 + tx + 32], gv_acc2);
        }

        float grad_P_ij = 0.0f;
        if (global_row_idx < num_tokens) {
            for (int d = 0; d < 64; ++d) {
                grad_P_ij += s_dO[tx][d] * s_V[ty][d];
            }
        }

        float local_p_ij = s_S[tx][ty];

        __syncthreads();

        if (global_row_idx < num_tokens) {
            s_S[tx][ty] = local_p_ij * (grad_P_ij - Di);
        }
        else {
            s_S[tx][ty] = 0.0f;
        }
        __syncthreads();

        if (global_row_idx < num_tokens) {
            for (int k = 0; k < BLOCK_SIZE; ++k) {
                float grad_S_ik = s_S[tx][k];
                reg_grad_Q_acc1 += grad_S_ik * s_K[k][ty] * scale;
                reg_grad_Q_acc2 += grad_S_ik * s_K[k][ty + 32] * scale;
            }
        }

        if (global_col_token < num_tokens) {
            float gk_acc1 = 0.0f;
            float gk_acc2 = 0.0f;
            for (int k = 0; k < BLOCK_SIZE; ++k) {
                float grad_S_kj = s_S[k][ty];
                gk_acc1 += grad_S_kj * s_Q[k][tx] * scale;
                gk_acc2 += grad_S_kj * s_Q[k][tx + 32] * scale;
            }
            atomicAdd(&d_grad_K[global_col_token * 64 + tx], gk_acc1);
            atomicAdd(&d_grad_K[global_col_token * 64 + tx + 32], gk_acc2);
        }

        __syncthreads();
    }

    if (global_row_idx < num_tokens) {
        d_grad_Q[global_row_idx * 64 + ty] = reg_grad_Q_acc1;
        d_grad_Q[global_row_idx * 64 + ty + 32] = reg_grad_Q_acc2;
    }
}

void launch_block_sparse_attn_backward_device(
    const float* d_grad_O, const float* d_Q, const float* d_K, const float* d_V,
    const float* d_O, const float* d_LSE, float* d_grad_Q, float* d_grad_K, float* d_grad_V,
    int num_tokens, int head_dim, int window_radius
) {
    int num_blocks = (num_tokens + BLOCK_SIZE - 1) / BLOCK_SIZE;
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    dim3 grid_size(num_blocks);
    dim3 block_size(BLOCK_SIZE, BLOCK_SIZE);

    block_sparse_attn_backward_kernel << <grid_size, block_size >> > (
        d_grad_O, d_Q, d_K, d_V, d_O, d_LSE, d_grad_Q, d_grad_K, d_grad_V,
        num_tokens, num_blocks, window_radius, scale
        );
}