#include "tokenizer.hpp"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>

__global__ void fused_tokenize_kernel(
    const char* d_sequence,
    int* d_tokens,
    int sequence_length,
    int kmer_size,
    int stride,
    int total_tokens
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= total_tokens) return;

    int token_id = 0;
    int start_pos = idx * stride;

    for (int i = 0; i < kmer_size; ++i) {
        char base = d_sequence[start_pos + i];
        int bit_val = 0;

        switch (base) {
        case 'A': case 'a': bit_val = 0; break; // 00
        case 'C': case 'c': bit_val = 1; break; // 01
        case 'G': case 'g': bit_val = 2; break; // 10
        case 'T': case 't': bit_val = 3; break; // 11
        default:            bit_val = 0; break; // Handle N or padding gracefully
        }

        token_id = (token_id << 2) | bit_val;
    }

    d_tokens[idx] = token_id;
}

std::vector<int> launch_fused_tokenizer(
    const std::string& fasta_sequence,
    int kmer_size,
    int stride,
    TokenizerMetrics& metrics
) {
    int seq_len = static_cast<int>(fasta_sequence.length());

    int total_tokens = (seq_len - kmer_size) / stride + 1;
    metrics.total_tokens_generated = total_tokens;

    char* d_sequence = nullptr;
    int* d_tokens = nullptr;
    cudaMalloc(&d_sequence, seq_len * sizeof(char));
    cudaMalloc(&d_tokens, total_tokens * sizeof(int));

    cudaMemcpy(d_sequence, fasta_sequence.c_str(), seq_len * sizeof(char), cudaMemcpyHostToDevice);

    int block_size = 256;
    int grid_size = (total_tokens + block_size - 1) / block_size;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    fused_tokenize_kernel << <grid_size, block_size >> > (
        d_sequence, d_tokens, seq_len, kmer_size, stride, total_tokens
        );

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    metrics.processing_time_ms = milliseconds;

    std::vector<int> h_tokens(total_tokens);
    cudaMemcpy(h_tokens.data(), d_tokens, total_tokens * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_sequence);
    cudaFree(d_tokens);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return h_tokens;
}