import torch
import numpy as np
import time
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from nucleobyte.__init__ import CUDABlockSparseAttention

def profile_pytorch_dense(Q, K, V, scale):
    """Measures raw execution performance of a standard dense attention pass."""

    for _ in range(5):
        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
        weights = torch.softmax(scores, dim=-1)
        out = torch.matmul(weights, V)
    
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    
    iterations = 20
    for _ in range(iterations):
        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
        weights = torch.softmax(scores, dim=-1)
        out = torch.matmul(weights, V)
        
    torch.cuda.synchronize()
    end_time = time.perf_counter()
    return (end_time - start_time) / iterations

def profile_cuda_sparse(cuda_layer, Q, K, V):
    """Measures raw execution performance of our zero-copy CUDA sparse attention."""

    for _ in range(5):
        out = cuda_layer(Q, K, V)
        
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    
    iterations = 20
    for _ in range(iterations):
        out = cuda_layer(Q, K, V)
        
    torch.cuda.synchronize()
    end_time = time.perf_counter()
    return (end_time - start_time) / iterations


if __name__ == "__main__":
    print("=" * 75)
    print(" NUCLEOBYTE SCALING HARNESS: DENSE ATTENTION VS CUSTOM CUDA BLOCK-SPARSE")
    print("=" * 75)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Target Execution Compute Device Hardware: {torch.cuda.get_device_name(0)}")
    
    HEAD_DIM = 64
    BLOCK_SIZE = 32
    WINDOW_RADIUS = 1
    SCALE = 1.0 / np.sqrt(HEAD_DIM)
    
    SEQUENCE_LENGTHS = [1024, 2048, 4096, 8192]
    
    print(f"\n{'Sequence Length':<18} | {'PyTorch Dense (s)':<22} | {'CUDA Block-Sparse (s)':<22} | {'Speedup Factor'}")
    print("-" * 80)
    
    cuda_attention = CUDABlockSparseAttention(window_radius=WINDOW_RADIUS, block_size=BLOCK_SIZE)
    
    for seq_len in SEQUENCE_LENGTHS:
        Q = torch.randn(seq_len, HEAD_DIM, dtype=torch.float32, device=device)
        K = torch.randn(seq_len, HEAD_DIM, dtype=torch.float32, device=device)
        V = torch.randn(seq_len, HEAD_DIM, dtype=torch.float32, device=device)
        
        dense_time = profile_pytorch_dense(Q, K, V, SCALE)
        sparse_time = profile_cuda_sparse(cuda_attention, Q, K, V)
        
        speedup = dense_time / sparse_time if sparse_time > 0 else 0.0
        
        print(f"{seq_len:<18} | {dense_time:<22.6f} | {sparse_time:<22.6f} | {speedup:.2f}x")
        
    print("-" * 80)
    print("[ANALYSIS COMPLETION] If scaling behaves correctly, speedup should amplify significantly at 4K and 8K horizons.")