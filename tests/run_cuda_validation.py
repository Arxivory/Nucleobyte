import torch
import torch.nn.functional as F
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from nucleobyte.__init__ import CUDABlockSparseAttention


def compute_dense_block_sparse_reference(Q, K, V, window_radius, block_size=32):
    """Ground truth dense masked softmax mathematical reference layer."""
    num_tokens, head_dim = Q.shape
    num_blocks = num_tokens // block_size
    scale = 1.0 / np.sqrt(head_dim)
    
    raw_scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
    
    block_mask = torch.zeros((num_blocks, num_blocks), dtype=torch.float32)
    for i in range(num_blocks):
        for j in range(num_blocks):
            if (j == 0) or (abs(i - j) <= window_radius):
                block_mask[i, j] = 1.0

    token_mask = block_mask.repeat_interleave(block_size, dim=0).repeat_interleave(block_size, dim=1)
    masked_scores = torch.where(token_mask == 1.0, raw_scores, torch.tensor(-float('inf'), device=Q.device))
    
    attention_weights = F.softmax(masked_scores, dim=-1)
    attention_weights = torch.nan_to_num(attention_weights, nan=0.0)
    
    return torch.matmul(attention_weights, V)


if __name__ == "__main__":
    print("=" * 70)
    print(" NUCLEOBYTE CORE HARNESS: PYTORCH REF VS NATIVE CUDA KERNEL VALIDATION")
    print("=" * 70)

    BLOCK_SIZE = 32
    HEAD_DIM = 64
    WINDOW_RADIUS = 1
    NUM_BLOCKS = 8
    NUM_TOKENS = NUM_BLOCKS * BLOCK_SIZE

    torch.manual_seed(42)
    Q = torch.randn(NUM_TOKENS, HEAD_DIM, dtype=torch.float32)
    K = torch.randn(NUM_TOKENS, HEAD_DIM, dtype=torch.float32)
    V = torch.randn(NUM_TOKENS, HEAD_DIM, dtype=torch.float32)

    print("Running PyTorch masked block reference calculations...")
    ref_output = compute_dense_block_sparse_reference(Q, K, V, WINDOW_RADIUS, BLOCK_SIZE)

    print("Launching custom C++/CUDA block-sparse attention kernel pipeline...")
    cuda_attention_layer = CUDABlockSparseAttention(window_radius=WINDOW_RADIUS, block_size=BLOCK_SIZE)
    cuda_output = cuda_attention_layer(Q, K, V)

    max_abs_error = torch.max(torch.abs(ref_output - cuda_output)).item()
    mean_abs_error = torch.mean(torch.abs(ref_output - cuda_output)).item()

    print("\n" + "-" * 50)
    print(" ERROR DIVERGENCE ANALYSIS METRICS:")
    print(f" -> Maximum Absolute Element-wise Error: {max_abs_error:.6e}")
    print(f" -> Mean Absolute Error Matrix Delta:   {mean_abs_error:.6e}")
    print("-" * 50)

    if max_abs_error < 1e-4:
        print("\n[SUCCESS] Verification complete! The CUDA kernel matches PyTorch exactly.")
    else:
        print("\n[WARNING] Numerical divergence exceeds expected thresholds.")
        print("Reviewing cross-warp online softmax synchronization variables recommended.")