import torch
import torch.nn.functional as F
import numpy as np

def compute_dense_block_sparse_reference(Q, K, V, window_radius, block_size=32):
    """
    Computes a ground-truth attention reference using a dense block mask.
    Matches the exact layout of the NucleoByte custom CUDA kernel.
    
    Shapes:
        Q, K, V: [Num_Tokens, Head_Dim]
    """
    num_tokens, head_dim = Q.shape
    num_blocks = num_tokens // block_size
    scale = 1.0 / np.sqrt(head_dim)
    
    raw_scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
    
    block_mask = torch.zeros((num_blocks, num_blocks), dtype=torch.float32)
    for i in range(num_blocks):
        for j in range(num_blocks):
            is_global_anchor = (j == 0)
            is_local_window  = (abs(i - j) <= window_radius)
            
            if is_global_anchor or is_local_window:
                block_mask[i, j] = 1.0

    token_mask = block_mask.repeat_interleave(block_size, dim=0).repeat_interleave(block_size, dim=1)
    
    masked_scores = torch.where(token_mask == 1.0, raw_scores, torch.tensor(-float('inf'), device=Q.device))
    
    attention_weights = F.softmax(masked_scores, dim=-1)
    
    attention_weights = torch.nan_to_num(attention_weights, nan=0.0)
    
    output = torch.matmul(attention_weights, V)
    
    return output, block_mask

if __name__ == "__main__":
    print("="*60)
    print(" NucleoByte Framework Validation Baseline Generator")
    print("="*60)
    
    BLOCK_SIZE = 32
    HEAD_DIM = 64
    WINDOW_RADIUS = 1
    NUM_BLOCKS = 8
    NUM_TOKENS = NUM_BLOCKS * BLOCK_SIZE
    
    print(f"Configuring Test Profile: Tokens={NUM_TOKENS} | Sequence Blocks={NUM_BLOCKS} | Window Radius={WINDOW_RADIUS}")
    
    torch.manual_seed(42)
    Q = torch.randn(NUM_TOKENS, HEAD_DIM)
    K = torch.randn(NUM_TOKENS, HEAD_DIM)
    V = torch.randn(NUM_TOKENS, HEAD_DIM)
    
    ref_output, mask_layout = compute_dense_block_sparse_reference(Q, K, V, WINDOW_RADIUS, BLOCK_SIZE)
    
    print("\nGenerated Block Structural Mask Layout Layout Target:")
    print(mask_layout.int().numpy())
    
    print(f"\nGround Truth Output Tensor Computed. Verification Matrix Shape: {list(ref_output.shape)}")
    print("Sample output rows preview (First 2 elements per token index):")
    print(ref_output[:4, :2].numpy())
    print("\n[READY] Reference module active. Awaiting Pybind11 integration hook testing.")