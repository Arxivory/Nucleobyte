import sys
import os
import torch
import torch.nn as nn
from torch.autograd import Function

try:
    import _core
except ImportError:
    sys.path.append(os.path.join(os.path.dirname(__file__), "..", "out", "build", "x64-Debug"))
    import _core

class CUDAKmerTokenizer:
    def __init__(self, kmer_size: int = 6, stride: int = 1):
        self.kmer_size = kmer_size
        self.stride = stride

    def tokenize(self, fasta_sequence: str):
        clean_sequence = fasta_sequence.replace("\n", "").replace(" ", "")
        tokens, metrics = _core.launch_fused_tokenizer(clean_sequence, self.kmer_size, self.stride)
        return tokens, metrics


class BlockSparseAttentionFunction(Function):
    @staticmethod
    def forward(ctx, Q_heads, K_heads, V_heads, window_radius, block_size):
        batch_size, num_heads, num_tokens, head_dim = Q_heads.shape
        O_heads = torch.zeros_like(Q_heads)

        LSE_heads = torch.empty((batch_size, num_heads, num_tokens), dtype=torch.float32, device=Q_heads.device)

        for b in range(batch_size):
            for head_idx in range(num_heads):
                _core.launch_block_sparse_attn_device(
                    Q_heads[b, head_idx].data_ptr(),
                    K_heads[b, head_idx].data_ptr(),
                    V_heads[b, head_idx].data_ptr(),
                    O_heads[b, head_idx].data_ptr(),
                    LSE_heads[b, head_idx].data_ptr(),
                    num_tokens, head_dim, window_radius
                )

        ctx.save_for_backward(Q_heads, K_heads, V_heads, LSE_heads)
        ctx.window_radius = window_radius
        ctx.block_size = block_size
        return O_heads

    @staticmethod
    def backward(ctx, grad_output):
        Q_heads, K_heads, V_heads, LSE_heads = ctx.saved_tensors
        batch_size, num_heads, num_tokens, head_dim = Q_heads.shape

        grad_Q = torch.zeros_like(Q_heads)
        grad_K = torch.zeros_like(K_heads)
        grad_V = torch.zeros_like(V_heads)

        grad_output_contig = grad_output.contiguous()

        for b in range(batch_size):
            for head_idx in range(num_heads):
                _core.launch_block_sparse_attn_backward_device(
                    grad_output_contig[b, head_idx].data_ptr(),
                    Q_heads[b, head_idx].data_ptr(),
                    K_heads[b, head_idx].data_ptr(),
                    V_heads[b, head_idx].data_ptr(),
                    LSE_heads[b, head_idx].data_ptr(),
                    grad_Q[b, head_idx].data_ptr(),
                    grad_K[b, head_idx].data_ptr(),
                    grad_V[b, head_idx].data_ptr(),
                    num_tokens, head_dim, ctx.window_radius
                )

        return grad_Q, grad_K, grad_V, None, None


class NucleoByteBlockSparseAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, window_radius: int = 1, block_size: int = 32):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.window_radius = window_radius
        self.block_size = block_size

        assert self.head_dim * num_heads == embed_dim, "embed_dim must be cleanly divisible by num_heads"

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_tokens, embed_dim = x.shape
        
        Q = self.q_proj(x) 
        K = self.k_proj(x) 
        V = self.v_proj(x) 

        Q_heads = Q.view(batch_size, num_tokens, self.num_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()
        K_heads = K.view(batch_size, num_tokens, self.num_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()
        V_heads = V.view(batch_size, num_tokens, self.num_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()

        O_heads = BlockSparseAttentionFunction.apply(
            Q_heads, K_heads, V_heads, self.window_radius, self.block_size
        )

        O_combined = O_heads.permute(0, 2, 1, 3).contiguous().view(batch_size, num_tokens, embed_dim)
        return self.out_proj(O_combined)


class NucleoByteFeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network processing token features independently.
    Expands representation space by 4x using GELU non-linearity before projecting back.
    """
    def __init__(self, embed_dim: int, ff_dim: int = None, dropout: float = 0.1):
        super().__init__()
        if ff_dim is None:
            ff_dim = 4 * embed_dim
        
        self.w_1 = nn.Linear(embed_dim, ff_dim)
        self.act = nn.GELU()
        self.w_2 = nn.Linear(ff_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w_2(self.act(self.w_1(x))))


class NucleoByteTransformerLayer(nn.Module):
    """
    Complete Pre-Layer Normalization (Pre-LN) DNA Transformer Layer.
    Structure:
        x = x + Attention(LayerNorm(x))
        x = x + FeedForward(LayerNorm(x))
    """
    def __init__(self, embed_dim: int, num_heads: int, window_radius: int = 1, block_size: int = 32, dropout: float = 0.1):
        super().__init__()
        self.attn_norm = nn.LayerNorm(embed_dim)
        self.attn = NucleoByteBlockSparseAttention(embed_dim, num_heads, window_radius, block_size)
        self.attn_dropout = nn.Dropout(dropout)

        self.ffn_norm = nn.LayerNorm(embed_dim)
        self.ffn = NucleoByteFeedForward(embed_dim, ff_dim=4 * embed_dim, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed_x = self.attn_norm(x)
        attn_out = self.attn(normed_x)
        x = x + self.attn_dropout(attn_out)

        normed_x = self.ffn_norm(x)
        ffn_out = self.ffn(normed_x)
        x = x + ffn_out

        return x