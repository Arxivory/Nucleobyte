import sys
import os
import numpy as np

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
        """Passes a raw genetic string directly into the custom CUDA 2-bit packing kernel."""
        clean_sequence = fasta_sequence.replace("\n", "").replace(" ", "")
        return _core.launch_fused_tokenizer(clean_sequence, self.kmer_size, self.stride)


class CUDABlockSparseAttention:
    def __init__(self, window_radius: int = 1, block_size: int = 32):
        self.window_radius = window_radius
        self.block_size = block_size

    def __call__(self, Q, K, V):
        is_torch = hasattr(Q, "detach")
        device = Q.device if is_torch else None

        if is_torch:
            Q_np = Q.detach().cpu().contiguous().numpy()
            K_np = K.detach().cpu().contiguous().numpy()
            V_np = V.detach().cpu().contiguous().numpy()
        else:
            Q_np = np.ascontiguousarray(Q, dtype=np.float32)
            K_np = np.ascontiguousarray(K, dtype=np.float32)
            V_np = np.ascontiguousarray(V, dtype=np.float32)

        num_tokens, head_dim = Q_np.shape
        out_np = np.empty_block = np.zeros_like(Q_np)

        _core.launch_block_sparse_attn_ptr(
            Q_np, K_np, V_np, out_np, num_tokens, head_dim, self.window_radius
        )

        if is_torch:
            import torch
            return torch.from_numpy(out_np).to(device)
        return out_np