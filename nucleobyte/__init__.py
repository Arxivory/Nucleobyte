import sys
import os
import numpy as np
import torch

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
        if hasattr(Q, "is_cuda") and Q.is_cuda:
            num_tokens, head_dim = Q.shape
            
            out = torch.zeros_like(Q)
            
            _core.launch_block_sparse_attn_device(
                Q.data_ptr(), K.data_ptr(), V.data_ptr(), out.data_ptr(),
                num_tokens, head_dim, self.window_radius
            )
            return out
        
        import numpy as np
        Q_np = np.ascontiguousarray(Q, dtype=np.float32)
        K_np = np.ascontiguousarray(K, dtype=np.float32)
        V_np = np.ascontiguousarray(V, dtype=np.float32)
        num_tokens, head_dim = Q_np.shape
        out_np = np.zeros_like(Q_np)
        
        raise NotImplementedError("For maximum performance, pass tensors already allocated on the GPU.")