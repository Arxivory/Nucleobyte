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
        """
        Executes the custom block-sparse hybrid attention kernel.
        Accepts PyTorch tensors or NumPy arrays of shape [Num_Tokens, Head_Dim].
        """
        is_torch = False
        device = None

        if hasattr(Q, "detach"):
            is_torch = True
            device = Q.device
            Q_np = Q.detach().cpu().numpy()
            K_np = K.detach().cpu().numpy()
            V_np = V.detach().cpu().numpy()
        else:
            Q_np = np.asarray(Q)
            K_np = np.asarray(K)
            V_np = np.asarray(V)

        num_tokens, head_dim = Q_np.shape

        h_Q = Q_np.flatten().tolist()
        h_K = K_np.flatten().tolist()
        h_V = V_np.flatten().tolist()

        h_O = _core.launch_block_sparse_attn(
            h_Q, h_K, h_V, num_tokens, head_dim, self.window_radius
        )

        out_np = np.array(h_O, dtype=np.float32).reshape(num_tokens, head_dim)

        if is_torch:
            import torch
            return torch.from_numpy(out_np).to(device)
        return out_np