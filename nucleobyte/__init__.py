import sys
import os

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
        """
        Passes a raw genetic string directly into the custom CUDA 2-bit packing kernel.
        Returns a tuple: (list_of_token_ids, TokenizerMetrics_object)
        """
        clean_sequence = fasta_sequence.replace("\n", "").replace(" ", "")
        return _core.launch_fused_tokenizer(clean_sequence, self.kmer_size, self.stride)