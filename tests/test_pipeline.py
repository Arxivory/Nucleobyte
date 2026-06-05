import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from nucleobyte import CUDAKmerTokenizer

def verify_tokenization_math():
    # Sequence: ATGC (4 characters)
    # A = 00 (0), T = 11 (3), G = 10 (2), C = 01 (1)
    # If K=2, Stride=1:
    # Token 1: AT -> 0011 in binary -> 3 in decimal
    # Token 2: TG -> 1110 in binary -> 14 in decimal
    # Token 3: GC -> 1001 in binary -> 9 in decimal
    test_sequence = "ATGC"
    
    print("Initializing NucleoByte CUDA Tokenizer (K=2, Stride=1)...")
    tokenizer = CUDAKmerTokenizer(kmer_size=2, stride=1)
    
    token_ids, metrics = tokenizer.tokenize(test_sequence)
    
    print("\n--- PERFORMANCE METRICS ---")
    print(f"Tokens Generated: {metrics.total_tokens_generated}")
    print(f"CUDA Kernel Execution Time: {metrics.processing_time_ms:.4f} ms")
    
    print("\n--- VALUE VERIFICATION ---")
    print(f"Expected IDs: [3, 14, 9]")
    print(f"Returned IDs: {token_ids}")
    
    if token_ids == [3, 14, 9]:
        print("\n SUCCESS: The 2-bit CUDA Bit-Shifting Kernel is mathematically flawless!")
    else:
        print("\n FAILURE: Mismatch detected in bitwise translation values.")

if __name__ == "__main__":
    verify_tokenization_math()