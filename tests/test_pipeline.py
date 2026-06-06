import torch
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from nucleobyte.__init__ import NucleoByteBlockSparseAttention, CUDAKmerTokenizer

if __name__ == "__main__":
    print("=" * 70)
    print(" NUCLEOBYTE END-TO-END PIPELINE VALIDATION ENGINE")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Target Compute Unit: {torch.cuda.get_device_name(0)}\n")

    mock_fasta_sequence = "ATCGCCGATCGAATTCCGGATCGATCGATCGA" * 64
    print(f"Raw Genomic Sequence Length: {len(mock_fasta_sequence)} base pairs.")

    tokenizer = CUDAKmerTokenizer(kmer_size=6, stride=1)
    tokens, metrics = tokenizer.tokenize(mock_fasta_sequence)
    print(f"[SUCCESS] Tokenizer packed {metrics.total_tokens_generated} tokens in {metrics.processing_time_ms:.4f} ms.")

    token_tensor = torch.tensor(tokens, dtype=torch.long, device=device)

    VOCAB_SIZE = 4096
    EMBED_DIM = 128
    NUM_HEADS = 4
    
    embedding_layer = torch.nn.Embedding(VOCAB_SIZE, EMBED_DIM).to(device)
    hidden_states = embedding_layer(token_tensor)
    print(f"Generated Hidden States Shape: {list(hidden_states.shape)}")

    attn_layer = NucleoByteBlockSparseAttention(
        embed_dim=EMBED_DIM, 
        num_heads=NUM_HEADS, 
        window_radius=1, 
        block_size=32
    ).to(device)

    output_states = attn_layer(hidden_states)
    print(f"Attention Layer Output Shape:   {list(output_states.shape)}")

    print("\nVerifying Backpropagation/Gradient Pass Compatibility...")
    loss = output_states.sum()
    loss.backward()
    
    print(f"-> Query Matrix Weight Gradient Shape: {list(attn_layer.q_proj.weight.grad.shape)}")
    print("-> Status: Gradient graphs constructed successfully!")
    print("\n[COMPLETE] End-to-End custom hardware integration block is functional!")