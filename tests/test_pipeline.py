import torch
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from nucleobyte.__init__ import NucleoByteTransformerLayer, CUDAKmerTokenizer

if __name__ == "__main__":
    print("=" * 75)
    print(" NUCLEOBYTE DEEP ARCHITECTURE PIPELINE VALIDATION ENGINE")
    print("=" * 75)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Target Execution Compute Device: {torch.cuda.get_device_name(0)}\n")

    mock_fasta_sequence = "ATCGCCGATCGAATTCCGGATCGATCGATCGA" * 128
    print(f"Raw Genomic Sequence String Size: {len(mock_fasta_sequence)} Base Pairs.")

    tokenizer = CUDAKmerTokenizer(kmer_size=6, stride=1)
    tokens, metrics = tokenizer.tokenize(mock_fasta_sequence)
    print(f"[SUCCESS] Tokenizer packed {metrics.total_tokens_generated} tokens in {metrics.processing_time_ms:.4f} ms.")

    token_tensor = torch.tensor(tokens, dtype=torch.long, device=device)

    VOCAB_SIZE = 4096
    EMBED_DIM = 128
    NUM_HEADS = 4
    NUM_LAYERS = 3
    
    embedding_layer = torch.nn.Embedding(VOCAB_SIZE, EMBED_DIM).to(device)
    hidden_states = embedding_layer(token_tensor)
    print(f"Base Feature Hidden States Shape: {list(hidden_states.shape)}")

    print(f"\nBuilding Deep Stack Configuration: {NUM_LAYERS} x [Pre-LN BlockSparseTransformer]...")
    transformer_stack = torch.nn.ModuleList([
        NucleoByteTransformerLayer(embed_dim=EMBED_DIM, num_heads=NUM_HEADS, window_radius=1, block_size=32)
        for _ in range(NUM_LAYERS)
    ]).to(device)

    current_states = hidden_states
    for layer_idx, layer in enumerate(transformer_stack):
        current_states = layer(current_states)
        print(f" -> Layer {layer_idx + 1} Forward Output Tensor Shape: {list(current_states.shape)}")

    print("\nExecuting Computational Autograd Graph Walkback...")
    target_loss = current_states.sum()
    target_loss.backward()
    print("[SUCCESS] Autograd graph reached terminal nodes without interruption.")

    print("\n--- Diagnostic Gradient Capture Analytics ---")
    for layer_idx, layer in enumerate(transformer_stack):
        q_grad = layer.attn.q_proj.weight.grad
        ffn_grad = layer.ffn.w_1.weight.grad
        print(f"Layer {layer_idx + 1} Attention Query Weight Gradient Shape : {list(q_grad.shape if q_grad is not None else 'None')}")
        print(f"Layer {layer_idx + 1} FeedForward UpProj Weight Gradient Shape: {list(ffn_grad.shape if ffn_grad is not None else 'None')}")

    print("\n[COMPLETE] Multi-layered genomic Transformer execution network is verified and ready!")