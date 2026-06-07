import os
import torch
import torch.nn.functional as F
import sys

from nucleobyte.__init__ import CUDAKmerTokenizer
from nucleobyte.model import NucleoByteLM

def decode_6mer(token_id: int) -> str:
    """
    Mathematical fallback decoder for a 4^6 (4096) vocab size mapping.
    Assumes standard lexicographical mapping: A=0, C=1, G=2, T=3
    """
    if token_id >= 4096:
        return "[SPECIAL]"
        
    bases = ['A', 'C', 'G', 'T']
    kmer = ""
    temp = token_id
    for _ in range(6):
        kmer = bases[temp % 4] + kmer
        temp //= 4
    return kmer

def run_genomic_inference():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Initializing inference on device: {device}")

    tokenizer = CUDAKmerTokenizer(kmer_size=6, stride=1)
    MASK_TOKEN_ID = 4099

    model = NucleoByteLM(vocab_size=4100, embed_dim=128, num_heads=2, num_layers=2).to(device)
    checkpoint_path = "nucleobyte_keratin_demo_weights.pt"
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Missing weights file '{checkpoint_path}'. Run memorization test first!")
        
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    print(f"[SUCCESS] Custom Block-Sparse Transformer weights loaded.\n")

    raw_dna_sequence = "CCCAGGGTCCGATGGGAAAGTGTAGCCTGC"
    print(f"Original Input DNA Sequence : {raw_dna_sequence}")

    tokens, metrics = tokenizer.tokenize(raw_dna_sequence)
    
    if not isinstance(tokens, torch.Tensor):
        input_tensor = torch.tensor([tokens], dtype=torch.long).to(device)
    else:
        input_tensor = tokens.unsqueeze(0).long().to(device)
    
    target_idx = input_tensor.size(1) // 2
    original_token_id = input_tensor[0, target_idx].item()
    
    input_tensor[0, target_idx] = MASK_TOKEN_ID
    print(f"Masked Token Index Position : {target_idx}")
    print(f"Target Ground-Truth 6-mer   : '{decode_6mer(original_token_id)}' (ID: {original_token_id})")

    with torch.no_grad():
        logits = model(input_tensor)
        
    masked_logits = logits[0, target_idx, :]
    probabilities = F.softmax(masked_logits, dim=-1)
    topk_probs, topk_indices = torch.topk(probabilities, k=5)

    print("\n=========================================================")
    print("            NUCLEOBYTE INFERENCE PREDICTIONS")
    print("=========================================================")
    
    for rank, (prob, idx) in enumerate(zip(topk_probs, topk_indices)):
        token_id = idx.item()
        predicted_kmer = decode_6mer(token_id)
        
        is_correct = " (GROUND TRUTH MATCH!)" if token_id == original_token_id else ""
        print(f"Rank {rank+1} | Prediction: '{predicted_kmer}' (ID: {token_id}) | Confidence: {prob.item()*100:.2f}%{is_correct}")
        
    print("=========================================================\n")

if __name__ == "__main__":
    run_genomic_inference()