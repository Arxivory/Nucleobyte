import torch
import torch.nn.functional as F
from __init__ import CUDAKmerTokenizer
from model import NucleoByteLM

def decode_token_to_kmer(token_id, kmer_size=6):
    """
    Decodes a vocabulary token ID (1 to 4096) back into a DNA k-mer string.
    Assumes standard base-4 alphabetical arrangement (A=0, C=1, G=2, T=3).
    """
    if token_id < 1 or token_id > 4096:
        if token_id == 0: return "[PAD]"
        if token_id == 4097: return "[MASK]"
        return "[UNK]"
        
    idx = token_id - 1
    bases = ['A', 'C', 'G', 'T']
    kmer = []
    
    for _ in range(kmer_size):
        kmer.append(bases[idx % 4])
        idx //= 4
        
    return "".join(reversed(kmer))


def run_masked_dna_inference(dna_sequence, mask_position=5, top_k=5):
    """
    Tokenizes a DNA sequence, masks out a chosen k-mer index, 
    and predicts the missing sequence using the custom trained weights.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing Inference Engine on: {device}")

    tokenizer = CUDAKmerTokenizer(kmer_size=6, stride=1)
    tokens, _ = tokenizer.tokenize(dna_sequence)
    
    if not isinstance(tokens, torch.Tensor):
        tokens = torch.tensor(tokens, dtype=torch.long)
        
    target_len = 4091
    if tokens.size(0) >= target_len:
        tokens = tokens[:target_len]
    else:
        tokens = torch.cat([tokens, torch.zeros(target_len - tokens.size(0), dtype=torch.long)])

    if mask_position >= tokens.size(0) or tokens[mask_position] == 0:
        raise ValueError(f"Selected mask position {mask_position} points to empty padding.")

    ground_truth_id = tokens[mask_position].item()
    original_kmer = decode_token_to_kmer(ground_truth_id)
    
    tokens[mask_position] = 4097
    
    input_tensor = tokens.unsqueeze(0).to(device)

    model = NucleoByteLM(vocab_size=4100, embed_dim=128, num_heads=2, num_layers=2).to(device)
    
    checkpoint_path = "nucleobyte_genomic_checkpoint.pt"
    try:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"[SUCCESS] Loaded pre-trained parameter weights from {checkpoint_path}")
    except FileNotFoundError:
        print(f"[WARNING] Checkpoint '{checkpoint_path}' not found. Running with randomized weights initialization.")

    model.eval()

    with torch.no_grad():
        logits = model(input_tensor)
        
        masked_logits = logits[0, mask_position]
        
        probabilities = F.softmax(masked_logits, dim=-1)
        
        top_probs, top_indices = torch.topk(probabilities, top_k)

    print("\n=========================================================")
    print("      NUCLEOBYTE GENOMIC MASKED PREDICTION REPORT")
    print("=========================================================")
    print(f"Input DNA Length     : {len(dna_sequence)} bases")
    print(f"Masked Token Index   : Position {mask_position}")
    print(f"True Target K-mer    : {original_kmer} (Token ID: {ground_truth_id})")
    print("---------------------------------------------------------")
    print(f"Top {top_k} Model Candidates Predicted:")
    
    for i in range(top_k):
        pred_id = top_indices[i].item()
        prob_val = top_probs[i].item() * 100
        decoded_str = decode_token_to_kmer(pred_id)
        
        match_flag = "[CORRECT]" if pred_id == ground_truth_id else ""
        print(f"  Rank {i+1} -> Code: {decoded_str:<8} | Prob: {prob_val:6.2f}% | Token ID: {pred_id:<4} {match_flag}")
    print("=========================================================\n")


if __name__ == "__main__":
    sample_dna = "ATGCGTACGTTAGCCTAGCCAAATTTGGGCCCGGATTTACGTAAAGGGTTTCCCAAAGGGTTTAAAACCCGGGTTT" * 60
    
    run_masked_dna_inference(sample_dna, mask_position=12, top_k=5)