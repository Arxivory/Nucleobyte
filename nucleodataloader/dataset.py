import torch
from torch.utils.data import Dataset, DataLoader
import random

class NucleoByteDataset(Dataset):
    """
    Optimized loader for processing raw FASTA/text genomic sequences
    and feeding them to the custom CUDA tokenizer infrastructure.
    """
    def __init__(self, fasta_file_path, max_sequence_length=4096):
        self.max_sequence_length = max_sequence_length
        self.sequences = []
        
        print(f"Parsing genomic corpus: {fasta_file_path}...")
        current_seq = []
        
        with open(fasta_file_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith(">"):
                    if current_seq:
                        self.sequences.append("".join(current_seq))
                        current_seq = []
                    continue
                clean_line = "".join([c for c in line.upper() if c in "ACGT"])
                current_seq.append(clean_line)
                
            if current_seq:
                self.sequences.append("".join(current_seq))

        print(f"[LOADED] Found {len(self.sequences)} continuous chromosomal blocks.")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        if len(seq) > self.max_sequence_length:
            start = 0
            seq = seq[start:start + self.max_sequence_length]
        return seq


class GenomicMLMCollator:
    """
    Handles on-the-fly CUDA tokenization and applies the 15% MLM masking matrix 
    directly before shipping numerical batches to the GPU execution engine.
    """
    def __init__(self, tokenizer, mask_token_id=4097, vocab_range=(1, 4096), mask_prob=0.15):
        self.tokenizer = tokenizer
        self.mask_token_id = mask_token_id
        self.min_vocab, self.max_vocab = vocab_range
        self.mask_prob = mask_prob

    def __call__(self, batch_sequences):
        token_batches = []
        
        for seq in batch_sequences:
            tokens, _ = self.tokenizer.tokenize(seq)
            if not isinstance(tokens, torch.Tensor):
                tokens = torch.tensor(tokens, dtype=torch.long)
            token_batches.append(tokens)
            
        max_len_in_batch = max(t.size(0) for t in token_batches)
        target_len = ((max_len_in_batch + 31) // 32) * 32 

        padded_tokens = []
        for t in token_batches:
            padding_size = target_len - t.size(0)
            padded_tokens.append(torch.nn.functional.pad(t, (0, padding_size), value=0))
                
        input_ids = torch.stack(padded_tokens)
        labels = input_ids.clone()

        probability_matrix = torch.full(labels.shape, self.mask_prob)
        probability_matrix[input_ids == 0] = 0.0
        masked_indices = torch.bernoulli(probability_matrix).bool()

        labels[~masked_indices] = -100

        indices_from_mask = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        input_ids[indices_from_mask] = self.mask_token_id

        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_from_mask
        random_words = torch.randint(self.min_vocab, self.max_vocab + 1, labels.shape, dtype=torch.long)
        input_ids[indices_random] = random_words[indices_random]

        return input_ids, labels