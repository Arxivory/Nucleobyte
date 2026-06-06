import os
import random
import torch
from torch.utils.data import DataLoader
import torch.optim as optim

from nucleobyte.__init__ import CUDAKmerTokenizer
from nucleodataloader.dataset import NucleoByteDataset, GenomicMLMCollator
from nucleobyte.model import NucleoByteLM

def generate_synthetic_fasta(filename="synthetic_genome.fasta", num_lines=20, line_length=2000):
    """Generates a mock FASTA file filled with random DNA bases for training validation."""
    print(f"Generating synthetic genomic data asset: {filename}...")
    bases = ['A', 'C', 'G', 'T']
    with open(filename, "w") as f:
        f.write(">Chromosome_Mock_Sequence_1\n")
        for _ in range(num_lines):
            line_chars = [random.choice(bases) for _ in range(line_length)]
            f.write("".join(line_chars) + "\n")
    print("[READY] Synthetic FASTA file created successfully.")


def execute_pretraining_run(fasta_file, epochs=3, batch_size=2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Targeting Training Hardware Device: {device}\n")

    tokenizer = CUDAKmerTokenizer(kmer_size=6, stride=1)
    dataset = NucleoByteDataset(fasta_file, max_sequence_length=4096)
    collator = GenomicMLMCollator(tokenizer)
    
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collator)

    model = NucleoByteLM(vocab_size=4100, embed_dim=128, num_heads=2, num_layers=2).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
    criterion = torch.nn.CrossEntropyLoss()

    model.train()
    print("\n=========================================================")
    print("      NUCLEOBYTE CORE PRE-TRAINING ENGINE STARTED")
    print("=========================================================")
    
    for epoch in range(epochs):
        total_loss = 0.0
        for step, (input_ids, labels) in enumerate(data_loader):
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            
            logits = model(input_ids)
            
            loss = criterion(logits.view(-1, 4100), labels.view(-1))
            
            loss.backward()
            
            optimizer.step()

            total_loss += loss.item()
            print(f"Epoch [{epoch+1}/{epochs}] | Step {step} | Batch MLM Loss: {loss.item():.4f}")
                
        print(f"--- Epoch {epoch+1} Complete. Avg Loss Matrix: {total_loss / len(data_loader):.4f} ---\n")

    print("[SUCCESS] Core pre-training verification execution completed!")
    
    torch.save(model.state_dict(), "nucleobyte_genomic_checkpoint.pt")
    print("Saved optimized weights to nucleobyte_genomic_checkpoint.pt")


if __name__ == "__main__":
    fasta_path = "synthetic_genome.fasta"
    
    if not os.path.exists(fasta_path):
        generate_synthetic_fasta(fasta_path, num_lines=15, line_length=3000)
        
    execute_pretraining_run(fasta_path, epochs=3, batch_size=2)