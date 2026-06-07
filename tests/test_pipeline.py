import os
import random
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

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


def execute_pretraining_run(fasta_file, epochs=3, batch_size=2, accumulation_steps=4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Targeting Training Hardware Device: {device}\n")

    tokenizer = CUDAKmerTokenizer(kmer_size=6, stride=1)
    dataset = NucleoByteDataset(fasta_file, max_sequence_length=4096)
    collator = GenomicMLMCollator(tokenizer)
    
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collator)

    model = NucleoByteLM(vocab_size=4100, embed_dim=128, num_heads=2, num_layers=2).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
    
    total_updates = (epochs * len(data_loader)) // accumulation_steps
    if (epochs * len(data_loader)) % accumulation_steps != 0:
        total_updates += 1

    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, total_updates), eta_min=1e-5)

    criterion = torch.nn.CrossEntropyLoss()

    model.train()
    print("\n=========================================================")
    print("      NUCLEOBYTE CORE PRE-TRAINING ENGINE STARTED")
    print("=========================================================")
    print(f"Gradient Accumulation: Active ({accumulation_steps} steps)")
    print(f"Total Parameter Optimization Updates Planned: {total_updates}\n")
    
    for epoch in range(epochs):
        total_loss = 0.0
        
        optimizer.zero_grad()
        
        for step, (input_ids, labels) in enumerate(data_loader):
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            
            logits = model(input_ids)
            
            raw_loss = criterion(logits.view(-1, 4100), labels.view(-1))
            
            loss = raw_loss / accumulation_steps
            
            loss.backward()
            
            total_loss += raw_loss.item()

            is_accumulation_boundary = (step + 1) % accumulation_steps == 0
            is_last_batch_of_epoch = (step + 1) == len(data_loader)

            if is_accumulation_boundary or is_last_batch_of_epoch:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Epoch [{epoch+1}/{epochs}] | Step {step} | Batch MLM Loss: {raw_loss.item():.4f} | LR: {current_lr:.6f} -> [UPDATING WEIGHTS]")
                
                optimizer.step()
                
                scheduler.step()
                
                optimizer.zero_grad()
            else:
                print(f"Epoch [{epoch+1}/{epochs}] | Step {step} | Batch MLM Loss: {raw_loss.item():.4f} -> [ACCUMULATING GRADIENTS]")
                
        print(f"--- Epoch {epoch+1} Complete. Avg Loss Matrix: {total_loss / len(data_loader):.4f} ---\n")

    print("[SUCCESS] Core pre-training verification execution completed!")
    
    torch.save(model.state_dict(), "nucleobyte_genomic_checkpoint.pt")
    print("Saved optimized weights to nucleobyte_genomic_checkpoint.pt")


if __name__ == "__main__":
    fasta_path = "synthetic_genome.fasta"
    
    if not os.path.exists(fasta_path):
        generate_synthetic_fasta(fasta_path, num_lines=15, line_length=3000)
        
    execute_pretraining_run(fasta_path, epochs=3, batch_size=2, accumulation_steps=4)