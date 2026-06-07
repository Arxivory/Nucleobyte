import os
import sys
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from nucleobyte.__init__ import CUDAKmerTokenizer
from nucleodataloader.dataset import NucleoByteDataset, GenomicMLMCollator
from nucleobyte.model import NucleoByteLM

def run_keratin_memorization_demo(fasta_file, epochs=150, batch_size=2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Targeting Hardware Device: {device}")

    tokenizer = CUDAKmerTokenizer(kmer_size=6, stride=1)
    dataset = NucleoByteDataset(fasta_file, max_sequence_length=512)
    collator = GenomicMLMCollator(tokenizer)
    
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)
    total_steps_per_epoch = len(data_loader)

    model = NucleoByteLM(vocab_size=4100, embed_dim=128, num_heads=2, num_layers=2).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=7e-4, weight_decay=0.01)
    
    total_updates = epochs * total_steps_per_epoch
    scheduler = CosineAnnealingLR(optimizer, T_max=total_updates, eta_min=1e-6)
    criterion = torch.nn.CrossEntropyLoss()

    model.train()
    print("\n=========================================================")
    print("      NUCLEOBYTE TRANSFORMATION CAPACITY MEMORIZATION TEST")
    print("=========================================================")
    print(f"Target Sequence        : {fasta_file} (Human Keratin Gene)")
    print(f"Total Batches Per Pass : {total_steps_per_epoch}")
    print(f"Total Epochs Scheduled : {epochs}")
    print("=========================================================\n")
    
    try:
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
                scheduler.step()
                
                total_loss += loss.item()
            
            avg_epoch_loss = total_loss / total_steps_per_epoch
            current_lr = optimizer.param_groups[0]['lr']
            print(f"\rOptimizing Sequence... Epoch [{epoch+1}/{epochs}] | Loss: {avg_epoch_loss:.4f} | LR: {current_lr:.7f}", end="")
            sys.stdout.flush()
            
        print(f"\n\n[SUCCESS] Memorization Test Complete. Final Converged Loss: {avg_epoch_loss:.4f}")

    except KeyboardInterrupt:
        print("\n[WARNING] Test aborted early by user.")

    torch.save(model.state_dict(), "nucleobyte_keratin_demo_weights.pt")
    print("Saved demo weights to nucleobyte_keratin_demo_weights.pt")


if __name__ == "__main__":
    fasta_path = "sampledata/keratin_sample.fasta"
    if not os.path.exists(fasta_path):
        raise FileNotFoundError(f"Missing file: {fasta_path}")
        
    run_keratin_memorization_demo(fasta_path, epochs=200, batch_size=2)