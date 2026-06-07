import torch
import torch.nn as nn
from nucleobyte.__init__ import NucleoByteTransformerLayer

class NucleoByteLM(nn.Module):
    """
    The complete Genomic Language Model combining your custom 
    Block-Sparse Transformer Stack with a token prediction projection head.
    """
    def __init__(self, vocab_size=4100, embed_dim=128, num_heads=2, num_layers=3, window_radius=1, block_size=32, max_seq_length=4096):
        super().__init__()
        
        self.token_embeddings = nn.Embedding(vocab_size, embed_dim)
        
        self.position_embeddings = nn.Embedding(max_seq_length, embed_dim)
        
        self.emb_norm = nn.LayerNorm(embed_dim)
        self.emb_dropout = nn.Dropout(0.1)
        
        self.layers = nn.ModuleList([
            NucleoByteTransformerLayer(
                embed_dim=embed_dim, 
                num_heads=num_heads, 
                window_radius=window_radius, 
                block_size=block_size
            ) for _ in range(num_layers)
        ])
        
        self.lm_head = nn.Linear(embed_dim, vocab_size)

    def forward(self, input_ids):
        seq_length = input_ids.size(1)
        
        positions = torch.arange(0, seq_length, dtype=torch.long, device=input_ids.device)
        positions = positions.unsqueeze(0).expand_as(input_ids)
        
        x = self.token_embeddings(input_ids) + self.position_embeddings(positions)
        
        x = self.emb_norm(x)
        x = self.emb_dropout(x)
        
        for layer in self.layers:
            x = layer(x)
            
        logits = self.lm_head(x)
        return logits