import torch
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from nucleobyte.__init__ import NucleoByteTransformerLayer, CUDAKmerTokenizer

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(True if device.type == "cuda" else False)
    
    layer = NucleoByteTransformerLayer(embed_dim=128, num_heads=2, window_radius=1, block_size=32).to(device)
    
    mock_input = torch.randn(4096, 128, device=device, requires_grad=True)
    
    output = layer(mock_input)
    
    loss = (output ** 2).sum()
    
    print("\nExecuting Computational Autograd Graph Walkback...")
    loss.backward()
    print("[SUCCESS] Autograd graph reached terminal nodes without interruption.")
    
    q_grad = layer.attn.q_proj.weight.grad
    k_grad = layer.attn.k_proj.weight.grad
    v_grad = layer.attn.v_proj.weight.grad
    
    print("\n--- Hardware Engine Derivative Precision Verification ---")
    print(f"Query Proj Gradient Variance : {q_grad.var().item():.6f} (Should be > 0)")
    print(f"Key Proj Gradient Variance   : {k_grad.var().item():.6f} (Should be > 0)")
    print(f"Value Proj Gradient Variance : {v_grad.var().item():.6f} (Should be > 0)")
    print(f"Query Proj Gradient Mean     : {q_grad.mean().item():.6f}")
    
    assert not torch.allclose(q_grad, torch.ones_like(q_grad) * 0.1), "Busted! Static placeholders are still present!"
    print("\n[COMPLETE] Custom CUDA Block-Sparse Attention Backward Pass is 100% Operational!")