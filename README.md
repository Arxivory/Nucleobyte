# NucleoByte: GPU-Accelerated Block-Sparse Genomic Language Model

NucleoByte is a high-performance, custom-engineered Transformer architecture built from scratch for genomic sequence modeling. It bypasses standard deep learning overhead by routing zero-copy, VRAM-resident tensor operations directly from PyTorch into native C++/CUDA hardware kernels.

---

## 1. The Core Problem

Standard Natural Language Processing (NLP) models face major scaling bottlenecks when applied to genomics:

* **The Context Trap:** DNA sequences are continuous, massive strings (often millions of base pairs) rather than brief, punctuated sentences. Standard dense $O(N^2)$ attention scaling becomes mathematically and financially prohibitive at scale.
* **Padding Waste:** Typical tokenizers or data collators force variable-length sequences to conform to fixed global lengths, allocating massive amounts of VRAM to zero-padding tokens that compute useless context, waste hardware cycles, and corrupt embedding tables with gradient noise.
* **Token Blur:** Biological representations (like $K$-mers) require massive vocabularies if computed dynamically via standard BPE. A hard-coded, bit-packed sliding window approach is required for native DNA parsing.

---

## 2. The Architectural Goal

NucleoByte optimizes genomic sequence training through a highly specialized pipeline:

1. **Fused 2-Bit Tokenization:** Native GPU string parsing that packs DNA bases into memory-efficient integer IDs on the fly.
2. **Dynamic Ceiling Hardware Alignment:** A custom data pipeline that dynamically pads batches up to the nearest multiple of the hardware block size ($32$), completely wiping out artificial padding overhead.
3. **Local Hybrid Block-Sparse Attention:** Restricting the model's receptive field to a tightly bounded local sliding neighborhood ($W$) to drop attention time and memory complexity from quadratic $O(N^2)$ to linear $O(N \cdot W)$.

---

## 3. Mathematical Foundations

### Localized Block-Sparse Attention Matrix

Instead of calculating the dense dot-product between all Queries ($Q$) and Keys ($K$) across the full sequence $N$, attention computation is structurally constrained using a window block radius $R$. For a block size $B = 32$, a token block index $i$ only computes attention scores against key block indices $j$ that satisfy:

$$|i - j| \le R$$

```
Dense Attention O(N²)           NucleoByte Block-Sparse O(N * W)
   [X X X X X X X X]                   [B B B . . . . .]  <- Block-Row i=0
   [X X X X X X X X]                   [B B B B . . . .]
   [X X X X X X X X]                   [B B B B B . . .]
   [X X X X X X X X]       ====>       [. B B B B B . .]  Window Radius (R) = 1
   [X X X X X X X X]                   [. . B B B B B .]  Block Size (B) = 32
   [X X X X X X X X]                   [. . . B B B B B]
   [X X X X X X X X]                   [. . . . B B B B]
   [X X X X X X X X]                   [. . . . . B B B]

```

### Softmax Normalization & Online Numerical Stability

To maintain absolute stability without standard PyTorch tracking arrays, the forward kernel handles FlashAttention-inspired online softmax scaling. For row block $i$ and column block $j$, the raw attention weights $S_{ij}$ are scaled by the dimension factor:

$$S_{ij} = \frac{Q_i K_j^T}{\sqrt{d_k}}$$

To prevent floating-point explosion during exponentiation ($\text{expf}$), the running maximum $m_i$, log-sum-exp tracking scale $d_i$, and running output aggregation $O_i$ are accumulated dynamically across active block boundaries:

$$m_i^{\text{new}} = \max(m_i^{\text{old}}, \max(S_{ij}))$$

$$d_i^{\text{new}} = d_i^{\text{old}} \cdot e^{(m_i^{\text{old}} - m_i^{\text{new}})} + \sum e^{(S_{ij} - m_i^{\text{new}})}$$

$$O_i^{\text{new}} = O_i^{\text{old}} \cdot e^{(m_i^{\text{old}} - m_i^{\text{new}})} + \sum e^{(S_{ij} - m_i^{\text{new}})} \cdot V_j$$

### Custom Backpropagation

Gradients are calculated inside a manual backpropagation CUDA kernel using `atomicAdd` routines to handle accumulation over shared key-value blocks. Softmax differentiation incorporates a pre-calculated row correction term $D_i$ mapped globally across dimensions:

$$
D_i = \sum_k \left( (\nabla O)_{ik} \cdot O_{ik} \right)
$$

$$
(\nabla S)_{ij} = P_{ij} \cdot \left( (\nabla O)_i V_j^T - D_i \right)
$$

---

## 4. Tech Stack

* **Frontend Framework:** PyTorch (custom `torch.autograd.Function` integration).
* **Acceleration Language:** Native C++17 / CUDA C (Shared Memory tiling, Warp Shuffle primitives, Warp-level reduction).
* **Interoperability Bridge:** `pybind11` (facilitating raw GPU device memory pointer exchange via `uintptr_t` casting).

---

## 5. Temporary Status & Milestone Validation

> **Project Status:** Active Development & Optimization Mode.

The core math engine, forward/backward loops, and hardware interfaces have been fully validated. To prove generalization capabilities under dynamic conditions, a validation run was executed using a shifting-window data load:

* **Corpus Source:** `gencode.v49.transcripts.fa` (Human genomic transcript mapping)
* **Dataset Scale:** 533,740 continuous chromosomal blocks / 266,870 batches per epoch
* **Architecture Size:** 3 Layers, 2 Heads, Embedding Dim 128, $K$-mer Size 6 ($4096$ Vocab + Padding/Mask)
* **Milestone Outcome:** Initialized at a random guess loss boundary of **`8.32`**. **Aborted early at Step 30,355 with training loss successfully reduced to `0.9340`.**

### Evaluation Output Verification

During subsequent Masked Language Modeling (MLM) extraction tests, the model accurately resolved structural context within local boundaries to pinpoint masked target segments:

```text
Original Input DNA Sequence : CCCAGGGTCCGATGGGAAAGTGTAGCCTGC
Masked Token Index Position : 12
Target Ground-Truth 6-mer   : 'TGGGAC' (ID: 3745)

=========================================================
               NUCLEOBYTE INFERENCE PREDICTIONS
=========================================================
Rank 1 | Prediction: 'TGGGAC' (ID: 3745) | Confidence: 10.16% (GROUND TRUTH MATCH!)
Rank 2 | Prediction: 'TGAAGT' (ID: 3595) | Confidence: 5.40%
Rank 3 | Prediction: 'GGCCCC' (ID: 2645) | Confidence: 3.12%

```

---

## 6. Project Directory Setup & Basic Usage

### Compilation

The engine requires an environment with standard PyTorch installation alongside the NVIDIA CUDA Toolkit (including `nvcc`). Build the PyBind core using standard C++ build automation:

```bash
mkdir out && cd out
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release

```

### Script Execution

The underlying layers handle standard PyTorch instantiation pipelines seamlessly:

```python
import torch
from nucleobyte.model import NucleoByteLM

# Initialize the validated hardware configuration 
model = NucleoByteLM(
    vocab_size=4100, 
    embed_dim=128, 
    num_heads=2, 
    num_layers=3, 
    window_radius=1, 
    block_size=32
).cuda()

# Synthetic token batch representing aligned sequence tokens
input_ids = torch.randint(1, 4096, (2, 4096), dtype=torch.long).cuda()

# Forward execution maps down directly into the block-sparse CUDA kernel
logits = model(input_ids)
print("Logits Shape:", logits.shape) # Expected: [2, 4096, 4100]

```
