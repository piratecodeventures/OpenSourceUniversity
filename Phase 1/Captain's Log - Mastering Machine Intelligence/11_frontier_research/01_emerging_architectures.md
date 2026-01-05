# emerging_architectures: Mamba, Liquid & S4 (Deep Dive)

## 📜 Story Mode: The Avant-Garde

> **Mission Date**: 2045.01.01
> **Location**: Deep Mind Research
> **Officer**: Kernel Architect
>
> **The Problem**: Transformers are $O(N^2)$.
> If I want to read a gene sequence (1M tokens), Transformers crash.
> RNNs are $O(N)$ but can't parallelize training.
> We need the speed of RNNs with the power of Attention.
>
> **The Solution**: **State Space Models (SSMs)**.
> Mamba. S4. Liquid Neural Networks.
> "Forget Attention. Use Selective State Spaces."
>
> *"Computer. Compile Selective Scan Kernel. Initialize Mamba Block."*

---

## 1. Problem Setup & Motivation

### The 6 Engineering Questions
1.  **WHAT**: Novel architectures that challenge the Transformer hegemony.
2.  **WHY**: Quadratic Scaling ($N^2$) is unsustainable for Long Context (DNA, Books, Code).
3.  **WHEN**: Sequence Length > 10k tokens. Edge Devices (Fast Inference).
4.  **WHERE**: `Mamba-Chat`, `Hyena`, `RWKV`.
5.  **WHO**: Albert Gu (Mamba), Ramin Hasani (Liquid NNs).
6.  **HOW**: Continuous Time Control Theory discretized for Deep Learning.

---

## 2. Mathematical Deep Dive: State Space Models (S4)

### 2.1 The Continuous System
$$ h'(t) = A h(t) + B x(t) $$
$$ y(t) = C h(t) $$
*   Maps input $x(t)$ to output $y(t)$ via latent state $h(t)$.
*   This is a **Convolution** in disguise! $y = x * K$.

### 2.2 The Selection Mechanism (Mamba)
Standard SSMs have time-invariant $A, B, C$.
**Mamba** makes them input-dependent:
$$ B = B(x_t), C = C(x_t), \Delta = \Delta(x_t) $$
*   This allows the model to "Select" what to remember and what to forget (Content-Aware).
*   **Result**: Linear Time Inference ($O(1)$ per step like RNN), Parallel Training ($O(N)$ via Scan).

---

## 3. The Ship's Code (Polyglot: Mamba Block)

```python
import torch
import torch.nn as nn
from mamba_ssm import Mamba

# LEVEL 2: Stacking Mamba Blocks
class MambaModel(nn.Module):
    def __init__(self, d_model, n_layers, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Stack of Mamba Layers
        self.layers = nn.ModuleList([
            Mamba(
                d_model=d_model, # Model dimension (D)
                d_state=16,      # SSM state expansion factor (N)
                d_conv=4,        # Local convolution width
                expand=2,        # Block expansion factor
            ) for _ in range(n_layers)
        ])
        
        self.norm_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm_f(x)
        return self.lm_head(x)
```

---

## 4. System Architecture: The Selective Scan

```mermaid
graph LR
    Input --> |Projection| B_C_Delta
    Input --> |Conv1d| ConvOut
    
    B_C_Delta & ConvOut --> |Selective Scan (Parallel)| SSM_State
    
    SSM_State --> |Gating| Output
```

**Hardware Aware**: The Scan operation must be implemented in **CUDA** to be fast. It minimizes HBM (Memory) reads, similar to FlashAttention.

---

## 13. Industry Interview Corner

### ❓ Real World Questions

**Q1: "Why is Mamba faster than Transformer at Inference?"**
*   **Answer**: "Transformer is **KV-Cache Bound**. It must read the entire history matrix for every new token ($O(N)$ per step). Mamba compresses history into a fixed-size state $h_t$ (just like an RNN). Inference is constant time $O(1)$ and uses constant memory."

**Q2: "What is the disadvantage of SSMs?"**
*   **Answer**: "In-Context Learning (Copying). Transformers are 'associative memories'—they can look back perfectly at a phone number 500 tokens ago. SSMs must compress that state, so they sometimes 'forget' exact details in long contexts compared to Attention."

---

## 14. Debug Your Thinking (Misconceptions)

> [!WARNING]
> **"RNNs are dead."**
> *   **Correction**: **LSTM** is dead. **Recurrence** is back. Mamba is essentially a "Super RNN" that can train in parallel.

> [!WARNING]
> **"Mamba replaces Transformers."**
> *   **Correction**: Not yet. **Hybrid Models** (Jamba: Mamba + Attention layers) are the current SOTA. You use Mamba for throughput and Attention for "Needle in a Haystack" recall.
