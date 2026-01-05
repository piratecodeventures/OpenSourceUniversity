# GenAI: Advanced GANs (The Forger)

## 📜 Story Mode: The Masterpiece

> **Mission Date**: 2043.10.20
>
> **The Problem**: My GAN faces look like blobs.
> I need 4K resolution. I need to change "Hair Color" without changing "Face Shape".
>
> **The Solution**: **StyleGAN**.
> I will inject "Style" at every layer.
> Coarse Style (Pose). Fine Style (Hair pores).
>
> *"Computer! Map Latent Z to Style W. Modulate Synthesis Network."*

---

## 1. Problem Setup

### Key Architectures
1.  **StyleGAN (NVIDIA)**: Uses Adaptive Instance Normalization (AdaIN) to control style at different scales.
2.  **CycleGAN**: Unpaired translation. Horse $\to$ Zebra without (Horse, Zebra) pairs.
    *   *Cycle Consistency*: $F(G(x)) \approx x$. (Translate English $\to$ French $\to$ English, should match).

---

## 2. Mathematical Formulation

### Cycle Consistency Loss
$$ L_{cyc}(G, F) = \mathbb{E}[||F(G(x)) - x||_1] + \mathbb{E}[||G(F(y)) - y||_1] $$
Ensures the mapping is reversible (Structure preserving).

---

## 3. The Ship's Code (Polyglot)

```python
import torch
import torch.nn as nn

# LEVEL 2: PyTorch (CycleGAN Generator Block - Residual)
class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3),
            nn.InstanceNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3),
            nn.InstanceNorm2d(dim)
        )

    def forward(self, x):
        return x + self.conv(x)
        return x + self.conv(x)
```

> [!TIP]
> **👁️ Visualizing the Cycle: A -> B -> A**
> Run this script to see the concept of Cycle Consistency (Reconstruction).
>
> ```python
> import matplotlib.pyplot as plt
> import numpy as np
>
> def plot_cycle_consistency():
>     # 1. Original Data (Domain A - Horses)
>     # Represented as a Sine Wave
>     x = np.linspace(0, 10, 100)
>     data_A = np.sin(x)
>     
>     # 2. Transform to Domain B (Zebras) = G(A)
>     # Fake transform: Invert it
>     data_B_fake = -data_A 
>     
>     # 3. Reconstruct Domain A = F(B)
>     # Ideally equal to A
>     data_A_recon = -data_B_fake
>     
>     # 4. Failed Cycle (Mode Collapse / Bad training)
>     data_A_bad = data_A * 0.5 + 0.2
>     
>     # Plot
>     fig, axes = plt.subplots(1, 3, figsize=(15, 4))
>     
>     axes[0].plot(x, data_A, 'b-', linewidth=3)
>     axes[0].set_title("Domain A (Original)\nExample: Horse")
>     axes[0].set_ylim(-1.5, 1.5)
>     
>     axes[1].plot(x, data_B_fake, 'r--', linewidth=3)
>     axes[1].set_title("Domain B (Glossy Translation)\nExample: Zebra")
>     axes[1].set_ylim(-1.5, 1.5)
>     
>     axes[2].plot(x, data_A, 'b-', alpha=0.3, linewidth=5, label='Original')
>     axes[2].plot(x, data_A_recon, 'g--', label='Good Cycle (Loss=0)')
>     axes[2].plot(x, data_A_bad, 'r:', label='Bad Cycle (High Loss)')
>     axes[2].set_title("Reconstruction (A -> B -> A)\nCycle Consistency Loss")
>     axes[2].legend()
>     axes[2].set_ylim(-1.5, 1.5)
>     
>     plt.show()
>     
> # Uncomment to run:
> # plot_cycle_consistency()
> ```

---

## 13. Assessment & Mastery Checks

**Q1: Style Mixing**
How does StyleGAN mix faces?
*   *Answer*: It uses Latent Code $w_1$ for coarse layers (Pose) and $w_2$ for fine layers (Colors). Result: Person A's pose, Person B's colors.

### 14. Common Misconceptions

> [!WARNING]
> **"CycleGAN creates new information."**
> *   **Correction**: It hallucinates textures. It cannot turn an Apple into an Orange if the geometric shape is too different.

### Concept Map
```mermaid
graph TD
    DataA[Domain A (Horse)] -- "Generator G" --> FakeB[Fake Zebra]
    FakeB -- "Generator F" --> ReconA[Reconstructed Horse]
    
    DataA -- "Compare" --> ReconA
    
    DataB[Domain B (Zebra)] -- "Generator F" --> FakeA[Fake Horse]
    FakeA -- "Generator G" --> ReconB[Reconstructed Zebra]
    
    DataB -- "Compare" --> ReconB
    
    Loss[Total Loss] --> AdvLoss[Adversarial (Fool D)]
    Loss --> CycLoss[Cycle (Reconstruction)]
    Loss --> Identity[Identity (Color Preservation)]
    
    style DataA fill:#f9f,stroke:#333
    style FakeB fill:#bbf,stroke:#333
    style ReconA fill:#bfb,stroke:#333
```
