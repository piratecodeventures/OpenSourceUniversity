# research_methods: Reproducibility & Publication (Deep Dive)

## 📜 Story Mode: The Scientist

> **Mission Date**: 2044.02.15
> **Location**: Research Archives
> **Officer**: Lead Reviewer
>
> **The Problem**: I downloaded the paper "SOTA ImageNet Model".
> I ran the code.
> It breaks.
> I fixed the code. The accuracy is 10% lower than reported.
> **Science is broken.**
>
> **The Solution**: **Reproducible Research**.
> Docker Containers. Seed Setting. Artifact Versioning.
> "Code is not enough. The Environment is the Code."
>
> *"Computer. Build Docker Image. Pull Weights Hash da32f. Run Evaluation."*

---

## 1. Problem Setup & Motivation

### The 6 Engineering Questions
1.  **WHAT**: The discipline of making scientific results technically reproducible by others.
2.  **WHY**: "Crisis of Reproducibility". Progress stops if we can't build on each other.
3.  **WHEN**: Writing Papers, releasing Open Source, handoff to Production.
4.  **WHERE**: `Docker`, `Hydra` (Config), `WandB` (Tracking), `PyTorch Lightning`.
5.  **WHO**: Joelle Pineau (Reproducibility Checklist), Papers With Code.
6.  **HOW**: Config Management $\to$ Containerization $\to$ Deterministic Algorithms.

---

## 2. Technical Deep Dive: Determinism

### 2.1 The Sources of Randomness
1.  **Data Loading**: Shuffling order.
2.  **Initialization**: Random weights ($N(0, 1)$).
3.  **CUDA**: Non-deterministic algorithms (Atomic Adds in float summation). Data races in parallel ops.
4.  **Environments**: Python 3.8 vs 3.9 sets dictionary order differently.

### 2.2 The Fix
```python
import torch
import numpy as np
import random
import os

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    # Critical for CUDA
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

---

## 3. The Ship's Code (Polyglot: Hydra Configuration)

```yaml
# config/config.yaml
defaults:
  - model: resnet
  - dataset: cifar10
  - optimizer: adam

hyperparameters:
  lr: 0.001
  batch_size: 64
  seed: 42
```

```python
import hydra
from omegaconf import DictConfig, OmegaConf

# LEVEL 2: Structured Experiments
@hydra.main(config_path="config", config_name="config")
def train(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    
    # 1. Init with Config
    seed_everything(cfg.hyperparameters.seed)
    model = instantiate(cfg.model)
    
    # 2. Log Config to WandB (The Evidence)
    wandb.init(config=cfg)
    
    # 3. Train...
    
if __name__ == "__main__":
    train()
```

---

## 4. System Architecture: The Research Repo

A "Gold Standard" repository structure:

```text
project/
├── configs/            # Hydra YAMLs
├── docker/             # Dockerfile
├── notebooks/          # Exploratory (Not for prod)
├── src/
│   ├── models/         # Architecture
│   ├── dataloaders/    # Pipelines
│   └── trainer.py      # Loop
├── tests/              # Pytest
├── requirements.txt    # Pinned versions (pip freeze)
└── README.md           # "How to run"
```

---

## 13. Industry Interview Corner

### ❓ Real World Questions

**Q1: "Why is `torch.backends.cudnn.benchmark = True` used?"**
*   **Answer**: "It auto-tunes convolution algorithms for your specific hardware. It makes training faster but **Non-Deterministic**. For research comparison, turn it OFF. For production training, turn it ON."

**Q2: "What is an Ablation Study?"**
*   **Answer**: "Removing one component at a time to prove its value. (e.g., 'Model without Attention', 'Model without Data Augmentation'). Proves that *your* contribution actually caused the improvement, not just hyperparameter tuning."

---

## 14. Debug Your Thinking (Misconceptions)

> [!WARNING]
> **"I linked the GitHub, so it's reproducible."**
> *   **Correction**: Did you include the `requirements.txt`? Did you link the specific **Git Commit Hash**? Did you host the Weights? Link Rot destroys half of ML papers in 5 years.

> [!WARNING]
> **"p-value < 0.05 means my model is better."**
> *   **Correction**: Statistical Significance testing in DL is rare but necessary. Running 1 seed vs 1 seed proves nothing. Run **5 seeds** and report Mean $\pm$ Std Dev. Interaction is only signficant if error bars don't overlap.
