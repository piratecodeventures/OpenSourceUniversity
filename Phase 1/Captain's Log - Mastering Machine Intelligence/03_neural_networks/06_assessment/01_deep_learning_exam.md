# Stage 2 Assessment: Deep Learning Mastery

## 🎓 The Exam

> **Officer Kael**: "You've built the engine (Core ML). You've built the brain (Neural Nets). You've scaled it to the stars (Distributed).
> Now, show me you can fly it."

This assessment covers **Phases 10, 11, and 11.5**.
Answer these without looking at the textbook.

---

## 📝 Part 1: Theory & Concepts (Interview Prep)

### Section A: The Basics
1.  **Backprop**: Explain Backpropagation to a 5-year-old.
2.  **Activation**: Why do we use ReLU instead of Sigmoid in deep networks?
3.  **Bias**: What happens to a neuron's decision boundary if you remove the Bias term?
4.  **Loss**: When would you use `BCEWithLogitsLoss` vs `CrossEntropyLoss`?

### Section B: Architectures (CNN/RNN/Transfomer)
5.  **CNN**: Why is a 1x1 Convolution useful?
6.  **Pooling**: Does Max Pooling provide Translation Invariance? Explain.
7.  **RNN**: Why do Vanishing Gradients affect RNNs more than CNNs?
8.  **Transformer**: What is the complexity of Self-Attention with respect to sequence length $N$? ($O(N), O(N^2), O(N^3)$?)
9.  **Positional Encoding**: Why do Transformers need it?

### Section C: Advanced & Scaling
10. **Batch Norm**: Does Batch Norm increase or decrease training time per epoch? What about total convergence time?
11. **Dropout**: If I use Dropout(0.5) during training, what must I do to the weights during testing?
12. **DDP**: In Distributed Data Parallel, what exactly is communicated between GPUs? (Gradients? Weights? Data?)
13. **Quantization**: Why does INT8 quantization speed up inference?
14. **GNN**: What is the "Oversmoothing" problem in Graph Neural Networks?

### Section D: Debugging (The "Real World")
15. **Scenario**: Your Loss is `NaN` after Batch 0. What are 3 possible reasons?
16. **Scenario**: Training Accuracy is 99%, Test Accuracy is 50%. What is this called, and name 3 fixes.
17. **Scenario**: You are fine-tuning ResNet. The GPU runs OOM (Out of Memory). Name 2 ways to fix this without changing the model architecture.

---

## 💻 Part 2: Coding Challenges

### Challenge 1: The Custom Layer
**Task**: Implement a **Swish Activation** function (`x * sigmoid(x)`) as a custom PyTorch `nn.Module`.
**Requirement**:
*   Must be differentiable.
*   Must work on GPU.

### Challenge 2: The Data Loader
**Task**: Write a PyTorch `Dataset` class for a folder of images.
**Requirement**:
*   Load image using PIL.
*   Apply random `HorizontalFlip`.
*   Normalize to mean=[0.485, ...] (ImageNet standards).
*   Return `(image_tensor, label_int)`.

### Challenge 3: The Trainer
**Task**: Write a generic `train_one_epoch` function.
**Requirement**:
*   Arguments: `model, loader, optimizer, loss_fn, device`.
*   Must handle `optimizer.zero_grad()`, `loss.backward()`, `optimizer.step()`.
*   Return average loss.

---

## 🛠️ Part 3: System Design

### Design 1: The YouTube Recommender
**Prompt**: "Design a Deep Learning system to recommend videos to users."
**Expectations**:
*   **Candidate Generation**: Two-Tower Model (User Tower / Video Tower)?
*   **Features**: Watch history (RNN/Transformer), Video Thumbnails (CNN).
*   **Scaling**: How to handle 1 Billion videos? (Approximate Nearest Neighbors).

---

## 🔑 Answer Key (Brief)

*   **Q2**: ReLU derivative is 1 (doesn't vanish). Sigmoid max derivative is 0.25 (vanishes).
*   **Q5**: Dimensionality reduction (change channel count) and adding non-linearity.
*   **Q8**: $O(N^2)$.
*   **Q10**: Increases time per epoch (overhead), but drastically decreases total convergence time (fewer epochs).
*   **Q12**: Gradients are averaged.
*   **Q15**: High Learning Rate, Exploding Gradient (No Clipping), Bad Data (NaN inputs).

---

## 💻 Part 4: Code Solutions (The Ship's Log)

### Challenge 1: Custom Swish (PyTorch)
```python
import torch
import torch.nn as nn

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)
```

### Challenge 2: Dataset (PyTorch)
```python
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
import os

class ImageFolderDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.files = os.listdir(root_dir)
        self.transform = T.Compose([
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def __len__(self):
        return len(self.files)
        
    def __getitem__(self, idx):
        path = os.path.join(self.root_dir, self.files[idx])
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        return img, 0 # Dummy label
```

### Challenge 3: Trainer (PyTorch)
```python
def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        out = model(x)
        loss = loss_fn(out, y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / len(loader)
```

---

> **Next Step**: Once you pass this, you are ready for **Stage 3: NLP**.
