# 🧠 Deep Vision Demo (Transfer Learning & Explainability)

> **Officer Kael's Log**: "The neural net works. But I need to know *what* it is looking at. Is it identifying the ship because of the hull, or the stars behind it?"

This project demonstrates **Modern Computer Vision** workflows using **PyTorch**.
It implements:
1.  **Transfer Learning**: Fine-tuning a pre-trained **ResNet18** on a custom dataset.
2.  **Explainable AI (XAI)**: Using **Grad-CAM** (Gradient-weighted Class Activation Mapping) to visualize heatmaps of model attention.

## 📂 Structure
*   `train.py`: Downloads ResNet18, replaces the head, and fine-tunes it.
*   `explain.py`: Loads an image, runs the model, and overlays a heatmap showing "hot" regions.

## 🚀 Quick Start

### 1. Install
```bash
pip install torch torchvision matplotlib numpy opencv-python
```

### 2. Run Training (Mock Data)
```bash
python train.py
# Downloads ResNet18 and trains on fake random images for demo.
```

### 3. Run Explainability
```bash
python explain.py
# Generates 'heatmap.png' showing where the model looked.
```

## 🧩 Key Concepts
*   **Transfer Learning**: We freeze the "Feature Extractor" (Layers 1-4) and only train the "Classifier" (Linear Layer). This allows training on small datasets.
*   **Grad-CAM**: We take the gradients of the target class (e.g., "Cat") flowing into the final Convolutional Layer. These gradients tell us which pixels were "important" for that decision.
