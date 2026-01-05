# 🛠️ The ML Diagnostics Toolkit (Stage 1 Project)

> **Officer Kael's Log**: "Theory is useless if you can't debug it. This toolkit visualizes the invisible math behind our engines."

This project serves as the Capstone for **Stage 1: Foundation Math & CS**. It implements 4 core modules from scratch.

## 📦 Modules

### 1. `gradient_descent_viz.py`
**The Optimizer**. A real-time visualization of Gradient Descent navigating a 2D loss surface.
*   **Features**: Switch between SGD, Momentum, and Adam. Adjustable Learning Rate.
*   **Math**: Visualizes $w_{new} = w_{old} - \eta \nabla L$.

### 2. `bayesian_classifier.py`
**The Probabilist**. A Naive Bayes text classifier built from first principles (no `sklearn`).
*   **Features**: Spams filtering logic using Log-Likelihoods to avoid underflow.
*   **Math**: $P(Spam|Word) \propto P(Word|Spam) P(Spam)$.

### 3. `svd_viz.py`
**The Compressor**. An interactive demo of Image Compression using Singular Value Decomposition.
*   **Features**: Slider to control 'k' (Rank). Shows memory savings vs quality loss.
*   **Math**: $A \approx U \Sigma_k V^T$.

### 4. `sorting_benchmark.py`
**The Engineer**. A benchmarking suite comparing QuickSort, MergeSort, and Python's Timsort on ML-style arrays.
*   **Features**: Plots Time vs N. Handles "Almost Sorted" data (common in Time Series).

## 🚀 How to Run

Requirements:
```bash
pip install numpy matplotlib scipy
```

Run a module:
```bash
python gradient_descent_viz.py
```
