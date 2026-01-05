# Linear Algebra: Vector Spaces and Basis

## 📜 Story Mode: The Universal Translator

> **Mission Date**: 2042.04.05
> **Location**: Deep Space Outpost "Vector Prime" - Comms Array
> **Officer**: Lead Engineer Kael
>
> **The Problem**: We are receiving a signal from an alien probe. They send coordinates like `[3, 5]`. We fly there, but find nothing.
>
> **The Realization**: We assumed their `[1, 0]` means "1 km North" and `[0, 1]` means "1 km East".
> But what if *their* `[1, 0]` means "1 km Northeast" and `[0, 1]` means "1 km Up"?
>
> We don't share the same **Basis**. We are traversing the same **Vector Space** (physical space), but we are using different maps (grid lines/coordinate systems) to describe it.
>
> **The Solution**: We need to perform a **Basis Change**. We need to express their basis vectors in terms of ours. Then we can translate every coordinate `[3, 5]` into our language.
>
> *"Computer! Analyze signal periodicity. Reconstruct their basis vectors. Construct the Transition Matrix!"*

---

## 1. Problem Setup & Motivation

### The 6 Engineering Questions
1.  **WHAT**: A **Vector Space** ($\mathbb{R}^n$) is the entire playground where vectors live. A **Basis** is a minimal set of vectors that can reach *any* point in that playground via scaling and adding (Linear Combination).
2.  **WHY**: Data compression. If we can find a "better basis" (like Fourier or Wavelet), we can describe complex signals with just a few numbers (coefficients).
3.  **WHEN**: Compression (JPEG), Signal Processing (FFT), Feature Extraction.
4.  **WHERE**:
    *   **Math**: Span, Independence, Basis.
    *   **Hardware**: Frequency Domain procession in DSP chips.
5.  **WHO**: Compression Engineers, Audio Engineers.
6.  **HOW**: Using **Change of Basis** matrices.

> [!NOTE]
> **🛑 Pause & Explain**
>
> **What is a "Basis"?**
>
> Imagine you have Legos.
> - **Span**: Anything you can build with those blocks.
> - **Basis**: The smallest set of block types you need to build *everything* in the set.
>
> If you have Red, Blue, and Green blocks, but Green can be made by mixing Red and Blue paint... then Green is redundant. It's **Dependent**. The Basis is just {Red, Blue}.

---

## 2. Mathematical Problem Formulation

### Definitions & Axioms
A set of vectors $\{ \mathbf{v}_1, \dots, \mathbf{v}_k \}$ is a **Basis** for space $V$ if:
1.  **Linear Independence**: No vector is a combination of the others.
2.  **Span**: Any vector in $V$ can be written as $c_1\mathbf{v}_1 + \dots + c_k\mathbf{v}_k$.

### Dimension
The **Dimension** is simply the number of vectors in the basis.
*   Line: 1 vector.
*   Plane: 2 vectors.
*   This Room: 3 vectors.

---

## 3. Step-by-Step Derivation

### Changing Your Perspective (Basis Change)
**Standard Basis (Ours)**: $\mathbf{e}_1 = [1, 0]$, $\mathbf{e}_2 = [0, 1]$.
**Alien Basis (Theirs)**: $\mathbf{u}_1 = [1, 1]$, $\mathbf{u}_2 = [-1, 1]$.

**Task**: The Alien says "Go `[2, 3]`". Where is that in our map?
Alien Vector $\mathbf{v}_{alien} = 2\mathbf{u}_1 + 3\mathbf{u}_2$.

**Calculation**:
Substitute $\mathbf{u}$ vectors:
$\mathbf{v}_{ours} = 2 \begin{bmatrix} 1 \\ 1 \end{bmatrix} + 3 \begin{bmatrix} -1 \\ 1 \end{bmatrix}$
$\mathbf{v}_{ours} = \begin{bmatrix} 2 \\ 2 \end{bmatrix} + \begin{bmatrix} -3 \\ 3 \end{bmatrix} = \begin{bmatrix} -1 \\ 5 \end{bmatrix}$.

**Result**: We fly to $[-1, 5]$.

---

## 4. Algorithm Construction

### Gram-Schmidt Process (Orthogonalization)
Problem: A random basis is messy (vectors assume weird angles). We prefer a **clean, perpendicular (Orthogonal)** basis.
**Algorithm**:
1.  Take first vector $\mathbf{v}_1$. Keep it.
2.  Take $\mathbf{v}_2$. Remove the "shadow" of $\mathbf{v}_1$ from it.
    *   $\mathbf{u}_2 = \mathbf{v}_2 - \text{proj}_{\mathbf{v}_1}(\mathbf{v}_2)$
3.  Repeat. All output vectors will be at $90^\circ$ to each other.

---

## 5. Worked Examples

### Example 1: One-Hot vs Embeddings
**Story**: Representing Words.
**One-Hot Basis**: 50,000 dimensions (Dictionary size).
*   "Cat": $[0, 1, 0, \dots]$
*   "Dog": $[0, 0, 1, \dots]$
*   Problem: These basis vectors are orthogonal. "Cat" and "Dog" share no relationship direction.

**Embedding Basis (Word2Vec)**: 300 dimensions.
*   "Cat" and "Dog" point in similar directions in this smaller, dense vector space.
*   We changed basis from "ID Space" to "Semantic Space".

---

## 6. Production-Grade Code

### The Ship's Code (Polyglot: Python vs NumPy vs PyTorch vs TensorFlow)

```python
import numpy as np
import torch
import tensorflow as tf

# LEVEL 0: Pure Python (The Gram-Schmidt Logic)
# Project v onto u: (v . u / u . u) * u
def project_pure(v, u):
    dot_Vu = sum(x*y for x, y in zip(v, u))
    dot_Uu = sum(y*y for y in u)
    scale = dot_Vu / dot_Uu
    return [scale * y for y in u]

def subtract_pure(v, u):
    return [x - y for x, y in zip(v, u)]

# LEVEL 1: NumPy (Basis Check)
def check_independence_numpy(matrix_vectors):
    # If Determinant != 0, they are independent (if square)
    # Or check Rank
    rank = np.linalg.matrix_rank(matrix_vectors)
    return rank == len(matrix_vectors)

# LEVEL 2: PyTorch (Manifold Learning)
def learn_new_basis_torch(data, new_dim=10):
    # This is essentially an Autoencoder or PCA
    # We learn a matrix W that maps input -> hidden basis
    layer = torch.nn.Linear(data.shape[1], new_dim)
    return layer(data)

# LEVEL 3: TensorFlow (Signal Basis - FFT)
def to_frequency_basis_tf(signal):
    # Changes basis from Time Domain to Frequency Domain
    return tf.signal.fft(tf.cast(signal, tf.complex64))
    # Changes basis from Time Domain to Frequency Domain
    return tf.signal.fft(tf.cast(signal, tf.complex64))

# LEVEL 4: Visualization (The Basis Grid)
def visualize_basis():
    """
    Shows how different basis vectors span the space.
    """
    import matplotlib.pyplot as plt
    
    # 1. Standard Basis [[1,0], [0,1]]
    origin = np.array([0, 0])
    e1 = np.array([1, 0])
    e2 = np.array([0, 1])
    
    # 2. Skewed Basis [[1,1], [-1,1]]
    u1 = np.array([1, 1])
    u2 = np.array([-1, 1])
    
    plt.figure(figsize=(10, 5))
    
    # Plot Standard
    plt.subplot(1, 2, 1)
    plt.quiver(*origin, *e1, color='r', scale=5, label='e1')
    plt.quiver(*origin, *e2, color='b', scale=5, label='e2')
    plt.grid(True)
    plt.title("Standard Basis (Grid)")
    
    # Plot Skewed
    plt.subplot(1, 2, 2)
    plt.quiver(*origin, *u1, color='orange', scale=5, label='u1')
    plt.quiver(*origin, *u2, color='purple', scale=5, label='u2')
    plt.grid(True)
    plt.title("Alien Basis (Skewed Grid)")
    
    plt.show()

---

## 7. System-Level Integration

```mermaid
graph TD
    Input[High-Dim Input (Images)] --> Encoder[Change of Basis (Encoder)]
    Encoder --> Latent[Latent Space (Compressed Basis)]
    Latent --> Decoder[Change of Basis (Decoder)]
    Decoder --> Output[Reconstructed Image]
```

---

## 8. Evaluation & Failure Analysis

### Failure Mode: Rank Collapse (Mode Collapse)
In GANs, if the generator maps all inputs to a low-dimensional subspace (the "Span" shrinks), it can only produce identical or very similar images. The "Basis" of the generator has collapsed.

---

## 9. Ethics, Safety & Risk Analysis

### The Basis of Bias
If your basis for "Resume Suitability" is derived from historical data, and historical data used "Name" or "Gender" as a strong signal, that bias becomes a fundamental axis of your space.
**Debiasing**: We technically perform a projection to remove the "Gender Axis" from the embedding space, making vectors orthogonal to it.

---

## 10. Advanced Theory & Research Depth

### Hilbert Spaces
Infinite-dimensional vector spaces. This is where Quantum Mechanics and Advanced Kernel Methods live.

### 📚 Deep Dive Resources
*   **Paper**: "Man is to Computer Programmer as Woman is to Homemaker? Debiasing Word Embeddings" (Bolukbasi et al., 2016) - Explicitly manipulating the basis to enforce fairness. [ArXiv:1607.06520](https://arxiv.org/abs/1607.06520)

---

## 11. Career & Mastery Signals

### Cadet (Junior)
*   Understands coordinates are relative to a frame of reference.

### Commander (Senior)
*   Thinks in terms of "Manifolds" (curved spaces) rather than flat vector spaces.
*   Can implement Gram-Schmidt stability fixes.

---

## 12. Industry Interview Corner

### ❓ Real World Questions
**Q1: "What guarantees that a set of vectors spans a full space?"**
*   **Answer**: "If you have $N$ vectors in $N$ dimensions, they Span the space IF AND ONLY IF they are Linearly Independent (Determinant $\neq$ 0)."

**Q2: "Why do we use Fourier Transforms in Convolutional Networks?"**
*   **Answer**: "Convolution in the Spatial Domain is equivalent to Multiplication in the Frequency Basis (Fourier Domain). Sometimes it's faster to compute there."

**Q3: "Explain Rank of a Matrix."**
*   **Answer**: "The Rank is the dimension of the actual information. A 100x100 matrix might only have Rank 2, meaning it effectively flattens all data onto a 2D sheet. It's the number of truly independent columns."

---

## 13. Debug Your Thinking (Common Misconceptions)

### ❌ Myth: "The origin (0,0) moves."
**✅ Truth**: In a Vector Space, the **Origin Must Be Fixed**. If you move the zero point, it's an Affine Space, not a Vector Space. You lose the ability to satisfy $0 \cdot \mathbf{v} = \mathbf{0}$.

### ❌ Myth: "Basis vectors must be orthogonal (90 degrees)."
**✅ Truth**: No. Orthogonal is *nice* (Orthonormal Basis), but not required. Any independent set works. (Though computationally, orthogonal is much more stable).

---

## 14. Assessment & Mastery Checks

**Q1: Independence**
Are $[1, 0]$ and $[2, 0]$ independent?
*   *Answer*: No. $2 \cdot [1, 0] = [2, 0]$. They lie on the same line.

---

## 15. Further Reading & Tooling
*   **Tool**: **Lasso Regression (L1)** - A technique that forces models to find a "Sparse Basis" (choosing only the most important features).

---

## 16. Concept Graph Integration
*   **Previous**: [Linear Transformations](01_foundation_math_cs/01_linear_algebra/05_linear_transformations.md).
*   **Next**: [SVD](01_foundation_math_cs/01_linear_algebra/07_svd.md) (Finding the 'Best' Basis).

### Concept Map
```mermaid
graph TD
    Space[Vector Space V] --> Basis[Basis Vectors]
    Basis --> Span[Span (The Grid)]
    Basis --> Dim[Dimension]
    
    Basis --> Ortho[Orthogonal Basis]
    Ortho --> Gram[Gram-Schmidt]
    
    Space --> Sub[Subspace]
    Sub --> Rank[Rank]
    Sub --> Null[Null Space]
    
    style Space fill:#f9f,stroke:#333
    style Basis fill:#bbf,stroke:#333
```
